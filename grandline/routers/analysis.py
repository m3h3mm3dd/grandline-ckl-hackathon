"""
Analysis router — all computed telemetry endpoints.

GET /analysis/{session_id}/laps                       → all lap summaries
GET /analysis/{session_id}/lap/{lap_index}            → full lap detail (frames)
GET /analysis/{session_id}/lap/{lap_index}/braking    → braking zones
GET /analysis/{session_id}/lap/{lap_index}/sectors    → sector breakdown
GET /analysis/{session_id}/lap/{lap_index}/corners    → per-corner analysis
GET /analysis/{session_id}/lap/{lap_index}/gg         → GG diagram data
GET /analysis/{session_id}/lap/{lap_index}/tyres      → tyre temp trend
GET /analysis/{session_id}/lap/{lap_index}/degradation → tyre degradation summary
GET /analysis/{session_id}/lap/{lap_index}/suspension → suspension / ride-height trend
GET /analysis/{session_id}/compare                    → distance-normalised comparison
GET /analysis/{session_id}/delta?lap_a=0&lap_b=1      → ΔT chart data
GET /analysis/{session_id}/theoretical_best           → purple lap
GET /analysis/{session_id}/corner_scoreboard          → all-lap corner table
GET /analysis/{session_id}/track                      → track boundary + corner map
"""

from fastapi import APIRouter, HTTPException, Query
from typing import Optional

from services.session_store import get_session
from services.metrics_engine import (
    compute_gg_diagram,
    detect_braking_zones,
    compute_sectors,
    compute_tyre_trend,
    compute_tyre_degradation,
    compute_suspension_report,
    compare_laps_distance,
    compare_laps,
    compute_delta_time,
    compute_theoretical_best,
)
from services.corner_detector import analyse_corners
from models.schemas import (
    LapSummary, LapDetail, BrakingZone, SectorSummary,
    GGDiagram, CornerAnalysis, LapCornerReport,
    TyreState, TyreDegradation, TrackData,
    DistanceComparison, LapComparison,
    DeltaTimeSeries, TheoreticalBest,
    CornerScoreboard, CornerScoreRow,
    SuspensionReport,
)

router = APIRouter()


# ── helpers ───────────────────────────────────────────────────────────────────

def _require_session(session_id: str):
    s = get_session(session_id)
    if not s:
        raise HTTPException(404, "Session not found")
    if not s._ready.is_set():
        raise HTTPException(202, "Session still processing — try again shortly")
    return s


def _require_lap(s, lap_index: int):
    if lap_index >= len(s.raw_laps):
        raise HTTPException(404, f"Lap {lap_index} not found (session has {len(s.raw_laps)} laps)")
    return s.raw_laps[lap_index]


# ── Lap summaries & detail ────────────────────────────────────────────────────

@router.get("/{session_id}/laps", response_model=list[LapSummary])
def get_laps(session_id: str):
    """Return summary for every lap in the session."""
    s = _require_session(session_id)
    return [d.summary for d in s.lap_details]


@router.get("/{session_id}/lap/{lap_index}", response_model=LapDetail)
def get_lap_detail(session_id: str, lap_index: int):
    """Full lap detail including all telemetry frames."""
    s = _require_session(session_id)
    if lap_index >= len(s.lap_details):
        raise HTTPException(404, "Lap not found")
    return s.lap_details[lap_index]


# ── Lap position path (for track map rendering) ───────────────────────────────

@router.get("/{session_id}/lap/{lap_index}/positions")
def get_lap_positions(
    session_id: str,
    lap_index: int,
    max_pts: int = Query(600, ge=50, le=2000, description="Max points returned"),
):
    """
    Return x, y, speed, throttle, brake for every frame in the lap,
    downsampled to max_pts. Used by the frontend to draw the car path
    colour-coded by speed/throttle, before the real-time stream starts.
    """
    s = _require_session(session_id)
    if lap_index >= len(s.lap_details):
        raise HTTPException(404, "Lap not found")
    frames = s.lap_details[lap_index].frames
    if not frames:
        return []
    # Downsample
    import numpy as np
    idx = np.round(np.linspace(0, len(frames) - 1, min(max_pts, len(frames)))).astype(int)
    return [
        {
            "t":        frames[i].t,
            "x":        frames[i].x,
            "y":        frames[i].y,
            "speed":    frames[i].speed,
            "throttle": frames[i].throttle,
            "brake":    frames[i].brake,
            "distance_m": frames[i].distance_m,
        }
        for i in idx
    ]


# ── Braking zones ─────────────────────────────────────────────────────────────

@router.get("/{session_id}/lap/{lap_index}/braking", response_model=list[BrakingZone])
def get_braking_zones(session_id: str, lap_index: int):
    """All braking zones detected in this lap, sorted by severity."""
    s = _require_session(session_id)
    frames = _require_lap(s, lap_index)
    zones = detect_braking_zones(frames)
    # Sort by peak brake pressure descending (heaviest first)
    zones.sort(key=lambda z: z.peak_brake, reverse=True)
    return zones


# ── Sectors ───────────────────────────────────────────────────────────────────

@router.get("/{session_id}/lap/{lap_index}/sectors", response_model=list[SectorSummary])
def get_sectors(session_id: str, lap_index: int):
    """3-sector breakdown for this lap (distance-based if GPS available)."""
    s = _require_session(session_id)
    frames = _require_lap(s, lap_index)
    return compute_sectors(frames, lap_index)


# ── Corner analysis ───────────────────────────────────────────────────────────

@router.get("/{session_id}/lap/{lap_index}/corners", response_model=LapCornerReport)
def get_corners(session_id: str, lap_index: int):
    """
    Per-corner analysis: entry/apex/exit speeds, peak brake, lateral g,
    trail-braking duration, and throttle at apex.
    """
    s = _require_session(session_id)
    frames = _require_lap(s, lap_index)

    corner_map = getattr(s, "corner_map", None)
    if corner_map is None or corner_map.centerline is None:
        raise HTTPException(
            503,
            "Track boundary not loaded — corner analysis unavailable. "
            "Ensure yas_marina_bnd.json is accessible via BND_PATH."
        )

    corners = analyse_corners(frames, corner_map, lap_index)
    return LapCornerReport(lap_index=lap_index, corners=corners)


# ── GG Diagram ────────────────────────────────────────────────────────────────

@router.get("/{session_id}/lap/{lap_index}/gg", response_model=GGDiagram)
def get_gg_diagram(session_id: str, lap_index: int):
    """
    Traction circle / GG diagram data.
    Provides lateral vs longitudinal acceleration scatter with
    friction circle envelope — key for analysing driver limit.
    """
    s = _require_session(session_id)
    frames = _require_lap(s, lap_index)
    return compute_gg_diagram(frames, lap_index)


# ── Tyre analysis ─────────────────────────────────────────────────────────────

@router.get("/{session_id}/lap/{lap_index}/tyres", response_model=list[TyreState])
def get_tyre_trend(
    session_id: str,
    lap_index: int,
    interval_s: float = Query(2.0, ge=0.5, le=10.0, description="Sample interval in seconds"),
):
    """
    Tyre temperature evolution through the lap.
    Downsampled to `interval_s` (default 2s) for efficient transfer.
    """
    s = _require_session(session_id)
    frames = _require_lap(s, lap_index)
    return compute_tyre_trend(frames, sample_interval_s=interval_s)


@router.get("/{session_id}/lap/{lap_index}/degradation", response_model=TyreDegradation)
def get_tyre_degradation(session_id: str, lap_index: int):
    """
    Lap-level tyre degradation summary: start/end avg temps,
    per-tyre peaks, and overheating / cold-start flags.
    """
    s = _require_session(session_id)
    frames = _require_lap(s, lap_index)
    return compute_tyre_degradation(frames, lap_index)


# ── Suspension / Ride Height ───────────────────────────────────────────────────

@router.get("/{session_id}/lap/{lap_index}/suspension", response_model=SuspensionReport)
def get_suspension(
    session_id: str,
    lap_index: int,
    interval_s: float = Query(1.0, ge=0.2, le=5.0, description="Sample interval in seconds"),
):
    """
    Suspension / ride-height time-series for a lap.

    Returns:
    - **damper_fl/fr/rl/rr** — per-wheel damper stroke (mm) from Badenia 560
    - **rh_front / rh_rear** — true centerline ride height (mm) from optical sensor
    - **summary** — per-lap averages and minimum ride heights

    The minimum ride height is particularly useful for detecting kerb rides and
    bottoming-out events that can damage the floor or affect aero balance.
    """
    s = _require_session(session_id)
    frames = _require_lap(s, lap_index)
    return compute_suspension_report(frames, lap_index, sample_interval_s=interval_s)


# ── Lap comparison ────────────────────────────────────────────────────────────

@router.get("/{session_id}/compare", response_model=DistanceComparison)
def compare(
    session_id: str,
    lap_a: int = Query(..., description="Reference lap index"),
    lap_b: int = Query(..., description="Comparison lap index"),
    n_points: int = Query(500, ge=100, le=2000, description="Distance grid resolution"),
):
    """
    Distance-normalised lap comparison (the correct engineering approach).
    Both laps are projected onto a shared distance axis so you see exactly
    where on-track one lap is faster or slower — braking points, apex speeds,
    exit traction.
    """
    s = _require_session(session_id)
    frames_a = _require_lap(s, lap_a)
    frames_b = _require_lap(s, lap_b)
    return compare_laps_distance(frames_a, frames_b, lap_a, lap_b, n_points=n_points)


@router.get("/{session_id}/compare/time", response_model=LapComparison)
def compare_time(
    session_id: str,
    lap_a: int = Query(..., description="Reference lap index"),
    lap_b: int = Query(..., description="Comparison lap index"),
):
    """
    Legacy time-normalised lap comparison.
    Prefer /compare (distance-based) for more accurate spatial analysis.
    """
    s = _require_session(session_id)
    frames_a = _require_lap(s, lap_a)
    frames_b = _require_lap(s, lap_b)
    return compare_laps(frames_a, frames_b, lap_a, lap_b)


# ── Track data ────────────────────────────────────────────────────────────────

@router.get("/{session_id}/track", response_model=TrackData)
def get_track(session_id: str):
    """
    Track boundary, centerline, and detected corner positions for Yas Marina.
    Use this to render the circuit map in the frontend.
    """
    s = _require_session(session_id)
    corner_map = getattr(s, "corner_map", None)
    if corner_map is None or corner_map.centerline is None:
        raise HTTPException(
            503,
            "Track boundary not loaded — ensure yas_marina_bnd.json is available."
        )
    track = corner_map.as_track_data()
    if track is None:
        raise HTTPException(503, "Track data could not be built")
    return track


# ── Session-level all-laps corner comparison ──────────────────────────────────

@router.get("/{session_id}/corners/all", response_model=list[LapCornerReport])
def get_all_corners(session_id: str):
    """
    Corner analysis across all laps — lets the frontend show
    corner-by-corner progression across the session.
    """
    s = _require_session(session_id)
    corner_map = getattr(s, "corner_map", None)
    if corner_map is None or corner_map.centerline is None:
        raise HTTPException(503, "Track boundary not loaded")

    reports = []
    for i, frames in enumerate(s.raw_laps):
        corners = analyse_corners(frames, corner_map, i)
        reports.append(LapCornerReport(lap_index=i, corners=corners))
    return reports


# ── Best lap quick-access ─────────────────────────────────────────────────────

@router.get("/{session_id}/best-lap", response_model=LapSummary)
def get_best_lap(session_id: str):
    """Return the summary of the fastest lap in the session."""
    s = _require_session(session_id)
    if not s.lap_details:
        raise HTTPException(404, "No laps found")
    best = min(s.lap_details, key=lambda d: d.summary.lap_time_s)
    return best.summary


# ── Delta time chart ──────────────────────────────────────────────────────────

@router.get("/{session_id}/delta", response_model=DeltaTimeSeries)
def get_delta_time(
    session_id: str,
    lap_a: int = Query(..., description="Reference lap index"),
    lap_b: int = Query(..., description="Comparison lap index"),
    n_points: int = Query(500, ge=100, le=1000),
):
    """
    Cumulative ΔT chart: time gained/lost between two laps at every metre of track.
    Positive = lap B is behind (slower); negative = lap B is ahead (faster).
    Mirrors the live delta chart used in real F1 & IndyCar broadcasts.
    """
    s = _require_session(session_id)
    frames_a = _require_lap(s, lap_a)
    frames_b = _require_lap(s, lap_b)
    return compute_delta_time(frames_a, frames_b, lap_a, lap_b, n_points=n_points)


# ── Theoretical best lap ──────────────────────────────────────────────────────

@router.get("/{session_id}/theoretical_best", response_model=TheoreticalBest)
def get_theoretical_best(
    session_id: str,
    n_sectors: int = Query(25, ge=5, le=100, description="Number of mini-sectors"),
):
    """
    'Purple lap' — theoretical fastest lap by combining best mini-sector times
    from across all laps in the session. Shows how much faster the car could
    go if every sector was driven at peak performance.
    """
    s = _require_session(session_id)
    if len(s.raw_laps) < 1:
        raise HTTPException(404, "No laps available")
    try:
        return compute_theoretical_best(s.raw_laps, n_sectors=n_sectors)
    except ValueError as e:
        raise HTTPException(422, str(e))


# ── Corner scoreboard ─────────────────────────────────────────────────────────

@router.get("/{session_id}/corner_scoreboard", response_model=CornerScoreboard)
def get_corner_scoreboard(
    session_id: str,
    lap_index: Optional[int] = Query(None, description="Specific lap — omit for best across all laps"),
):
    """
    Corner-by-corner efficiency table. If lap_index is given, returns that lap's
    data. Otherwise returns the best metric for each corner across all laps —
    the 'ideal corner card' showing peak performance at every turn.
    """
    s = _require_session(session_id)
    corner_map = getattr(s, "corner_map", None)
    if corner_map is None or corner_map.centerline is None:
        raise HTTPException(503, "Track boundary not loaded")

    laps_to_analyse = (
        [lap_index] if lap_index is not None
        else list(range(len(s.raw_laps)))
    )

    # Gather corner analysis per lap
    all_rows: list[CornerScoreRow] = []
    for li in laps_to_analyse:
        if li >= len(s.raw_laps):
            continue
        corners = analyse_corners(s.raw_laps[li], corner_map, li)
        for c in corners:
            all_rows.append(CornerScoreRow(
                corner_id=c.corner_id,
                distance_m=round(c.distance_m, 1),
                direction=c.direction,
                min_speed=round(c.min_speed_kph, 1),
                entry_speed=round(c.entry_speed_kph, 1),
                exit_speed=round(c.exit_speed_kph, 1),
                peak_brake=round(c.peak_brake, 3),
                peak_lat_g=round(c.peak_lat_g, 2),
                lap_index=li,
            ))

    # If multi-lap: keep only the best (highest apex speed) row per corner
    if lap_index is None and all_rows:
        best: dict[str, CornerScoreRow] = {}
        for row in all_rows:
            if row.corner_id not in best or row.min_speed > best[row.corner_id].min_speed:
                best[row.corner_id] = row
        all_rows = sorted(best.values(), key=lambda r: r.distance_m)

    return CornerScoreboard(
        session_id=session_id,
        laps=laps_to_analyse,
        rows=all_rows,
    )
