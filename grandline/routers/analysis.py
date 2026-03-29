from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, HTTPException, Query

from services.session_store import get_session
from services.metrics_engine import (
    compute_lap_detail, detect_braking_zones,
    compute_sectors, compare_laps,
    compute_delta_time, compute_theoretical_best,
)
from services.corner_detector import analyse_corners
from models.schemas import (
    LapDetail, LapSummary, BrakingZone,
    SectorSummary, LapComparison,
    TrackData, DeltaTimeSeries, TheoreticalBest,
    CornerScoreboard, CornerScoreRow,
)

router = APIRouter()


async def _require_ready(session_id: str):
    s = get_session(session_id)
    if not s:
        raise HTTPException(404, "Session not found")
    if not s._ready.is_set():
        raise HTTPException(202, "Session still processing — try again shortly")
    return s


# ── Core lap endpoints ─────────────────────────────────────────────────────────

@router.get("/{session_id}/laps", response_model=list[LapSummary])
async def list_laps(session_id: str):
    s = await _require_ready(session_id)
    return [d.summary for d in s.lap_details]


@router.get("/{session_id}/laps/{lap_index}", response_model=LapDetail)
async def get_lap(session_id: str, lap_index: int):
    s = await _require_ready(session_id)
    if lap_index >= len(s.lap_details):
        raise HTTPException(404, f"Lap {lap_index} not found (session has {len(s.lap_details)} laps)")
    return s.lap_details[lap_index]


@router.get("/{session_id}/laps/{lap_index}/frames")
async def get_lap_frames(
    session_id: str,
    lap_index: int,
    downsample: int = Query(1, ge=1, le=50, description="Return every Nth frame (1=all)"),
):
    """
    Returns raw telemetry frames for a lap.
    Use downsample=10 for real-time replay (10 Hz from 100 Hz source).
    Use downsample=1 for full-resolution chart rendering.
    """
    s = await _require_ready(session_id)
    if lap_index >= len(s.lap_details):
        raise HTTPException(404, "Lap not found")
    frames = s.lap_details[lap_index].frames[::downsample]
    return {"lap_index": lap_index, "frame_count": len(frames), "frames": frames}


@router.get("/{session_id}/laps/{lap_index}/braking", response_model=list[BrakingZone])
async def get_braking_zones(session_id: str, lap_index: int):
    s = await _require_ready(session_id)
    if lap_index >= len(s.raw_laps):
        raise HTTPException(404, "Lap not found")
    return detect_braking_zones(s.raw_laps[lap_index])


@router.get("/{session_id}/laps/{lap_index}/sectors", response_model=list[SectorSummary])
async def get_sectors(session_id: str, lap_index: int):
    s = await _require_ready(session_id)
    if lap_index >= len(s.raw_laps):
        raise HTTPException(404, "Lap not found")
    return compute_sectors(s.raw_laps[lap_index], lap_index)


@router.get("/{session_id}/laps/{lap_index}/positions")
async def get_lap_positions(
    session_id: str,
    lap_index: int,
    max_pts: int = Query(800, ge=50, le=2000, description="Max number of position points to return"),
):
    """Return {x, y, speed} list for the full lap GPS path, downsampled to max_pts."""
    s = await _require_ready(session_id)
    if lap_index >= len(s.raw_laps):
        raise HTTPException(404, "Lap not found")
    frames = s.raw_laps[lap_index]
    step = max(1, len(frames) // max_pts)
    return [
        {"x": round(f.x, 2), "y": round(f.y, 2), "speed": round(f.speed, 1)}
        for f in frames[::step]
    ]


# ── Track geometry ─────────────────────────────────────────────────────────────

@router.get("/{session_id}/track", response_model=TrackData)
async def get_track(session_id: str):
    """Return track boundaries, centerline, and corner markers for map rendering."""
    s = await _require_ready(session_id)
    if not s.corner_map:
        raise HTTPException(503, "Corner map not available for this session")
    td = s.corner_map.as_track_data()
    if not td:
        raise HTTPException(
            503,
            "Track data not yet available — boundary file (yas_marina_bnd.json) may be missing",
        )
    return td


# ── Camera list ────────────────────────────────────────────────────────────────

@router.get("/{session_id}/cameras")
async def get_cameras(session_id: str):
    """Return list of available camera topics decoded from this session's MCAP file."""
    s = await _require_ready(session_id)
    return {"cameras": s.available_cameras or ["camera_fl"]}


# ── Corner scoreboard ──────────────────────────────────────────────────────────

@router.get("/{session_id}/corner_scoreboard", response_model=CornerScoreboard)
async def get_corner_scoreboard(
    session_id: str,
    lap_index: Optional[int] = Query(
        None,
        description="Lap index to analyse; defaults to the fastest lap",
    ),
):
    """Per-corner metrics table: apex speed, entry/exit speed, peak brake, lateral G."""
    s = await _require_ready(session_id)
    if not s.corner_map or not s.corner_map.corners:
        raise HTTPException(503, "Corner map not available — boundary file may be missing")
    if not s.raw_laps:
        raise HTTPException(404, "No laps available in this session")

    # Default: use the fastest lap
    if lap_index is None:
        lap_index = min(
            range(len(s.lap_details)),
            key=lambda i: s.lap_details[i].summary.lap_time_s,
        )
    if lap_index >= len(s.raw_laps):
        raise HTTPException(404, f"Lap {lap_index} not found")

    corners = analyse_corners(s.raw_laps[lap_index], s.corner_map, lap_index)
    rows = [
        CornerScoreRow(
            corner_id=c.corner_id,
            distance_m=c.distance_m,
            direction=next(
                (cm.direction for cm in s.corner_map.corners if cm.id == c.corner_id),
                "left",
            ),
            min_speed=c.min_speed_kph,
            entry_speed=c.entry_speed_kph,
            exit_speed=c.exit_speed_kph,
            peak_brake=c.entry_brake,
            peak_lat_g=c.max_lat_acc,
            lap_index=lap_index,
        )
        for c in corners
    ]
    return CornerScoreboard(session_id=session_id, laps=[lap_index], rows=rows)


# ── Delta time chart ───────────────────────────────────────────────────────────

@router.get("/{session_id}/delta", response_model=DeltaTimeSeries)
async def get_delta(
    session_id: str,
    lap_a: int = Query(..., description="Reference lap index"),
    lap_b: int = Query(..., description="Comparison lap index"),
    n_points: int = Query(400, ge=50, le=1000, description="Number of distance grid points"),
):
    """Distance-normalised cumulative time delta between two laps (ΔT chart)."""
    s = await _require_ready(session_id)
    n = len(s.raw_laps)
    if lap_a >= n or lap_b >= n:
        raise HTTPException(404, f"Lap index out of range (session has {n} laps)")
    return compute_delta_time(s.raw_laps[lap_a], s.raw_laps[lap_b], lap_a, lap_b, n_points)


# ── Theoretical best (purple lap) ─────────────────────────────────────────────

@router.get("/{session_id}/theoretical_best", response_model=TheoreticalBest)
async def get_theoretical_best(session_id: str):
    """'Purple lap' — fastest possible time by combining best mini-sectors across all laps."""
    s = await _require_ready(session_id)
    if not s.raw_laps:
        raise HTTPException(404, "No laps available in this session")
    return compute_theoretical_best(s.raw_laps)


# ── Legacy time-based comparison (kept for backward compatibility) ─────────────

@router.get("/{session_id}/compare", response_model=LapComparison)
async def compare(
    session_id: str,
    lap_a: int = Query(..., description="Reference lap index"),
    lap_b: int = Query(..., description="Comparison lap index"),
):
    s = await _require_ready(session_id)
    n = len(s.raw_laps)
    if lap_a >= n or lap_b >= n:
        raise HTTPException(404, f"Lap index out of range (session has {n} laps)")
    if lap_a == lap_b:
        raise HTTPException(400, "lap_a and lap_b must be different")
    return compare_laps(s.raw_laps[lap_a], s.raw_laps[lap_b], lap_a, lap_b)
