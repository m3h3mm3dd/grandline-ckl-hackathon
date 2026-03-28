from fastapi import APIRouter, HTTPException

from services.session_store import get_session
from services.ai_coach import (
    coach_single_lap, coach_lap_comparison, coach_followup,
    coach_wheel_to_wheel,
)
from services.metrics_engine import (
    compute_gg_diagram, compute_tyre_degradation, compare_laps_distance
)
from services.corner_detector import analyse_corners
from models.schemas import CoachRequest, CoachResponse

router = APIRouter()


async def _require_ready(session_id: str):
    s = get_session(session_id)
    if not s:
        raise HTTPException(404, "Session not found")
    if not s._ready.is_set():
        raise HTTPException(202, "Session still processing")
    return s


@router.post("/debrief", response_model=CoachResponse)
async def debrief(req: CoachRequest):
    """Post-lap or full-session debrief from the AI race engineer."""
    s = await _require_ready(req.session_id)
    n = len(s.lap_details)

    if req.compare_lap is not None:
        lap_a = req.lap_index if req.lap_index is not None else 0
        lap_b = req.compare_lap
        if lap_a >= n or lap_b >= n:
            raise HTTPException(404, "Lap index out of range")

        comparison = compare_laps_distance(
            s.raw_laps[lap_a], s.raw_laps[lap_b], lap_a, lap_b
        )

        # Enrich with corners and tyre data if available and requested
        corner_map = getattr(s, "corner_map", None)
        corners_a = corners_b = None
        if req.include_corners and corner_map and corner_map.centerline is not None:
            corners_a = analyse_corners(s.raw_laps[lap_a], corner_map, lap_a)
            corners_b = analyse_corners(s.raw_laps[lap_b], corner_map, lap_b)

        tyre_a = tyre_b = None
        if req.include_tyres:
            tyre_a = compute_tyre_degradation(s.raw_laps[lap_a], lap_a)
            tyre_b = compute_tyre_degradation(s.raw_laps[lap_b], lap_b)

        return coach_lap_comparison(
            comparison,
            corners_a=corners_a, corners_b=corners_b,
            tyre_a=tyre_a, tyre_b=tyre_b,
            cache_key=(req.session_id, "compare", lap_a, lap_b),
        )

    # Single lap or best lap
    lap_idx = req.lap_index
    if lap_idx is None:
        lap_idx = min(range(n), key=lambda i: s.lap_details[i].summary.lap_time_s)
    if lap_idx >= n:
        raise HTTPException(404, "Lap not found")

    detail = s.lap_details[lap_idx]
    raw    = s.raw_laps[lap_idx]

    # Enrich with GG, corners, tyres
    gg     = compute_gg_diagram(raw, lap_idx)
    tyre   = compute_tyre_degradation(raw, lap_idx) if req.include_tyres else None

    corner_map = getattr(s, "corner_map", None)
    corners = None
    if req.include_corners and corner_map and corner_map.centerline is not None:
        corners = analyse_corners(raw, corner_map, lap_idx)

    # Use wheel-to-wheel specialised coach for race scenario
    scenario = s.meta.scenario
    if scenario == "wheel_to_wheel":
        return coach_wheel_to_wheel(detail, corners=corners, gg=gg)

    return coach_single_lap(
        detail, corners=corners, gg=gg, tyre=tyre,
        cache_key=(req.session_id, "single", lap_idx),
    )


@router.post("/ask", response_model=CoachResponse)
async def ask(req: CoachRequest):
    """Follow-up question to the engineer."""
    if not req.question:
        raise HTTPException(400, "question is required")
    s = await _require_ready(req.session_id)

    lap_summary = None
    corners = None
    if req.lap_index is not None and req.lap_index < len(s.lap_details):
        lap_summary = s.lap_details[req.lap_index].summary
        corner_map = getattr(s, "corner_map", None)
        if req.include_corners and corner_map and corner_map.centerline is not None:
            corners = analyse_corners(s.raw_laps[req.lap_index], corner_map, req.lap_index)

    return coach_followup([], req.question, lap_summary, corners=corners)
