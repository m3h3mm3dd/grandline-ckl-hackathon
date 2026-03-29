"""
SSE stream — real-time lap replay with onboard camera and AI tips.

Connect: GET /stream/{session_id}/lap/{lap_index}?speed=1.0

Events emitted:
  frame         → TelemetryFrame JSON  (~10 Hz)
  camera_frame  → { t, jpeg_b64 }      (~2 Hz, actual onboard footage)
  coach_tip     → { t, tip }           (every ~8 s)
  lap_end       → LapSummary JSON
"""

from __future__ import annotations

import asyncio
import base64
import bisect
import json
import logging
from typing import AsyncGenerator

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse

from services.session_store import get_session
from services.coaching_engine import build_coaching_engine

log = logging.getLogger(__name__)
router = APIRouter()

_STREAM_HZ       = 100    # send all 100 Hz frames for smooth 90+ fps playback
_CAMERA_EVERY_N  = 5      # camera every 5 telemetry frames → 20 Hz camera stream


def _sse(event: str, data: dict | str) -> str:
    payload = data if isinstance(data, str) else json.dumps(data)
    return f"event: {event}\ndata: {payload}\n\n"


def _nearest_camera_frame(
    cam_frames: list[tuple[float, bytes]],
    target_ts: float,
) -> bytes | None:
    """Binary-search for the camera frame closest to target_ts."""
    if not cam_frames:
        return None
    timestamps = [f[0] for f in cam_frames]
    idx = bisect.bisect_left(timestamps, target_ts)
    if idx == 0:
        return cam_frames[0][1]
    if idx >= len(cam_frames):
        return cam_frames[-1][1]
    before = cam_frames[idx - 1]
    after  = cam_frames[idx]
    return (before if abs(before[0] - target_ts) <= abs(after[0] - target_ts) else after)[1]


async def _generate(
    session_id: str,
    lap_index: int,
    speed_factor: float,
    camera: str = "camera_fl",
    start_frame: int = 0,
) -> AsyncGenerator[str, None]:
    s = get_session(session_id)
    if not s or not s._ready.is_set():
        yield _sse("error", {"message": "session not ready"})
        return

    if lap_index >= len(s.lap_details):
        yield _sse("error", {"message": "lap not found"})
        return

    detail     = s.lap_details[lap_index]
    frames     = detail.frames
    # Get camera frames for the requested camera
    cam_laps   = s.camera_laps_by_topic.get(camera, [])
    cam_frames = cam_laps[lap_index] if lap_index < len(cam_laps) else []

    log.info("Stream: session=%s lap=%d camera=%s frames=%d cam_frames=%d start_frame=%d",
             session_id, lap_index, camera, len(frames), len(cam_frames), start_frame)

    # Downsample telemetry to stream Hz (source is ~100 Hz)
    step = max(1, int(100 / _STREAM_HZ))
    all_stream_frames = frames[::step]
    total_frames = len(all_stream_frames)

    # Skip to start_frame for speed-change resume without restarting from lap 0
    clipped_start = max(0, min(start_frame, total_frames - 1))
    stream_frames = all_stream_frames[clipped_start:]

    if not stream_frames:
        yield _sse("error", {"message": "no frames"})
        return

    # Send metadata so frontend knows total frame count and lap distance
    lap_dist = getattr(detail.summary, "lap_dist_m", None) or 0.0
    yield _sse("stream_meta", {
        "total_frames": total_frames,
        "lap_dist_m": round(lap_dist, 1),
        "start_frame": clipped_start,
    })

    # ── Build rule-based coaching engine (always on, no API key needed) ──
    try:
        coach = build_coaching_engine(s, lap_index)
    except Exception as e:
        log.warning("CoachingEngine init failed: %s", e)
        coach = None

    # ── Reference speed profile for HUD (best lap speed at each distance) ──
    ref_profile = coach._ref if (coach and coach._ref) else None

    interval    = (1.0 / _STREAM_HZ) / speed_factor
    cam_counter = 0

    for i, frame in enumerate(stream_frames):
        # Telemetry frame — augment with reference speed for HUD
        frame_dict = frame.model_dump()
        if ref_profile is not None:
            d = frame.distance_m or 0.0
            frame_dict["ref_speed"] = round(ref_profile.speed_at(d), 1)
        else:
            frame_dict["ref_speed"] = None
        # Absolute frame index (for seek/resume support)
        frame_dict["frame_idx"] = clipped_start + i
        yield _sse("frame", frame_dict)

        # Camera frame (every _CAMERA_EVERY_N telemetry frames)
        cam_counter += 1
        if cam_counter >= _CAMERA_EVERY_N:
            cam_counter = 0
            jpeg = _nearest_camera_frame(cam_frames, frame.ts)
            if jpeg:
                b64 = base64.b64encode(jpeg).decode("ascii")
                yield _sse("camera_frame", {"t": frame.t, "jpeg_b64": b64})

        # Rule-based coaching events — always active
        if coach:
            try:
                events = coach.process_frame(frame)
                for ev in events:
                    yield _sse("coach_tip", {"t": frame.t, **ev})
            except Exception as e:
                log.warning("Coaching event error: %s", e)

        await asyncio.sleep(interval)

    yield _sse("lap_end", detail.summary.model_dump())


@router.get("/{session_id}/cameras")
async def list_cameras(session_id: str):
    """List available camera feeds for this session."""
    s = get_session(session_id)
    if not s:
        raise HTTPException(404, "Session not found")
    if not s._ready.is_set():
        raise HTTPException(202, "Session still processing")
    return {"cameras": s.available_cameras or ["camera_fl"], "default": "camera_fl"}


@router.get("/{session_id}/lap/{lap_index}")
async def stream_lap(
    session_id: str,
    lap_index: int,
    speed: float = Query(1.0, ge=0.1, le=10.0, description="Replay speed factor"),
    camera: str = Query("camera_fl", description="Camera to stream (camera_fl, camera_r, etc)"),
    start_frame: int = Query(0, ge=0, description="Frame index to resume from (speed-change seek)"),
):
    s = get_session(session_id)
    if not s:
        raise HTTPException(404, "Session not found")

    return StreamingResponse(
        _generate(session_id, lap_index, speed, camera, start_frame),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "X-Accel-Buffering": "no",
            "Content-Encoding": "identity",  # prevent any proxy/middleware gzip buffering
        },
    )
