"""
Preload service — auto-detects and processes the hackathon MCAP files at startup.

Set environment variable PRELOAD_DATA_DIR to the directory containing the 3 MCAP files.
The session store will be populated with stable, predictable session IDs so the
frontend (Lovable) can reference them without requiring the user to upload files.

Session ID mapping:
  good_lap       → "preload-good-lap"
  fast_laps      → "preload-fast-laps"
  wheel_to_wheel → "preload-wheel-to-wheel"
"""

from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path
from typing import Optional

from services.session_store import _sessions, Session
from services.corner_detector import CornerMap

log = logging.getLogger(__name__)

# Stable session IDs for the preloaded hackathon data
PRELOAD_IDS = {
    "good_lap":       "preload-good-lap",
    "fast_laps":      "preload-fast-laps",
    "wheel_to_wheel": "preload-wheel-to-wheel",
}

# File name fragments to scenario mapping
_SCENARIO_FRAGMENTS = {
    "good_lap":       "good_lap",
    "fast_laps":      "fast_laps",
    "wheel_to_wheel": "wheel_to_wheel",
}


def _identify_scenario(filename: str) -> Optional[str]:
    fn = filename.lower().replace("-", "_")
    for scenario, fragment in _SCENARIO_FRAGMENTS.items():
        if fragment in fn:
            return scenario
    return None


async def preload_hackathon_data(
    data_dir: Optional[str] = None,
    bnd_path: Optional[Path] = None,
) -> None:
    """
    Scan data_dir for MCAP files matching the hackathon scenarios.
    Register each as a preloaded session and kick off background processing.

    This is called once at FastAPI startup via lifespan.
    """
    dir_path = Path(data_dir or os.getenv("PRELOAD_DATA_DIR", "data/mcap"))

    if not dir_path.exists():
        log.info(
            "Preload dir %s not found — skipping preload. "
            "Set PRELOAD_DATA_DIR to enable auto-loading of hackathon MCAP files.",
            dir_path,
        )
        return

    mcap_files = sorted(dir_path.glob("*.mcap"))
    if not mcap_files:
        log.info("No MCAP files found in %s — skipping preload.", dir_path)
        return

    log.info("Found %d MCAP files in %s, beginning preload…", len(mcap_files), dir_path)

    # Build corner map once (shared across sessions)
    corner_map = CornerMap(bnd_path)

    tasks = []
    for mcap_path in mcap_files:
        scenario = _identify_scenario(mcap_path.name)
        if scenario is None:
            log.warning("  Skipping unrecognised file: %s", mcap_path.name)
            continue

        session_id = PRELOAD_IDS[scenario]

        # Don't re-register if already loaded (e.g. hot-reload)
        if session_id in _sessions:
            log.info("  Session %s already loaded — skipping", session_id)
            continue

        log.info("  Registering %s → %s", mcap_path.name, session_id)
        s = Session(session_id, mcap_path.name, mcap_path)
        s.corner_map = corner_map   # attach shared corner map
        s.meta_preloaded = True     # flag for SessionMeta
        _sessions[session_id] = s

        # Schedule background decode
        tasks.append(asyncio.create_task(
            _process_with_corner_map(s, corner_map),
            name=f"preload-{session_id}",
        ))

    if tasks:
        log.info("  %d MCAP files queued for background processing", len(tasks))
    else:
        log.info("  All matching files already loaded")


async def _process_with_corner_map(s: Session, corner_map: CornerMap) -> None:
    """
    Decode the MCAP file in a thread pool, then assign distances
    using the corner map and update lap details accordingly.
    """
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, _decode_and_enrich, s, corner_map)
    s._ready.set()
    log.info("  ✓ %s ready — %d laps", s.filename, len(s.raw_laps))


def _decode_and_enrich(s: Session, corner_map: CornerMap) -> None:
    """Runs in thread pool. Decodes MCAP, detects laps, assigns distances."""
    from services.mcap_reader import stream_frames
    from services.lap_detector import detect_laps
    from services.metrics_engine import compute_lap_detail

    log.info("    Decoding %s …", s.mcap_path.name)
    frames = list(stream_frames(s.mcap_path))
    log.info("    %d raw frames", len(frames))

    # Detect laps
    bnd_path = Path(os.getenv("BND_PATH", "data/yas_marina_bnd.json"))
    s.raw_laps = detect_laps(frames, bnd_path if bnd_path.exists() else None)
    log.info("    %d laps detected", len(s.raw_laps))

    # Assign GPS arc-length distance to every frame in every lap
    if corner_map.centerline is not None:
        for lap_frames in s.raw_laps:
            corner_map.assign_distances(lap_frames)

    # Compute lap details with enriched distance data
    s.lap_details = [
        compute_lap_detail(lap, i)
        for i, lap in enumerate(s.raw_laps)
    ]

    # Load camera frames from MCAP and bin them into laps by timestamp
    s._load_camera_frames()

    # GPS-calibrate corner map using the first lap's GPS trace (most accurate)
    if s.raw_laps and corner_map.centerline is not None:
        try:
            corner_map.calibrate_from_gps(s.raw_laps[0])
            log.info("    Corner map GPS-calibrated: %d corners", len(corner_map.corners))
        except Exception as e:
            log.warning("    Corner calibration failed: %s", e)
