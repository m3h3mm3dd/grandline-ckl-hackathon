from __future__ import annotations

import asyncio
import logging
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from services.mcap_reader import stream_frames, stream_camera_frames, RawFrame
from services.lap_detector import detect_laps
from services.metrics_engine import compute_lap_detail, LapDetail
from models.schemas import SessionMeta

log = logging.getLogger(__name__)

UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", "uploads"))
BND_PATH   = Path(os.getenv("BND_PATH", "data/yas_marina_bnd.json"))

_SCENARIO_MAP = {
    "good_lap":       "good_lap",
    "fast_laps":      "fast_laps",
    "wheel_to_wheel": "wheel_to_wheel",
}


def _guess_scenario(filename: str) -> str:
    fn = filename.lower()
    for key in _SCENARIO_MAP:
        if key.replace("_", "") in fn.replace("_", "").replace("-", ""):
            return _SCENARIO_MAP[key]
    return "unknown"


class Session:
    def __init__(self, session_id: str, filename: str, mcap_path: Path):
        self.session_id  = session_id
        self.filename    = filename
        self.mcap_path   = mcap_path
        self.uploaded_at = datetime.now(timezone.utc).isoformat()
        self.raw_laps:    list[list[RawFrame]] = []
        self.lap_details: list[LapDetail] = []
        self._ready       = asyncio.Event()
        self.corner_map   = None
        self.meta_preloaded: bool = False
        # Camera frames per lap: list[list[tuple[float, bytes]]]
        # Each inner list is sorted by timestamp; bytes are raw JPEG
        self.camera_laps: list[list[tuple[float, bytes]]] = []

    async def process(self):
        if self.corner_map is None:
            from services.corner_detector import CornerMap
            self.corner_map = CornerMap(BND_PATH if BND_PATH.exists() else None)
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._decode)
        self._ready.set()

    def _decode(self):
        log.info("Decoding %s …", self.mcap_path.name)
        frames = list(stream_frames(self.mcap_path))
        log.info("  %d raw frames", len(frames))

        self.raw_laps = detect_laps(frames, BND_PATH if BND_PATH.exists() else None)

        if self.corner_map is not None and self.corner_map.centerline is not None:
            for lap_frames in self.raw_laps:
                self.corner_map.assign_distances(lap_frames)

        self.lap_details = [
            compute_lap_detail(lap, i)
            for i, lap in enumerate(self.raw_laps)
        ]

        # Load camera frames and distribute into laps
        self._load_camera_frames()
        log.info("  %d laps decoded", len(self.raw_laps))

    def _load_camera_frames(self):
        """
        Read the front-left camera topic and bin frames into laps by timestamp.
        Stores ~10 Hz JPEG frames per lap for SSE streaming.
        """
        if not self.raw_laps:
            self.camera_laps = []
            return

        # Build lap time-windows
        lap_windows = [
            (lap[0].ts, lap[-1].ts)
            for lap in self.raw_laps
        ]
        buckets: list[list[tuple[float, bytes]]] = [[] for _ in self.raw_laps]

        try:
            for ts, jpeg in stream_camera_frames(self.mcap_path):
                for i, (t_start, t_end) in enumerate(lap_windows):
                    if t_start <= ts <= t_end:
                        buckets[i].append((ts, jpeg))
                        break
        except Exception as e:
            log.warning("Camera frame load failed: %s", e)

        self.camera_laps = buckets
        total = sum(len(b) for b in buckets)
        log.info("  %d camera frames loaded across %d laps", total, len(buckets))

    async def wait_ready(self, timeout: float = 120.0):
        await asyncio.wait_for(self._ready.wait(), timeout=timeout)

    @property
    def meta(self) -> SessionMeta:
        duration = 0.0
        if self.raw_laps:
            first = self.raw_laps[0][0].ts
            last  = self.raw_laps[-1][-1].ts
            duration = last - first
        return SessionMeta(
            session_id=self.session_id,
            filename=self.filename,
            duration_s=round(duration, 2),
            lap_count=len(self.raw_laps),
            scenario=_guess_scenario(self.filename),
            uploaded_at=self.uploaded_at,
            preloaded=self.meta_preloaded,
        )


# ── Global registry ───────────────────────────────────────────────────────────

_sessions: dict[str, Session] = {}


def create_session(filename: str, mcap_path: Path) -> Session:
    sid = str(uuid.uuid4())
    s = Session(sid, filename, mcap_path)
    _sessions[sid] = s
    return s


def get_session(session_id: str) -> Optional[Session]:
    return _sessions.get(session_id)


def list_sessions() -> list[SessionMeta]:
    return [s.meta for s in _sessions.values()]


def delete_session(session_id: str) -> bool:
    s = _sessions.pop(session_id, None)
    if s and s.mcap_path.exists() and not s.meta_preloaded:
        s.mcap_path.unlink(missing_ok=True)
    return s is not None
