"""
Corner detection and per-corner lap analysis for Yas Marina Circuit.

Approach:
1. Load track boundary JSON → compute centerline as midpoint of inner/outer.
2. Compute arc-length distance along centerline.
3. Compute signed curvature of the centerline (κ = dθ/ds).
4. Find local maxima of |κ| above a threshold → corner peaks.
5. Define corner entry/apex/exit windows from curvature profile.
6. For each lap, snap GPS frames to nearest centerline point → distance_m.
7. Extract speed/brake/throttle at entry/apex/exit for each corner.

Yas Marina Corner Knowledge (21 turns):
  T1  — right, end of main straight (~200m from SF line)
  T2  — right, continuation of T1
  T3  — right, leads onto back straight
  T5  — right, start of marina chicane
  T6  — left, marina
  T7  — right, marina exit
  T8  — right hairpin (slowest corner)
  T9  — left
  T11 — right
  T14 — left
  T17 — right
  T19 — right esses
  T20 — left esses
  T21 — right, final corner onto main straight
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Optional

import numpy as np

from services.mcap_reader import RawFrame
from models.schemas import CornerAnalysis, CornerMarker, TrackData, TrackPoint

log = logging.getLogger(__name__)

# Curvature threshold for corner detection (1/m)
_CURVATURE_THRESHOLD = 0.015
# Minimum distance between corners (m) — prevents double-counting
_MIN_CORNER_SEPARATION_M = 60.0
# Corner window: frames within this distance of apex form the corner
_CORNER_WINDOW_M = 80.0
# Minimum speed drop to qualify as a "real" corner (vs. straight kink)
_MIN_SPEED_DROP_KPH = 10.0


def _compute_arc_length(pts: np.ndarray) -> np.ndarray:
    """Cumulative arc-length for a 2D polyline (N,2)."""
    diffs = np.diff(pts, axis=0)
    segs  = np.linalg.norm(diffs, axis=1)
    return np.concatenate([[0.0], np.cumsum(segs)])


def _compute_curvature(pts: np.ndarray, arc: np.ndarray) -> np.ndarray:
    """
    Signed curvature κ = dθ/ds at each point.
    Positive = left turn, negative = right turn (in standard orientation).
    """
    n = len(pts)
    kappa = np.zeros(n)
    for i in range(1, n - 1):
        # Central difference on heading angle
        v1 = pts[i] - pts[i - 1]
        v2 = pts[i + 1] - pts[i]
        th1 = math.atan2(v1[1], v1[0])
        th2 = math.atan2(v2[1], v2[0])
        dth = th2 - th1
        # Wrap to [-π, π]
        dth = (dth + math.pi) % (2 * math.pi) - math.pi
        ds = arc[i + 1] - arc[i - 1]
        if ds > 0:
            kappa[i] = dth / ds
    return kappa


def _local_maxima(arr: np.ndarray, min_dist: int = 5) -> list[int]:
    """Indices of local maxima in 1D array with minimum separation."""
    peaks = []
    for i in range(1, len(arr) - 1):
        if arr[i] >= arr[i - 1] and arr[i] >= arr[i + 1]:
            if not peaks or (i - peaks[-1]) >= min_dist:
                peaks.append(i)
    return peaks


class CornerMap:
    """
    Loaded once from track boundary file.
    Provides: centerline, arc-length, corner list, GPS snapping.
    """

    def __init__(self, bnd_path: Optional[Path] = None):
        self.centerline: Optional[np.ndarray] = None
        self.arc:        Optional[np.ndarray] = None
        self.curvature:  Optional[np.ndarray] = None
        self.corners:    list[CornerMarker]   = []
        self.lap_distance_m: float = 0.0
        self._inner: Optional[np.ndarray] = None
        self._outer: Optional[np.ndarray] = None

        if bnd_path and bnd_path.exists():
            self._build(bnd_path)

    def _build(self, bnd_path: Path):
        try:
            with open(bnd_path) as f:
                bnd = json.load(f)

            # Support nested boundaries dict (yas_marina_bnd.json format)
            bnd_data = bnd.get("boundaries", bnd)
            inner = np.array(
                bnd_data.get("left_border",
                bnd_data.get("inner",
                bnd_data.get("left", []))), dtype=float)
            outer = np.array(
                bnd_data.get("right_border",
                bnd_data.get("outer",
                bnd_data.get("right", []))), dtype=float)

            if len(inner) < 4 or len(outer) < 4:
                log.warning("Boundary too short; corner map disabled")
                return

            # Resample both to same length
            n = min(len(inner), len(outer))
            inner = inner[:n]
            outer = outer[:n]

            self._inner = inner
            self._outer = outer
            self.centerline = (inner + outer) / 2.0
            self.arc = _compute_arc_length(self.centerline)
            self.lap_distance_m = float(self.arc[-1])
            self.curvature = _compute_curvature(self.centerline, self.arc)

            self._detect_corners()
            log.info(
                "CornerMap built: %.0fm lap, %d corners",
                self.lap_distance_m, len(self.corners)
            )
        except Exception as e:
            log.warning("CornerMap build failed: %s", e)

    def _detect_corners(self):
        abs_kappa = np.abs(self.curvature)
        # min separation in index units (approx)
        avg_seg = self.lap_distance_m / len(self.centerline)
        min_idx_sep = max(1, int(_MIN_CORNER_SEPARATION_M / avg_seg))

        peaks = _local_maxima(abs_kappa, min_dist=min_idx_sep)
        # Filter below threshold
        peaks = [p for p in peaks if abs_kappa[p] >= _CURVATURE_THRESHOLD]

        # Sort by arc position
        peaks.sort(key=lambda i: self.arc[i])

        self.corners = []
        for t_num, peak_idx in enumerate(peaks, start=1):
            cx, cy = self.centerline[peak_idx]
            kappa   = self.curvature[peak_idx]
            direction = "left" if kappa > 0 else "right"
            self.corners.append(CornerMarker(
                id=f"T{t_num}",
                x=round(float(cx), 2),
                y=round(float(cy), 2),
                distance_m=round(float(self.arc[peak_idx]), 1),
                direction=direction,
            ))

    def snap_distance(self, x: float, y: float) -> float:
        """
        Snap a GPS point (x, y) to the nearest centerline point
        and return the arc-length distance (m).
        """
        if self.centerline is None:
            return 0.0
        pt = np.array([x, y])
        dists = np.linalg.norm(self.centerline - pt, axis=1)
        idx = int(np.argmin(dists))
        return float(self.arc[idx])

    def assign_distances(self, frames: list[RawFrame]) -> None:
        """
        Fill frame.distance_m in-place for a list of RawFrames.
        Uses nearest-centerline snapping.
        """
        if self.centerline is None:
            return
        pts = np.array([[f.x, f.y] for f in frames])
        # Batch nearest-neighbour
        all_cl = self.centerline
        for i, (px, py) in enumerate(pts):
            dists = np.linalg.norm(all_cl - np.array([px, py]), axis=1)
            frames[i].distance_m = float(self.arc[int(np.argmin(dists))])

    def as_track_data(self, max_pts: int = 700) -> Optional[TrackData]:
        if self.centerline is None:
            return None

        def _sample(arr: np.ndarray) -> np.ndarray:
            if len(arr) <= max_pts:
                return arr
            idx = np.round(np.linspace(0, len(arr) - 1, max_pts)).astype(int)
            return arr[idx]

        inner = _sample(self._inner)
        outer = _sample(self._outer)
        cl    = _sample(self.centerline)

        return TrackData(
            inner=[TrackPoint(x=round(float(p[0]), 2), y=round(float(p[1]), 2))
                   for p in inner],
            outer=[TrackPoint(x=round(float(p[0]), 2), y=round(float(p[1]), 2))
                   for p in outer],
            centerline=[TrackPoint(x=round(float(p[0]), 2), y=round(float(p[1]), 2))
                        for p in cl],
            corners=self.corners,
            lap_distance_m=round(self.lap_distance_m, 1),
        )


# ── Per-corner lap analysis ───────────────────────────────────────────────────

def analyse_corners(
    frames: list[RawFrame],
    corner_map: CornerMap,
    lap_index: int,
) -> list[CornerAnalysis]:
    """
    For each corner, find the entry/apex/exit from the distance-stamped frames.
    Returns a list of CornerAnalysis objects.
    """
    if not corner_map.corners or not frames:
        return []

    # Ensure distances are assigned
    if frames[0].distance_m is None:
        corner_map.assign_distances(frames)

    results: list[CornerAnalysis] = []

    for corner in corner_map.corners:
        apex_d = corner.distance_m

        # Window of frames within ±CORNER_WINDOW_M of apex
        window = [
            f for f in frames
            if f.distance_m is not None
            and abs(f.distance_m - apex_d) <= _CORNER_WINDOW_M
        ]

        if len(window) < 5:
            continue

        speeds = [f.speed for f in window]
        min_speed = min(speeds)
        max_speed = max(speeds)

        # Not a real braking corner if speed barely changes
        if max_speed - min_speed < _MIN_SPEED_DROP_KPH:
            continue

        # Apex = frame closest to corner apex distance
        apex_frame = min(window, key=lambda f: abs(f.distance_m - apex_d))

        # Entry = first frame in window (furthest before apex)
        entry_frame = window[0]
        # Exit = last frame in window
        exit_frame = window[-1]

        # Peak brake in entry half
        entry_half = window[: len(window) // 2]
        peak_brake = max((f.brake for f in entry_half), default=0.0)

        # Peak lateral g
        max_lat = max((abs(f.lat_acc) for f in window), default=0.0)

        # Trail braking duration: brake > 0.05 AND |steering| > 0.05 rad
        trail_frames = [
            f for f in entry_half
            if f.brake > 0.05 and abs(f.steering) > 0.05
        ]
        trail_duration = (
            trail_frames[-1].ts - trail_frames[0].ts
            if len(trail_frames) >= 2 else 0.0
        )

        results.append(CornerAnalysis(
            corner_id=corner.id,
            lap_index=lap_index,
            entry_speed_kph=round(entry_frame.speed, 1),
            apex_speed_kph=round(apex_frame.speed, 1),
            exit_speed_kph=round(exit_frame.speed, 1),
            min_speed_kph=round(min_speed, 1),
            entry_brake=round(peak_brake, 3),
            max_lat_acc=round(max_lat, 3),
            trail_brake_duration_s=round(trail_duration, 3),
            throttle_at_apex=round(apex_frame.throttle, 3),
            distance_m=round(apex_d, 1),
        ))

    return results
