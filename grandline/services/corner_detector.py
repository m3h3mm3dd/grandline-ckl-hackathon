"""
Corner detection and per-corner lap analysis for Yas Marina Circuit.

Approach:
1. Load track boundary JSON → compute centerline via KDTree nearest-neighbor matching.
2. Apply Gaussian smoothing → compute signed curvature κ = dθ/ds.
3. Find local maxima of |κ| with scipy.signal.find_peaks → corner peaks.
4. OPTIONAL: calibrate_from_gps() re-runs detection using the actual GPS racing
   line for more accurate corner positions. Called after session loads.
5. For each lap, snap GPS frames to nearest centerline point → distance_m.
6. Extract speed/brake/throttle at entry/apex/exit for each corner.

Yas Marina A2RL Circuit (inner layout, ~3 km):
  Group 1  — T1/T2/T3   first corner complex (~490–543 m)
  Group 2  — T5-T9      hairpin/marina complex (~1505–1580 m)
  Group 3  — T10-T21    final chicane complex (~2727–2957 m)
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Optional

import numpy as np

try:
    from scipy.ndimage import gaussian_filter1d
    from scipy.signal import find_peaks
    from scipy.spatial import KDTree
    _SCIPY = True
except ImportError:
    _SCIPY = False

from services.mcap_reader import RawFrame
from models.schemas import CornerAnalysis, CornerMarker, TrackData, TrackPoint

log = logging.getLogger(__name__)

# Curvature threshold for corner detection (1/m)
_CURVATURE_THRESHOLD = 0.008
# Minimum distance between corner peaks (m)
_MIN_CORNER_SEPARATION_M = 80.0
# Corner window for per-lap analysis (m on either side of apex)
_CORNER_WINDOW_M = 80.0
# Minimum speed drop to qualify as a "real" corner (vs. straight kink)
_MIN_SPEED_DROP_KPH = 8.0
# Gaussian smoothing sigma for curvature computation (in centerline index units)
_SMOOTH_SIGMA = 5


def _compute_arc_length(pts: np.ndarray) -> np.ndarray:
    """Cumulative arc-length for a 2D polyline (N,2)."""
    diffs = np.diff(pts, axis=0)
    segs  = np.linalg.norm(diffs, axis=1)
    return np.concatenate([[0.0], np.cumsum(segs)])


def _compute_curvature(pts: np.ndarray, arc: np.ndarray) -> np.ndarray:
    """Signed curvature κ = dθ/ds. Positive = left, negative = right."""
    n = len(pts)
    kappa = np.zeros(n)
    for i in range(1, n - 1):
        v1 = pts[i] - pts[i - 1]
        v2 = pts[i + 1] - pts[i]
        th1 = math.atan2(v1[1], v1[0])
        th2 = math.atan2(v2[1], v2[0])
        dth = (th2 - th1 + math.pi) % (2 * math.pi) - math.pi
        ds = arc[i + 1] - arc[i - 1]
        if ds > 0:
            kappa[i] = dth / ds
    return kappa


def _find_curvature_peaks(
    kappa: np.ndarray,
    arc: np.ndarray,
    height: float = _CURVATURE_THRESHOLD,
    min_sep_m: float = _MIN_CORNER_SEPARATION_M,
) -> list[int]:
    """Return indices of curvature peaks using scipy.signal.find_peaks."""
    abs_k = np.abs(kappa)
    avg_seg = float(arc[-1]) / max(len(arc) - 1, 1)
    min_sep_idx = max(1, int(min_sep_m / avg_seg))

    if _SCIPY:
        peaks, _ = find_peaks(abs_k, height=height, distance=min_sep_idx)
        return list(peaks)
    else:
        # Fallback: simple greedy local maxima
        peaks = []
        for i in range(1, len(abs_k) - 1):
            if abs_k[i] >= abs_k[i - 1] and abs_k[i] >= abs_k[i + 1]:
                if abs_k[i] >= height:
                    if not peaks or (i - peaks[-1]) >= min_sep_idx:
                        peaks.append(i)
        return peaks


class CornerMap:
    """
    Loaded once from track boundary file.
    Provides: centerline, arc-length, corner list, GPS snapping.

    Call calibrate_from_gps(frames) after the first lap is processed for
    GPS-accurate corner positions.
    """

    def __init__(self, bnd_path: Optional[Path] = None):
        self.centerline: Optional[np.ndarray] = None
        self.arc:        Optional[np.ndarray] = None
        self.curvature:  Optional[np.ndarray] = None
        self.corners:    list[CornerMarker]   = []
        self.lap_distance_m: float = 0.0
        self._inner: Optional[np.ndarray] = None
        self._outer: Optional[np.ndarray] = None
        self._cl_tree: Optional[object] = None  # KDTree over centerline
        self._gps_calibrated: bool = False      # True after first GPS calibration

        if bnd_path and bnd_path.exists():
            self._build(bnd_path)

    def _build(self, bnd_path: Path):
        try:
            with open(bnd_path) as f:
                bnd = json.load(f)

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

            self._inner = inner

            # ── Proper centerline via KDTree nearest-neighbor matching ────────
            # The inner/outer start at DIFFERENT positions on the circuit
            # (gap = ~190m). For each inner point find the closest outer point
            # so the midpoint is a genuine cross-track centre and the rendered
            # polygons are properly aligned.
            if _SCIPY:
                outer_tree = KDTree(outer)
                _, idxs = outer_tree.query(inner)
                paired_outer = outer[idxs]
            else:
                # Fallback: roll outer so its start aligns with inner start
                dists = np.linalg.norm(outer - inner[0], axis=1)
                offset = int(np.argmin(dists))
                outer_rolled = np.roll(outer, -offset, axis=0)
                n = min(len(inner), len(outer_rolled))
                paired_outer = outer_rolled[:n]
                inner = inner[:n]
                self._inner = inner  # keep trimmed version

            # Store ALIGNED outer for proper polygon rendering
            self._outer = paired_outer

            self.centerline = (inner + paired_outer) / 2.0

            # ── Arc length ────────────────────────────────────────────────────
            self.arc = _compute_arc_length(self.centerline)
            self.lap_distance_m = float(self.arc[-1])

            # ── Smooth centerline before curvature ───────────────────────────
            if _SCIPY:
                cl_smooth = np.column_stack([
                    gaussian_filter1d(self.centerline[:, 0], sigma=_SMOOTH_SIGMA),
                    gaussian_filter1d(self.centerline[:, 1], sigma=_SMOOTH_SIGMA),
                ])
                arc_smooth = _compute_arc_length(cl_smooth)
            else:
                cl_smooth = self.centerline
                arc_smooth = self.arc

            self.curvature = _compute_curvature(cl_smooth, arc_smooth)

            # ── Build KDTree for fast snap_distance() ─────────────────────────
            if _SCIPY:
                self._cl_tree = KDTree(self.centerline)

            self._detect_corners()
            log.info(
                "CornerMap built: %.0fm lap, %d corners detected from boundary",
                self.lap_distance_m, len(self.corners)
            )
        except Exception as e:
            log.warning("CornerMap build failed: %s", e, exc_info=True)

    def _detect_corners(self):
        """Detect corners from boundary-derived centerline curvature."""
        if self.curvature is None or self.arc is None:
            return
        peaks = _find_curvature_peaks(self.curvature, self.arc)
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

    def calibrate_from_gps(self, frames: list[RawFrame]) -> None:
        """
        Re-detect corners using the actual GPS racing line.
        More accurate than boundary curvature because it reflects where the
        car actually turns.  Call this after the first lap has been decoded.
        """
        if self.centerline is None or not _SCIPY:
            return
        if self._gps_calibrated:
            return  # already calibrated from first session

        gps_frames = [f for f in frames if f.x != 0 or f.y != 0]
        if len(gps_frames) < 200:
            return

        pts    = np.array([[f.x, f.y] for f in gps_frames])
        speeds = np.array([f.speed for f in gps_frames])

        # Smooth the GPS path
        pts_s = np.column_stack([
            gaussian_filter1d(pts[:, 0], sigma=5),
            gaussian_filter1d(pts[:, 1], sigma=5),
        ])

        arc = _compute_arc_length(pts_s)
        kappa = _compute_curvature(pts_s, arc)

        peaks = _find_curvature_peaks(
            kappa, arc,
            height=_CURVATURE_THRESHOLD,
            min_sep_m=_MIN_CORNER_SEPARATION_M,
        )
        if not peaks:
            log.warning("GPS corner calibration: no peaks found, keeping boundary corners")
            return

        # Sort by arc distance
        peaks = sorted(peaks, key=lambda p: arc[p])

        # Map GPS positions → centerline positions via KDTree
        cl_tree = self._cl_tree or KDTree(self.centerline)
        arc_cl = _compute_arc_length(self.centerline)
        abs_k = np.abs(kappa)

        # Build list: (cl_dist, cx, cy, direction) sorted by centerline position
        raw: list[tuple[float, float, float, str]] = []
        for p in peaks:
            gps_xy = pts[p]
            _, cl_idx = cl_tree.query(gps_xy)
            cl_dist = float(arc_cl[cl_idx])
            cx = float(self.centerline[cl_idx, 0])
            cy = float(self.centerline[cl_idx, 1])
            direction = "left" if kappa[p] > 0 else "right"
            raw.append((cl_dist, cx, cy, direction))

        # Sort by centerline distance for consistent T1, T2 … labelling
        raw.sort(key=lambda r: r[0])

        # Deduplicate: keep first when two corners < MIN_SEP/2 apart
        deduped: list[tuple[float, float, float, str]] = []
        for entry in raw:
            if not deduped or (entry[0] - deduped[-1][0]) > _MIN_CORNER_SEPARATION_M / 2:
                deduped.append(entry)

        prev_count = len(self.corners)
        self.corners = [
            CornerMarker(
                id=f"T{i + 1}",
                x=round(cx, 2),
                y=round(cy, 2),
                distance_m=round(cl_dist, 1),
                direction=direction,
            )
            for i, (cl_dist, cx, cy, direction) in enumerate(deduped)
        ]

        self._gps_calibrated = True
        log.info(
            "GPS-calibrated CornerMap: %d corners (was %d from boundary)",
            len(self.corners), prev_count,
        )

    def snap_distance(self, x: float, y: float) -> float:
        """
        Snap a GPS point (x, y) to the nearest centerline point
        and return the arc-length distance (m).
        """
        if self.centerline is None:
            return 0.0
        pt = np.array([x, y])
        if self._cl_tree is not None:
            _, idx = self._cl_tree.query(pt)
        else:
            dists = np.linalg.norm(self.centerline - pt, axis=1)
            idx = int(np.argmin(dists))
        return float(self.arc[idx])

    def assign_distances(self, frames: list[RawFrame]) -> None:
        """Fill frame.distance_m in-place for a list of RawFrames."""
        if self.centerline is None:
            return
        if self._cl_tree is not None:
            pts = np.array([[f.x, f.y] for f in frames])
            _, idxs = self._cl_tree.query(pts)
            for i, idx in enumerate(idxs):
                frames[i].distance_m = float(self.arc[idx])
        else:
            all_cl = self.centerline
            for i, f in enumerate(frames):
                dists = np.linalg.norm(all_cl - np.array([f.x, f.y]), axis=1)
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

        # Entry / exit
        entry_frame = window[0]
        exit_frame  = window[-1]

        # Peak brake in entry half
        entry_half = window[: len(window) // 2]
        peak_brake = max((f.brake for f in entry_half), default=0.0)

        # Peak lateral g
        max_lat = max((abs(f.lat_acc) for f in window), default=0.0)

        # Trail braking duration
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
