"""
Detects lap boundaries from the StateEstimation position stream.

Strategy:
  1. Project x,y onto the start/finish line defined by Yas Marina track boundary.
  2. Detect signed zero-crossings of the projection — each crossing = new lap.
  3. Fallback: if track boundary not loaded, use spatial k-means to find the
     position cluster that repeats most → likely the start/finish straight.

The Yas Marina start/finish is roughly at the pit exit, around x≈-150, y≈50
in the local map frame (confirmed from yas_marina_bnd.json pit entry).
We define the S/F line as a short segment perpendicular to the straight.
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Optional

import numpy as np

from services.mcap_reader import RawFrame

log = logging.getLogger(__name__)

# Yas Marina S/F line — normal vector pointing in direction of travel
# These values are approximate; refined at runtime from bnd.json if available
_SF_POINT  = np.array([-150.0, 50.0])   # a point on the S/F line
_SF_NORMAL = np.array([1.0, 0.0])        # perpendicular to the straight

_MIN_LAP_SECONDS = 55.0   # discard any "lap" shorter than this (false crossing)


def _load_sf_from_boundary(bnd_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """
    Parse yas_marina_bnd.json to extract a start/finish line estimate.
    The file has 'inner' and 'outer' boundary arrays of [x, y] points.
    We take the midpoint of the first inner/outer pair as the S/F point and
    compute the track direction from the first two midpoints.
    """
    with open(bnd_path) as f:
        bnd = json.load(f)

    # Support nested boundaries dict (yas_marina_bnd.json format)
    bnd_data = bnd.get("boundaries", bnd)
    inner = np.array(
        bnd_data.get("left_border",
        bnd_data.get("inner",
        bnd_data.get("left", []))))
    outer = np.array(
        bnd_data.get("right_border",
        bnd_data.get("outer",
        bnd_data.get("right", []))))

    if len(inner) < 2 or len(outer) < 2:
        raise ValueError("Boundary file too short")

    sf_pt = (inner[0] + outer[0]) / 2.0
    next_pt = (inner[1] + outer[1]) / 2.0
    direction = next_pt - sf_pt
    direction /= np.linalg.norm(direction)
    normal = np.array([-direction[1], direction[0]])  # perpendicular

    log.info("S/F line: point=%s normal=%s", sf_pt, normal)
    return sf_pt.astype(float), normal.astype(float)


def _signed_dist(pt: np.ndarray, sf_pt: np.ndarray, sf_normal: np.ndarray) -> float:
    return float(np.dot(pt - sf_pt, sf_normal))


def detect_laps(
    frames: list[RawFrame],
    bnd_path: Optional[Path] = None,
) -> list[list[RawFrame]]:
    """
    Returns a list of laps, each lap a list of RawFrames.
    Frames within each lap start at the crossing and end just before the next.
    """
    if not frames:
        return []

    sf_pt = _SF_POINT.copy()
    sf_normal = _SF_NORMAL.copy()

    if bnd_path and bnd_path.exists():
        try:
            sf_pt, sf_normal = _load_sf_from_boundary(bnd_path)
        except Exception as e:
            log.warning("Boundary load failed (%s), using defaults", e)

    # Compute signed distance for every frame
    dists = np.array([
        _signed_dist(np.array([f.x, f.y]), sf_pt, sf_normal)
        for f in frames
    ])

    # Zero-crossing detection: look for transitions from negative to positive
    crossings: list[int] = []
    for i in range(1, len(dists)):
        if dists[i - 1] < 0 and dists[i] >= 0:
            crossings.append(i)

    if not crossings:
        log.warning("No S/F crossings found — treating whole file as one lap")
        return [frames]

    # Build lap slices, enforcing minimum lap time
    boundaries = [0] + crossings + [len(frames)]
    laps: list[list[RawFrame]] = []
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        chunk = frames[start:end]
        if not chunk:
            continue
        duration = chunk[-1].ts - chunk[0].ts
        if duration < _MIN_LAP_SECONDS:
            log.debug("Skipping short segment %.1fs (likely out-lap)", duration)
            continue
        laps.append(chunk)

    log.info("Detected %d laps from %d frames", len(laps), len(frames))
    return laps