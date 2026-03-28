"""
Rule-based race coaching engine — zero API keys required.

Generates three event types during lap replay:
  feedback       — post-corner analysis, fired ~100 m after each apex
  recommendation — corner approach tip, fired ~280 m before each apex
  motivation     — lap delta updates, sector PBs, flag calls

All metrics compared against the session's personal-best lap.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

log = logging.getLogger(__name__)


# ── Tuning constants ──────────────────────────────────────────────────────────
_REC_FIRE_M:          float = 280.0   # how far before apex to fire recommendation
_REC_COOLDOWN_M:      float = 180.0   # min gap between two recommendations
_FB_FIRE_M:           float = 120.0   # how far after apex to fire feedback
_SPD_NOTICE_KPH:      float = 3.0     # min speed delta worth calling out
_SECTOR_INTERVAL:     float = 0.25    # 4 sectors per lap


# ── Reference profile ─────────────────────────────────────────────────────────

@dataclass
class ReferenceProfile:
    """
    Distance-indexed arrays built from the best lap.

    The reference axis is `distance_m` (centerline position 0…lap_len)
    — the same coordinate used by corner markers and TelemetryFrame.distance_m.
    This lets us call ref.speed_at(frame.distance_m) directly.
    """
    dist:     np.ndarray   # (N,) centerline distance (m)
    speed:    np.ndarray   # (N,) km/h
    throttle: np.ndarray   # (N,) 0–1
    brake:    np.ndarray   # (N,) 0–1
    lap_time: float        # seconds (best lap duration)
    lap_dist: float        # total metres (centerline length)

    @classmethod
    def build(cls, frames: list) -> Optional["ReferenceProfile"]:
        """Build from a list of RawFrame objects (session.raw_laps[i])."""
        if not frames:
            return None

        # ── Distance axis: prefer stored distance_m (set by corner_map.assign_distances)
        raw_dist = [getattr(f, "distance_m", None) for f in frames]
        has_dist = sum(1 for d in raw_dist if d is not None) > len(frames) // 2

        if has_dist:
            # Fill any None gaps with linear interpolation / forward fill
            dist_arr = np.array([d if d is not None else 0.0 for d in raw_dist], dtype=float)
            # Forward fill zeros caused by None
            mask = dist_arr == 0.0
            if mask[0]:
                dist_arr[0] = 0.0
            for i in range(1, len(dist_arr)):
                if mask[i]:
                    dist_arr[i] = dist_arr[i - 1]
        else:
            # Fall back: integrate speed × dt
            log.warning("ReferenceProfile: distance_m not available, integrating speed×dt")
            ts  = np.array([f.ts for f in frames], dtype=float)
            # Detect nanoseconds (ROS ns timestamps are > 1e12 when relative)
            dt = np.diff(ts, prepend=ts[0])
            if ts[-1] > 1e12:          # nanosecond epoch timestamps
                dt = dt * 1e-9
            spd = np.array([f.speed for f in frames], dtype=float) / 3.6
            dist_arr = np.cumsum(np.abs(spd * dt))

        speed    = np.array([f.speed    for f in frames], dtype=float)
        throttle = np.array([f.throttle for f in frames], dtype=float)
        brake    = np.array([f.brake    for f in frames], dtype=float)

        # ── Lap time in seconds ──
        ts_raw = np.array([f.ts for f in frames], dtype=float)
        lap_time = float(ts_raw[-1] - ts_raw[0])
        if lap_time > 1e9:            # nanoseconds → seconds
            lap_time *= 1e-9
        elif lap_time < 0.1:          # suspiciously short — fallback
            lap_time = 105.0

        lap_dist = float(dist_arr[-1]) if dist_arr[-1] > 0 else 5800.0

        log.info("ReferenceProfile built: dist_axis=%s  lap=%.1fs  %.0fm  %d frames",
                 "distance_m" if has_dist else "integrated",
                 lap_time, lap_dist, len(frames))

        return cls(
            dist=dist_arr, speed=speed, throttle=throttle,
            brake=brake, lap_time=lap_time, lap_dist=lap_dist,
        )

    # ── Lookup helpers ────────────────────────────────────────────────────────

    def speed_at(self, d: float) -> float:
        d = float(d) % self.lap_dist if self.lap_dist > 0 else float(d)
        return float(np.interp(d, self.dist, self.speed))

    def throttle_at(self, d: float) -> float:
        d = float(d) % self.lap_dist if self.lap_dist > 0 else float(d)
        return float(np.interp(d, self.dist, self.throttle))

    def brake_at(self, d: float) -> float:
        d = float(d) % self.lap_dist if self.lap_dist > 0 else float(d)
        return float(np.interp(d, self.dist, self.brake))

    def time_at_dist(self, d: float) -> float:
        """Approximate elapsed time to reach distance d (using speed profile)."""
        d = min(float(d), self.lap_dist)
        if not hasattr(self, "_time_arr"):
            spd_ms = self.speed / 3.6
            spd_ms[spd_ms < 1.0] = 1.0  # avoid /0
            dd = np.diff(self.dist, prepend=self.dist[0])
            dd[0] = 0.0
            self._time_arr = np.cumsum(dd / spd_ms)
        return float(np.interp(d, self.dist, self._time_arr))


# ── Corner state machine ───────────────────────────────────────────────────────

@dataclass
class _CornerState:
    corner_id:     str
    apex_dist:     float
    direction:     str
    rec_fired:     bool  = False
    fb_fired:      bool  = False
    entry_speed:   float = 0.0
    entry_brake:   float = 0.0
    apex_speed:    float = 0.0
    apex_throttle: float = 0.0
    exit_speed:    float = 0.0


# ── Coaching engine ───────────────────────────────────────────────────────────

class CoachingEngine:
    """
    Instantiate once per lap replay. Call process_frame() on every frame.
    Returns list of coaching event dicts (possibly empty).

    Event schema:
      { kind, corner, headline, detail, icon, delta_s }
    """

    def __init__(
        self,
        ref:      Optional[ReferenceProfile],
        corners:  list,       # CornerMarker objects (must have .distance_m and .id)
        lap_dist: float = 5800.0,
    ):
        self._ref       = ref
        self._lap_dist  = lap_dist or 5800.0
        self._last_rec_dist:  float     = -999.0
        self._sector_fired:   set[int]  = set()
        self._frame_count:    int       = 0
        self._last_d:         float     = 0.0

        # Build per-corner state, sorted by apex_dist
        self._corners_sorted: list[_CornerState] = []
        for c in corners:
            d = getattr(c, "distance_m", None)
            if d is not None:
                self._corners_sorted.append(_CornerState(
                    corner_id=str(c.id),
                    apex_dist=float(d),
                    direction=getattr(c, "direction", "unknown"),
                ))
        self._corners_sorted.sort(key=lambda cs: cs.apex_dist)

        log.info("CoachingEngine: %d corners  ref=%s  lap_dist=%.0fm",
                 len(self._corners_sorted), "YES" if ref else "NONE", self._lap_dist)

    # ── Main per-frame call ───────────────────────────────────────────────────

    def process_frame(self, frame) -> list[dict]:
        events: list[dict] = []
        self._frame_count += 1

        # ── Get current distance along lap ──
        # Prefer frame.distance_m (set by corner_map.assign_distances in backend)
        d_raw = getattr(frame, "distance_m", None)
        if d_raw is not None and d_raw > 0:
            d = float(d_raw)
        else:
            # Fallback: integrate speed × 50ms (assumes 20Hz)
            spd_ms = max(getattr(frame, "speed", 0), 0) / 3.6
            self._last_d = self._last_d + spd_ms * 0.05
            d = self._last_d

        # ── Fire opening message ──
        if self._frame_count == 5:
            events.append(_evt(
                "motivation", None,
                "LIGHTS OUT — ATTACK",
                "Push to the first corner, get on the reference line early. "
                "Your best lap is waiting.",
                "🟢", None,
            ))

        # ── Per-corner events ─────────────────────────────────────────────────
        for cs in self._corners_sorted:
            dist_to_apex = cs.apex_dist - d

            # Recommendation window: 0–280m before apex
            if (not cs.rec_fired
                    and 0 < dist_to_apex <= _REC_FIRE_M
                    and d - self._last_rec_dist > _REC_COOLDOWN_M):
                ev = self._make_recommendation(frame, cs, dist_to_apex, d)
                if ev:
                    events.append(ev)
                    cs.rec_fired = True
                    cs.entry_speed = getattr(frame, "speed", 0)
                    cs.entry_brake = getattr(frame, "brake", 0)
                    self._last_rec_dist = d

            # Apex snapshot: within ±25m of apex
            if (abs(dist_to_apex) < 25 and cs.apex_speed == 0.0):
                cs.apex_speed    = getattr(frame, "speed", 0)
                cs.apex_throttle = getattr(frame, "throttle", 0)

            # Feedback: apex has passed and we're 120m+ beyond it
            if (not cs.fb_fired and dist_to_apex < -_FB_FIRE_M):
                cs.exit_speed = getattr(frame, "speed", 0)
                ev = self._make_feedback(frame, cs, d)
                if ev:
                    events.append(ev)
                cs.fb_fired = True

        # ── Sector motivation ────────────────────────────────────────────────
        prog   = d / self._lap_dist if self._lap_dist > 0 else 0.0
        sector = int(prog / _SECTOR_INTERVAL)
        if 0 < sector <= 3 and sector not in self._sector_fired:
            self._sector_fired.add(sector)
            ev = self._make_sector_motivation(d, sector)
            if ev:
                events.append(ev)

        self._last_d = d
        return events

    # ── Event builders ────────────────────────────────────────────────────────

    def _make_recommendation(
        self, frame, cs: _CornerState, dist_to_apex: float, d: float
    ) -> Optional[dict]:
        ref_spd = self._ref.speed_at(d) if self._ref else 0.0
        cur_spd = float(getattr(frame, "speed", 0))
        delta   = cur_spd - ref_spd        # positive = you're carrying more speed
        turn    = cs.direction
        dist_m  = int(dist_to_apex)
        corner  = cs.corner_id

        if delta > _SPD_NOTICE_KPH:
            headline = f"{corner} IN {dist_m}M — BRAKE EARLIER"
            detail   = (f"Carrying +{delta:.0f} km/h vs best lap. "
                        f"Move the brake point 10–15m earlier "
                        f"and trail the brake through the {turn} apex.")
            icon = "🟡"
        elif delta < -_SPD_NOTICE_KPH:
            headline = f"{corner} IN {dist_m}M — PUSH THE ENTRY"
            detail   = (f"{abs(delta):.0f} km/h short of best entry speed. "
                        f"Hold full throttle longer, delay the brake "
                        f"and trust the car into the {turn} corner.")
            icon = "🔵"
        else:
            headline = f"{corner} IN {dist_m}M — ON REFERENCE ✓"
            detail   = (f"Entry speed matching best lap. "
                        f"Hit the apex, get back to full throttle as early as you can.")
            icon = "✅"

        return _evt("recommendation", corner, headline, detail, icon, round(delta * 0.02, 2))

    def _make_feedback(
        self, frame, cs: _CornerState, d: float
    ) -> Optional[dict]:
        corner = cs.corner_id
        if cs.apex_speed < 1.0:
            return None  # never captured the apex

        ref_apex = self._ref.speed_at(cs.apex_dist)          if self._ref else cs.apex_speed
        ref_exit = self._ref.speed_at(cs.apex_dist + 80)     if self._ref else cs.exit_speed
        ref_thr  = self._ref.throttle_at(cs.apex_dist)       if self._ref else 0.5

        apex_delta = cs.apex_speed  - ref_apex    # positive = faster
        exit_delta = cs.exit_speed  - ref_exit
        thr_pct    = int(cs.apex_throttle * 100)
        ref_thr_pct = int(ref_thr * 100)

        # Pick the more significant story: apex vs exit
        if abs(exit_delta) >= abs(apex_delta):
            if exit_delta > _SPD_NOTICE_KPH:
                headline = f"{corner} ✓ GREAT EXIT +{exit_delta:.0f} km/h"
                detail   = (f"Exit speed {exit_delta:.0f} km/h above reference. "
                            f"Carry that momentum — don't lift until the next braking zone.")
                icon = "🟢"
            elif exit_delta < -_SPD_NOTICE_KPH:
                if thr_pct < ref_thr_pct - 12:
                    reason = f"throttle only {thr_pct}% at apex vs {ref_thr_pct}% reference — commit earlier"
                elif cs.entry_brake > 0.7:
                    reason = "over-braking into the corner — ease off by 10%"
                else:
                    reason = "too much mid-corner rotation — straighten your entry line"
                headline = f"{corner} SLOW EXIT −{abs(exit_delta):.0f} km/h"
                detail   = f"Exit speed {abs(exit_delta):.0f} km/h short of reference. Cause: {reason}."
                icon = "🔴"
            else:
                headline = f"{corner} CLEAN — MATCHED"
                detail   = f"Exit speed within {abs(exit_delta):.1f} km/h of best. Solid corner."
                icon = "✅"
        else:
            if apex_delta > _SPD_NOTICE_KPH:
                headline = f"{corner} FAST APEX +{apex_delta:.0f} km/h"
                detail   = (f"Apex speed {apex_delta:.0f} km/h over reference. "
                            f"Use all the road on exit — open the steering wheel early.")
                icon = "🟢"
            elif apex_delta < -_SPD_NOTICE_KPH:
                headline = f"{corner} SLOW APEX −{abs(apex_delta):.0f} km/h"
                detail   = (f"Apex speed {abs(apex_delta):.0f} km/h under reference. "
                            f"{'Releasing brake too early — stay on the pedal longer.' if cs.entry_brake < 0.4 else 'Too much entry braking — smooth the application.'}")
                icon = "🟡"
            else:
                headline = f"{corner} APEX MATCHED ✓"
                detail   = f"Apex speed on reference. Focus on exit — full throttle as early as possible."
                icon = "✅"

        # Append throttle note if significantly off
        if thr_pct < ref_thr_pct - 15:
            detail += f" Note: throttle {thr_pct}% at apex (ref {ref_thr_pct}%) — you can commit earlier."

        time_est = round(-(exit_delta / max(cs.exit_speed, 60)) * 0.6, 2)
        return _evt("feedback", corner, headline, detail, icon, time_est)

    def _make_sector_motivation(self, d: float, sector: int) -> Optional[dict]:
        names = {1: "SECTOR 1", 2: "SECTOR 2", 3: "SECTOR 3"}
        name  = names.get(sector, f"SECTOR {sector}")

        if not self._ref:
            return _evt("motivation", None,
                        f"{name} COMPLETE",
                        "No reference lap yet — this lap builds your baseline.",
                        "⏱", None)

        ref_t   = self._ref.time_at_dist(d)
        # Estimate current elapsed time proportionally
        est_lap = self._ref.lap_time * (d / self._lap_dist) if self._lap_dist > 0 else ref_t
        delta   = est_lap - ref_t   # positive = we're slower (took longer)

        if delta < -0.30:
            headline = f"{name} — AHEAD OF BEST 🔥"
            detail   = f"{abs(delta):.2f}s clear of reference split. Maintain the pace — don't over-drive."
            icon = "🔥"
        elif delta < 0.15:
            headline = f"{name} — ON PACE ✓"
            detail   = f"Within {abs(delta):.2f}s of reference. Personal best lap is on the cards."
            icon = "🟢"
        elif delta < 0.60:
            headline = f"{name} — +{delta:.2f}s"
            detail   = f"Slightly off reference. Push the next sector — one clean corner recovers this."
            icon = "🟡"
        else:
            headline = f"{name} — +{delta:.2f}s BEHIND"
            detail   = f"Need to find {delta:.2f}s. Identify one place you can attack harder."
            icon = "🔴"

        return _evt("motivation", None, headline, detail, icon, round(delta, 2))


# ── Helpers ───────────────────────────────────────────────────────────────────

def _evt(
    kind:    str,
    corner:  Optional[str],
    headline: str,
    detail:   str,
    icon:     str,
    delta_s:  Optional[float],
) -> dict:
    return {
        "kind":     kind,
        "corner":   corner,
        "headline": str(headline),
        "detail":   str(detail),
        "icon":     str(icon),
        "delta_s":  delta_s,
    }


def build_coaching_engine(session, lap_index: int) -> CoachingEngine:
    """
    Factory: build a CoachingEngine for a lap in a session.
    Uses the personal-best lap (by time) as reference,
    falling back to 2nd-best if lap_index IS the best.
    """
    if not session:
        return CoachingEngine(None, [], 5800.0)

    n = len(session.lap_details)
    if n == 0:
        return CoachingEngine(None, [], 5800.0)

    # Find personal best lap
    times = sorted(
        [(session.lap_details[i].summary.lap_time_s, i) for i in range(n)]
    )
    best_idx = times[0][1]
    if best_idx == lap_index and len(times) > 1:
        best_idx = times[1][1]

    ref_frames = session.raw_laps[best_idx] if best_idx < len(session.raw_laps) else []
    ref = ReferenceProfile.build(ref_frames) if ref_frames else None

    corner_map = getattr(session, "corner_map", None)
    corners    = corner_map.corners if (corner_map and hasattr(corner_map, "corners")) else []

    lap_dist = session.lap_details[lap_index].summary.lap_distance_m or 5800.0

    engine = CoachingEngine(ref=ref, corners=corners, lap_dist=lap_dist)
    log.info("CoachingEngine lap=%d ref=lap%d corners=%d dist=%.0fm",
             lap_index, best_idx, len(corners), lap_dist)
    return engine
