from __future__ import annotations

import math
from typing import Optional

import numpy as np

from services.mcap_reader import RawFrame
from models.schemas import (
    LapSummary, LapDetail, TelemetryFrame,
    BrakingZone, SectorSummary, LapComparison,
    GGPoint, GGDiagram,
    TyreState, TyreDegradation,
    DistanceComparison,
    DeltaTimeSeries, TheoreticalBest, MiniSectorBest,
    SuspensionState, SuspensionSummary, SuspensionReport,
)

# Yas Marina sector definitions — distance-based (% of lap)
_SECTOR_SPLITS = [0.33, 0.66, 1.0]

# Braking zone thresholds
_BRAKE_THRESHOLD      = 0.15
_MIN_BRAKING_DURATION = 0.2   # seconds

# Tyre temperature limits (°C)
_TYRE_COLD_LIMIT = 70.0
_TYRE_HOT_LIMIT  = 120.0

# GG diagram: downsample to keep payload manageable
_GG_DOWNSAMPLE = 5   # take every Nth frame


def _to_schema_frame(f: RawFrame, t: float) -> TelemetryFrame:
    return TelemetryFrame(
        t=round(t, 4),
        ts=f.ts,
        x=round(f.x, 3),
        y=round(f.y, 3),
        z=round(f.z, 3),
        vx=round(f.vx, 3),
        vy=round(f.vy, 3),
        speed=round(f.speed, 2),
        throttle=round(max(0.0, min(1.0, f.throttle)), 4),
        brake=round(max(0.0, min(1.0, f.brake)), 4),
        steering=round(f.steering, 4),
        gear=f.gear,
        rpm=round(f.rpm, 1),
        slip_angle=round(f.slip_angle, 5),
        lat_acc=round(f.lat_acc, 4),
        lon_acc=round(f.lon_acc, 4),
        distance_m=round(f.distance_m, 1) if f.distance_m is not None else None,
        tyre_temp_fl=round(f.tyre_temp_fl, 1) if f.tyre_temp_fl else None,
        tyre_temp_fr=round(f.tyre_temp_fr, 1) if f.tyre_temp_fr else None,
        tyre_temp_rl=round(f.tyre_temp_rl, 1) if f.tyre_temp_rl else None,
        tyre_temp_rr=round(f.tyre_temp_rr, 1) if f.tyre_temp_rr else None,
        brake_press_fl=round(f.brake_press_fl, 1) if f.brake_press_fl else None,
        brake_press_fr=round(f.brake_press_fr, 1) if f.brake_press_fr else None,
        ride_height_fl=round(f.ride_height_fl, 2) if f.ride_height_fl else None,
        ride_height_fr=round(f.ride_height_fr, 2) if f.ride_height_fr else None,
        ride_height_rl=round(f.ride_height_rl, 2) if f.ride_height_rl else None,
        ride_height_rr=round(f.ride_height_rr, 2) if f.ride_height_rr else None,
        rh_front=round(f.rh_front, 2) if f.rh_front else None,
        rh_rear=round(f.rh_rear, 2) if f.rh_rear else None,
        wheel_load_fl=round(f.wheel_load_fl, 0) if f.wheel_load_fl else None,
        wheel_load_fr=round(f.wheel_load_fr, 0) if f.wheel_load_fr else None,
        wheel_load_rl=round(f.wheel_load_rl, 0) if f.wheel_load_rl else None,
        wheel_load_rr=round(f.wheel_load_rr, 0) if f.wheel_load_rr else None,
        brake_disc_temp_fl=round(f.brake_disc_temp_fl, 1) if f.brake_disc_temp_fl else None,
        brake_disc_temp_fr=round(f.brake_disc_temp_fr, 1) if f.brake_disc_temp_fr else None,
        car_flag=f.car_flag,
        track_flag=f.track_flag,
        session_type=f.session_type,
        oil_temp=round(f.oil_temp, 1),
        water_temp=round(f.water_temp, 1),
        fuel_l=round(f.fuel_l, 2) if f.fuel_l is not None else None,
        cpu_usage=f.cpu_usage,
        gpu_usage=f.gpu_usage,
        cpu_temp=round(f.cpu_temp, 1),
        gpu_temp=round(f.gpu_temp, 1),
        wheel_slip=round(f.wheel_slip, 4),
    )


def compute_lap_detail(frames: list[RawFrame], lap_index: int) -> LapDetail:
    if not frames:
        raise ValueError("Empty lap")

    t0 = frames[0].ts
    schema_frames = [_to_schema_frame(f, f.ts - t0) for f in frames]

    speeds    = np.array([f.speed    for f in frames])
    brakes    = np.array([f.brake    for f in frames])
    throttles = np.array([f.throttle for f in frames])
    lat_acc   = np.array([f.lat_acc  for f in frames])
    lon_acc   = np.array([f.lon_acc  for f in frames])

    lap_time = frames[-1].ts - frames[0].ts

    # Lap distance from GPS arc-length (if available)
    lap_distance = None
    distances = [f.distance_m for f in frames if f.distance_m is not None]
    if distances:
        lap_distance = round(max(distances), 1)

    summary = LapSummary(
        lap_index=lap_index,
        lap_time_s=round(lap_time, 3),
        max_speed_kph=round(float(speeds.max()), 1),
        avg_speed_kph=round(float(speeds.mean()), 1),
        max_lat_acc=round(float(np.abs(lat_acc).max()), 2),
        max_lon_acc=round(float(np.abs(lon_acc).max()), 2),
        max_brake=round(float(brakes.max()), 3),
        avg_throttle=round(float(throttles.mean()), 3),
        top_gear=int(max(f.gear for f in frames)),
        frame_count=len(frames),
        lap_distance_m=lap_distance,
    )

    return LapDetail(summary=summary, frames=schema_frames)


def detect_braking_zones(frames: list[RawFrame]) -> list[BrakingZone]:
    if not frames:
        return []

    t0 = frames[0].ts
    zones: list[BrakingZone] = []
    in_zone = False
    zone_start = 0

    for i, f in enumerate(frames):
        braking = f.brake > _BRAKE_THRESHOLD
        if braking and not in_zone:
            in_zone = True
            zone_start = i
        elif not braking and in_zone:
            in_zone = False
            chunk = frames[zone_start:i]
            duration = chunk[-1].ts - chunk[0].ts
            if duration >= _MIN_BRAKING_DURATION:
                entry_speed = chunk[0].speed
                min_speed   = min(c.speed for c in chunk)
                peak_brake  = max(c.brake for c in chunk)
                zones.append(BrakingZone(
                    t_start=round(chunk[0].ts - t0, 3),
                    t_end=round(chunk[-1].ts - t0, 3),
                    x=round(chunk[0].x, 2),
                    y=round(chunk[0].y, 2),
                    entry_speed_kph=round(entry_speed, 1),
                    min_speed_kph=round(min_speed, 1),
                    peak_brake=round(peak_brake, 3),
                    duration_s=round(duration, 3),
                    distance_m=round(chunk[0].distance_m, 1) if chunk[0].distance_m else None,
                ))

    return zones


def compute_sectors(frames: list[RawFrame], lap_index: int) -> list[SectorSummary]:
    """
    Split lap into 3 sectors by distance (if available) or frame count.
    Distance-based is more accurate for comparing laps of different durations.
    """
    if not frames:
        return []

    n = len(frames)
    sectors: list[SectorSummary] = []

    # Use distance if available, else frame index
    has_distance = frames[0].distance_m is not None
    if has_distance:
        total_d = max(f.distance_m for f in frames if f.distance_m is not None)
        splits  = [total_d * p for p in _SECTOR_SPLITS]
        sector_frames: list[list[RawFrame]] = [[] for _ in _SECTOR_SPLITS]
        cur_sector = 0
        for f in frames:
            if f.distance_m is None:
                continue
            while cur_sector < len(splits) - 1 and f.distance_m > splits[cur_sector]:
                cur_sector += 1
            sector_frames[cur_sector].append(f)
    else:
        sector_frames = []
        prev_cut = 0
        for pct in _SECTOR_SPLITS:
            cut = min(int(n * pct), n)
            sector_frames.append(frames[prev_cut:cut])
            prev_cut = cut

    for s_idx, chunk in enumerate(sector_frames):
        if not chunk:
            continue
        speeds = [f.speed for f in chunk]
        sectors.append(SectorSummary(
            sector=s_idx + 1,
            lap_index=lap_index,
            time_s=round(chunk[-1].ts - chunk[0].ts, 3),
            avg_speed_kph=round(float(np.mean(speeds)), 1),
            max_speed_kph=round(float(np.max(speeds)), 1),
        ))

    return sectors


# ── GG Diagram ────────────────────────────────────────────────────────────────

def compute_gg_diagram(frames: list[RawFrame], lap_index: int) -> GGDiagram:
    """
    Build a traction-circle scatter plot for the lap.
    Returns sampled (lat_acc, lon_acc) pairs plus envelope metrics.
    """
    sampled = frames[::_GG_DOWNSAMPLE]
    t0 = frames[0].ts

    points = [
        GGPoint(
            lat_acc=round(f.lat_acc, 3),
            lon_acc=round(f.lon_acc, 3),
            speed=round(f.speed, 1),
            t=round(f.ts - t0, 3),
        )
        for f in sampled
        if abs(f.lat_acc) < 50 and abs(f.lon_acc) < 50  # sanity filter
    ]

    lat_arr = np.array([p.lat_acc for p in points])
    lon_arr = np.array([p.lon_acc for p in points])

    max_lat   = float(np.abs(lat_arr).max()) if len(lat_arr) else 0.0
    max_lon_acc = float(lon_arr[lon_arr > 0].max()) if any(lon_arr > 0) else 0.0
    max_lon_dec = float(np.abs(lon_arr[lon_arr < 0]).max()) if any(lon_arr < 0) else 0.0

    # Friction circle radius: 95th-percentile combined g-force
    combined = np.sqrt(lat_arr**2 + lon_arr**2)
    friction_r = float(np.percentile(combined, 95)) if len(combined) else 0.0

    return GGDiagram(
        lap_index=lap_index,
        points=points,
        max_lat_g=round(max_lat, 3),
        max_lon_acc_g=round(max_lon_acc, 3),
        max_lon_dec_g=round(max_lon_dec, 3),
        friction_circle_radius=round(friction_r, 3),
    )


# ── Tyre analysis ─────────────────────────────────────────────────────────────

def compute_tyre_trend(frames: list[RawFrame], sample_interval_s: float = 2.0) -> list[TyreState]:
    """
    Downsample tyre temperature data to ~every 2s.
    Returns time-series of tyre temps through the lap.
    """
    if not frames:
        return []

    states: list[TyreState] = []
    t0 = frames[0].ts
    last_sample_t = -999.0

    for f in frames:
        t_rel = f.ts - t0
        if t_rel - last_sample_t < sample_interval_s:
            continue
        last_sample_t = t_rel

        has_temps = any([
            f.tyre_temp_fl, f.tyre_temp_fr,
            f.tyre_temp_rl, f.tyre_temp_rr
        ])
        if not has_temps:
            continue

        temps = [t for t in [
            f.tyre_temp_fl, f.tyre_temp_fr,
            f.tyre_temp_rl, f.tyre_temp_rr
        ] if t is not None]
        avg = float(np.mean(temps)) if temps else None

        states.append(TyreState(
            t=round(t_rel, 2),
            distance_m=round(f.distance_m, 1) if f.distance_m else None,
            temp_fl=round(f.tyre_temp_fl, 1) if f.tyre_temp_fl else None,
            temp_fr=round(f.tyre_temp_fr, 1) if f.tyre_temp_fr else None,
            temp_rl=round(f.tyre_temp_rl, 1) if f.tyre_temp_rl else None,
            temp_rr=round(f.tyre_temp_rr, 1) if f.tyre_temp_rr else None,
            avg_temp=round(avg, 1) if avg else None,
        ))

    return states


def compute_tyre_degradation(frames: list[RawFrame], lap_index: int) -> TyreDegradation:
    """Summarise tyre state at start/end of lap and flag overheating / cold start."""
    all_temps = [
        [f.tyre_temp_fl, f.tyre_temp_fr, f.tyre_temp_rl, f.tyre_temp_rr]
        for f in frames
    ]

    def _avg(temps):
        valid = [t for t in temps if t is not None]
        return float(np.mean(valid)) if valid else None

    def _peak(attr):
        vals = [getattr(f, attr) for f in frames if getattr(f, attr) is not None]
        return round(float(max(vals)), 1) if vals else None

    start_avg = _avg(all_temps[0]) if all_temps else None
    end_avg   = _avg(all_temps[-1]) if all_temps else None

    peak_fl = _peak("tyre_temp_fl")
    peak_fr = _peak("tyre_temp_fr")
    peak_rl = _peak("tyre_temp_rl")
    peak_rr = _peak("tyre_temp_rr")

    all_peak = [p for p in [peak_fl, peak_fr, peak_rl, peak_rr] if p is not None]
    overheating = any(p > _TYRE_HOT_LIMIT for p in all_peak) if all_peak else False
    cold_start  = (start_avg is not None and start_avg < _TYRE_COLD_LIMIT)

    return TyreDegradation(
        lap_index=lap_index,
        start_avg_temp=round(start_avg, 1) if start_avg else None,
        end_avg_temp=round(end_avg, 1) if end_avg else None,
        peak_temp_fl=peak_fl,
        peak_temp_fr=peak_fr,
        peak_temp_rl=peak_rl,
        peak_temp_rr=peak_rr,
        overheating=overheating,
        cold_start=cold_start,
    )


# ── Suspension / Ride Height ──────────────────────────────────────────────────

def compute_suspension_report(
    frames: list[RawFrame],
    lap_index: int,
    sample_interval_s: float = 1.0,
) -> SuspensionReport:
    """
    Downsample suspension data to ~every 1s and compute per-lap summary stats.

    Signal notes:
    - ride_height_fl/fr/rl/rr  = damper stroke per wheel (mm) from Badenia 560
    - rh_front / rh_rear       = true centerline ride height from optical sensor (mm)

    If the MCAP file contains these topics the values will be non-None;
    otherwise the summary.has_data flag will be False.
    """
    trend: list[SuspensionState] = []
    t0 = frames[0].ts if frames else 0.0
    last_t = -999.0

    for f in frames:
        t_rel = f.ts - t0
        if t_rel - last_t < sample_interval_s:
            continue
        last_t = t_rel

        has = any([
            f.ride_height_fl, f.ride_height_fr,
            f.ride_height_rl, f.ride_height_rr,
            f.rh_front, f.rh_rear,
        ])
        if not has:
            continue

        trend.append(SuspensionState(
            t=round(t_rel, 2),
            distance_m=round(f.distance_m, 1) if f.distance_m is not None else None,
            damper_fl=round(f.ride_height_fl, 2) if f.ride_height_fl else None,
            damper_fr=round(f.ride_height_fr, 2) if f.ride_height_fr else None,
            damper_rl=round(f.ride_height_rl, 2) if f.ride_height_rl else None,
            damper_rr=round(f.ride_height_rr, 2) if f.ride_height_rr else None,
            rh_front=round(f.rh_front, 2) if f.rh_front else None,
            rh_rear=round(f.rh_rear, 2) if f.rh_rear else None,
        ))

    def _avg(attr):
        vals = [getattr(f, attr) for f in frames if getattr(f, attr) is not None and getattr(f, attr) > 0]
        return round(float(np.mean(vals)), 2) if vals else None

    def _min(attr):
        vals = [getattr(f, attr) for f in frames if getattr(f, attr) is not None and getattr(f, attr) > 0]
        return round(float(np.min(vals)), 2) if vals else None

    summary = SuspensionSummary(
        lap_index=lap_index,
        avg_damper_fl=_avg("ride_height_fl"),
        avg_damper_fr=_avg("ride_height_fr"),
        avg_damper_rl=_avg("ride_height_rl"),
        avg_damper_rr=_avg("ride_height_rr"),
        avg_rh_front=_avg("rh_front"),
        avg_rh_rear=_avg("rh_rear"),
        min_rh_front=_min("rh_front"),
        min_rh_rear=_min("rh_rear"),
        has_data=len(trend) > 0,
    )

    return SuspensionReport(summary=summary, trend=trend)


# ── Distance-normalised lap comparison ───────────────────────────────────────

def compare_laps_distance(
    frames_a: list[RawFrame],
    frames_b: list[RawFrame],
    lap_index_a: int,
    lap_index_b: int,
    n_points: int = 500,
) -> DistanceComparison:
    """
    Interpolate both laps onto a shared DISTANCE grid (not time).
    This is the correct engineering approach: you want to compare
    what happens at the SAME POINT on track, not the same moment in time.
    """
    detail_a = compute_lap_detail(frames_a, lap_index_a)
    detail_b = compute_lap_detail(frames_b, lap_index_b)

    def _dist_array(frames: list[RawFrame]) -> tuple[np.ndarray, ...]:
        """Return (distance, speed, throttle, brake) as parallel arrays."""
        has_dist = frames[0].distance_m is not None
        if has_dist:
            d = np.array([f.distance_m for f in frames])
        else:
            # Fall back: integrate speed × dt as proxy distance
            t = np.array([f.ts for f in frames])
            spd_ms = np.array([f.speed / 3.6 for f in frames])
            dt = np.diff(t, prepend=t[0])
            d = np.cumsum(spd_ms * dt)

        sp = np.array([f.speed    for f in frames])
        th = np.array([f.throttle for f in frames])
        br = np.array([f.brake    for f in frames])
        return d, sp, th, br

    d_a, sp_a, th_a, br_a = _dist_array(frames_a)
    d_b, sp_b, th_b, br_b = _dist_array(frames_b)

    # Common distance grid: 0 → min(max_dist_a, max_dist_b)
    max_common = min(d_a.max(), d_b.max())
    grid = np.linspace(0.0, max_common, n_points)

    def _interp(d, arr):
        # Sort by distance (should already be monotonic but handle edge cases)
        order = np.argsort(d)
        return np.interp(grid, d[order], arr[order])

    i_sp_a = _interp(d_a, sp_a)
    i_sp_b = _interp(d_b, sp_b)
    i_th_a = _interp(d_a, th_a)
    i_th_b = _interp(d_b, th_b)
    i_br_a = _interp(d_a, br_a)
    i_br_b = _interp(d_b, br_b)

    speed_delta = (i_sp_b - i_sp_a).round(2).tolist()

    return DistanceComparison(
        lap_a=detail_a.summary,
        lap_b=detail_b.summary,
        delta_s=round(detail_b.summary.lap_time_s - detail_a.summary.lap_time_s, 3),
        distance_grid=grid.round(1).tolist(),
        speed_a=i_sp_a.round(2).tolist(),
        speed_b=i_sp_b.round(2).tolist(),
        throttle_a=i_th_a.round(3).tolist(),
        throttle_b=i_th_b.round(3).tolist(),
        brake_a=i_br_a.round(3).tolist(),
        brake_b=i_br_b.round(3).tolist(),
        speed_delta=speed_delta,
        braking_zones_a=detect_braking_zones(frames_a),
        braking_zones_b=detect_braking_zones(frames_b),
        sectors_a=compute_sectors(frames_a, lap_index_a),
        sectors_b=compute_sectors(frames_b, lap_index_b),
    )


# Legacy time-based comparison (backward compat)
def compare_laps(
    frames_a: list[RawFrame],
    frames_b: list[RawFrame],
    lap_index_a: int,
    lap_index_b: int,
) -> "LapComparison":
    from models.schemas import LapComparison
    detail_a = compute_lap_detail(frames_a, lap_index_a)
    detail_b = compute_lap_detail(frames_b, lap_index_b)

    def normalised_speeds(frames: list[RawFrame]) -> np.ndarray:
        t0 = frames[0].ts
        t_end = frames[-1].ts
        if t_end == t0:
            return np.zeros(500)
        ts = np.array([(f.ts - t0) / (t_end - t0) for f in frames])
        sp = np.array([f.speed for f in frames])
        grid = np.linspace(0, 1, 500)
        return np.interp(grid, ts, sp)

    sp_a = normalised_speeds(frames_a)
    sp_b = normalised_speeds(frames_b)
    delta = (sp_b - sp_a).round(2).tolist()

    return LapComparison(
        lap_a=detail_a.summary,
        lap_b=detail_b.summary,
        delta_s=round(detail_b.summary.lap_time_s - detail_a.summary.lap_time_s, 3),
        speed_delta=delta,
        braking_zones_a=detect_braking_zones(frames_a),
        braking_zones_b=detect_braking_zones(frames_b),
        sectors_a=compute_sectors(frames_a, lap_index_a),
        sectors_b=compute_sectors(frames_b, lap_index_b),
    )


# ── Delta time series ─────────────────────────────────────────────────────────

def compute_delta_time(
    frames_a: list[RawFrame],
    frames_b: list[RawFrame],
    lap_index_a: int,
    lap_index_b: int,
    n_points: int = 500,
) -> DeltaTimeSeries:
    """
    Compute cumulative time delta between two laps on a shared distance axis.

    Method: for each tiny distance element dd, time = dd / speed.
    Integrate across the lap to get t(distance) for each lap, then
    delta_t = t_b(d) - t_a(d).  Positive delta means lap B is behind
    (slower) at that point on track — mirrors the F1 ΔT chart.
    """
    detail_a = compute_lap_detail(frames_a, lap_index_a)
    detail_b = compute_lap_detail(frames_b, lap_index_b)

    def _dist_speed(frames: list[RawFrame]):
        has_dist = frames[0].distance_m is not None
        if has_dist:
            d = np.array([f.distance_m for f in frames])
        else:
            t   = np.array([f.ts for f in frames])
            spd = np.array([f.speed / 3.6 for f in frames])
            dt  = np.diff(t, prepend=t[0])
            d   = np.cumsum(spd * dt)
        spd = np.array([f.speed for f in frames])
        return d, spd

    d_a, sp_a = _dist_speed(frames_a)
    d_b, sp_b = _dist_speed(frames_b)

    max_common = min(d_a.max(), d_b.max())
    grid = np.linspace(0.0, max_common, n_points)
    dd   = np.diff(grid, prepend=grid[0])

    def _cumtime(d, spd):
        # Interpolate speed onto shared grid; clamp to 1 km/h minimum
        order  = np.argsort(d)
        v_kmh  = np.interp(grid, d[order], spd[order])
        v_ms   = np.maximum(v_kmh / 3.6, 0.5)   # avoid /0
        return np.cumsum(dd / v_ms)

    t_a = _cumtime(d_a, sp_a)
    t_b = _cumtime(d_b, sp_b)
    delta = (t_b - t_a).round(3)

    def _spd_on_grid(d, spd):
        order = np.argsort(d)
        return np.interp(grid, d[order], spd[order])

    return DeltaTimeSeries(
        lap_a=detail_a.summary,
        lap_b=detail_b.summary,
        total_delta_s=round(detail_b.summary.lap_time_s - detail_a.summary.lap_time_s, 3),
        distance_grid=grid.round(1).tolist(),
        delta_t=delta.tolist(),
        speed_a=_spd_on_grid(d_a, sp_a).round(2).tolist(),
        speed_b=_spd_on_grid(d_b, sp_b).round(2).tolist(),
    )


# ── Theoretical best lap ──────────────────────────────────────────────────────

def compute_theoretical_best(
    all_laps: list[list[RawFrame]],
    n_sectors: int = 25,
) -> TheoreticalBest:
    """
    'Purple lap' — split each lap into n_sectors mini-sectors by distance,
    pick the fastest sector time from any lap, sum them.
    """
    if not all_laps:
        raise ValueError("No laps provided")

    # Build distance arrays per lap
    def _dist_time(frames: list[RawFrame]):
        has_dist = frames[0].distance_m is not None
        if has_dist:
            # distance_m is centerline position, NOT cumulative.
            # Convert to cumulative by integrating v*dt (more reliable for sectors).
            t_arr = np.array([f.ts for f in frames])
            spd   = np.array([f.speed / 3.6 for f in frames])
            dt    = np.diff(t_arr, prepend=t_arr[0])
            d     = np.cumsum(spd * dt)
        else:
            t_arr = np.array([f.ts for f in frames])
            spd   = np.array([f.speed / 3.6 for f in frames])
            dt    = np.diff(t_arr, prepend=t_arr[0])
            d     = np.cumsum(spd * dt)
        t = np.array([f.ts - frames[0].ts for f in frames])
        return d, t

    dist_times = [_dist_time(lap) for lap in all_laps]

    # Find lap distance range shared by all laps
    max_common = min(d.max() for d, _ in dist_times)

    # Sector boundaries on shared distance axis
    sector_edges = np.linspace(0.0, max_common, n_sectors + 1)

    best_sectors: list[MiniSectorBest] = []
    best_time = 0.0

    for s_idx in range(n_sectors):
        d_start = sector_edges[s_idx]
        d_end   = sector_edges[s_idx + 1]
        best_st = None
        best_lap_i = 0

        for lap_i, (d, t) in enumerate(dist_times):
            order = np.argsort(d)
            d_s, t_s = d[order], t[order]
            # Time to reach d_start and d_end by linear interpolation
            if d_s.max() < d_end:
                continue
            t_in  = float(np.interp(d_start, d_s, t_s))
            t_out = float(np.interp(d_end,   d_s, t_s))
            st = t_out - t_in
            if best_st is None or st < best_st:
                best_st    = st
                best_lap_i = lap_i

        if best_st is None:
            best_st = 0.0
        best_time += best_st
        best_sectors.append(MiniSectorBest(
            sector_idx=s_idx,
            best_lap=best_lap_i,
            sector_time_s=round(best_st, 4),
            d_start=round(d_start, 1),
            d_end=round(d_end, 1),
        ))

    # Best real lap time
    real_times = [lap[-1].ts - lap[0].ts for lap in all_laps]
    best_real  = min(real_times)

    return TheoreticalBest(
        theoretical_time_s=round(best_time, 3),
        best_real_time_s=round(best_real, 3),
        time_saved_s=round(best_real - best_time, 3),
        n_sectors=n_sectors,
        sectors=best_sectors,
        sector_laps=[s.best_lap for s in best_sectors],
    )
