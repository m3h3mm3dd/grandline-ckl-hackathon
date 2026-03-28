from __future__ import annotations
from typing import Literal, Optional
from pydantic import BaseModel, Field


# ── Telemetry frame (one row decoded from StateEstimation @ 100 Hz) ───────────

class TelemetryFrame(BaseModel):
    t:          float   # seconds since lap start
    ts:         float   # original ROS timestamp (nanoseconds)

    # Position (map frame, metres)
    x:          float
    y:          float
    z:          float

    # Velocity
    vx:         float   # longitudinal m/s
    vy:         float   # lateral m/s
    speed:      float   # |v| km/h

    # Driver inputs
    throttle:   float   # 0–1
    brake:      float   # 0–1
    steering:   float   # rad

    # Powertrain
    gear:       int
    rpm:        float

    # Dynamics
    slip_angle: float   # rad
    lat_acc:    float   # m/s² (Kistler)
    lon_acc:    float   # m/s²

    # Distance along lap (computed from GPS arc-length)
    distance_m: Optional[float] = None

    # Tyres (optional — not all files have every field)
    tyre_temp_fl: Optional[float] = None
    tyre_temp_fr: Optional[float] = None
    tyre_temp_rl: Optional[float] = None
    tyre_temp_rr: Optional[float] = None

    brake_press_fl: Optional[float] = None
    brake_press_fr: Optional[float] = None

    # Suspension (Badenia 560)
    ride_height_fl: Optional[float] = None
    ride_height_fr: Optional[float] = None
    ride_height_rl: Optional[float] = None
    ride_height_rr: Optional[float] = None

    # Wheel loads
    wheel_load_fl: Optional[float] = None
    wheel_load_fr: Optional[float] = None
    wheel_load_rl: Optional[float] = None
    wheel_load_rr: Optional[float] = None

    # Brake disc temps
    brake_disc_temp_fl: Optional[float] = None
    brake_disc_temp_fr: Optional[float] = None

    # ── Race control & car system state ──────────────────────────────────────
    car_flag:    int = 0   # 0=none 1=green 2=yellow 3=red 4=checkered
    track_flag:  int = 0   # bitmask from rc_status_01
    session_type: int = 0  # 0=unknown 1=practice 2=qualifying 3=race
    oil_temp:    float = 0.0   # °C (ICEStatus02)
    water_temp:  float = 0.0   # °C (ICEStatus02)
    fuel_l:      Optional[float] = None  # litres remaining (ICEStatus01)
    cpu_usage:   int = 0   # % (HLMsg06 — AI computer)
    gpu_usage:   int = 0   # %
    cpu_temp:    float = 0.0   # °C
    gpu_temp:    float = 0.0   # °C
    wheel_slip:  float = 0.0   # 0–1 normalised, derived from wheel speeds


# ── Lap ───────────────────────────────────────────────────────────────────────

class LapSummary(BaseModel):
    lap_index:    int
    lap_time_s:   float
    max_speed_kph: float
    avg_speed_kph: float
    max_lat_acc:  float
    max_lon_acc:  float
    max_brake:    float
    avg_throttle: float
    top_gear:     int
    frame_count:  int
    lap_distance_m: Optional[float] = None  # total GPS arc-length


class LapDetail(BaseModel):
    summary: LapSummary
    frames:  list[TelemetryFrame]


# ── Session ───────────────────────────────────────────────────────────────────

class SessionMeta(BaseModel):
    session_id: str
    filename:   str
    duration_s: float
    lap_count:  int
    scenario:   Literal["good_lap", "fast_laps", "wheel_to_wheel", "unknown"]
    uploaded_at: str
    preloaded:  bool = False   # True if auto-loaded from hackathon data dir


# ── Analysis ──────────────────────────────────────────────────────────────────

class BrakingZone(BaseModel):
    t_start: float
    t_end:   float
    x:       float
    y:       float
    entry_speed_kph: float
    min_speed_kph:   float
    peak_brake:      float
    duration_s:      float
    distance_m:      Optional[float] = None   # distance along lap at entry


class SectorSummary(BaseModel):
    sector:       int
    lap_index:    int
    time_s:       float
    avg_speed_kph: float
    max_speed_kph: float


# ── GG Diagram (traction circle) ─────────────────────────────────────────────

class GGPoint(BaseModel):
    lat_acc:  float   # m/s²  (positive = left)
    lon_acc:  float   # m/s²  (positive = forward/acceleration)
    speed:    float   # kph at this moment
    t:        float   # lap time


class GGDiagram(BaseModel):
    lap_index: int
    points:    list[GGPoint]
    max_lat_g: float
    max_lon_acc_g: float   # max braking decel (positive)
    max_lon_dec_g: float   # max traction (positive)
    friction_circle_radius: float   # estimated limit g


# ── Corner analysis ───────────────────────────────────────────────────────────

class CornerAnalysis(BaseModel):
    corner_id:     str    # e.g. "T1", "T5"
    lap_index:     int
    entry_speed_kph: float
    apex_speed_kph:  float
    exit_speed_kph:  float
    min_speed_kph:   float
    entry_brake:     float   # peak brake at corner entry
    max_lat_acc:     float   # peak lateral load through corner
    trail_brake_duration_s: float  # time with both brake > 0.05 AND steering > 0.05
    throttle_at_apex: float  # throttle at apex point
    distance_m:      float   # distance along lap at corner entry


class LapCornerReport(BaseModel):
    lap_index: int
    corners:   list[CornerAnalysis]


# ── Tyre analysis ─────────────────────────────────────────────────────────────

class TyreState(BaseModel):
    t:        float
    distance_m: Optional[float] = None
    temp_fl:  Optional[float] = None
    temp_fr:  Optional[float] = None
    temp_rl:  Optional[float] = None
    temp_rr:  Optional[float] = None
    avg_temp: Optional[float] = None   # mean of all 4


class TyreDegradation(BaseModel):
    lap_index: int
    start_avg_temp: Optional[float] = None
    end_avg_temp:   Optional[float] = None
    peak_temp_fl:   Optional[float] = None
    peak_temp_fr:   Optional[float] = None
    peak_temp_rl:   Optional[float] = None
    peak_temp_rr:   Optional[float] = None
    overheating:    bool = False   # any tyre exceeded 120°C
    cold_start:     bool = False   # started below 70°C


# ── Track data ────────────────────────────────────────────────────────────────

class TrackPoint(BaseModel):
    x: float
    y: float


class CornerMarker(BaseModel):
    id:    str    # "T1" … "T21"
    x:     float  # centerline position
    y:     float
    distance_m: float
    direction: Literal["left", "right"]


class TrackData(BaseModel):
    inner: list[TrackPoint]
    outer: list[TrackPoint]
    centerline: list[TrackPoint]
    corners:    list[CornerMarker]
    lap_distance_m: float   # total centerline arc-length


# ── Delta time series ─────────────────────────────────────────────────────────

class DeltaTimeSeries(BaseModel):
    """Cumulative time delta between two laps, projected on a shared distance axis."""
    lap_a:         LapSummary
    lap_b:         LapSummary
    total_delta_s: float              # lap_b_time - lap_a_time (negative = B is faster)
    distance_grid: list[float]        # metres, shared x-axis
    delta_t:       list[float]        # seconds — positive means lap B is behind (slower)
    speed_a:       list[float]        # kph on shared grid (reference lap)
    speed_b:       list[float]        # kph on shared grid (comparison lap)


# ── Theoretical best lap ──────────────────────────────────────────────────────

class MiniSectorBest(BaseModel):
    sector_idx:  int
    best_lap:    int    # lap index that produced this best sector time
    sector_time_s: float
    d_start:     float
    d_end:       float

class TheoreticalBest(BaseModel):
    """'Purple lap' — the fastest possible lap by combining best mini-sectors."""
    theoretical_time_s: float
    best_real_time_s:   float        # actual best lap for comparison
    time_saved_s:       float        # how much faster than best real lap
    n_sectors:          int
    sectors:            list[MiniSectorBest]
    sector_laps:        list[int]    # which lap index contributed each sector


# ── Corner scoreboard ─────────────────────────────────────────────────────────

class CornerScoreRow(BaseModel):
    corner_id:    str
    distance_m:   float
    direction:    str
    min_speed:    float   # apex speed km/h
    entry_speed:  float
    exit_speed:   float
    peak_brake:   float   # 0-1
    peak_lat_g:   float   # m/s²
    lap_index:    int     # which lap this row is for

class CornerScoreboard(BaseModel):
    """Per-corner best metrics, optionally across all laps."""
    session_id: str
    laps:       list[int]
    rows:       list[CornerScoreRow]


# ── Distance-normalised lap comparison ───────────────────────────────────────

class DistanceComparison(BaseModel):
    lap_a: LapSummary
    lap_b: LapSummary
    delta_s: float              # lap_b_time - lap_a_time
    distance_grid: list[float]  # metres, shared x-axis (500 points)
    speed_a: list[float]        # kph interpolated on distance_grid
    speed_b: list[float]
    throttle_a: list[float]
    throttle_b: list[float]
    brake_a: list[float]
    brake_b: list[float]
    speed_delta: list[float]    # speed_b - speed_a (positive = B faster)
    braking_zones_a: list[BrakingZone]
    braking_zones_b: list[BrakingZone]
    sectors_a: list[SectorSummary]
    sectors_b: list[SectorSummary]


# Legacy time-based comparison (kept for backward compatibility)
class LapComparison(BaseModel):
    lap_a: LapSummary
    lap_b: LapSummary
    delta_s:        float
    speed_delta:    list[float]
    braking_zones_a: list[BrakingZone]
    braking_zones_b: list[BrakingZone]
    sectors_a: list[SectorSummary]
    sectors_b: list[SectorSummary]


# ── AI Coach ──────────────────────────────────────────────────────────────────

class CoachRequest(BaseModel):
    session_id: str
    lap_index:  Optional[int] = None     # None → whole session debrief
    compare_lap: Optional[int] = None   # triggers lap comparison coaching
    question:   Optional[str] = None    # follow-up question
    include_corners: bool = True        # enrich prompt with corner data
    include_tyres:   bool = True        # enrich prompt with tyre data


class CoachMessage(BaseModel):
    role:    Literal["engineer", "driver"]
    content: str


class CoachResponse(BaseModel):
    messages: list[CoachMessage]
    coaching_points: list[str]   # structured bullets for UI cards


# ── Real-time stream frame ────────────────────────────────────────────────────

class RealtimeEvent(BaseModel):
    type:     Literal["frame", "coach_tip", "lap_complete", "session_end"]
    t:        float
    payload:  dict
