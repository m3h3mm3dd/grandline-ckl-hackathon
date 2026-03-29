from __future__ import annotations

import logging
from pathlib import Path
from typing import Generator, Optional

import numpy as np
from mcap.reader import make_reader
from mcap_ros2.decoder import DecoderFactory

log = logging.getLogger(__name__)

# ── Topic names (exactly as in the MCAP files) ────────────────────────────────
TOPIC_STATE      = "/constructor0/state_estimation"
TOPIC_KISTLER_ACC  = "/constructor0/can/kistler_acc_body"
TOPIC_KISTLER_CORR = "/constructor0/can/kistler_correvit"
TOPIC_TPMS_F     = "/constructor0/can/badenia_560_tpms_front"
TOPIC_TPMS_R     = "/constructor0/can/badenia_560_tpms_rear"
TOPIC_SURF_F     = "/constructor0/can/badenia_560_tyre_surface_temp_front"
TOPIC_SURF_R     = "/constructor0/can/badenia_560_tyre_surface_temp_rear"
TOPIC_RIDE_F     = "/constructor0/can/badenia_560_ride_front"
TOPIC_RIDE_R     = "/constructor0/can/badenia_560_ride_rear"
TOPIC_WHEEL_LOAD = "/constructor0/can/badenia_560_wheel_load"
TOPIC_BRAKE_TEMP = "/constructor0/can/badenia_560_brake_disk_temp"
TOPIC_CAMERA_FL  = "/constructor0/sensor/camera_fl/compressed_image"
TOPIC_CAMERA_R   = "/constructor0/sensor/camera_r/compressed_image"
TOPIC_GPS        = "/constructor0/vectornav/raw/gps"
TOPIC_RC         = "/constructor0/can/rc_status_01"
TOPIC_ICE1       = "/constructor0/can/ice_status_01"
TOPIC_ICE2       = "/constructor0/can/ice_status_02"
TOPIC_HL6        = "/constructor0/can/hl_msg_06"
TOPIC_WSS        = "/constructor0/can/wheels_speed_01"

_TELEMETRY_TOPICS = {
    TOPIC_STATE, TOPIC_KISTLER_ACC, TOPIC_KISTLER_CORR,
    TOPIC_TPMS_F, TOPIC_TPMS_R, TOPIC_SURF_F, TOPIC_SURF_R,
    TOPIC_RIDE_F, TOPIC_RIDE_R, TOPIC_WHEEL_LOAD, TOPIC_BRAKE_TEMP,
    TOPIC_RC, TOPIC_ICE1, TOPIC_ICE2, TOPIC_HL6, TOPIC_WSS,
}


def _ros_ts(msg) -> float:
    """Extract seconds from a ROS2 header stamp."""
    return float(msg.header.stamp.sec) + float(msg.header.stamp.nanosec) * 1e-9


class RawFrame:
    """One telemetry sample assembled from multiple ROS2 topics."""
    __slots__ = (
        "ts", "x", "y", "z",
        "vx", "vy", "vz", "speed",
        "throttle", "brake", "steering",
        "gear", "rpm",
        "slip_angle", "lat_acc", "lon_acc",
        "tyre_temp_fl", "tyre_temp_fr", "tyre_temp_rl", "tyre_temp_rr",
        "brake_press_fl", "brake_press_fr",
        "ride_height_fl", "ride_height_fr", "ride_height_rl", "ride_height_rr",
        "rh_front", "rh_rear",          # true ride height from optical sensor (mm)
        "wheel_load_fl", "wheel_load_fr", "wheel_load_rl", "wheel_load_rr",
        "brake_disc_temp_fl", "brake_disc_temp_fr",
        "distance_m",
        # ── New fields from extra CAN topics ──
        "car_flag",       # rc_status_01: 0=none 1=green 2=yellow 3=red 4=checkered
        "track_flag",     # rc_status_01: bitmask
        "session_type",   # rc_status_01: 0=unknown 1=practice 2=quali 3=race
        "oil_temp",       # ice_status_02: °C
        "water_temp",     # ice_status_02: °C
        "fuel_l",         # ice_status_01: litres remaining
        "cpu_usage",      # hl_msg_06: %
        "gpu_usage",      # hl_msg_06: %
        "cpu_temp",       # hl_msg_06: °C
        "gpu_temp",       # hl_msg_06: °C
        "wheel_slip",     # derived from wss vs speed: max slip across 4 wheels 0..1
    )

    def __init__(self, ts: float):
        self.ts = ts
        self.x = self.y = self.z = 0.0
        self.vx = self.vy = self.vz = 0.0
        self.speed = 0.0
        self.throttle = self.brake = 0.0
        self.steering = 0.0
        self.gear = 0
        self.rpm = 0.0
        self.slip_angle = 0.0
        self.lat_acc = self.lon_acc = 0.0
        self.tyre_temp_fl = self.tyre_temp_fr = None
        self.tyre_temp_rl = self.tyre_temp_rr = None
        self.brake_press_fl = self.brake_press_fr = None
        self.ride_height_fl = self.ride_height_fr = None
        self.ride_height_rl = self.ride_height_rr = None
        self.rh_front = self.rh_rear = None   # centerline optical ride height (mm)
        self.wheel_load_fl = self.wheel_load_fr = None
        self.wheel_load_rl = self.wheel_load_rr = None
        self.brake_disc_temp_fl = self.brake_disc_temp_fr = None
        self.distance_m = None
        self.car_flag = 0
        self.track_flag = 0
        self.session_type = 0
        self.oil_temp = 0.0
        self.water_temp = 0.0
        self.fuel_l = None
        self.cpu_usage = 0
        self.gpu_usage = 0
        self.cpu_temp = 0.0
        self.gpu_temp = 0.0
        self.wheel_slip = 0.0


def stream_frames(mcap_path: str | Path) -> Generator[RawFrame, None, None]:
    """
    Yields RawFrame objects in time order by decoding ROS2 messages.

    StateEstimation fires at ~100 Hz and is the primary clock.
    CAN topics (Kistler, TPMS, ride, wheel load, brake temp) are merged
    into frames using the most recent value seen.
    """
    side: dict = {}

    with open(mcap_path, "rb") as f:
        reader = make_reader(f, decoder_factories=[DecoderFactory()])
        for schema, channel, message, ros_msg in reader.iter_decoded_messages(
            topics=list(_TELEMETRY_TOPICS)
        ):
            topic = channel.topic
            try:
                if topic == TOPIC_STATE:
                    # ── StateEstimation ───────────────────────────────────────
                    ts = _ros_ts(ros_msg)
                    frame = RawFrame(ts=ts)
                    frame.x = float(ros_msg.x_m)
                    frame.y = float(ros_msg.y_m)
                    frame.z = float(ros_msg.z_m)
                    frame.vx = float(ros_msg.vx_mps)
                    frame.vy = float(ros_msg.vy_mps)
                    frame.vz = float(ros_msg.vz_mps)
                    frame.speed = float(ros_msg.v_mps) * 3.6   # → km/h
                    frame.throttle = float(ros_msg.gas)         # 0–1
                    frame.brake = float(ros_msg.brake)           # 0–1
                    frame.gear = int(ros_msg.gear)
                    frame.rpm = float(ros_msg.rpm)
                    frame.slip_angle = float(ros_msg.beta_rad)
                    frame.lat_acc = float(ros_msg.ay_mps2)
                    frame.lon_acc = float(ros_msg.ax_mps2)
                    frame.steering = float(ros_msg.delta_wheel_rad)
                    frame.brake_press_fl = float(ros_msg.cba_actual_pressure_fl_pa)
                    frame.brake_press_fr = float(ros_msg.cba_actual_pressure_fr_pa)
                    # Merge latest CAN side-data
                    for k, v in side.items():
                        if hasattr(frame, k):
                            setattr(frame, k, v)
                    yield frame

                elif topic == TOPIC_KISTLER_ACC:
                    side["lon_acc"] = float(ros_msg.acc_x_body)
                    side["lat_acc"] = float(ros_msg.acc_y_body)

                elif topic == TOPIC_KISTLER_CORR:
                    side["slip_angle"] = float(ros_msg.angle_cor)

                elif topic == TOPIC_TPMS_F:
                    # TPMS internal tyre temps (°C) — fallback if no surface temp
                    side.setdefault("tyre_temp_fl", float(ros_msg.tpr4_temp_fl))
                    side.setdefault("tyre_temp_fr", float(ros_msg.tpr4_temp_fr))

                elif topic == TOPIC_TPMS_R:
                    side.setdefault("tyre_temp_rl", float(ros_msg.tpr4_temp_rl))
                    side.setdefault("tyre_temp_rr", float(ros_msg.tpr4_temp_rr))

                elif topic == TOPIC_SURF_F:
                    # Infrared surface temps — average of outer/center/inner zones
                    side["tyre_temp_fl"] = (
                        float(ros_msg.outer_fl) + float(ros_msg.center_fl) + float(ros_msg.inner_fl)
                    ) / 3.0
                    side["tyre_temp_fr"] = (
                        float(ros_msg.outer_fr) + float(ros_msg.center_fr) + float(ros_msg.inner_fr)
                    ) / 3.0

                elif topic == TOPIC_SURF_R:
                    side["tyre_temp_rl"] = (
                        float(ros_msg.outer_rl) + float(ros_msg.center_rl) + float(ros_msg.inner_rl)
                    ) / 3.0
                    side["tyre_temp_rr"] = (
                        float(ros_msg.outer_rr) + float(ros_msg.center_rr) + float(ros_msg.inner_rr)
                    ) / 3.0

                elif topic == TOPIC_RIDE_F:
                    side["ride_height_fl"] = float(ros_msg.damper_stroke_fl)
                    side["ride_height_fr"] = float(ros_msg.damper_stroke_fr)
                    side["rh_front"] = float(ros_msg.ride_height_front)

                elif topic == TOPIC_RIDE_R:
                    side["ride_height_rl"] = float(ros_msg.damper_stroke_rl)
                    side["ride_height_rr"] = float(ros_msg.damper_stroke_rr)
                    side["rh_rear"] = float(ros_msg.ride_height_rear)

                elif topic == TOPIC_WHEEL_LOAD:
                    # Note: msg order is fl, fr, rr, rl (not fl, fr, rl, rr)
                    side["wheel_load_fl"] = float(ros_msg.load_wheel_fl)
                    side["wheel_load_fr"] = float(ros_msg.load_wheel_fr)
                    side["wheel_load_rr"] = float(ros_msg.load_wheel_rr)
                    side["wheel_load_rl"] = float(ros_msg.load_wheel_rl)

                elif topic == TOPIC_BRAKE_TEMP:
                    side["brake_disc_temp_fl"] = float(ros_msg.brake_disk_temp_fl)
                    side["brake_disc_temp_fr"] = float(ros_msg.brake_disk_temp_fr)

                elif topic == TOPIC_RC:
                    side["car_flag"]    = int(ros_msg.rc_car_flag)
                    side["track_flag"]  = int(ros_msg.rc_track_flag)
                    side["session_type"] = int(ros_msg.rc_session_type)

                elif topic == TOPIC_ICE1:
                    v = float(ros_msg.ice_available_fuel_l)
                    if v > 0:
                        side["fuel_l"] = v

                elif topic == TOPIC_ICE2:
                    side["oil_temp"]   = float(ros_msg.ice_oil_temp_deg_c)
                    side["water_temp"] = float(ros_msg.ice_water_temp_deg_c)

                elif topic == TOPIC_HL6:
                    side["cpu_usage"] = int(ros_msg.hl_cpu_usage)
                    side["gpu_usage"] = int(ros_msg.hl_gpu_usage)
                    side["cpu_temp"]  = float(ros_msg.hl_pc_temp)   # hl_cpu_temp in v7
                    try:
                        side["gpu_temp"] = float(ros_msg.hl_gpu_temp)
                    except Exception:
                        pass

                elif topic == TOPIC_WSS:
                    # Convert rad/s → km/h using approx wheel radius 0.33 m
                    WHEEL_R = 0.33
                    fl = float(ros_msg.wss_speed_fl_rad_s) * WHEEL_R * 3.6
                    fr = float(ros_msg.wss_speed_fr_rad_s) * WHEEL_R * 3.6
                    rl = float(ros_msg.wss_speed_rl_rad_s) * WHEEL_R * 3.6
                    rr = float(ros_msg.wss_speed_rr_rad_s) * WHEEL_R * 3.6
                    # Slip = max deviation from average wheel speed (normalised 0-1)
                    avg = (fl + fr + rl + rr) / 4.0
                    if avg > 5.0:  # only meaningful above 5 km/h
                        slip = max(abs(fl-avg), abs(fr-avg), abs(rl-avg), abs(rr-avg)) / avg
                        side["wheel_slip"] = round(min(slip, 1.0), 4)

            except Exception as e:
                log.debug("Decode error on %s: %s", topic, e)
                continue


def stream_camera_frames(
    mcap_path: str | Path,
    topic: str = TOPIC_CAMERA_FL,
) -> Generator[tuple[float, bytes], None, None]:
    """
    Yields (timestamp_sec, jpeg_bytes) for every camera frame on the given topic.
    Default topic is the front-left camera (~1 Hz in the hackathon dataset).
    """
    with open(mcap_path, "rb") as f:
        reader = make_reader(f, decoder_factories=[DecoderFactory()])
        for schema, channel, message, ros_msg in reader.iter_decoded_messages(
            topics=[topic]
        ):
            try:
                ts = _ros_ts(ros_msg)
                # ros_msg.data is a sequence of uint8 — convert to bytes
                raw = ros_msg.data
                # data may be bytes, bytearray, list[int], or numpy array
                if isinstance(raw, (bytes, bytearray)):
                    jpeg = bytes(raw)
                else:
                    jpeg = bytes(bytearray(int(b) for b in raw))
                if jpeg:
                    yield ts, jpeg
            except Exception as e:
                log.debug("Camera frame decode error: %s", e)
                continue
