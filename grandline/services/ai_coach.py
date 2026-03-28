from __future__ import annotations

import json
import os
import logging
from typing import Optional

import anthropic

from models.schemas import (
    LapSummary, LapDetail, LapComparison, DistanceComparison,
    BrakingZone, SectorSummary, CoachResponse, CoachMessage,
    CornerAnalysis, GGDiagram, TyreDegradation,
)

log = logging.getLogger(__name__)

# Model selection — intentionally tiered to keep costs low
# Tips (frequent, short): haiku  ~$0.0003 per call
# Debriefs (rare, rich):  sonnet  ~$0.006  per call
_MODEL_TIPS    = "claude-haiku-4-5-20251001"
_MODEL_DEBRIEF = "claude-sonnet-4-6"

_client: Optional[anthropic.Anthropic] = None

# In-memory cache: (session_id, lap_index, mode) → CoachResponse
# Prevents re-computing expensive debriefs for the same lap twice
_debrief_cache: dict[tuple, "CoachResponse"] = {}


def _get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        _client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    return _client

_SYSTEM_PROMPT = """You are a world-class race engineer with 20 years in top-level motorsport — Formula 1, IndyCar, and autonomous racing.
You speak directly, use precise technical language, and always reference actual numbers from the data.
You never pad responses. Every sentence has a purpose.

Structure your response as:
1. One sharp overall assessment (2–3 sentences max).
2. Up to 4 numbered coaching points — each tied to a specific metric, corner, or event in the data.
3. One clear priority action for the next lap (be specific: "Brake 15m later into T8" not "brake later").

Tone: professional, honest, direct. Like Toto Wolff briefing a driver before the final stint, not a YouTube tutorial.
When you have corner data, name the corners. When you have tyre data, reference temperatures.
Never say "it seems" or "it appears" — state findings as facts derived from the telemetry."""


# ── Formatting helpers ────────────────────────────────────────────────────────

def _fmt_lap(s: LapSummary, label: str = "") -> str:
    tag = f"[{label}] " if label else ""
    dist = f" | {s.lap_distance_m:.0f}m" if s.lap_distance_m else ""
    return (
        f"{tag}Lap {s.lap_index}: {s.lap_time_s:.3f}s{dist} | "
        f"Vmax {s.max_speed_kph:.0f} kph | Vavg {s.avg_speed_kph:.0f} kph | "
        f"Max lat-g {s.max_lat_acc:.2f} m/s² | Max lon-g {s.max_lon_acc:.2f} m/s² | "
        f"Avg throttle {s.avg_throttle*100:.0f}% | Max brake {s.max_brake*100:.0f}% | "
        f"Top gear {s.top_gear}"
    )


def _fmt_braking_zones(zones: list[BrakingZone], label: str) -> str:
    if not zones:
        return f"  {label}: no significant braking zones detected"
    # Sort by entry speed descending (heaviest stops first)
    top = sorted(zones, key=lambda z: z.entry_speed_kph, reverse=True)[:6]
    lines = [f"  {label} braking zones ({len(zones)} total, showing heaviest):"]
    for z in top:
        dist_str = f" @{z.distance_m:.0f}m" if z.distance_m else ""
        lines.append(
            f"    t={z.t_start:.1f}s{dist_str}  "
            f"{z.entry_speed_kph:.0f}→{z.min_speed_kph:.0f} kph  "
            f"peak_brake={z.peak_brake*100:.0f}%  dur={z.duration_s:.2f}s"
        )
    return "\n".join(lines)


def _fmt_sectors(sectors: list[SectorSummary], label: str) -> str:
    lines = [f"  {label} sectors:"]
    for sec in sectors:
        lines.append(
            f"    S{sec.sector}: {sec.time_s:.3f}s  "
            f"avg={sec.avg_speed_kph:.0f} kph  max={sec.max_speed_kph:.0f} kph"
        )
    return "\n".join(lines)


def _fmt_corners(corners: list[CornerAnalysis], label: str = "") -> str:
    if not corners:
        return ""
    tag = f"{label} " if label else ""
    lines = [f"  {tag}corner analysis ({len(corners)} corners):"]
    for c in corners:
        trail = f"  trail-brake={c.trail_brake_duration_s:.2f}s" if c.trail_brake_duration_s > 0.05 else ""
        lines.append(
            f"    {c.corner_id}: entry={c.entry_speed_kph:.0f} apex={c.apex_speed_kph:.0f} "
            f"exit={c.exit_speed_kph:.0f} kph | "
            f"brake={c.entry_brake*100:.0f}% | lat={c.max_lat_acc:.2f} m/s² | "
            f"throttle@apex={c.throttle_at_apex*100:.0f}%{trail}"
        )
    return "\n".join(lines)


def _fmt_gg(gg: GGDiagram) -> str:
    return (
        f"  GG diagram: max_lat={gg.max_lat_g:.2f} m/s² | "
        f"max_brake_decel={gg.max_lon_dec_g:.2f} m/s² | "
        f"max_accel={gg.max_lon_acc_g:.2f} m/s² | "
        f"friction_circle_95pct={gg.friction_circle_radius:.2f} m/s²"
    )


def _fmt_tyres(tyre: TyreDegradation) -> str:
    if tyre.start_avg_temp is None:
        return "  Tyre temps: no data"
    parts = [
        f"  Tyres: start={tyre.start_avg_temp:.0f}°C avg → end={tyre.end_avg_temp:.0f}°C avg"
    ]
    peaks = [
        ("FL", tyre.peak_temp_fl), ("FR", tyre.peak_temp_fr),
        ("RL", tyre.peak_temp_rl), ("RR", tyre.peak_temp_rr),
    ]
    peak_str = "  ".join(f"{name}={v:.0f}°C" for name, v in peaks if v)
    if peak_str:
        parts.append(f"  Peak temps: {peak_str}")
    if tyre.overheating:
        parts.append("  ⚠ OVERHEATING detected (>120°C)")
    if tyre.cold_start:
        parts.append("  ⚠ COLD START (<70°C) — first lap tyre warm-up required")
    return "\n".join(parts)


def _extract_coaching_points(text: str) -> list[str]:
    """Pull numbered points from the model response for structured UI display."""
    lines = text.split("\n")
    points = []
    for line in lines:
        stripped = line.strip()
        if stripped and stripped[0].isdigit() and len(stripped) > 2 and stripped[1:3] in (". ", ") "):
            points.append(stripped[3:].strip() if len(stripped) > 3 else stripped)
    return points[:4]


def _call_claude(user_prompt: str, max_tokens: int = 600, model: str = _MODEL_DEBRIEF) -> str:
    response = _get_client().messages.create(
        model=model,
        max_tokens=max_tokens,
        system=_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_prompt}],
    )
    return response.content[0].text


# ── Public coaching functions ─────────────────────────────────────────────────

def coach_single_lap(
    detail: LapDetail,
    corners: Optional[list[CornerAnalysis]] = None,
    gg: Optional[GGDiagram] = None,
    tyre: Optional[TyreDegradation] = None,
    cache_key: Optional[tuple] = None,
) -> CoachResponse:
    if cache_key and cache_key in _debrief_cache:
        log.info("Returning cached debrief for %s", cache_key)
        return _debrief_cache[cache_key]

    s = detail.summary
    sections = [f"Lap debrief — {_fmt_lap(s)}"]

    if gg:
        sections.append(_fmt_gg(gg))
    if tyre:
        sections.append(_fmt_tyres(tyre))
    if corners:
        sections.append(_fmt_corners(corners))

    sections.append(
        f"\nBraking: max {s.max_brake*100:.0f}%, throttle avg {s.avg_throttle*100:.0f}%\n"
        f"Lateral load: {s.max_lat_acc:.2f} m/s² peak | Long: {s.max_lon_acc:.2f} m/s²\n"
        f"{s.frame_count} telemetry samples over {s.lap_time_s:.3f}s\n\n"
        f"Give me a focused debrief on this lap's strengths and where time is left on the table."
    )

    text = _call_claude("\n".join(sections), model=_MODEL_DEBRIEF)
    result = CoachResponse(
        messages=[CoachMessage(role="engineer", content=text)],
        coaching_points=_extract_coaching_points(text),
    )
    if cache_key:
        _debrief_cache[cache_key] = result
    return result


def coach_lap_comparison(
    comparison: DistanceComparison,
    corners_a: Optional[list[CornerAnalysis]] = None,
    corners_b: Optional[list[CornerAnalysis]] = None,
    tyre_a: Optional[TyreDegradation] = None,
    tyre_b: Optional[TyreDegradation] = None,
    cache_key: Optional[tuple] = None,
) -> CoachResponse:
    if cache_key and cache_key in _debrief_cache:
        log.info("Returning cached comparison for %s", cache_key)
        return _debrief_cache[cache_key]
    delta_sign = "faster" if comparison.delta_s < 0 else "slower"
    delta_abs  = abs(comparison.delta_s)

    sections = [
        f"Distance-normalised lap comparison (real engineering approach — same track position, not same time):\n",
        f"{_fmt_lap(comparison.lap_a, 'LAP A')}",
        f"{_fmt_lap(comparison.lap_b, 'LAP B')}",
        f"\nLap B is {delta_abs:.3f}s {delta_sign} than Lap A.",
        f"\nSpeed delta statistics (distance grid, positive = Lap B faster):",
        f"  max gain: +{max(comparison.speed_delta):.1f} kph",
        f"  max loss: {min(comparison.speed_delta):.1f} kph",
        f"  avg: {sum(comparison.speed_delta)/len(comparison.speed_delta):.2f} kph",
        f"\n{_fmt_braking_zones(comparison.braking_zones_a, 'LAP A')}",
        f"{_fmt_braking_zones(comparison.braking_zones_b, 'LAP B')}",
        f"\n{_fmt_sectors(comparison.sectors_a, 'LAP A')}",
        f"{_fmt_sectors(comparison.sectors_b, 'LAP B')}",
    ]

    if corners_a and corners_b:
        # Show corner-by-corner delta (matched by corner_id)
        corner_b_map = {c.corner_id: c for c in corners_b}
        lines = ["\nCorner-by-corner comparison (Lap A → Lap B):"]
        for ca in corners_a:
            cb = corner_b_map.get(ca.corner_id)
            if cb:
                apex_delta = cb.apex_speed_kph - ca.apex_speed_kph
                exit_delta = cb.exit_speed_kph - ca.exit_speed_kph
                sign_a = "+" if apex_delta >= 0 else ""
                sign_e = "+" if exit_delta >= 0 else ""
                lines.append(
                    f"  {ca.corner_id}: apex {sign_a}{apex_delta:.1f} kph | "
                    f"exit {sign_e}{exit_delta:.1f} kph | "
                    f"brake A={ca.entry_brake*100:.0f}% B={cb.entry_brake*100:.0f}%"
                )
        sections.append("\n".join(lines))

    if tyre_a:
        sections.append(f"\n{_fmt_tyres(tyre_a).replace('Tyres:', 'LAP A tyres:')}")
    if tyre_b:
        sections.append(_fmt_tyres(tyre_b).replace("Tyres:", "LAP B tyres:"))

    sections.append(
        "\nIdentify the specific corners where time is gained/lost, "
        "what the driver should change at each, and the single biggest opportunity."
    )

    text = _call_claude("\n".join(sections), max_tokens=900, model=_MODEL_DEBRIEF)
    result = CoachResponse(
        messages=[CoachMessage(role="engineer", content=text)],
        coaching_points=_extract_coaching_points(text),
    )
    if cache_key:
        _debrief_cache[cache_key] = result
    return result


def coach_realtime_tip(
    recent_frames_summary: dict,
    track_position_pct: float,
    upcoming_corner: Optional[str] = None,
) -> str:
    """
    Returns a short (<25 word) real-time tip based on current telemetry.
    Designed to be called every ~8s during live replay.
    """
    corner_hint = f"\nUpcoming: {upcoming_corner}" if upcoming_corner else ""
    prompt = (
        f"Current telemetry at {track_position_pct*100:.0f}% of lap:{corner_hint}\n"
        f"Speed: {recent_frames_summary.get('speed', 0):.0f} kph\n"
        f"Throttle: {recent_frames_summary.get('throttle', 0)*100:.0f}%\n"
        f"Brake: {recent_frames_summary.get('brake', 0)*100:.0f}%\n"
        f"Lat-acc: {recent_frames_summary.get('lat_acc', 0):.2f} m/s²\n"
        f"Gear: {recent_frames_summary.get('gear', 0)}\n\n"
        f"Give ONE short real-time coaching tip (max 20 words). No preamble. Be specific."
    )

    try:
        return _call_claude(prompt, max_tokens=60, model=_MODEL_TIPS)
    except Exception as e:
        log.warning("Realtime tip failed: %s", e)
        return ""


def coach_wheel_to_wheel(
    detail: LapDetail,
    corners: Optional[list[CornerAnalysis]] = None,
    gg: Optional[GGDiagram] = None,
) -> CoachResponse:
    """
    Specialised coaching for wheel-to-wheel race scenarios.
    Analyses defensive/offensive lines, overtaking braking patterns,
    and tyre/slip management under race pressure.
    """
    s = detail.summary
    sections = [
        "Race scenario debrief — wheel-to-wheel analysis:\n",
        f"{_fmt_lap(s)}",
    ]

    if gg:
        sections.append(
            f"{_fmt_gg(gg)}\n"
            f"  → Under race pressure, friction circle usage indicates defensive vs attacking line."
        )

    if corners:
        # Find corners with unusual braking (potentially late/early for overtaking)
        aggressive = [c for c in corners if c.entry_brake > 0.7]
        if aggressive:
            sections.append(
                f"\n  High-pressure braking zones detected at: "
                + ", ".join(c.corner_id for c in aggressive)
            )
        sections.append(_fmt_corners(corners, "Race"))

    sections.append(
        f"\nThis is wheel-to-wheel race data. Analyse:\n"
        f"1. Overtaking/defending behaviour at braking zones\n"
        f"2. Tyre and slip management under race stress\n"
        f"3. Where the racing line deviates from optimal (defensive/offensive)\n"
        f"4. Risk vs reward at each high-pressure braking event"
    )

    text = _call_claude("\n".join(sections), max_tokens=800)
    return CoachResponse(
        messages=[CoachMessage(role="engineer", content=text)],
        coaching_points=_extract_coaching_points(text),
    )


def coach_followup(
    history: list[dict],
    question: str,
    lap_summary: Optional[LapSummary] = None,
    corners: Optional[list[CornerAnalysis]] = None,
) -> CoachResponse:
    """Free-form follow-up question in an ongoing coaching conversation."""
    messages = list(history) + [{"role": "user", "content": question}]

    context_parts = []
    if lap_summary:
        context_parts.append(f"Lap context: {_fmt_lap(lap_summary)}")
    if corners:
        context_parts.append(_fmt_corners(corners))

    if context_parts and messages:
        messages[0]["content"] = "\n".join(context_parts) + "\n\n" + messages[0]["content"]

    response = _get_client().messages.create(
        model=_MODEL_DEBRIEF,
        max_tokens=500,
        system=_SYSTEM_PROMPT,
        messages=messages,
    )
    text = response.content[0].text
    return CoachResponse(
        messages=[CoachMessage(role="engineer", content=text)],
        coaching_points=_extract_coaching_points(text),
    )
