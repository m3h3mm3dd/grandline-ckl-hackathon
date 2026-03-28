# Grandline — AI Race Engineer

> Real A2RL autonomous racing telemetry. Real AI coaching. Race engineer in your pocket.

Built for the **Constructor GenAI Hackathon 2026 — Autonomous Track**.

---

## What It Does

Grandline ingests raw MCAP telemetry from the A2RL autonomous racing car at Yas Marina and turns it into actionable engineering intelligence:

- **Distance-normalised lap comparison** — compares laps at the same track position, not the same moment in time (the correct engineering approach)
- **Per-corner analysis** — entry/apex/exit speeds, trail-braking duration, lateral G load, throttle at apex for every corner
- **GG diagram** — traction circle / friction circle envelope showing how close to the limit the car operates
- **Tyre thermal analysis** — temperature evolution through the lap, overheating flags, cold-start detection
- **AI Race Engineer** (Claude claude-opus-4-5) — scenario-aware coaching that references actual numbers, corner names, and tyre states
- **Live SSE streaming** — real-time telemetry replay with periodic AI tips injected mid-lap
- **Auto-preload** — drops the 3 hackathon MCAP files at startup so judges can demo instantly

---

## Quick Start

### With Docker (recommended)

```bash
# Clone and enter the repo
git clone https://github.com/m3h3mm3dd/grandline-ckl-hackathon
cd grandline-ckl-hackathon

# Build image
docker build -t grandline .

# Run with hackathon data mounted
docker run -p 8000:8000 \
  -e ANTHROPIC_API_KEY=your_key \
  -e PRELOAD_DATA_DIR=/app/data/mcap \
  -e BND_PATH=/app/data/yas_marina_bnd.json \
  -v /path/to/mcap/files:/app/data/mcap \
  -v /path/to/yas_marina_bnd.json:/app/data/yas_marina_bnd.json \
  grandline
```

The server auto-processes all 3 MCAP files on startup. Check `/preloaded` to see when they're ready.

### Without Docker

```bash
cd grandline-ckl-hackathon
pip install -r requirements.txt

cd backend
export ANTHROPIC_API_KEY=your_key
export PRELOAD_DATA_DIR=../data/mcap
export BND_PATH=../data/yas_marina_bnd.json
uvicorn main:app --host 0.0.0.0 --port 8000
```

---

## Preloaded Session IDs

Once the MCAP files are processed, use these stable IDs in all API calls:

| Scenario | Session ID |
|----------|-----------|
| Good lap (reference) | `preload-good-lap` |
| Fast laps | `preload-fast-laps` |
| Wheel-to-wheel race | `preload-wheel-to-wheel` |

Check readiness: `GET /preloaded`

---

## API Reference

### Sessions
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/sessions/` | POST | Upload an MCAP file |
| `/sessions/` | GET | List all sessions |
| `/sessions/{id}` | GET | Session metadata |
| `/sessions/{id}/ready` | GET | Check if processing complete |
| `/preloaded` | GET | Status of all preloaded sessions |

### Analysis
| Endpoint | Description |
|----------|-------------|
| `GET /analysis/{id}/laps` | All lap summaries |
| `GET /analysis/{id}/lap/{n}` | Full lap detail + all frames |
| `GET /analysis/{id}/lap/{n}/braking` | Braking zones, sorted by severity |
| `GET /analysis/{id}/lap/{n}/sectors` | 3-sector breakdown (distance-based) |
| `GET /analysis/{id}/lap/{n}/corners` | Per-corner analysis (all detected corners) |
| `GET /analysis/{id}/lap/{n}/gg` | GG diagram / traction circle data |
| `GET /analysis/{id}/lap/{n}/tyres` | Tyre temperature time-series |
| `GET /analysis/{id}/lap/{n}/degradation` | Tyre degradation summary |
| `GET /analysis/{id}/compare?lap_a=0&lap_b=1` | Distance-normalised lap comparison |
| `GET /analysis/{id}/corners/all` | Corner analysis across all laps |
| `GET /analysis/{id}/best-lap` | Fastest lap summary |
| `GET /analysis/{id}/track` | Track boundary + centerline + corner markers |

### AI Coach
| Endpoint | Description |
|----------|-------------|
| `POST /coach/debrief` | Full AI debrief (single lap or comparison) |
| `POST /coach/ask` | Follow-up question to the engineer |

### Live Streaming
| Endpoint | Description |
|----------|-------------|
| `GET /stream/{id}/lap/{n}?speed=1.0` | SSE stream: frames + AI tips + lap_end event |

---

## Architecture

```
backend/
  main.py               FastAPI app + startup preload
grandline/
  routers/
    sessions.py         MCAP upload + session management
    analysis.py         All telemetry analysis endpoints
    coach.py            AI coaching endpoints
    stream.py           SSE real-time replay
  services/
    mcap_reader.py      Binary CDR decoder for all MCAP topics
    lap_detector.py     S/F line crossing detection
    corner_detector.py  Curvature-based corner detection + GPS snapping
    metrics_engine.py   GG diagram, distance comparison, tyre analysis
    ai_coach.py         Claude prompts with corner + tyre context
    session_store.py    In-memory session registry
    preload.py          Hackathon MCAP auto-loader
  models/
    schemas.py          Pydantic models for all API responses
```

---

## Data Decoded

From the A2RL dataset topics:

- **StateEstimation @100Hz** — position, velocity, slip, steering, pedals, gear, RPM, brake pressures
- **Kistler IMU @250Hz** — body-frame lateral + longitudinal acceleration (used for GG diagram)
- **Kistler Correvit @250Hz** — optical ground speed + slip angle
- **TPMS @50Hz** — tyre pressures and temperatures (FL/FR/RL/RR)
- **Badenia 560 Ride @100Hz** — damper strokes / ride heights
- **Badenia 560 Wheel Load @100Hz** — per-wheel vertical loads
- **Badenia 560 Brake Disc Temp @20Hz** — brake disc temperatures

---

## Scoring Checklist

- ✅ **Working app that runs** — FastAPI backend, real MCAP decoding, Claude AI
- ✅ **Real logic, not static** — every number derived from actual telemetry
- ✅ **Clear business use case** — AI race engineer for autonomous + human drivers
- ✅ **Runs on judges' machines** — Docker + preload = zero manual steps
- ✅ **Code quality** — typed, structured, tested imports
- ✅ **GitHub repo** — full source included
