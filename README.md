# Grandline — AI Race Engineer 🏁

> Real A2RL autonomous racing telemetry. Real AI coaching. Track intelligence in your pocket.

Built for the **Constructor GenAI Hackathon 2026 — Autonomous Track**.

---

## Why Grandline

Grandline turns raw autonomous racing telemetry into engineering insight that is actually usable during analysis and demo time. Instead of showing generic charts, it reconstructs laps, aligns them by **distance along the track**, analyzes corners and tyre behavior, and generates coaching that references the real data.

This makes it useful for:

- lap comparison
- driving and control analysis
- tyre and grip monitoring
- judge-friendly demos with real data
- AI-assisted race debriefs

---

## Key Features ⚙️

- **Distance-normalised lap comparison**  
  Compare laps at the same track position instead of the same timestamp.

- **Per-corner analysis**  
  Entry, apex, and exit speed, trail braking behavior, throttle at apex, and lateral load.

- **GG diagram**  
  Visualizes the traction envelope and how close the car operates to the grip limit.

- **Tyre thermal analysis**  
  Temperature evolution, overheating flags, and cold-start detection.

- **AI Race Engineer**  
  Scenario-aware debriefs and coaching grounded in actual numbers, corner behavior, and tyre state.

- **Live replay with SSE streaming**  
  Replays telemetry in real time and injects AI tips during the lap.

- **Auto-preload for demo readiness**  
  Loads the three hackathon sessions automatically at startup so the app is ready in seconds.

---

## Demo Screenshots 📸

> Add your screenshots to a folder like `docs/images/` and update the filenames below.

### Main dashboard
![Grandline dashboard](docs/images/dashboard-main.png)

Track map, onboard view, live telemetry, and lap summary in one screen.

### Fast lap analysis
![Fast lap analysis](docs/images/dashboard-fast-lap.png)

High-speed segment playback with synchronized speed, throttle, and vehicle state.

### Telemetry panel
![Telemetry panel](docs/images/telemetry-panel.png)

GG diagram, tyre temperatures, slip angle, and session statistics.

---

## Dataset & Provenance 🗂️

Grandline is built on the **A2RL autonomous racing dataset for Yas Marina Circuit (Abu Dhabi)** provided during the **Constructor GenAI Hackathon 2026**.

### Dataset contents

The shared hackathon data includes:

- **3 MCAP files** with real autonomous racing sessions
- **high-frequency telemetry** up to 250 Hz
- **onboard camera feeds**
- **per-wheel measurements** including brake pressure, wheel speed, tyre temperature, tyre pressure, and wheel loads
- **track boundary data** via `yas_marina_bnd.json`
- **ROS 2 message definitions** for decoding all data types
- **GPS, IMU, suspension, ride height, and powertrain-related signals**

### Dataset link

Dataset package used during development:

`https://eu1-s3.virtuozzo.com/dev-auto-mobility-data/hackathon.tar.xz`

### Sessions used in the app

| Scenario | Session ID |
|----------|------------|
| Good lap (reference) | `preload-good-lap` |
| Fast laps | `preload-fast-laps` |
| Wheel-to-wheel race | `preload-wheel-to-wheel` |

---

## Quick Start 🚀

### With Docker

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

## API Reference 📚

### Sessions

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/sessions/` | POST | Upload an MCAP file |
| `/sessions/` | GET | List all sessions |
| `/sessions/{id}` | GET | Session metadata |
| `/sessions/{id}/ready` | GET | Check if processing is complete |
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

## Architecture 🧠

```text
backend/
  main.py               FastAPI app + startup preload

grandline/
  routers/
    sessions.py         MCAP upload + session management
    analysis.py         Telemetry analysis endpoints
    coach.py            AI coaching endpoints
    stream.py           SSE real-time replay

  services/
    mcap_reader.py      Binary CDR decoder for MCAP topics
    lap_detector.py     Start/finish line crossing detection
    corner_detector.py  Curvature-based corner detection + GPS snapping
    metrics_engine.py   GG diagram, lap comparison, tyre analysis
    ai_coach.py         Claude prompts with corner + tyre context
    session_store.py    In-memory session registry
    preload.py          Hackathon session auto-loader

  models/
    schemas.py          Pydantic response models
```

---

## Telemetry Decoded 📡

From the A2RL dataset topics:

- **StateEstimation @100Hz** for position, velocity, slip, steering, pedals, gear, RPM, and brake pressures
- **Kistler IMU @250Hz** for lateral and longitudinal acceleration used in the GG diagram
- **Kistler Correvit @250Hz** for optical ground speed and slip angle
- **TPMS @50Hz** for tyre pressures and temperatures
- **Badenia 560 Ride @100Hz** for damper strokes and ride heights
- **Badenia 560 Wheel Load @100Hz** for per-wheel vertical loads
- **Badenia 560 Brake Disc Temp @20Hz** for brake disc temperatures

---

## Tech Stack 🛠️

- **Backend:** FastAPI
- **Streaming:** Server-Sent Events
- **Models:** Pydantic
- **AI:** Claude (`claude-opus-4-5`)
- **Data format:** MCAP + ROS 2 message definitions
- **Deployment:** Docker

---

## Repository Structure Suggestions 🗂️

A cleaner repository structure can make the project easier to navigate and more polished for reviewers:

```text
backend/
grandline/
data/
docs/
  images/
README.md
requirements.txt
Dockerfile
.env.example
```

Optional additions that would make the repo look even stronger:

- `docs/images/` for README screenshots
- `.env.example` for environment variable setup
- `LICENSE`
- `CONTRIBUTING.md`

---

## Team 👥

| Name | Role |
|------|------|
| Turan Hasanzade | UI, data infrastructure, backend feedback/debrief features |
| Mahammad Mammadli | Backend lead, core backend implementation |
| Suad Huseynli | Frontend and backend telemetry support |

---

## Contact 📬

For questions, feedback, or collaboration:

- Suad Huseynli — `suadhuseynli11@gmail.com`
- Mahammad Mammadli — `mmammadli@constructor.university`
- Turan Hasanzade — `thasanzade@constructor.university`

You can also open an issue in the repository.

---

## Acknowledgements 🙌

This project was built during the **Constructor GenAI Hackathon 2026** using the **A2RL Yas Marina autonomous racing dataset**.

Thanks to the organizers and dataset providers for making the telemetry, track boundary data, ROS 2 message definitions, and session scenarios available.

---

## Scoring Checklist ✅

- ✅ **Working app that runs**  
  FastAPI backend, real MCAP decoding, AI-powered analysis, and a flow that judges can actually test.

- ✅ **Real logic, not static**  
  The numbers are derived from actual telemetry, not hardcoded demo values.

- ✅ **Clear business and technical value**  
  Grandline acts as an AI race engineer for autonomous racing analysis, with room to extend into motorsport tooling, simulation workflows, and performance diagnostics.

- ✅ **Runs on judges' machines**  
  Docker support plus preload logic means the project can be launched quickly without setup chaos.

- ✅ **Clean engineering structure**  
  Typed models, separated services, organized routers, and a backend layout that shows deliberate system design.

- ✅ **Real data provenance**  
  Built around A2RL Yas Marina autonomous racing telemetry provided for the hackathon.

- ✅ **Strong demo experience**  
  Dashboard views, telemetry panels, lap comparison, and AI-generated feedback make the project easy to understand in motion, not just in theory.

- ✅ **GitHub-ready presentation**  
  Clear documentation, dataset context, team roles, contact details, and screenshots give the repository a polished finish.

---

## Final Pitch 🎯

**Grandline is not just a telemetry viewer.** It is a compact AI-assisted race engineering workspace built on real autonomous racing data, designed to turn raw laps into decisions.

From GG diagrams and tyre temperatures to per-corner insights and distance-aligned comparisons, the project aims to feel both technically credible and demo-ready. The result is something that looks sharp on the surface, while still having real engineering bones underneath it.

That balance is the sweet spot: not a static hackathon poster, not an overbuilt research maze, but a system that can explain, compare, replay, and coach using real racing telemetry. 📈🏎️