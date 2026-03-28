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