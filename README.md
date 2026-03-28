Grandline — AI Race Engineer

Real A2RL autonomous racing telemetry. Real AI coaching. Track intelligence in your pocket.

Built for the Constructor GenAI Hackathon 2026 — Autonomous Track.

⸻

Why Grandline

Grandline turns raw autonomous racing telemetry into engineering insight that is actually usable during analysis and demo time. Instead of showing generic charts, it reconstructs laps, aligns them by distance along the track, analyzes corners and tyre behavior, and generates coaching that references the real data.

This makes it useful for:
	•	lap comparison
	•	driving and control analysis
	•	tyre and grip monitoring
	•	judge-friendly demos with real data
	•	AI-assisted race debriefs

⸻

Key Features
	•	Distance-normalised lap comparison
Compare laps at the same track position instead of the same timestamp.
	•	Per-corner analysis
Entry, apex, and exit speed, trail braking behavior, throttle at apex, and lateral load.
	•	GG diagram
Visualizes the traction envelope and how close the car operates to the grip limit.
	•	Tyre thermal analysis
Temperature evolution, overheating flags, and cold-start detection.
	•	AI Race Engineer
Scenario-aware debriefs and coaching grounded in actual numbers, corner behavior, and tyre state.
	•	Live replay with SSE streaming
Replays telemetry in real time and injects AI tips during the lap.
	•	Auto-preload for demo readiness
Loads the three hackathon sessions automatically at startup so the app is ready in seconds.

⸻

Demo Screenshots

Put your screenshots into something like docs/images/ and then use them here.

Main dashboard

![Grandline dashboard](docs/images/dashboard-main.png)

Fast lap view

![Fast lap analysis](docs/images/dashboard-fast-lap.png)

Telemetry side panel

![Telemetry panel](docs/images/telemetry-panel.png)

Suggested captions
	•	Main dashboard: track map, onboard view, live telemetry, and lap summary in one screen
	•	Fast lap view: high-speed segment playback with synchronized speed, throttle, and vehicle state
	•	Telemetry panel: GG diagram, tyre temperatures, slip angle, and session statistics

⸻

Dataset

Grandline is built on the A2RL autonomous racing dataset for Yas Marina Circuit (Abu Dhabi) provided during the Constructor GenAI Hackathon 2026.

Dataset contents

The shared hackathon data includes:
	•	3 MCAP files with real autonomous racing sessions
	•	high-frequency telemetry up to 250 Hz
	•	onboard camera feeds
	•	per-wheel measurements including brake pressure, wheel speed, tyre temperature, tyre pressure, and wheel loads
	•	track boundary data via yas_marina_bnd.json
	•	ROS 2 message definitions for decoding all data types
	•	GPS, IMU, suspension, ride height, and powertrain-related signals

Sessions used in the app

Scenario	Session ID
Good lap (reference)	preload-good-lap
Fast laps	preload-fast-laps
Wheel-to-wheel race	preload-wheel-to-wheel

Dataset origin

You should definitely keep this section. It makes the repo look more serious and answers the first question every judge or recruiter will ask: where did the data come from?

Use something like this in the final repo:

Dataset source: A2RL autonomous racing telemetry from Yas Marina Circuit, distributed for the Constructor GenAI Hackathon 2026.

Data package used in development: `hackathon.tar.xz`
Track boundary file: `yas_marina_bnd.json`

Dataset access

The hackathon dataset used during development is available here:

Dataset link: https://eu1-s3.virtuozzo.com/dev-auto-mobility-data/hackathon.tar.xz

If redistribution rules change later, this can be replaced with a note explaining access limitations.

⸻

Quick Start

With Docker

git clone https://github.com/m3h3mm3dd/grandline-ckl-hackathon
cd grandline-ckl-hackathon

docker build -t grandline .

docker run -p 8000:8000 \
  -e ANTHROPIC_API_KEY=your_key \
  -e PRELOAD_DATA_DIR=/app/data/mcap \
  -e BND_PATH=/app/data/yas_marina_bnd.json \
  -v /path/to/mcap/files:/app/data/mcap \
  -v /path/to/yas_marina_bnd.json:/app/data/yas_marina_bnd.json \
  grandline

Then open the backend and check preload readiness:

GET /preloaded

Local run

Recommended Python version: 3.12 or 3.13

cd grandline-ckl-hackathon
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip setuptools wheel
pip install -r requirements.txt

cd backend
export ANTHROPIC_API_KEY=your_key
export PRELOAD_DATA_DIR=../data/mcap
export BND_PATH=../data/yas_marina_bnd.json
uvicorn main:app --host 0.0.0.0 --port 8000


⸻

API Reference

Sessions

Endpoint	Method	Description
/sessions/	POST	Upload an MCAP file
/sessions/	GET	List all sessions
/sessions/{id}	GET	Session metadata
/sessions/{id}/ready	GET	Check if processing complete
/preloaded	GET	Status of all preloaded sessions

Analysis

Endpoint	Description
GET /analysis/{id}/laps	All lap summaries
GET /analysis/{id}/lap/{n}	Full lap detail + all frames
GET /analysis/{id}/lap/{n}/braking	Braking zones, sorted by severity
GET /analysis/{id}/lap/{n}/sectors	3-sector breakdown (distance-based)
GET /analysis/{id}/lap/{n}/corners	Per-corner analysis (all detected corners)
GET /analysis/{id}/lap/{n}/gg	GG diagram / traction circle data
GET /analysis/{id}/lap/{n}/tyres	Tyre temperature time-series
GET /analysis/{id}/lap/{n}/degradation	Tyre degradation summary
GET /analysis/{id}/compare?lap_a=0&lap_b=1	Distance-normalised lap comparison
GET /analysis/{id}/corners/all	Corner analysis across all laps
GET /analysis/{id}/best-lap	Fastest lap summary
GET /analysis/{id}/track	Track boundary + centerline + corner markers

AI Coach

Endpoint	Description
POST /coach/debrief	Full AI debrief (single lap or comparison)
POST /coach/ask	Follow-up question to the engineer

Live Streaming

Endpoint	Description
GET /stream/{id}/lap/{n}?speed=1.0	SSE stream: frames + AI tips + lap_end event


⸻

Architecture

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


⸻

Telemetry Decoded

From the A2RL dataset topics:
	•	StateEstimation @100Hz for position, velocity, slip, steering, pedals, gear, RPM, and brake pressures
	•	Kistler IMU @250Hz for lateral and longitudinal acceleration used in the GG diagram
	•	Kistler Correvit @250Hz for optical ground speed and slip angle
	•	TPMS @50Hz for tyre pressures and temperatures
	•	Badenia 560 Ride @100Hz for damper strokes and ride heights
	•	Badenia 560 Wheel Load @100Hz for per-wheel vertical loads
	•	Badenia 560 Brake Disc Temp @20Hz for brake disc temperatures

⸻

Tech Stack
	•	Backend: FastAPI
	•	Streaming: Server-Sent Events
	•	Models: Pydantic
	•	AI: Claude (claude-opus-4-5)
	•	Data format: MCAP + ROS 2 message definitions
	•	Deployment: Docker

⸻

Why This Is Technically Strong
	•	Uses real telemetry, not mocked values
	•	Applies the correct comparison method with distance-normalized lap alignment
	•	Combines vehicle dynamics, tyres, braking, and corner-level metrics in one workflow
	•	Produces judge-friendly outputs without hiding the engineering depth
	•	Is structured for extension into simulation, motorsport tooling, or autonomous systems analysis

⸻

Repository Structure Suggestions

To make the repo feel cleaner, consider this layout:

backend/
frontend/
docs/
  images/
  architecture/
data/
README.md
LICENSE
.env.example

And add:
	•	.env.example
	•	docs/images/ for screenshots
	•	LICENSE
	•	CONTRIBUTING.md if you want the repo to look extra polished

⸻

Credits and Acknowledgements

This project was built during the Constructor GenAI Hackathon 2026 using the A2RL Yas Marina autonomous racing dataset.

Thanks to the organizers and dataset providers for making the telemetry, track boundary data, ROS 2 message definitions, and session scenarios available.

If the official dataset page, event page, or organizer names can be publicly listed, add them here.

⸻

Team

Name	Role
Turan Hasanzade	UI, data infrastructure, backend feedback/debrief features
Mahammad Mammadli	Most of the backend implementation
Suad Huseynli	Frontend and backend telemetry support


⸻

Contact

Yes, add a contact section, but make it your team contact, not the original dataset publishers unless they explicitly asked for that.

Use something like:

For questions, feedback, or collaboration:
- Suad Huseynli — suadhuseynli11@gmail.com
- Mahammad Mammadli — mmammadli@constructor.university
- Turan Hasanzade — thasanzade@constructor.university
- GitHub Issues: https://github.com/m3h3mm3dd/grandline-ckl-hackathon/issues

If you want to acknowledge dataset maintainers, put them under Acknowledgements, not as your project contact.

⸻

Final Notes

This repo will look much stronger if you add these finishing touches:
	1.	Dataset origin section
So people instantly understand where the data came from and why it matters.
	2.	Real screenshots with short captions
Because your app already looks strong, and visuals make the repo feel alive.
	3.	Credits and acknowledgements separated from your own contact section
This keeps the project identity clean and professional.
	4.	A more vivid scoring/value section
So the README feels like a finished product page, not just a technical note.

⸻

Scoring Checklist
	•	✅ Working app that runs
FastAPI backend, real MCAP decoding, Claude-powered analysis, and a flow that judges can actually test.
	•	✅ Real logic, not static
The numbers are derived from actual telemetry, not hardcoded demo glitter.
	•	✅ Clear business and technical value
Grandline acts as an AI race engineer for autonomous racing analysis, with room to extend into motorsport tooling, simulation workflows, and performance diagnostics.
	•	✅ Runs on judges’ machines
Docker support plus preload logic means the project can be launched quickly without setup drama.
	•	✅ Clean engineering structure
Typed models, separated services, organized routers, and a backend layout that shows deliberate system design.
	•	✅ Real data provenance
Built around A2RL Yas Marina autonomous racing telemetry provided for the hackathon.
	•	✅ Strong demo experience
Dashboard views, telemetry panels, lap comparison, and AI-generated feedback make the project easy to understand in motion, not just in theory.
	•	✅ GitHub-ready presentation
Clear documentation, dataset context, team roles, contact details, and screenshots give the repository a polished finish.

⸻

Final Pitch

🏁 Grandline is not just a telemetry viewer. It is a compact AI-assisted race engineering workspace built on real autonomous racing data, designed to turn raw laps into decisions.

From GG diagrams and tyre temperatures to per-corner insights and distance-aligned comparisons, the project aims to feel both technically credible and demo-ready. The result is something that looks sharp on the surface, while still having real engineering bones underneath it.

That balance is the sweet spot: not a static hackathon poster, not an overbuilt research maze, but a system that can explain, compare, replay, and coach using real racing telemetry. 🚦📈🧠