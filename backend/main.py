import os
import asyncio
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Allow imports from the grandline/ package when this file is run from backend/
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "grandline"))

from routers import sessions, analysis, coach, stream

# Basic logging setup for the backend
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(name)s — %(message)s",
)
log = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── Startup ───────────────────────────────────────────────────────────────

    # Make sure the upload folder exists before the app starts handling requests
    os.makedirs(os.getenv("UPLOAD_DIR", "uploads"), exist_ok=True)

    # Auto-load the hackathon MCAP files.
    # Resolution order for data dir:
    #   1. PRELOAD_DATA_DIR env var (explicit override)
    #   2. ../data/hackathon  (data folder at project root, next to backend/)
    #   3. data/mcap          (legacy: data folder inside backend/)
    _here = Path(__file__).parent
    _default_dirs = [
        _here.parent / "data" / "hackathon",
        _here / "data" / "mcap",
    ]

    # First try to get the preload folder from environment variables
    preload_dir = os.getenv("PRELOAD_DATA_DIR")

    # If not provided, search the default candidate directories
    if not preload_dir:
        for _d in _default_dirs:
            if _d.exists():
                preload_dir = str(_d)
                log.info("Auto-detected data directory: %s", preload_dir)
                break

    # Decide which BND file to use.
    # Priority:
    #   1. BND_PATH environment variable
    #   2. yas_marina_bnd.json inside the detected preload directory
    #   3. fallback legacy/default path
    _bnd_env = os.getenv("BND_PATH")
    if _bnd_env:
        bnd_path = Path(_bnd_env)
    elif preload_dir and (Path(preload_dir) / "yas_marina_bnd.json").exists():
        bnd_path = Path(preload_dir) / "yas_marina_bnd.json"
    else:
        bnd_path = _here.parent / "data" / "hackathon" / "yas_marina_bnd.json"

    # If a preload directory was found, start background preloading of hackathon data
    if preload_dir:
        try:
            from services.preload import preload_hackathon_data
            await preload_hackathon_data(preload_dir, bnd_path)
            log.info("Preload initiated — sessions processing in background")
        except Exception as e:
            log.warning("Preload failed: %s", e)
    else:
        # If no preload directory exists, the user can still upload MCAP files manually
        log.info(
            "No data directory found. "
            "Upload MCAP files via POST /sessions to get started."
        )

    # Hand control back to FastAPI while the app runs
    yield

    # ── Shutdown ──────────────────────────────────────────────────────────────
    log.info("Shutting down pitlane backend")


app = FastAPI(
    title="pitlane",
    description=(
        "Race engineer in your pocket — real A2RL autonomous racing telemetry, "
        "AI coaching powered by Claude, distance-normalised lap comparison, "
        "GG diagram, per-corner analysis, and live replay streaming."
    ),
    version="2.0.0",
    lifespan=lifespan,
)

# GZipMiddleware is intentionally disabled because it buffers SSE streams
# and breaks real-time event delivery.
# app.add_middleware(GZipMiddleware, minimum_size=1000)

# Enable CORS so frontend clients can access the backend API
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "*").split(","),
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register API route groups
app.include_router(sessions.router, prefix="/sessions", tags=["sessions"])
app.include_router(analysis.router, prefix="/analysis", tags=["analysis"])
app.include_router(coach.router,    prefix="/coach",    tags=["coach"])
app.include_router(stream.router,   prefix="/stream",   tags=["stream"])


@app.get("/health")
def health():
    # Check how many sessions exist and how many are already processed/ready
    from services.session_store import _sessions
    ready = sum(1 for s in _sessions.values() if s._ready.is_set())
    return {
        "status": "ok",
        "sessions_total": len(_sessions),
        "sessions_ready": ready,
    }


@app.get("/preloaded")
def preloaded_sessions():
    """Return the stable IDs for all preloaded hackathon sessions."""
    from services.preload import PRELOAD_IDS
    from services.session_store import _sessions

    # Build a response showing whether each preloaded session is ready
    # and, if ready, how many laps it contains
    result = {}
    for scenario, sid in PRELOAD_IDS.items():
        s = _sessions.get(sid)
        result[scenario] = {
            "session_id": sid,
            "ready": s._ready.is_set() if s else False,
            "lap_count": len(s.raw_laps) if (s and s._ready.is_set()) else None,
        }
    return result