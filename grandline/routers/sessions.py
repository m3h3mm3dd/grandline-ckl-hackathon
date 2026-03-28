import shutil
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile, BackgroundTasks
from fastapi.responses import JSONResponse

from services.session_store import (
    create_session, get_session, list_sessions, delete_session, UPLOAD_DIR
)
from models.schemas import SessionMeta

router = APIRouter()


@router.post("/", response_model=SessionMeta, status_code=201)
async def upload_session(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
):
    if not file.filename.endswith(".mcap"):
        raise HTTPException(400, "Only .mcap files are accepted")

    dest = UPLOAD_DIR / file.filename
    with dest.open("wb") as f:
        shutil.copyfileobj(file.file, f)

    session = create_session(file.filename, dest)
    background_tasks.add_task(session.process)

    return session.meta


@router.get("/", response_model=list[SessionMeta])
def get_sessions():
    return list_sessions()


@router.get("/{session_id}", response_model=SessionMeta)
def get_session_meta(session_id: str):
    s = get_session(session_id)
    if not s:
        raise HTTPException(404, "Session not found")
    return s.meta


@router.get("/{session_id}/ready")
async def session_ready(session_id: str):
    s = get_session(session_id)
    if not s:
        raise HTTPException(404, "Session not found")
    ready = s._ready.is_set()
    return {"ready": ready, "lap_count": len(s.raw_laps) if ready else 0}


@router.delete("/{session_id}", status_code=204)
def remove_session(session_id: str):
    if not delete_session(session_id):
        raise HTTPException(404, "Session not found")