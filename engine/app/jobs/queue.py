"""
Minimal job queue wrapper.

Key properties:
  - Jobs run in a background daemon thread (SYNCHRONOUS mode).
  - A threading.Semaphore(1) serializes pipeline jobs — only one runs at a time.
    This prevents concurrent runs from competing for CPU/RAM on the same machine.
  - Job status is persisted to disk (JSON) so uvicorn --reload or process restarts
    don't make in-flight jobs invisible to the frontend.
  - No env vars are read or written here.
"""
import json
import os
import threading
from typing import Optional

from app.config import TMP_DIR

# Serialize pipeline execution — only one job at a time on this machine.
_pipeline_semaphore = threading.Semaphore(1)

# Status persisted per-job to {TMP_DIR}/{job_id}/status.json
def _status_path(job_id: str) -> str:
    return os.path.join(TMP_DIR, job_id, "status.json")

def _write_status(job_id: str, status: str, result_url: str | None = None, error: str | None = None):
    os.makedirs(os.path.join(TMP_DIR, job_id), exist_ok=True)
    with open(_status_path(job_id), "w") as f:
        json.dump({"job_id": job_id, "status": status, "result_url": result_url, "error": error}, f)

def _read_status(job_id: str) -> dict:
    path = _status_path(job_id)
    if not os.path.exists(path):
        return {"job_id": job_id, "status": "not_found", "result_url": None, "error": None}
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return {"job_id": job_id, "status": "not_found", "result_url": None, "error": None}


def _run_and_deduct_task(job_id: str, api_key: str, **kwargs):
    """Wrapper: acquires semaphore, runs pipeline, releases semaphore, deducts credit."""
    _write_status(job_id, "running")
    try:
        # Wait for any already-running job to finish before starting
        with _pipeline_semaphore:
            from app.pipeline.orchestrator import run_pipeline
            result = run_pipeline(job_id, **kwargs)
        _write_status(job_id, "done", result_url=f"/output/{job_id}.mp4")
    except Exception as e:
        import traceback
        traceback.print_exc()
        _write_status(job_id, "failed", error=str(e))
        return

    # Deduct credit on success
    if api_key:
        try:
            from app.db.session import SessionLocal
            from app.db.models import ApiKey
            db = SessionLocal()
            try:
                db_key = db.query(ApiKey).filter(ApiKey.key == api_key).first()
                if db_key and db_key.credits_remaining > 0:
                    db_key.credits_remaining -= 1
                    db.commit()
            finally:
                db.close()
        except Exception:
            pass  # Credit deduction failure should never mask a successful job


def enqueue_job(
    job_id: str,
    image: Optional[str],
    audio: Optional[str],
    text: Optional[str],
    identity_name: Optional[str] = None,
    emotion: str = "neutral",
    gaze_target: str = "camera",
    api_key: str = None,
) -> str:
    _write_status(job_id, "queued")
    thread = threading.Thread(
        target=_run_and_deduct_task,
        args=(job_id, api_key),
        kwargs={
            "image": image,
            "audio": audio,
            "text": text,
            "identity_name": identity_name,
            "emotion": emotion,
            "gaze_target": gaze_target,
        },
        daemon=True,
    )
    thread.start()
    return job_id


def get_job_status(job_id: str) -> dict:
    return _read_status(job_id)
