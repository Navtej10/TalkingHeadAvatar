"""
Minimal job queue wrapper. Uses RQ+Redis by default; swap this module for
Celery or a cloud task queue in production without touching the API layer.

For Phase 1 (baseline), this can even run synchronously in-process — flip
`SYNCHRONOUS` to True for local dev without Redis running.
"""
from typing import Optional
from rq import Queue
from redis import Redis
from app.pipeline.orchestrator import run_pipeline
from app.config import REDIS_URL

SYNCHRONOUS = True

_MOCK_JOBS = {}

# Setup redis connection and RQ queue
redis_conn = None
queue = None
if not SYNCHRONOUS:
    redis_conn = Redis.from_url(REDIS_URL)
    queue = Queue(connection=redis_conn)

def _run_and_deduct_task(job_id: str, api_key: str, **kwargs):
    """Wrapper to execute pipeline and deduct credits on success."""
    _MOCK_JOBS[job_id] = "running"
    try:
        result = run_pipeline(job_id, **kwargs)
        _MOCK_JOBS[job_id] = "done"
    except Exception as e:
        _MOCK_JOBS[job_id] = "failed"
        import traceback
        traceback.print_exc()
        raise e
    
    # Deduct credit if successful
    if api_key:
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
            
    return result

def enqueue_job(job_id: str, image: Optional[str], audio: Optional[str], text: Optional[str], identity_name: Optional[str] = None, emotion: str = "neutral", gaze_target: str = "camera", api_key: str = None) -> str:
    if SYNCHRONOUS:
        import threading
        _MOCK_JOBS[job_id] = "queued"
        # Fallback for synchronous execution if explicitly flipped back
        thread = threading.Thread(target=_run_and_deduct_task, args=(job_id, api_key), kwargs={
            "image": image,
            "audio": audio,
            "text": text,
            "identity_name": identity_name,
            "emotion": emotion,
            "gaze_target": gaze_target
        })
        thread.daemon = True
        thread.start()
    else:
        # Push to RQ queue
        queue.enqueue(
            _run_and_deduct_task, 
            args=(job_id, api_key), 
            kwargs={
                "image": image,
                "audio": audio,
                "text": text,
                "identity_name": identity_name,
                "emotion": emotion,
                "gaze_target": gaze_target
            },
            job_id=job_id,
            result_ttl=86400  # Keep results in redis for 24 hours
        )

    return job_id

def get_job_status(job_id: str) -> dict:
    from rq.job import Job
    from rq.exceptions import NoSuchJobError
    
    if SYNCHRONOUS:
        status = _MOCK_JOBS.get(job_id, "failed")
        return {
            "job_id": job_id, 
            "status": status,
            "result_url": f"/output/{job_id}.mp4" if status == "done" else None,
            "error": "Pipeline failed to produce output" if status == "failed" else None
        }
        
    try:
        job = Job.fetch(job_id, connection=redis_conn)
        
        status_map = {
            "queued": "queued",
            "started": "running",
            "finished": "done",
            "failed": "failed",
            "deferred": "queued",
            "scheduled": "queued"
        }
        
        api_status = status_map.get(job.get_status(), "failed")
        
        return {
            "job_id": job_id, 
            "status": api_status,
            "result_url": job.result if job.is_finished else None,
            "error": str(job.exc_info) if job.is_failed else None
        }
    except NoSuchJobError:
        return {"job_id": job_id, "status": "not_found", "result_url": None, "error": None}
