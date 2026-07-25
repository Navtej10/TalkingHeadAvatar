import uuid
from typing import Optional

from fastapi import APIRouter, UploadFile, File, Form, Depends, HTTPException
from app.api.auth import verify_api_key
from pydantic import BaseModel

from app.jobs.queue import enqueue_job, get_job_status

router = APIRouter()


class JobResponse(BaseModel):
    job_id: str
    status: str


class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    result_url: Optional[str] = None
    error: Optional[str] = None


@router.post("")
def create_job(
    image: Optional[UploadFile] = File(None),
    audio: Optional[UploadFile] = File(None),
    text: Optional[str] = Form(None),
    identity_name: Optional[str] = Form(None),
    emotion: Optional[str] = Form("neutral"),
    gaze_target: Optional[str] = Form("camera"),
    api_key: str = Depends(verify_api_key)
):
    """
    Creates a new talking-head generation job.
    Accepts either an image upload or an identity_name.
    """
    if not image and not identity_name:
        raise HTTPException(status_code=400, detail="Must provide either 'image' or 'identity_name'.")
    if not audio and not text:
        raise HTTPException(status_code=400, detail="Must provide either 'audio' or 'text'.")

    # Validate emotion
    valid_emotions = ["neutral", "happy", "serious", "surprised"]
    if emotion not in valid_emotions:
        raise HTTPException(status_code=400, detail=f"Invalid emotion. Must be one of: {valid_emotions}")

    # Generate a unique ID for this job
    job_id = uuid.uuid4().hex[:8]
    
    # Save UploadFiles to disk synchronously before passing to queue
    import os
    import shutil
    from app.config import INPUT_DIR
    
    os.makedirs(INPUT_DIR, exist_ok=True)
    
    saved_image_path = None
    if image:
        saved_image_path = f"{INPUT_DIR}/{job_id}_image.jpg"
        with open(saved_image_path, "wb") as f:
            shutil.copyfileobj(image.file, f)
            
    saved_audio_path = None
    if audio:
        saved_audio_path = f"{INPUT_DIR}/{job_id}_audio.wav"
        with open(saved_audio_path, "wb") as f:
            shutil.copyfileobj(audio.file, f)

    # Submit to queue (currently blocks if SYNCHRONOUS=True)
    enqueue_job(job_id, image=saved_image_path, audio=saved_audio_path, text=text, identity_name=identity_name, emotion=emotion, gaze_target=gaze_target, api_key=api_key)

    return {"job_id": job_id, "status": "queued"}


@router.get("/{job_id}", response_model=JobStatusResponse)
def get_job(job_id: str, api_key: str = Depends(verify_api_key)):
    """
    Polls the current status of a job.
    """
    status = get_job_status(job_id)
    return JobStatusResponse(**status)
