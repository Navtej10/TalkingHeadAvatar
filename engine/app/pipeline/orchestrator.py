"""
Runs all pipeline stages in order. This is the one place that knows the
full sequence — individual stages don't know about each other.

Swap, reorder, or skip stages here as the pipeline evolves (e.g. skip
stage8/stage7 entirely on the cpu_dev profile — already handled inside
those stages via config, but you could also do it here for clarity).
"""
import os
import time
import cv2
from datetime import datetime
from typing import Optional

from app.config import INPUT_DIR, TMP_DIR
from app.core.identity_store import load_identity
from app.models.validation import validate_all_checkpoints
from app.pipeline.base import PipelineStageError
from app.pipeline.stage1_face_processing import FaceProcessingStage
from app.pipeline.stage2_identity_encoding import IdentityEncodingStage
from app.pipeline.stage2b_audio_prep import AudioPreparationStage
from app.pipeline.stage3_audio_encoding import AudioEncodingStage
from app.pipeline.stage4_motion_prediction import MotionPredictionStage
from app.pipeline.stage5_generation import GenerationStage
from app.pipeline.stage6_lip_refinement import LipRefinementStage
from app.pipeline.stage7_face_restoration import FaceRestorationStage
from app.pipeline.stage8_frame_interpolation import FrameInterpolationStage
from app.pipeline.stage9_temporal_stabilization import TemporalStabilizationStage
from app.pipeline.stage10_assembly import AssemblyStage

STAGES = [
    FaceProcessingStage(),
    IdentityEncodingStage(),
    AudioPreparationStage(),
    AudioEncodingStage(),
    MotionPredictionStage(),
    GenerationStage(),
    LipRefinementStage(),
    FaceRestorationStage(),
    FrameInterpolationStage(),
    TemporalStabilizationStage(),
    AssemblyStage(),
]

# Full map: stage number -> stage name (used by skip_to_stage parameter)
_STAGE_MAP = {
    1: "face_processing",
    2: "identity_encoding",
    3: "audio_encoding",
    4: "motion_prediction",
    5: "generation",
    6: "lip_refinement",
    7: "face_restoration",
    8: "frame_interpolation",
    9: "temporal_stabilization",
    10: "assembly",
}


def run_pipeline(
    job_id: str,
    image: Optional[str] = None,
    audio: Optional[str] = None,
    text: Optional[str] = None,
    identity_name: Optional[str] = None,
    emotion: str = "neutral",
    gaze_target: str = "camera",
    # Dev-only: resume from a specific stage using a previously-cached job.
    # Only pass these from test scripts — never from the production API path.
    skip_to_stage: Optional[int] = None,
    cached_job_id: Optional[str] = None,
) -> str:


    os.makedirs(INPUT_DIR, exist_ok=True)
    os.makedirs(f"{TMP_DIR}/{job_id}", exist_ok=True)

    context = {
        "job_id": job_id,
        "image_path": image,
        "audio_path": audio,
        "text": text,
        "emotion": emotion,
        "gaze_target": gaze_target,
    }

    # If using a pre-saved identity, inject it directly and skip Stages 1 & 2
    stages_to_run = list(STAGES)
    if identity_name:
        embedding, thumbnail_path = load_identity(identity_name)
        aligned_face = cv2.imread(thumbnail_path)
        context["face"] = {
            "aligned_face": aligned_face,
            "landmarks": None,
            "pose": None,
            "crop_box": None,
        }
        context["identity"] = {
            "name": identity_name,
            "embedding": embedding,
        }
        stages_to_run = [s for s in STAGES if s.name not in ("face_processing", "identity_encoding")]

    # ── Dev resume: skip_to_stage + cached_job_id (test scripts only) ──────────
    if skip_to_stage is not None and cached_job_id is not None:
        target_name = _STAGE_MAP.get(skip_to_stage)
        if target_name is None:
            raise ValueError(f"Invalid skip_to_stage: {skip_to_stage}. Valid stages are: {list(_STAGE_MAP.keys())}")
            
        if identity_name and image is None and skip_to_stage in [1, 2]:
            raise ValueError("Cannot skip to stage 1 or 2 with an identity_name but no input image.")

        if "audio" not in context:
            context["audio"] = {}
        if not context["audio"].get("waveform_path"):
            context["audio"]["waveform_path"] = (
                audio or f"{TMP_DIR}/{cached_job_id}/tts.wav"
            )

        # Only inject cached raw frames when skipping past generation (Stage 5)
        if skip_to_stage >= 6:
            if "frames" not in context:
                context["frames"] = {}
            context["frames"]["raw"] = f"{TMP_DIR}/{cached_job_id}/frames_raw"

        start_appending = False
        new_stages = []
        for s in stages_to_run:
            if s.name == target_name:
                start_appending = True
            if start_appending:
                new_stages.append(s)
        
        # Only prepend face_processing so context['face'] is available if not already populated
        face_stage = next((s for s in STAGES if s.name == "face_processing"), None)
        if face_stage and "face" not in context and (not new_stages or new_stages[0].name != "face_processing"):
            new_stages.insert(0, face_stage)
        stages_to_run = new_stages
        print(f"[dev] Resuming from Stage {skip_to_stage} ({target_name}) using cached job {cached_job_id}")

    validate_all_checkpoints(stages_to_run)

    # ── Run stages ─────────────────────────────────────────────────────────────
    for stage in stages_to_run:
        start_time = time.time()
        timestamp_start = datetime.now().strftime("%H:%M:%S")
        print(f"[{timestamp_start}] STARTING stage: {stage.name}")

        try:
            context = stage.run(context)
        except Exception as e:
            elapsed = time.time() - start_time
            timestamp_end = datetime.now().strftime("%H:%M:%S")
            print(f"[{timestamp_end}] FAILED stage: {stage.name} ({elapsed:.1f}s)")
            raise PipelineStageError(stage.name, job_id, e) from e

        elapsed = time.time() - start_time
        timestamp_end = datetime.now().strftime("%H:%M:%S")
        print(f"[{timestamp_end}] FINISHED stage: {stage.name} ({elapsed:.1f}s)")

    return context["output_path"]
