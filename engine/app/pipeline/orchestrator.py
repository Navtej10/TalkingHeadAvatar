"""
Runs all pipeline stages in order. This is the one place that knows the
full sequence — individual stages don't know about each other.

Swap, reorder, or skip stages here as the pipeline evolves (e.g. skip
stage8/stage7 entirely on the cpu_dev profile — already handled inside
those stages via config, but you could also do it here for clarity).
"""
import shutil
from typing import Optional

from app.config import INPUT_DIR, TMP_DIR
from app.pipeline.stage1_face_processing import FaceProcessingStage
from app.pipeline.stage2_identity_encoding import IdentityEncodingStage
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
    AudioEncodingStage(),
    MotionPredictionStage(),
    GenerationStage(),
    LipRefinementStage(),
    FaceRestorationStage(),
    TemporalStabilizationStage(),
    FrameInterpolationStage(),
    AssemblyStage(),
]


def run_pipeline(job_id: str, image: Optional[str]=None, audio: Optional[str]=None, text: Optional[str]=None, identity_name: Optional[str]=None, emotion: str="neutral", gaze_target: str="camera") -> str:
    import os
    from app.core.identity_store import load_identity
    import cv2

    os.makedirs(INPUT_DIR, exist_ok=True)
    os.makedirs(f"{TMP_DIR}/{job_id}", exist_ok=True)

    # API layer now saves files directly to INPUT_DIR and passes absolute paths.
    image_path = image
    audio_path = audio

    context = {
        "job_id": job_id,
        "image_path": image_path,
        "audio_path": audio_path,
        "text": text,
        "emotion": emotion,
        "gaze_target": gaze_target,
    }
    
    # If using a pre-saved identity, inject it directly and skip Stage 1 & 2
    stages_to_run = STAGES
    if identity_name:
        embedding, thumbnail_path = load_identity(identity_name)
        
        # Inject the mock aligned_face for generator stages
        aligned_face = cv2.imread(thumbnail_path)
        
        context["face"] = {
            "aligned_face": aligned_face,
            "landmarks": None,  # Will bypass refine cropping
            "pose": None,
            "crop_box": None
        }
        context["identity"] = {
            "name": identity_name,
            "embedding": embedding
        }
        
        # Skip FaceProcessingStage and IdentityEncodingStage
        stages_to_run = [s for s in STAGES if s.name not in ("face_processing", "identity_encoding")]

    for stage in stages_to_run:
        context = stage.run(context)

    return context["output_path"]
