"""
Stage 5 - Talking Head Generation (the core generator)
identity + audio + motion -> raw frames.

This is the highest-leverage stage to get right first. Wire in whichever
generator wins the Phase 0 bake-off (see docs/ROADMAP.md):
  - LivePortrait (already used in InterviewAI)
  - SadTalker
  - MuseTalk

Keep this stage's interface stable even if the underlying model changes.
"""
from app.pipeline.base import PipelineStage
from app.models.registry import get_model
from app.config import TMP_DIR
import os
import cv2


class GenerationStage(PipelineStage):
    name = "generation"

    def run(self, context: dict) -> dict:
        face = context["face"]
        audio_waveform_path = context["audio"]["waveform_path"]
        job_id = context.get("job_id", "unknown")

        generator = get_model("generation", "liveportrait")
        
        motion_latents = context.get("motion", {}).get("latents")
        
        raw_frames = generator.generate(
            aligned_face=face["aligned_face"], 
            audio_waveform_path=audio_waveform_path, 
            motion=motion_latents
        )

        frames_dir_path = f"{TMP_DIR}/{job_id}/frames_raw"
        os.makedirs(frames_dir_path, exist_ok=True)
        
        # Store raw frames as a path to a frame-dump directory
        for i, frame in enumerate(raw_frames):
            frame_path = os.path.join(frames_dir_path, f"frame_{i:04d}.png")
            cv2.imwrite(frame_path, frame)

        if "frames" not in context:
            context["frames"] = {}
            
        context["frames"]["raw"] = frames_dir_path
        return context
