"""
Stage 7 - Face Restoration
Restore high-frequency detail (skin texture, eyelashes, hair, teeth) lost
during generation, so output doesn't look soft/plastic.

Reference models: GFPGAN, CodeFormer
"""
from app.pipeline.base import PipelineStage
from app.config import get_active_profile, TMP_DIR
from app.models.registry import get_model
import os
import cv2
import glob


class FaceRestorationStage(PipelineStage):
    name = "face_restoration"

    def __init__(self):
        super().__init__()
        self.profile = get_active_profile()
        if self.profile.enable_restoration:
            self.restorer = get_model("face_restoration", "gfpgan")
        else:
            self.restorer = None

    def run(self, context: dict) -> dict:
        if not self.profile.enable_restoration:
            return context

        job_id = context.get("job_id", "unknown")
        
        # fallback to raw if refined is not available
        input_frames_dir = context["frames"].get("refined", context["frames"].get("raw"))
        if not input_frames_dir:
            raise ValueError(f"Job {job_id}: No frames available for face restoration.")

        restored_frames_dir = f"{TMP_DIR}/{job_id}/frames_restored"
        os.makedirs(restored_frames_dir, exist_ok=True)

        frame_files = sorted(glob.glob(os.path.join(input_frames_dir, "*.png")))
        for f in frame_files:
            frame = cv2.imread(f)
            if frame is None:
                continue
                
            restored_frame = self.restorer.restore_frame(frame)
            
            basename = os.path.basename(f)
            cv2.imwrite(os.path.join(restored_frames_dir, basename), restored_frame)

        context["frames"]["restored"] = restored_frames_dir
        return context
