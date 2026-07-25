"""
Stage 8 - Frame Interpolation
Raise generator FPS (often 12-25) to a smooth 30/60 FPS.

Reference model: RIFE
"""
from app.pipeline.base import PipelineStage
from app.config import get_active_profile, TMP_DIR
from app.models.registry import get_model
import os


class FrameInterpolationStage(PipelineStage):
    name = "frame_interpolation"

    def __init__(self):
        super().__init__()
        self.profile = get_active_profile()
        if self.profile.enable_interpolation:
            self.interpolator = get_model("frame_interpolation", "rife")
        else:
            self.interpolator = None

    def run(self, context: dict) -> dict:
        if not self.profile.enable_interpolation:
            return context

        job_id = context.get("job_id", "unknown")

        # Fallback through priority chain: stabilized -> restored -> refined -> raw
        input_frames_dir = context["frames"].get("stabilized",
                           context["frames"].get("restored",
                           context["frames"].get("refined",
                           context["frames"].get("raw"))))
                           
        if not input_frames_dir:
            raise ValueError(f"Job {job_id}: No frames available for frame interpolation.")

        interpolated_frames_dir = f"{TMP_DIR}/{job_id}/frames_interpolated"
        os.makedirs(interpolated_frames_dir, exist_ok=True)

        import cv2
        import shutil
        
        frames = sorted([f for f in os.listdir(input_frames_dir) if f.endswith(('.png', '.jpg'))])
        if not frames:
            raise ValueError(f"Job {job_id}: No frame files found in {input_frames_dir}")
            
        source_fps = context.get("source_fps", 15) # Default generator FPS baseline
        target_fps = self.profile.target_fps
        ratio = target_fps / source_fps
        
        if ratio <= 1.0:
            for f in frames:
                shutil.copy(os.path.join(input_frames_dir, f), os.path.join(interpolated_frames_dir, f))
        else:
            num_intermediate = int(ratio) - 1
            output_idx = 0
            
            for i in range(len(frames) - 1):
                img0_path = os.path.join(input_frames_dir, frames[i])
                img1_path = os.path.join(input_frames_dir, frames[i+1])
                
                img0 = cv2.imread(img0_path)
                img1 = cv2.imread(img1_path)
                
                cv2.imwrite(os.path.join(interpolated_frames_dir, f"{output_idx:06d}.png"), img0)
                output_idx += 1
                
                for step in range(1, num_intermediate + 1):
                    t = step / (num_intermediate + 1)
                    mid_img = self.interpolator.interpolate(img0, img1, t)
                    cv2.imwrite(os.path.join(interpolated_frames_dir, f"{output_idx:06d}.png"), mid_img)
                    output_idx += 1
            
            if frames:
                last_img = cv2.imread(os.path.join(input_frames_dir, frames[-1]))
                cv2.imwrite(os.path.join(interpolated_frames_dir, f"{output_idx:06d}.png"), last_img)

        context["frames"]["interpolated"] = interpolated_frames_dir
        return context
