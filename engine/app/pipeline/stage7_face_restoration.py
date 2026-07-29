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

        source_path = context.get("images", {}).get("source")
        source_img = cv2.imread(source_path) if source_path and os.path.exists(source_path) else None
        
        tgt_mean, tgt_std = None, None
        noise_std = 0.0
        
        import numpy as np
        if source_img is not None:
            # 1. Color Grading Target (Reinhard)
            src_lab = cv2.cvtColor(source_img, cv2.COLOR_BGR2LAB).astype(np.float32)
            tgt_mean, tgt_std = cv2.meanStdDev(src_lab)
            
            # 2. Noise Estimation (using Laplacian variance proxy)
            gray = cv2.cvtColor(source_img, cv2.COLOR_BGR2GRAY)
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            # Standard deviation of laplacian gives a rough proxy of high-frequency noise/texture
            # We scale it down because we only want subtle film grain
            noise_std = np.clip(np.std(laplacian) * 0.1, 1.0, 15.0)

        sharpness_scores = []
        frame_files = sorted(glob.glob(os.path.join(input_frames_dir, "*.png")))
        total_frames = len(frame_files)
        
        for f_idx, f in enumerate(frame_files):
            if f_idx > 0 and f_idx % 10 == 0:
                print(f"[{self.name}] processed {f_idx}/{total_frames} frames")
            frame = cv2.imread(f)
            if frame is None:
                continue
                
            restored_frame = self.restorer.restore_frame(frame)
            
            # Apply Grading
            if tgt_mean is not None and tgt_std is not None:
                res_lab = cv2.cvtColor(restored_frame, cv2.COLOR_BGR2LAB).astype(np.float32)
                res_mean, res_std = cv2.meanStdDev(res_lab)
                res_std[res_std == 0] = 1e-5
                
                out_lab = (res_lab - res_mean.reshape(1,1,3)) * (tgt_std.reshape(1,1,3) / res_std.reshape(1,1,3)) + tgt_mean.reshape(1,1,3)
                out_lab = np.clip(out_lab, 0, 255).astype(np.uint8)
                restored_frame = cv2.cvtColor(out_lab, cv2.COLOR_LAB2BGR)
                
            # Apply Noise Injection
            if noise_std > 0:
                noise = np.random.normal(0, noise_std, restored_frame.shape).astype(np.float32)
                restored_frame = np.clip(restored_frame.astype(np.float32) + noise, 0, 255).astype(np.uint8)
                
            # Calculate Sharpness (No-Reference Metric)
            gray = cv2.cvtColor(restored_frame, cv2.COLOR_BGR2GRAY)
            sharpness = np.var(cv2.Laplacian(gray, cv2.CV_64F))
            sharpness_scores.append(sharpness)
            
            basename = os.path.basename(f)
            cv2.imwrite(os.path.join(restored_frames_dir, basename), restored_frame)

        if "metadata" not in context:
            context["metadata"] = {}
        context["metadata"]["mean_sharpness_score"] = float(np.mean(sharpness_scores)) if sharpness_scores else 0.0

        context["frames"]["restored"] = restored_frames_dir
        return context
