"""
Stage 6 - Lip Refinement
Fix lip edges, teeth, mouth closure, phoneme accuracy on the raw generator
output.

Reference models: MuseTalk refiner, Wav2Lip (as a post-process pass)
"""
from app.pipeline.base import PipelineStage
from app.config import get_active_profile, TMP_DIR
from app.models.registry import get_model
import os
import cv2
import glob


class LipRefinementStage(PipelineStage):
    name = "lip_refinement"

    def run(self, context: dict) -> dict:
        profile = get_active_profile()
        if not profile.enable_lip_refinement:
            return context

        # Use Wav2Lip on cpu_dev, otherwise MuseTalk
        model_name = "wav2lip" if profile.name == "cpu_dev" else "musetalk"
        refiner = get_model("lip_refinement", model_name)

        job_id = context.get("job_id", "unknown")
        raw_frames_dir = context["frames"]["raw"]
        audio_waveform_path = context["audio"]["waveform_path"]
        landmarks = context["face"]["landmarks"]
        crop_box = context["face"]["crop_box"]

        refined_frames_dir = f"{TMP_DIR}/{job_id}/frames_refined"
        os.makedirs(refined_frames_dir, exist_ok=True)

        if landmarks and crop_box:
            # Transform landmarks to generated frame space (max_res x max_res)
            cx1, cy1, cx2, cy2 = crop_box
            cw = cx2 - cx1
            ch = cy2 - cy1
            max_res = profile.max_resolution

            def transform(pt):
                return [
                    (pt[0] - cx1) / cw * max_res,
                    (pt[1] - cy1) / ch * max_res
                ]

            # InsightFace 5-point: 3 is left mouth, 4 is right mouth
            t_left = transform(landmarks[3])
            t_right = transform(landmarks[4])

            cx = (t_left[0] + t_right[0]) / 2.0
            cy = (t_left[1] + t_right[1]) / 2.0
            width = abs(t_right[0] - t_left[0]) * 2.0
            height = width * 0.8

            x1 = max(0, int(cx - width / 2))
            y1 = max(0, int(cy - height / 2))
            x2 = int(cx + width / 2)
            y2 = int(cy + height / 2)
        else:
            x1, y1, x2, y2 = 0, 0, 0, 0

        frame_files = sorted(glob.glob(os.path.join(raw_frames_dir, "*.png")))
        for f in frame_files:
            frame = cv2.imread(f)
            if frame is None:
                continue

            if landmarks:
                h, w = frame.shape[:2]
                bx2 = min(w, x2)
                by2 = min(h, y2)
                bx1 = max(0, x1)
                by1 = max(0, y1)

                if bx2 > bx1 and by2 > by1:
                    mouth_crop = frame[by1:by2, bx1:bx2]
                    
                    # Call refiner (a Phase 1 mock just returns it unchanged)
                    refined_crop = refiner.refine_mouth(mouth_crop, audio_waveform_path)
                    
                    # Composite back using selected blend mode
                    if refined_crop.shape == mouth_crop.shape:
                        blend_mode = getattr(profile, "lip_blend_mode", "feather")
                        if blend_mode == "poisson":
                            import numpy as np
                            center = (bx1 + (bx2 - bx1) // 2, by1 + (by2 - by1) // 2)
                            mask = 255 * np.ones(refined_crop.shape, refined_crop.dtype)
                            try:
                                frame = cv2.seamlessClone(refined_crop, frame, mask, center, cv2.NORMAL_CLONE)
                            except Exception:
                                frame[by1:by2, bx1:bx2] = refined_crop
                        else:
                            import numpy as np
                            mask = np.ones((by2 - by1, bx2 - bx1), dtype=np.float32)
                            # Create a soft feathered edge
                            mask[0:3, :] = 0; mask[-3:, :] = 0
                            mask[:, 0:3] = 0; mask[:, -3:] = 0
                            mask_blur = cv2.GaussianBlur(mask, (15, 15), 0)
                            mask_blur = np.expand_dims(mask_blur, axis=-1)
                            
                            frame_crop = frame[by1:by2, bx1:bx2].astype(np.float32)
                            refined_f = refined_crop.astype(np.float32)
                            blended = refined_f * mask_blur + frame_crop * (1 - mask_blur)
                            frame[by1:by2, bx1:bx2] = blended.astype(np.uint8)
            
            basename = os.path.basename(f)
            cv2.imwrite(os.path.join(refined_frames_dir, basename), frame)

        context["frames"]["refined"] = refined_frames_dir
        return context
