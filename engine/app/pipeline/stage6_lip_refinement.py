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

        max_retries = 2
        audio_offset_ms = 0.0
        final_offset = 0.0
        
        syncnet = get_model("sync_scorer", "syncnet")
        
        for attempt in range(max_retries + 1):
            asymmetric_frames = 0
            frame_files = sorted(glob.glob(os.path.join(raw_frames_dir, "*.png")))
            for f_idx, f in enumerate(frame_files):
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
                        
                        # Extract visemes if available
                        visemes = None
                        speech_features = context.get("speech_features")
                        if isinstance(speech_features, dict):
                            visemes = speech_features.get("discrete")
                        
                        # Call refiner (a Phase 1 mock just returns it unchanged)
                        # We pass audio_offset_ms to apply the time shift
                        refined_crop = refiner.refine_mouth(mouth_crop, audio_waveform_path, visemes=visemes, audio_offset_ms=audio_offset_ms)
                        
                        # 1. Color Transfer (Reinhard)
                        def reinhard_color_transfer(src, tgt):
                            import cv2
                            import numpy as np
                            src_lab = cv2.cvtColor(src, cv2.COLOR_BGR2LAB).astype(np.float32)
                            tgt_lab = cv2.cvtColor(tgt, cv2.COLOR_BGR2LAB).astype(np.float32)
                            
                            src_mean, src_std = cv2.meanStdDev(src_lab)
                            tgt_mean, tgt_std = cv2.meanStdDev(tgt_lab)
                            
                            src_std[src_std == 0] = 1e-5
                            
                            out_lab = (src_lab - src_mean.reshape(1,1,3)) * (tgt_std.reshape(1,1,3) / src_std.reshape(1,1,3)) + tgt_mean.reshape(1,1,3)
                            out_lab = np.clip(out_lab, 0, 255).astype(np.uint8)
                            return cv2.cvtColor(out_lab, cv2.COLOR_LAB2BGR)
    
                        if isinstance(refined_crop, list):
                            # For simplicity assume single image return from mock or we only handle the first
                            refined_crop = refined_crop[0] if len(refined_crop) > 0 else mouth_crop
                            
                        refined_crop = reinhard_color_transfer(refined_crop, mouth_crop)
                        
                        # 2. Get Face Parsing Mask
                        parser = get_model("face_parsing", "bisenet")
                        parse_mask_full = parser.parse_face(frame)
                        parse_crop = parse_mask_full[by1:by2, bx1:bx2]
                        
                        # Composite back using selected blend mode
                        if refined_crop.shape == mouth_crop.shape:
                            blend_mode = getattr(profile, "lip_blend_mode", "feather")
                            if blend_mode == "poisson":
                                import numpy as np
                                import cv2
                                center = (bx1 + (bx2 - bx1) // 2, by1 + (by2 - by1) // 2)
                                
                                # Hair-aware mask for Poisson
                                mask = 255 * np.ones(refined_crop.shape[:2], dtype=np.uint8)
                                mask[parse_crop == 17] = 0
                                mask = cv2.merge([mask, mask, mask])
                                
                                try:
                                    frame = cv2.seamlessClone(refined_crop, frame, mask, center, cv2.NORMAL_CLONE)
                                except Exception:
                                    frame[by1:by2, bx1:bx2] = refined_crop
                            else:
                                import numpy as np
                                import cv2
                                mask = np.ones((by2 - by1, bx2 - bx1), dtype=np.float32)
                                # Create a soft feathered edge
                                mask[0:3, :] = 0; mask[-3:, :] = 0
                                mask[:, 0:3] = 0; mask[:, -3:] = 0
                                mask_blur = cv2.GaussianBlur(mask, (15, 15), 0)
                                
                                # Hair-aware mask modification
                                mask_blur[parse_crop == 17] = 0.0
                                
                                mask_blur = np.expand_dims(mask_blur, axis=-1)
                                
                                frame_crop = frame[by1:by2, bx1:bx2].astype(np.float32)
                                refined_f = refined_crop.astype(np.float32)
                                blended = refined_f * mask_blur + frame_crop * (1 - mask_blur)
                                frame[by1:by2, bx1:bx2] = blended.astype(np.uint8)

                    # 3. Eye Refinement
                    t_lefteye = transform(landmarks[0])
                    t_righteye = transform(landmarks[1])
                    
                    eye_width = width * 0.4
                    eye_height = eye_width * 0.5
                    
                    def crop_eye(center):
                        ex1 = max(0, int(center[0] - eye_width / 2))
                        ey1 = max(0, int(center[1] - eye_height / 2))
                        ex2 = min(w, int(center[0] + eye_width / 2))
                        ey2 = min(h, int(center[1] + eye_height / 2))
                        if ex2 > ex1 and ey2 > ey1:
                            return frame[ey1:ey2, ex1:ex2], (ex1, ey1, ex2, ey2)
                        raise ValueError(f"Failed to crop eye at center {center}. Out of bounds.")
                    
                    left_eye_crop, l_box = crop_eye(t_lefteye)
                    right_eye_crop, r_box = crop_eye(t_righteye)
                    
                    if left_eye_crop is not None and right_eye_crop is not None:
                        # Eye GAN has been removed entirely per instructions.
                        # No fallback is provided.
                        pass
                            
                            
                        # 4. Asymmetry Check
                        # Calculate heuristic asymmetry (vertical difference + some mock deviation)
                        y_diff = abs(t_lefteye[1] - t_righteye[1])
                        # If the baseline distance is large, or generated frame deviated:
                        asym_score = y_diff * (1.0 + np.random.uniform(-0.1, 0.2))
                        if asym_score > 10.0:
                            asymmetric_frames += 1
                
                basename = os.path.basename(f)
                cv2.imwrite(os.path.join(refined_frames_dir, basename), frame)

            # Post-check execution
            final_offset = syncnet.evaluate(refined_frames_dir, audio_waveform_path)
            
            if abs(final_offset) > 80.0 and attempt < max_retries:
                # Apply shift and retry
                audio_offset_ms -= final_offset # Shift in opposite direction of offset
                continue
            else:
                break

        if "metadata" not in context:
            context["metadata"] = {}
        context["metadata"]["sync_offset_ms"] = final_offset
        context["metadata"]["eye_asymmetry_score"] = asymmetric_frames
        context["frames"]["refined"] = refined_frames_dir
        return context
