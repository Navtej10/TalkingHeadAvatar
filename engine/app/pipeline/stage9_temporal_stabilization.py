"""
Stage 9 - Temporal Stabilization
Reduce frame-to-frame identity flicker, color shifts, jitter. Humans are
very sensitive to this, so don't skip it even in an early version.

Approach: EMA smoothing over landmarks/latents between consecutive frames.
"""
from app.pipeline.base import PipelineStage
from app.config import get_active_profile, TMP_DIR
import os
import cv2
import glob
import numpy as np


class TemporalStabilizationStage(PipelineStage):
    name = "temporal_stabilization"

    def __init__(self, ema_decay: float = 0.8):
        super().__init__()
        self.ema_decay = ema_decay

    def run(self, context: dict) -> dict:
        job_id = context.get("job_id", "unknown")
        
        # Fallback through priority chain: restored -> refined -> raw
        input_frames_dir = context["frames"].get("restored", 
                           context["frames"].get("refined", 
                           context["frames"].get("raw")))
                           
        if not input_frames_dir:
            raise ValueError(f"Job {job_id}: No frames available for temporal stabilization.")

        stabilized_frames_dir = f"{TMP_DIR}/{job_id}/frames_stabilized"
        os.makedirs(stabilized_frames_dir, exist_ok=True)

        frame_files = sorted(glob.glob(os.path.join(input_frames_dir, "*.png")))
        
        SMOOTHING_RADIUS = 5

        transforms = []
        
        if len(frame_files) > 1:
            prev = cv2.imread(frame_files[0])
            prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
            
            for i in range(1, len(frame_files)):
                curr = cv2.imread(frame_files[i])
                curr_gray = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)
                
                prev_pts = cv2.goodFeaturesToTrack(prev_gray, maxCorners=200, qualityLevel=0.01, minDistance=30, blockSize=3)
                
                if prev_pts is not None:
                    curr_pts, status, err = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_pts, None)
                    
                    idx = np.where(status == 1)[0]
                    prev_pts = prev_pts[idx]
                    curr_pts = curr_pts[idx]
                    
                    if len(prev_pts) > 3:
                        m, _ = cv2.estimateAffinePartial2D(prev_pts, curr_pts)
                        if m is not None:
                            dx = m[0, 2]
                            dy = m[1, 2]
                            da = np.arctan2(m[1, 0], m[0, 0])
                        else:
                            dx, dy, da = 0, 0, 0
                    else:
                        dx, dy, da = 0, 0, 0
                else:
                    dx, dy, da = 0, 0, 0
                
                transforms.append([dx, dy, da])
                prev_gray = curr_gray
                
        if not transforms:
            import shutil
            for f in frame_files:
                basename = os.path.basename(f)
                shutil.copy(f, os.path.join(stabilized_frames_dir, basename))
            context["frames"]["stabilized"] = stabilized_frames_dir
            return context
            
        transforms = np.array(transforms)
        trajectory = np.cumsum(transforms, axis=0)
        smoothed_trajectory = np.copy(trajectory)
        
        for i in range(3):
            padded = np.pad(trajectory[:, i], (SMOOTHING_RADIUS, SMOOTHING_RADIUS), mode='edge')
            smoothed = np.convolve(padded, np.ones(2 * SMOOTHING_RADIUS + 1) / (2 * SMOOTHING_RADIUS + 1), mode='valid')
            smoothed_trajectory[:, i] = smoothed
            
        difference = smoothed_trajectory - trajectory
        transforms_smooth = transforms + difference
        
        frame_0 = cv2.imread(frame_files[0])
        cv2.imwrite(os.path.join(stabilized_frames_dir, os.path.basename(frame_files[0])), frame_0)
        
        for i in range(len(transforms_smooth)):
            dx, dy, da = transforms_smooth[i]
            
            m = np.zeros((2, 3), np.float32)
            m[0, 0] = np.cos(da)
            m[0, 1] = -np.sin(da)
            m[1, 0] = np.sin(da)
            m[1, 1] = np.cos(da)
            m[0, 2] = dx
            m[1, 2] = dy
            
            frame = cv2.imread(frame_files[i+1])
            h, w = frame.shape[:2]
            
            stabilized_frame = cv2.warpAffine(frame, m, (w, h))
            
            cv2.imwrite(os.path.join(stabilized_frames_dir, os.path.basename(frame_files[i+1])), stabilized_frame)

        context["frames"]["stabilized"] = stabilized_frames_dir
        return context
