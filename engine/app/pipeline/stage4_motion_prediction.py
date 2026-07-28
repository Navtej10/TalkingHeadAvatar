"""
Stage 4 - Motion Prediction
Predict a latent motion representation (blink, gaze, jaw, head/neck sway)
from identity + speech features, to be decoded into animation by Stage 5.

Often bundled with the generator itself (e.g. LivePortrait computes motion
and generation together) — if so, this stage can become a thin pass-through
that just prepares inputs for stage5.
"""
from app.pipeline.base import PipelineStage
from app.config import get_active_profile


class MotionPredictionStage(PipelineStage):
    name = "motion_prediction"

    def run(self, context: dict) -> dict:
        profile = get_active_profile()
        
        # If disabled for this profile, just pass through
        if not profile.enable_motion_latents:
            return context

        audio_path = context["audio"]["waveform_path"]

        # Calculate duration of the audio to match frames
        import ffmpeg
        try:
            probe = ffmpeg.probe(audio_path)
            audio_stream = next((s for s in probe['streams'] if s['codec_type'] == 'audio'), None)
            duration = float(audio_stream['duration']) if audio_stream else 0.0
        except Exception:
            duration = 1.0  # fallback
            
        num_frames = int(duration * profile.target_fps)
        if num_frames == 0:
            num_frames = 1
            
        driving_video_path = context.get("driving_video_path")
        if not driving_video_path:
            # Fallback to a test video for now
            import os
            driving_video_path = os.path.abspath(os.path.join(
                os.path.dirname(__file__), 
                "../../../data/input/demo_vid.mp4"
            ))
            
        motion_latents = self._extract_driving_video_frames(
            video_path=driving_video_path,
            num_frames=num_frames
        )

        if "motion" not in context:
            context["motion"] = {}
        context["motion"]["latents"] = motion_latents
        
        return context

    def _extract_driving_video_frames(self, video_path: str, num_frames: int):
        import cv2
        import numpy as np
        
        cap = cv2.VideoCapture(video_path)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # LivePortrait's motion_extractor expects 256x256 frames
            frame_resized = cv2.resize(frame, (256, 256))
            frames.append(frame_resized)
        cap.release()
        
        if not frames:
            raise ValueError(f"Could not read any frames from driving video: {video_path}")
            
        # Reconcile length with audio duration (num_frames) by ping-pong looping the video
        # This prevents sharp, continuous looping/jumping at the end of the video.
        matched_frames = []
        n = len(frames)
        for i in range(num_frames):
            if n == 1:
                idx = 0
            else:
                cycle_len = 2 * (n - 1)
                pos = i % cycle_len
                idx = pos if pos < n else cycle_len - pos
            matched_frames.append(frames[idx])
            
        return matched_frames
