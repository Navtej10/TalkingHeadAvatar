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
            
        emotion = context.get("emotion", "neutral")
        gaze_target = context.get("gaze_target", "camera")
        
        original_pose = None
        if "face" in context and context["face"].get("pose") is not None:
            original_pose = context["face"]["pose"]
            
        speech_features = context.get("speech_features")
            
        motion_latents = self._generate_dynamic_latents(
            num_frames=num_frames, 
            speech_features=speech_features,
            target_fps=profile.target_fps,
            emotion=emotion, 
            gaze_target=gaze_target, 
            original_pose=original_pose
        )

        if "motion" not in context:
            context["motion"] = {}
        context["motion"]["latents"] = motion_latents
        
        return context

    def _generate_dynamic_latents(self, num_frames: int, speech_features=None, target_fps=30, emotion: str = "neutral", gaze_target: str = "camera", original_pose=None):
        import numpy as np
        
        latents = np.zeros((num_frames, 21, 3))
        
        base_latent = np.zeros((21, 3))
        if emotion == "happy":
            base_latent[0, 1] = 0.5
        elif emotion == "serious":
            base_latent[0, 1] = -0.2
        elif emotion == "surprised":
            base_latent[1, 1] = 0.8
            
        if gaze_target == "camera" and original_pose is not None:
            pitch, yaw, roll = original_pose
            base_latent[2, 0] = -pitch * 0.1
            base_latent[2, 1] = -yaw * 0.1
            
        latents += base_latent
        
        if speech_features is not None:
            energy = np.linalg.norm(speech_features, axis=1)
            if energy.max() > 0:
                energy = (energy - energy.min()) / (energy.max() - energy.min())
            
            if len(energy) != num_frames:
                x_old = np.linspace(0, 1, len(energy))
                x_new = np.linspace(0, 1, num_frames)
                energy = np.interp(x_new, x_old, energy)
        else:
            energy = np.zeros(num_frames)
            
        t = np.linspace(0, num_frames / target_fps, num_frames)
        base_bob = np.sin(2 * np.pi * 0.5 * t) * 0.05
        pitch_bob = base_bob * (0.5 + energy * 0.5) 
        yaw_sway = np.cos(2 * np.pi * 0.2 * t) * 0.05 * (0.5 + energy * 0.5)
        
        latents[:, 2, 0] += pitch_bob
        latents[:, 2, 1] += yaw_sway
        
        blink_prob = 1.0 / (4.0 * target_fps)
        is_blink = np.random.rand(num_frames) < blink_prob
        blink_filter = np.array([0.5, 1.0, 0.5])
        blink_signal = np.convolve(is_blink.astype(float), blink_filter, mode='same')
        blink_signal = np.clip(blink_signal, 0, 1)
        
        latents[:, 11, 1] += blink_signal * 0.8
        
        latents += np.random.randn(*latents.shape) * 1e-4
        
        assert np.sum(np.abs(np.diff(latents, axis=0))) > 0, "Motion latents must be time-varying"
        
        return [np.copy(latents[i]) for i in range(num_frames)]
