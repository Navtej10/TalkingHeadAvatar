"""
Stage 4 - Audio-to-Motion Prediction

Converts audio features (produced by Stage 3) into a per-frame sequence of
LivePortrait keypoint deltas (shape: [21, 3] per frame).

This completely removes the driving-video dependency that was causing the
distorted/melted-face artefacts.  All motion is derived from the audio signal:

  - Jaw open/close  ← mel-spectrogram energy per frame
  - Jaw shape       ← viseme class (0-4) from Wav2Vec2 CTC phoneme output
  - Eye blinks      ← biologically-realistic Poisson-distributed timing
  - Brow motion     ← prosodic stress peaks
  - Head sway       ← low-frequency sinusoidal + smoothed noise (no driving video)

Output put into context:
    context["motion"]["latents"] = List[np.ndarray shape (21,3)]
"""
from app.pipeline.base import PipelineStage
from app.config import get_active_profile
import os


class MotionPredictionStage(PipelineStage):
    name = "motion_prediction"

    def run(self, context: dict) -> dict:
        profile = get_active_profile()

        if not profile.enable_motion_latents:
            return context

        audio_path = context["audio"]["waveform_path"]
        speech_features = context.get("speech_features", {})

        # ── Determine frame count from audio duration ──────────────────────────
        import ffmpeg
        try:
            probe = ffmpeg.probe(audio_path)
            audio_stream = next(
                (s for s in probe["streams"] if s["codec_type"] == "audio"), None
            )
            duration = float(audio_stream["duration"]) if audio_stream else 1.0
        except Exception:
            duration = 1.0

        num_frames = max(1, int(duration * profile.target_fps))

        # ── Generate audio-driven keypoint deltas ──────────────────────────────
        from app.models.audio_to_motion import audio_to_motion_deltas

        job_id = context.get("job_id", "unknown")
        # Use job_id hash as seed for reproducibility within a job
        seed = int.from_bytes(job_id.encode()[:4].ljust(4, b"\x00"), "big") % (2 ** 31)

        motion_latents = audio_to_motion_deltas(
            audio_waveform_path=audio_path,
            speech_features=speech_features,
            num_frames=num_frames,
            fps=profile.target_fps,
            seed=seed,
        )

        if "motion" not in context:
            context["motion"] = {}
        context["motion"]["latents"] = motion_latents

        print(
            f"[motion_prediction] Produced {len(motion_latents)} audio-driven "
            f"keypoint-delta frames for job {job_id}",
            flush=True,
        )
        return context
