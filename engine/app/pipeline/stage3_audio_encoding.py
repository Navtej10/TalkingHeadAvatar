"""
Stage 3 - Audio Encoding
Extract phoneme/prosody/emotion/timing features from speech rather than
driving the generator with raw waveform.

If `text` is given instead of `audio`, route it through TTS first
(reuse the existing edge-tts integration from InterviewAI here).

Reference model: Wav2Vec2 / HuBERT
"""
from app.pipeline.base import PipelineStage
from app.config import TMP_DIR
import os
import asyncio
import ffmpeg
import edge_tts


_W2V2_PROCESSOR = None
_W2V2_MODEL = None

class AudioEncodingStage(PipelineStage):
    name = "audio_encoding"

    def __init__(self):
        super().__init__()
        from app.config import get_active_profile
        self.profile = get_active_profile()

    def run(self, context: dict) -> dict:
        audio_path = context.get("audio_path")
        text = context.get("text")
        job_id = context.get("job_id", "unknown")

        if not audio_path and not text:
            raise ValueError("either audio or text is required")

        if "audio" not in context:
            context["audio"] = {}

        from app.core.cache import get_file_hash, get_text_hash, get_cache, set_cache
        cache_key = None
        try:
            if audio_path:
                cache_key = f"audio_{get_file_hash(audio_path)}"
            elif text:
                cache_key = f"audio_text_{get_text_hash(text)}"
        except Exception as e:
            print(f"Warning: Failed to generate cache key for audio: {e}")

        if cache_key:
            cached_data = get_cache(cache_key)
            if cached_data is not None:
                context["audio"]["waveform_path"] = cached_data["waveform_path"]
                context["speech_features"] = cached_data["speech_features"]
                return context

        if audio_path:
            waveform_path = f"{TMP_DIR}/{job_id}/waveform.wav"
            try:
                ffmpeg.input(audio_path).output(waveform_path, ac=1, ar='16k').overwrite_output().run(quiet=True)
            except ffmpeg.Error as e:
                raise RuntimeError(f"FFmpeg error: {e.stderr.decode() if e.stderr else str(e)}")
            context["audio"]["waveform_path"] = waveform_path
        else:
            tts_path = f"{TMP_DIR}/{job_id}/tts.wav"
            self._tts(text, tts_path)
            context["audio"]["waveform_path"] = tts_path

        context["speech_features"] = self.encode_audio(context["audio"]["waveform_path"])
        
        if cache_key:
            set_cache(cache_key, {
                "waveform_path": context["audio"]["waveform_path"],
                "speech_features": context["speech_features"]
            })
            
        return context

    def _tts(self, text: str, output_path: str):
        async def _tts_async():
            communicate = edge_tts.Communicate(text, "en-US-AriaNeural")
            await communicate.save(output_path)
        
        asyncio.run(_tts_async())

    def encode_audio(self, audio_path: str):
        import torch
        import librosa
        import numpy as np
        import scipy.interpolate
        
        global _W2V2_PROCESSOR, _W2V2_MODEL
        if _W2V2_PROCESSOR is None or _W2V2_MODEL is None:
            try:
                from transformers import Wav2Vec2Processor, Wav2Vec2Model
                _W2V2_PROCESSOR = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
                _W2V2_MODEL = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
            except Exception as e:
                raise RuntimeError(f"Failed to load Wav2Vec2 model: {e}")
                
        device = torch.device(self.profile.device)
        dtype = torch.float16 if "cuda" in str(device).lower() else torch.float32
        
        _W2V2_MODEL.to(device, dtype=dtype)
        _W2V2_MODEL.eval()
        
        try:
            speech, sr = librosa.load(audio_path, sr=16000)
        except Exception as e:
            raise RuntimeError(f"Failed to read audio file {audio_path}: {e}")
            
        duration = len(speech) / sr
        target_fps = self.profile.target_fps
        total_frames = int(duration * target_fps)
        if total_frames == 0:
            total_frames = 1
            
        inputs = _W2V2_PROCESSOR(speech, sampling_rate=sr, return_tensors="pt").input_values
        inputs = inputs.to(device, dtype=dtype)
        
        with torch.no_grad():
            outputs = _W2V2_MODEL(inputs)
            hidden_states = outputs.last_hidden_state
            
        hidden_states = hidden_states.squeeze(0).cpu().float().numpy()
        seq_len, dim = hidden_states.shape
        
        if seq_len == 1:
            resampled_states = np.repeat(hidden_states, total_frames, axis=0)
        else:
            x_old = np.linspace(0, 1, seq_len)
            x_new = np.linspace(0, 1, total_frames)
            interp_func = scipy.interpolate.interp1d(x_old, hidden_states, axis=0, kind='linear', fill_value="extrapolate")
            resampled_states = interp_func(x_new)
        
        return resampled_states
