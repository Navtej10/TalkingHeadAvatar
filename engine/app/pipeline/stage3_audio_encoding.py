"""
Stage 3 - Audio Encoding
Extract phoneme/prosody/emotion/timing features from speech rather than
driving the generator with raw waveform.

Reference model: Wav2Vec2 / HuBERT
"""
from app.pipeline.base import PipelineStage
from app.config import TMP_DIR
import os
import asyncio


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
        job_id = context.get("job_id", "unknown")

        if not audio_path:
            raise ValueError("audio_path is required for Stage 3")

        if "audio" not in context:
            context["audio"] = {}

        from app.core.cache import get_file_hash, get_cache, set_cache
        cache_key = None
        try:
            cache_key = f"audio_{get_file_hash(audio_path)}"
        except Exception as e:
            print(f"Warning: Failed to generate cache key for audio: {e}")

        if cache_key:
            cached_data = get_cache(cache_key)
            if cached_data is not None:
                if os.path.exists(cached_data["waveform_path"]):
                    context["audio"]["waveform_path"] = cached_data["waveform_path"]
                    context["speech_features"] = cached_data["speech_features"]
                    return context

        context["audio"]["waveform_path"] = audio_path
        context["speech_features"] = self.encode_audio(context["audio"]["waveform_path"])
        
        if cache_key:
            set_cache(cache_key, {
                "waveform_path": context["audio"]["waveform_path"],
                "speech_features": context["speech_features"]
            })
            
        return context

    def encode_audio(self, audio_path: str):
        import torch
        import librosa
        import numpy as np
        import scipy.interpolate
        
        global _W2V2_PROCESSOR, _W2V2_MODEL
        # We also add CTC model for discrete viseme mapping
        if _W2V2_PROCESSOR is None or _W2V2_MODEL is None or getattr(self, '_ctc_model', None) is None:
            print("[audio_encoding] Wav2Vec2 model: LOADED FRESH (first use this process)")
            import time
            from huggingface_hub.utils import HfHubHTTPError
            
            max_retries = 3
            backoff_factor = 5
            
            hf_token = os.environ.get("HF_TOKEN")
            
            for attempt in range(max_retries + 1):
                try:
                    from transformers import Wav2Vec2Processor, Wav2Vec2Model, Wav2Vec2ForCTC
                    _W2V2_PROCESSOR = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h", token=hf_token)
                    _W2V2_MODEL = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h", token=hf_token)
                    self._ctc_model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h", token=hf_token)
                    break # Success
                except Exception as e:
                    # Check if it's a 429
                    is_429 = "429" in str(e) or (isinstance(e, HfHubHTTPError) and e.response.status_code == 429)
                    if is_429 and attempt < max_retries:
                        sleep_time = backoff_factor * (2 ** attempt)
                        print(f"HuggingFace rate limit (429) hit. Retrying in {sleep_time}s...")
                        time.sleep(sleep_time)
                    else:
                        raise RuntimeError(f"Failed to load Wav2Vec2 model: {e}")
        else:
            print("[audio_encoding] Wav2Vec2 model: served from cache (already loaded)")
                
        device = torch.device(self.profile.device)
        dtype = torch.float16 if "cuda" in str(device).lower() else torch.float32
        
        _W2V2_MODEL.to(device, dtype=dtype)
        _W2V2_MODEL.eval()
        self._ctc_model.to(device, dtype=dtype)
        self._ctc_model.eval()
        
        try:
            speech, sr = librosa.load(audio_path, sr=16000)
        except Exception as e:
            raise RuntimeError(f"Failed to read audio file {audio_path}: {e}")
            
        duration = len(speech) / sr
        
        import hashlib
        try:
            with open(audio_path, "rb") as f:
                file_hash = hashlib.md5(f.read()).hexdigest()[:16]
            print(f"[audio_encoding] Processing audio: path={audio_path}, duration={duration:.2f}s, hash={file_hash}")
        except Exception as e:
            print(f"[audio_encoding] Processing audio: path={audio_path}, duration={duration:.2f}s, hash=ERROR({e})")
            
        target_fps = self.profile.target_fps
        total_frames = int(duration * target_fps)
        if total_frames == 0:
            total_frames = 1
            
        inputs = _W2V2_PROCESSOR(speech, sampling_rate=sr, return_tensors="pt").input_values
        inputs = inputs.to(device, dtype=dtype)
        
        with torch.no_grad():
            outputs = _W2V2_MODEL(inputs)
            hidden_states = outputs.last_hidden_state
            
            ctc_outputs = self._ctc_model(inputs)
            logits = ctc_outputs.logits
            predicted_ids = torch.argmax(logits, dim=-1)
            
        hidden_states = hidden_states.squeeze(0).cpu().float().numpy()
        seq_len, dim = hidden_states.shape
        
        predicted_ids = predicted_ids.squeeze(0).cpu().numpy()
        vocab = _W2V2_PROCESSOR.tokenizer.convert_ids_to_tokens(predicted_ids)
        
        # Simple character-to-viseme mapping
        # 0: silence/generic, 1: bilabial (P/B/M), 2: labiodental (F/V), 3: open vowel (A/E/I), 4: rounded vowel (O/U/W)
        viseme_sequence = np.zeros(len(vocab), dtype=np.int32)
        for i, token in enumerate(vocab):
            t = token.upper()
            if t in ["P", "B", "M"]:
                viseme_sequence[i] = 1
            elif t in ["F", "V"]:
                viseme_sequence[i] = 2
            elif t in ["A", "E", "I"]:
                viseme_sequence[i] = 3
            elif t in ["O", "U", "W"]:
                viseme_sequence[i] = 4
        
        if seq_len == 1:
            resampled_states = np.repeat(hidden_states, total_frames, axis=0)
            resampled_visemes = np.repeat(viseme_sequence, total_frames, axis=0)
        else:
            x_old = np.linspace(0, 1, seq_len)
            x_new = np.linspace(0, 1, total_frames)
            
            interp_func = scipy.interpolate.interp1d(x_old, hidden_states, axis=0, kind='linear', fill_value="extrapolate")
            resampled_states = interp_func(x_new)
            
            # Nearest neighbor for discrete visemes
            interp_func_viseme = scipy.interpolate.interp1d(x_old, viseme_sequence, kind='nearest', fill_value="extrapolate")
            resampled_visemes = interp_func_viseme(x_new).astype(np.int32)
        
        return {
            "continuous": resampled_states,
            "discrete": resampled_visemes
        }
