import os
import sys
import numpy as np
import pytest
from scipy.io import wavfile

# Ensure app is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "engine")))

from app.pipeline.stage3_audio_encoding import AudioEncodingStage
from app.config import PROFILES

def test_audio_encoding_stage(tmp_path):
    # Create a dummy 16k sine wave
    sample_rate = 16000
    duration = 2.0  # seconds
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    audio_data = np.sin(2 * np.pi * 440 * t)
    
    dummy_wav_path = str(tmp_path / "dummy.wav")
    wavfile.write(dummy_wav_path, sample_rate, audio_data.astype(np.float32))
    
    # Init stage
    stage = AudioEncodingStage()
    
    # Call encode directly since we already have the waveform
    features = stage.encode_audio(dummy_wav_path)
    
    assert features is not None
    
    target_fps = stage.profile.target_fps
    expected_frames = int(duration * target_fps)
    
    # Check shape [T, hidden_dim]
    assert len(features.shape) == 2
    assert features.shape[0] == expected_frames
    assert features.shape[1] == 768  # wav2vec2-base-960h dim
    
    # Check variance
    variance = np.var(features, axis=0).mean()
    assert variance > 0.0, "Features should have non-zero variance over time"
