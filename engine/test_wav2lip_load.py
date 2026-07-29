import os
import sys
import torch
import numpy as np

from app.models.registry import load_wav2lip

print("Loading Wav2Lip model...")
wrapper = load_wav2lip()
print("Model loaded successfully!")

print("Creating dummy tensors for forward pass...")
# Wav2Lip forward expects (audio_sequences, face_sequences)
# audio_sequences: (B, 1, 80, 16) -> mel spectrogram
# face_sequences: (B, 6, H, W) -> concatenated masked and full face (6 channels)
B, H, W = 1, 96, 96
dummy_mel = torch.randn(B, 1, 80, 16).to(wrapper.device, dtype=wrapper.dtype)
dummy_face = torch.randn(B, 6, H, W).to(wrapper.device, dtype=wrapper.dtype)

print("Running dummy forward pass...")
with torch.no_grad():
    out = wrapper.models["generator"](dummy_mel, dummy_face)
print("Forward pass successful! Output shape:", out.shape)
