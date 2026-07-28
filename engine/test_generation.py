import os
import cv2
import numpy as np
import torch
from app.pipeline.stage5_generation import GenerationStage

import unittest.mock as mock

class MockProfile:
    enable_motion_latents = True
    target_fps = 30
    device = "cpu"
    custom_lora_run_name = None

def generate_with_scale(scale_factor, out_dir):
    with mock.patch("app.config.get_active_profile", return_value=MockProfile()):
        stage5 = GenerationStage()
        face_img = cv2.imread("tests/fixtures/test_face.jpg")
        if face_img is None:
            print("No test face found!")
            return
            
        # Get real motion latents from stage 4 instead of synthetic mock
        from app.pipeline.stage4_motion_prediction import MotionPredictionStage
        stage4 = MotionPredictionStage()
        ctx4 = {
            "audio": {"waveform_path": "tests/fixtures/test_audio.wav"}
        }
        ctx4 = stage4.run(ctx4)
        latents = ctx4["motion"]["latents"]

        context = {
            "face": {"aligned_face": face_img},
            "audio": {"waveform_path": "tests/fixtures/test_audio.wav"},
            "job_id": f"test_motion_{scale_factor}",
            "motion": {"latents": latents}
        }

        out_ctx = stage5.run(context)
        print(f"Generated frames for scale {scale_factor} at: {out_ctx['frames']['raw']}")

if __name__ == "__main__":
    generate_with_scale(1.0, "normal")
    generate_with_scale(0.0, "near_zero")
