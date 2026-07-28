import numpy as np
from app.pipeline.stage4_motion_prediction import MotionPredictionStage

class MockProfile:
    enable_motion_latents = True
    target_fps = 30

import unittest.mock as mock
with mock.patch("app.pipeline.stage4_motion_prediction.get_active_profile", return_value=MockProfile()):
    stage4 = MotionPredictionStage()
    context = {
        "audio": {"waveform_path": "test.wav"},
        "emotion": "neutral",
        "gaze_target": "camera",
        "face": {"pose": (0.0, 0.0, 0.0)},
    }
    out_ctx = stage4.run(context)
    latents = out_ctx["motion"]["latents"]
    arr = np.array(latents)
    print("Max value:", arr.max())
    print("Min value:", arr.min())
    print("Mean abs value:", np.abs(arr).mean())
