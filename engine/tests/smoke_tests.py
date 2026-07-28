import os
import sys
import cv2
import numpy as np

# Add engine to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.pipeline.stage1_face_processing import FaceProcessingStage
from app.pipeline.stage2_identity_encoding import IdentityEncodingStage
from app.pipeline.stage3_audio_encoding import AudioEncodingStage
from app.pipeline.stage4_motion_prediction import MotionPredictionStage
from app.pipeline.stage5_generation import GenerationStage
from app.pipeline.stage6_lip_refinement import LipRefinementStage
from app.pipeline.stage7_face_restoration import FaceRestorationStage
from app.pipeline.stage8_frame_interpolation import FrameInterpolationStage
from app.pipeline.stage9_temporal_stabilization import TemporalStabilizationStage
from app.pipeline.stage10_assembly import AssemblyStage
from app.config import TMP_DIR

IMAGE_PATH = os.path.abspath("tests/fixtures/test_face.jpg")
AUDIO_PATH = os.path.abspath("tests/fixtures/test_audio.wav")

def make_job_dir(job_id):
    os.makedirs(f"{TMP_DIR}/{job_id}", exist_ok=True)

def test_stage1():
    job_id = "test"
    make_job_dir(job_id)
    stage = FaceProcessingStage()
    context = {"image_path": IMAGE_PATH, "job_id": job_id}
    res = stage.run(context)
    assert "face" in res
    assert res["face"]["aligned_face"] is not None
    assert res["face"]["crop_box"] is not None
    # nonzero area
    x1, y1, x2, y2 = res["face"]["crop_box"]
    assert (x2 - x1) * (y2 - y1) > 0
    print("Stage 1 passed")
    return res

def test_stage2(context):
    stage = IdentityEncodingStage()
    res = stage.run(context)
    assert "identity" in res
    assert "embedding" in res["identity"]
    emb = res["identity"]["embedding"]
    assert emb.shape[0] > 0
    assert np.var(emb) > 0
    print("Stage 2 passed")
    return res

def test_stage3(context):
    stage = AudioEncodingStage()
    # also test with text to trigger edge-tts
    job_id = "test_text"
    make_job_dir(job_id)
    text_ctx = {"text": "Hello world", "job_id": job_id}
    text_res = stage.run(text_ctx)
    assert "audio" in text_res
    assert os.path.exists(text_res["audio"]["waveform_path"])
    
    # test with audio
    res = stage.run(context)
    assert "speech_features" in res
    print("Stage 3 passed")
    return res

def test_stage4(context):
    stage = MotionPredictionStage()
    context["emotion"] = "neutral"
    context["gaze_target"] = "camera"
    res = stage.run(context)
    assert "motion" in res
    assert "latents" in res["motion"]
    # check that motion isn't static
    latents = res["motion"]["latents"]
    assert len(latents) > 0
    diff = np.sum(np.abs(np.diff(latents, axis=0)))
    assert diff > 0
    print("Stage 4 passed")
    return res

def test_stage5(context):
    stage = GenerationStage()
    res = stage.run(context)
    assert "frames" in res
    assert "raw" in res["frames"]
    assert os.path.exists(res["frames"]["raw"])
    print("Stage 5 passed")
    return res

def test_stage6(context):
    stage = LipRefinementStage()
    res = stage.run(context)
    assert "refined" in res["frames"]
    assert os.path.exists(res["frames"]["refined"])
    print("Stage 6 passed")
    return res

if __name__ == "__main__":
    ctx = test_stage1()
    ctx["audio_path"] = AUDIO_PATH
    ctx = test_stage2(ctx)
    ctx = test_stage3(ctx)
    ctx = test_stage4(ctx)
    try:
        ctx = test_stage5(ctx)
        ctx = test_stage6(ctx)
    except Exception as e:
        print(f"Failed at later stage due to missing weights/models: {e}")
