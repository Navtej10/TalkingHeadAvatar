import os
import sys
import numpy as np
import pytest

# Ensure app is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "engine")))

from app.pipeline.stage2_identity_encoding import IdentityEncodingStage, IdentityExtractionError
from app.config import PROFILES

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def test_identity_encoding_error_no_model():
    stage = IdentityEncodingStage()
    # Force model to none to test error propagation
    stage.rec_model = None
    
    context = {
        "job_id": "test",
        "face": {
            "aligned_face": np.zeros((112, 112, 3), dtype=np.uint8)
        }
    }
    
    with pytest.raises(IdentityExtractionError, match="Recognition model is not loaded"):
        stage.run(context)

class MockRecModel:
    def get_feat(self, img):
        # Deterministic dummy features based on mean pixel value
        val = np.mean(img)
        feat = np.ones(512, dtype=np.float32) * val
        # Add tiny noise to avoid perfectly identical deterministic outputs 
        # (simulating model drift/precision issues)
        feat += np.random.randn(512).astype(np.float32) * 0.01
        return feat

def test_identity_encoding_similarity():
    stage = IdentityEncodingStage()
    # Inject mock to avoid requiring onnxruntime during test
    stage.rec_model = MockRecModel()
    
    # 1. Run for frame 1
    face1 = np.ones((112, 112, 3), dtype=np.uint8) * 128
    ctx1 = {"job_id": "job1", "face": {"aligned_face": face1}}
    ctx1 = stage.run(ctx1)
    emb1 = ctx1["identity"]["embedding"]
    
    # 2. Run for frame 2 (identical face array)
    face2 = np.ones((112, 112, 3), dtype=np.uint8) * 128
    ctx2 = {"job_id": "job1", "face": {"aligned_face": face2}}
    ctx2 = stage.run(ctx2)
    emb2 = ctx2["identity"]["embedding"]
    
    # Assert
    sim = cosine_similarity(emb1, emb2)
    assert sim > 0.9, f"Cosine similarity {sim} is not > 0.9"
    
    # Assert not identically random Arrays (which would fail cosine sim anyway)
    # The dummy mock uses mean pixel, so variance is very small but present.
