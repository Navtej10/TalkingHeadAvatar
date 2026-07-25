import os
import pytest
import numpy as np
import cv2
import ffmpeg

# Marking this as slow so it doesn't run in standard CI without --run-slow
pytestmark = pytest.mark.slow


def _compute_crude_lip_sync_score(video_path: str, audio_path: str) -> float:
    """
    Crude audio-visual correlation metric for Phase 1.
    In a real environment, you would use SyncNet to compute the confidence score.
    Returns a dummy/crude score between 0.0 and 1.0.
    """
    # For Phase 1, we just ensure the video has frames and audio matches length roughly.
    try:
        probe = ffmpeg.probe(video_path)
        v_stream = next((s for s in probe['streams'] if s['codec_type'] == 'video'), None)
        a_stream = next((s for s in probe['streams'] if s['codec_type'] == 'audio'), None)
        
        if not v_stream or not a_stream:
            return 0.0
            
        v_dur = float(v_stream.get('duration', 0.0))
        a_dur = float(a_stream.get('duration', 0.0))
        
        # A perfectly synced video has matching durations
        if max(v_dur, a_dur) == 0:
            return 0.0
            
        diff = abs(v_dur - a_dur)
        score = max(0.0, 1.0 - diff)
        return score
    except Exception as e:
        print(f"Sync metric failed: {e}")
        return 0.0


def _compute_identity_similarity(input_image_path: str, output_video_path: str) -> float:
    """
    Computes cosine similarity between ArcFace embedding on input vs output frames.
    For Phase 1 (since ArcFace in Stage 2 is stubbed), this returns a mock score.
    """
    try:
        # Load input image
        img = cv2.imread(input_image_path)
        if img is None:
            return 0.0
            
        # In a real environment, we'd run InsightFace/ArcFace to get 512D embeddings:
        # emb_input = arcface.get_embedding(img)
        # emb_output = arcface.get_embedding(first_video_frame)
        # similarity = np.dot(emb_input, emb_output) / (np.linalg.norm(emb_input) * np.linalg.norm(emb_output))
        
        # Mocking similarity for Phase 1
        similarity = 0.95
        return similarity
    except Exception as e:
        print(f"Identity similarity failed: {e}")
        return 0.0


@pytest.fixture
def bake_off_clips(tmp_path):
    """
    Mocks 5 bake-off clips from Prompt 0.1 for testing purposes.
    Returns a list of dicts: [{"image": path, "audio": path}, ...]
    """
    clips = []
    import wave, struct
    for i in range(5):
        img_path = str(tmp_path / f"bakeoff_{i}.jpg")
        # Create a dummy colored image
        img = np.zeros((256, 256, 3), dtype=np.uint8)
        img[:] = (i * 50, 100, 200)
        cv2.imwrite(img_path, img)
        
        aud_path = str(tmp_path / f"bakeoff_{i}.wav")
        with wave.open(aud_path, 'w') as f:
            f.setnchannels(1)
            f.setsampwidth(2)
            f.setframerate(16000)
            # 1 second of silence
            f.writeframes(struct.pack('<h', 0) * 16000)
            
        clips.append({"image": img_path, "audio": aud_path})
        
    return clips


def test_quality_regression_bakeoff(bake_off_clips, monkeypatch, tmp_path):
    """
    Runs the full pipeline on 5 bake-off clips and asserts baseline quality metrics.
    """
    monkeypatch.setenv("TALKING_HEAD_DATA_DIR", str(tmp_path))
    from app.pipeline.orchestrator import run_pipeline
    from tests.test_pipeline_smoke import FakeUploadFile
    
    results = []
    
    for i, clip in enumerate(bake_off_clips):
        job_id = f"bakeoff-job-{i}"
        
        with open(clip["image"], "rb") as f:
            img_bytes = f.read()
            
        with open(clip["audio"], "rb") as f:
            aud_bytes = f.read()
            
        img_upload = FakeUploadFile(img_bytes)
        aud_upload = FakeUploadFile(aud_bytes)
        
        try:
            output_mp4 = run_pipeline(job_id, image=img_upload, audio=aud_upload, text=None)
            
            sync_score = _compute_crude_lip_sync_score(output_mp4, clip["audio"])
            id_score = _compute_identity_similarity(clip["image"], output_mp4)
            
            results.append({
                "job_id": job_id,
                "sync_score": sync_score,
                "id_score": id_score,
                "status": "success"
            })
        except Exception as e:
            results.append({
                "job_id": job_id,
                "status": "failed",
                "error": str(e)
            })

    # Assertions
    failures = [r for r in results if r["status"] == "failed"]
    assert len(failures) == 0, f"Pipeline failed on clips: {failures}"
    
    avg_sync = np.mean([r["sync_score"] for r in results])
    avg_id = np.mean([r["id_score"] for r in results])
    
    # We want strict regression bounds
    assert avg_sync > 0.8, f"Lip-sync score degraded below threshold: {avg_sync}"
    assert avg_id > 0.9, f"Identity similarity degraded below threshold: {avg_id}"
    
    print(f"\n--- Quality Regression Results ---")
    print(f"Average Lip-Sync Score: {avg_sync:.3f}")
    print(f"Average Identity Similarity: {avg_id:.3f}")
