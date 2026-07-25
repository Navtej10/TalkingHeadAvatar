import os
import time
import pytest
from fastapi.testclient import TestClient
from app.main import app


def test_api_job_flow(tmp_path, monkeypatch):
    monkeypatch.setenv("TALKING_HEAD_DATA_DIR", str(tmp_path))
    
    client = TestClient(app)
    
    fixture_path = os.path.join(os.path.dirname(__file__), "fixtures", "test_face.jpg")
    audio_fixture = os.path.join(os.path.dirname(__file__), "fixtures", "test_audio.wav")
    
    with open(fixture_path, "rb") as f_img, open(audio_fixture, "rb") as f_aud:
        response = client.post(
            "/jobs",
            files={
                "image": ("test_face.jpg", f_img, "image/jpeg"),
                "audio": ("test_audio.wav", f_aud, "audio/wav")
            }
        )
    
    assert response.status_code == 200
    data = response.json()
    assert "job_id" in data
    job_id = data["job_id"]
    
    timeout = 60
    start_time = time.time()
    
    while True:
        res = client.get(f"/jobs/{job_id}")
        assert res.status_code == 200
        status_data = res.json()
        status = status_data["status"]
        
        if status in ["done", "failed"]:
            break
            
        if time.time() - start_time > timeout:
            pytest.fail("Timeout waiting for job to finish")
            
        time.sleep(1)
        
    assert status == "done", f"Job failed: {status_data.get('error')}"
    
    result_url = status_data.get("result_url")
    assert result_url is not None
    assert os.path.exists(result_url)
