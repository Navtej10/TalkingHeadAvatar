"""
Smoke test: confirms the orchestrator runs all 10 stages end-to-end without
error using the stub implementations. This should pass on day one, before
any real model is wired in — it's a scaffold sanity check, not a quality
test.
"""
import io
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.pipeline.orchestrator import run_pipeline  # noqa: E402


class FakeUploadFile:
    def __init__(self, content: bytes):
        self.file = io.BytesIO(content)


def test_pipeline_runs_end_to_end(tmp_path, monkeypatch):
    monkeypatch.setenv("TALKING_HEAD_DATA_DIR", str(tmp_path))
    # reload config-dependent modules would be needed for a real test env;
    # this smoke test illustrates the shape of a pipeline test.
    fixture_path = os.path.join(os.path.dirname(__file__), "fixtures", "test_face.jpg")
    with open(fixture_path, "rb") as f:
        image_bytes = f.read()
    image = FakeUploadFile(image_bytes)
    
    audio_fixture = os.path.join(os.path.dirname(__file__), "fixtures", "test_audio.wav")
    with open(audio_fixture, "rb") as f:
        audio_bytes = f.read()
    audio = FakeUploadFile(audio_bytes)

    output_path = run_pipeline("test-job-1", image=image, audio=audio, text=None)

    assert output_path.endswith("test-job-1.mp4")
    assert os.path.exists(output_path)

    import ffmpeg
    probe = ffmpeg.probe(output_path)
    video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
    assert video_stream is not None
    
    frames = int(video_stream.get('nb_frames', 0))
    duration = float(video_stream.get('duration', 0.0))
    
    assert frames > 0, f"Video must have frames, got {frames}"
    assert duration > 0.0, f"Video must have duration, got {duration}"
