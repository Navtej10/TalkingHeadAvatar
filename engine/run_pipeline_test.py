import os
import sys

from app.pipeline.orchestrator import run_pipeline

fixture_path = os.path.abspath(os.path.join("tests", "fixtures", "test_face.jpg"))
audio_fixture = os.path.abspath(os.path.join("tests", "fixtures", "test_audio.wav"))

print("Running pipeline directly...")
out_path = run_pipeline(
    job_id="test_local_job",
    image=fixture_path,
    audio=audio_fixture
)
print("Done. Output:", out_path)
