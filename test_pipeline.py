import os
import sys

# Add engine to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'engine')))

from engine.app.pipeline.orchestrator import run_pipeline

image_path = os.path.abspath("tests/fixtures/test_face.jpg")
audio_path = os.path.abspath("tests/fixtures/test_audio.wav")

try:
    print("Starting pipeline...")
    output = run_pipeline("test_job_1", image=image_path, audio=audio_path)
    print("Pipeline finished successfully:", output)
except Exception as e:
    import traceback
    traceback.print_exc()
    print("Pipeline failed:", e)
