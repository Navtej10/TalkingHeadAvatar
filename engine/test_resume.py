"""
Dev-only resume script. Pass skip_to_stage and cached_job_id as direct
function arguments — no env vars needed or used.
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from app.pipeline.orchestrator import run_pipeline

image_path = os.path.abspath("../data/input/a95a61cf_image.jpg")
audio_path = os.path.abspath("../data/input/a95a61cf_audio.wav")

try:
    print("Starting pipeline resume from Stage 3...")
    output = run_pipeline(
        "a95a61cf_stage3plus",
        image=image_path,
        audio=audio_path,
        skip_to_stage=3,
        cached_job_id="a95a61cf",
    )
    print("Pipeline completed successfully:", output)
except Exception as e:
    import traceback
    traceback.print_exc()
    print("Pipeline failed:", e)
