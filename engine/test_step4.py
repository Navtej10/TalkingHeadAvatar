import os
import sys
import time

from app.pipeline.stage1_face_processing import FaceProcessingStage
from app.pipeline.stage2_identity_encoding import IdentityEncodingStage
from app.pipeline.stage2b_audio_prep import AudioPreparationStage
from app.pipeline.stage3_audio_encoding import AudioEncodingStage
from app.models.validation import validate_all_checkpoints

# Two different audio files
audio1 = "D:/Navtej/TalkingHeadAvatar/engine/vendor/MuseTalk/data/audio/eng.wav"
audio2 = "D:/Navtej/TalkingHeadAvatar/engine/vendor/MuseTalk/data/audio/sun.wav"
face_img = "D:/Navtej/TalkingHeadAvatar/engine/test_img.jpg"

print("Validating checkpoints...")
validate_all_checkpoints()

s1 = FaceProcessingStage()
s2 = IdentityEncodingStage()
s2b = AudioPreparationStage()
s3 = AudioEncodingStage()

print("\n=== JOB 1 ===")
context1 = {
    "job_id": "job_1111",
    "audio_path": audio1,
}
print("Running Stage 3 (Audio Encoding)...")
context1 = s3.run(context1)

print("\n=== JOB 2 ===")
context2 = {
    "job_id": "job_2222",
    "audio_path": audio2,
}
print("Running Stage 3 (Audio Encoding)...")
context2 = s3.run(context2)

print("\nFinished both jobs.")
