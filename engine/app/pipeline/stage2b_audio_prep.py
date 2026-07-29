"""
Stage 2b - Audio Preparation
This stage resolves text into audio using TTS if text is provided and audio is not.
It ensures that subsequent stages (Stage 3) always have a valid `audio_path` to work with.
"""
from app.pipeline.base import PipelineStage
from app.config import TMP_DIR
import os
import asyncio
import edge_tts

class AudioPreparationStage(PipelineStage):
    name = "audio_preparation"

    def run(self, context: dict) -> dict:
        audio_path = context.get("audio_path")
        text = context.get("text")
        job_id = context.get("job_id", "unknown")

        if not audio_path and not text:
            raise ValueError("either audio or text is required")
            
        if not audio_path and text:
            tts_path = f"{TMP_DIR}/{job_id}/tts.wav"
            self._tts(text, tts_path)
            context["audio_path"] = tts_path
            # We don't overwrite text in context; downstream stages like lip sync might want it
            # But Stage 3 will now use the generated audio_path.

        return context

    def _tts(self, text: str, output_path: str):
        async def _tts_async():
            communicate = edge_tts.Communicate(text, "en-US-AriaNeural")
            await communicate.save(output_path)
        
        asyncio.run(_tts_async())
