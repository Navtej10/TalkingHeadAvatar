"""
Stage 10 - Video Assembly
Mux stabilized frames with audio, encode to MP4.
"""
import os
import shutil
import ffmpeg
from app.pipeline.base import PipelineStage
from app.config import OUTPUT_DIR, TMP_DIR, get_active_profile


class AssemblyStage(PipelineStage):
    name = "assembly"

    def run(self, context: dict) -> dict:
        job_id = context["job_id"]
        
        # Priority order for frames
        frame_priorities = ["interpolated", "stabilized", "restored", "refined", "raw"]
        
        frames_dir = None
        if "frames" in context:
            for key in frame_priorities:
                if key in context["frames"]:
                    frames_dir = context["frames"][key]
                    break
                    
        if frames_dir is None:
            raise ValueError(f"Job {job_id}: No frames found to assemble.")

        audio_path = context["audio"]["waveform_path"]
        
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        output_path = f"{OUTPUT_DIR}/{job_id}.mp4"
        
        profile = get_active_profile()

        # Mux frames + audio into an MP4
        video_input = ffmpeg.input(os.path.join(frames_dir, "frame_%04d.png"), framerate=profile.target_fps)
        audio_input = ffmpeg.input(audio_path)
        
        try:
            ffmpeg.output(
                video_input, 
                audio_input, 
                output_path, 
                vcodec='libx264', 
                pix_fmt='yuv420p',
                acodec='aac'
            ).overwrite_output().run(quiet=True)
        except ffmpeg.Error as e:
            raise RuntimeError(f"FFmpeg error: {e.stderr.decode() if e.stderr else str(e)}")

        context["output_path"] = output_path
        
        # Cleanup
        if not os.environ.get("TALKING_HEAD_KEEP_TMP"):
            tmp_job_dir = f"{TMP_DIR}/{job_id}"
            if os.path.exists(tmp_job_dir):
                shutil.rmtree(tmp_job_dir, ignore_errors=True)
                
        return context
