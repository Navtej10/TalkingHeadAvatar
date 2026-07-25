"""
Hardware/quality profiles. Pick one via TALKING_HEAD_PROFILE env var.

cpu_dev     - local development on CPU, small/fast models, low res
gpu_single  - single consumer GPU (e.g. RTX 4090), production inference
gpu_cloud   - rented multi-GPU, batch jobs / fine-tuning workloads
"""
import os
from dataclasses import dataclass


@dataclass
class Profile:
    name: str
    device: str
    max_resolution: int
    target_fps: int
    enable_interpolation: bool
    enable_restoration: bool
    enable_motion_latents: bool
    enable_lip_refinement: bool
    batch_size: int
    custom_lora_run_name: str = "default_avatar"
    codeformer_fidelity: float = 0.5
    lip_blend_mode: str = "feather"


PROFILES = {
    "cpu_dev": Profile(
        name="cpu_dev",
        device="cpu",
        max_resolution=256,
        target_fps=15,
        enable_interpolation=False,
        enable_restoration=False,
        enable_motion_latents=True,
        enable_lip_refinement=False,
        batch_size=1,
        codeformer_fidelity=0.5,
        lip_blend_mode="feather",
    ),
    "gpu_single": Profile(
        name="gpu_single",
        device="cuda:0",
        max_resolution=512,
        target_fps=30,
        enable_interpolation=True,
        enable_restoration=True,
        enable_motion_latents=True,
        enable_lip_refinement=True,
        batch_size=4,
        codeformer_fidelity=0.7,
        lip_blend_mode="poisson",
    ),
    "gpu_cloud": Profile(
        name="gpu_cloud",
        device="cuda",
        max_resolution=1024,
        target_fps=60,
        enable_interpolation=True,
        enable_restoration=True,
        enable_motion_latents=True,
        enable_lip_refinement=True,
        batch_size=16,
        codeformer_fidelity=0.7,
        lip_blend_mode="poisson",
    ),
}


def get_active_profile() -> Profile:
    name = os.environ.get("TALKING_HEAD_PROFILE", "cpu_dev")
    if name not in PROFILES:
        raise ValueError(f"Unknown profile '{name}'. Options: {list(PROFILES)}")
    return PROFILES[name]


# Working directories
DATA_DIR = os.environ.get("TALKING_HEAD_DATA_DIR", "../data")
INPUT_DIR = f"{DATA_DIR}/input"
OUTPUT_DIR = f"{DATA_DIR}/output"
TMP_DIR = f"{DATA_DIR}/tmp"
MODELS_CACHE_DIR = os.environ.get("TALKING_HEAD_MODELS_DIR", "../models_cache")
REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///./talkinghead.db")
