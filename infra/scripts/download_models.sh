#!/usr/bin/env bash
# Pulls pretrained weights into models_cache/. Fill in per model chosen in
# Phase 0 (see docs/ROADMAP.md). Each of these ships as its own repo with
# its own weight-download instructions - vendor them here so the rest of
# the team/CI has one script to run.
set -euo pipefail

MODELS_DIR="$(dirname "$0")/../../models_cache"
mkdir -p "$MODELS_DIR"

echo "Downloading LivePortrait..."
LIVEPORTRAIT_DIR="$MODELS_DIR/liveportrait"
if [ ! -d "$LIVEPORTRAIT_DIR" ]; then
    git clone https://github.com/KwaiVGI/LivePortrait "$LIVEPORTRAIT_DIR"
fi
pip install -r "$LIVEPORTRAIT_DIR/requirements.txt"
huggingface-cli download KwaiVGI/LivePortrait --local-dir "$LIVEPORTRAIT_DIR/weights/" --exclude "*.git*"

echo "Downloading MuseTalk..."
MUSETALK_DIR="$MODELS_DIR/musetalk"
if [ ! -d "$MUSETALK_DIR" ]; then
    git clone https://github.com/TMElyralab/MuseTalk "$MUSETALK_DIR"
fi
pip install -r "$MUSETALK_DIR/requirements.txt"
huggingface-cli download TMElyralab/MuseTalk --local-dir "$MUSETALK_DIR/weights/" --exclude "*.git*"

echo "Downloading Wav2Lip..."
WAV2LIP_DIR="$MODELS_DIR/wav2lip"
if [ ! -d "$WAV2LIP_DIR" ]; then
    git clone https://github.com/Rudrabha/Wav2Lip "$WAV2LIP_DIR"
fi
pip install -r "$WAV2LIP_DIR/requirements.txt"
# Assuming standard weights location for Wav2Lip (often via google drive, but we'll use a placeholder huggingface repo if available, or just mock the download step for Phase 1)
# huggingface-cli download some_repo/Wav2Lip --local-dir "$WAV2LIP_DIR/checkpoints/"

echo "Downloading GFPGAN..."
GFPGAN_DIR="$MODELS_DIR/gfpgan"
if [ ! -d "$GFPGAN_DIR" ]; then
    git clone https://github.com/TencentARC/GFPGAN "$GFPGAN_DIR"
fi
pip install -r "$GFPGAN_DIR/requirements.txt"
# Download pre-trained GFPGANv1.4 model
wget https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth -O "$GFPGAN_DIR/experiments/pretrained_models/GFPGANv1.4.pth" || echo "Note: wget failed, please download GFPGANv1.4 manually."

echo "Downloading RIFE..."
RIFE_DIR="$MODELS_DIR/rife"
if [ ! -d "$RIFE_DIR" ]; then
    git clone https://github.com/hzwer/ECCV2022-RIFE "$RIFE_DIR"
fi
pip install -r "$RIFE_DIR/requirements.txt"
# Assuming standard weights for RIFE
# huggingface-cli download hzwer/ECCV2022-RIFE --local-dir "$RIFE_DIR/checkpoints/"

echo "TODO: clone/download other chosen models into $MODELS_DIR, e.g.:"
echo "  - SadTalker:     https://github.com/OpenTalker/SadTalker"
echo "  - InsightFace:   pip install insightface (downloads weights on first use)"
