import os
import sys
import urllib.request
import hashlib

# Expected Models Database
# Note: For massive models we just print download commands/URLs and exit non-zero if missing.
MODELS_DB = {
    "GFPGAN": {
        "url": "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth",
        "path": "gfpgan/weights/GFPGANv1.4.pth",
        "size": 348632874,
        "auto_download": True
    },
    "LivePortrait_Appearance": {
        "url": "https://huggingface.co/KwaiVGI/LivePortrait/resolve/main/base_models/appearance_feature_extractor.pth",
        "path": "liveportrait/weights/appearance_feature_extractor.pth",
        "auto_download": False,
        "instructions": "wget -c https://huggingface.co/KwaiVGI/LivePortrait/resolve/main/base_models/appearance_feature_extractor.pth -O models_cache/liveportrait/weights/appearance_feature_extractor.pth"
    },
    "LivePortrait_Motion": {
        "url": "https://huggingface.co/KwaiVGI/LivePortrait/resolve/main/base_models/motion_extractor.pth",
        "path": "liveportrait/weights/motion_extractor.pth",
        "auto_download": False,
        "instructions": "wget -c https://huggingface.co/KwaiVGI/LivePortrait/resolve/main/base_models/motion_extractor.pth -O models_cache/liveportrait/weights/motion_extractor.pth"
    },
    "LivePortrait_Warping": {
        "url": "https://huggingface.co/KwaiVGI/LivePortrait/resolve/main/base_models/warping_module.pth",
        "path": "liveportrait/weights/warping_module.pth",
        "auto_download": False,
        "instructions": "wget -c https://huggingface.co/KwaiVGI/LivePortrait/resolve/main/base_models/warping_module.pth -O models_cache/liveportrait/weights/warping_module.pth"
    },
    "LivePortrait_Spade": {
        "url": "https://huggingface.co/KwaiVGI/LivePortrait/resolve/main/base_models/spade_generator.pth",
        "path": "liveportrait/weights/spade_generator.pth",
        "auto_download": False,
        "instructions": "wget -c https://huggingface.co/KwaiVGI/LivePortrait/resolve/main/base_models/spade_generator.pth -O models_cache/liveportrait/weights/spade_generator.pth"
    },
    "LivePortrait_Stitching": {
        "url": "https://huggingface.co/KwaiVGI/LivePortrait/resolve/main/base_models/stitching_retargeting_module.pth",
        "path": "liveportrait/weights/stitching_retargeting_module.pth",
        "auto_download": False,
        "instructions": "wget -c https://huggingface.co/KwaiVGI/LivePortrait/resolve/main/base_models/stitching_retargeting_module.pth -O models_cache/liveportrait/weights/stitching_retargeting_module.pth"
    },
    "MuseTalk_UNet": {
        "url": "https://huggingface.co/TencentARC/MuseTalk/resolve/main/musetalk/unet.pth",
        "path": "musetalk/weights/unet.pth",
        "auto_download": False,
        "instructions": "wget -c https://huggingface.co/TencentARC/MuseTalk/resolve/main/musetalk/unet.pth -O models_cache/musetalk/weights/unet.pth"
    },
    "MuseTalk_VAE": {
        "url": "https://huggingface.co/TencentARC/MuseTalk/resolve/main/musetalk/vae.pth",
        "path": "musetalk/weights/vae.pth",
        "auto_download": False,
        "instructions": "wget -c https://huggingface.co/TencentARC/MuseTalk/resolve/main/musetalk/vae.pth -O models_cache/musetalk/weights/vae.pth"
    },
    "MuseTalk_Projector": {
        "url": "https://huggingface.co/TencentARC/MuseTalk/resolve/main/musetalk/audio_projector.pth",
        "path": "musetalk/weights/audio_projector.pth",
        "auto_download": False,
        "instructions": "wget -c https://huggingface.co/TencentARC/MuseTalk/resolve/main/musetalk/audio_projector.pth -O models_cache/musetalk/weights/audio_projector.pth"
    },
    "Wav2Lip": {
        "url": "https://iiitaphyd-my.sharepoint.com/:u:/g/personal/radhika_seth_research_iiit_ac_in/Eb3LEzbfuIlCgK5W4X_U9-UBXjFmZ_4jXyN7aZl-3c8AOA?e=O5T9mS",
        "path": "wav2lip/weights/wav2lip.pth",
        "auto_download": False,
        "instructions": "Download wav2lip.pth manually from official repo links and place it at models_cache/wav2lip/weights/wav2lip.pth"
    },
    "CodeFormer": {
        "url": "https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth",
        "path": "CodeFormer/weights/CodeFormer/codeformer.pth",
        "auto_download": False,
        "instructions": "wget -c https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth -O models_cache/CodeFormer/weights/CodeFormer/codeformer.pth"
    },
    "RIFE": {
        "url": "https://github.com/hzwer/arXiv2020-RIFE/releases/download/v4.6/flownet.pth",
        "path": "rife/weights/flownet.pth",
        "auto_download": False,
        "instructions": "wget -c https://github.com/hzwer/arXiv2020-RIFE/releases/download/v4.6/flownet.pth -O models_cache/rife/weights/flownet.pth"
    }
}


def download_resumable(url, out_path, expected_size=None, expected_hash=None):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    
    existing_size = os.path.getsize(out_path) if os.path.exists(out_path) else 0
    
    if expected_size and existing_size == expected_size:
        print(f"[{out_path}] Already fully downloaded (size matches).")
        return True
        
    headers = {}
    if existing_size > 0:
        headers["Range"] = f"bytes={existing_size}-"
        print(f"[{out_path}] Resuming download from {existing_size} bytes...")
    else:
        print(f"[{out_path}] Starting fresh download...")
        
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req) as response:
            mode = "ab" if existing_size > 0 else "wb"
            with open(out_path, mode) as f:
                while True:
                    chunk = response.read(8192)
                    if not chunk:
                        break
                    f.write(chunk)
    except urllib.error.HTTPError as e:
        if e.code == 416: # Range Not Satisfiable (already fully downloaded)
            print(f"[{out_path}] Download already complete according to server.")
        else:
            print(f"[{out_path}] HTTP Error {e.code}: {e.reason}")
            return False
    except Exception as e:
        print(f"[{out_path}] Download failed: {e}")
        return False
        
    # Validation
    final_size = os.path.getsize(out_path)
    if expected_size and final_size != expected_size:
        print(f"[{out_path}] Error: Size mismatch. Expected {expected_size}, got {final_size}")
        return False
        
    if expected_hash:
        sha256 = hashlib.sha256()
        with open(out_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256.update(chunk)
        if sha256.hexdigest() != expected_hash:
            print(f"[{out_path}] Error: Hash mismatch!")
            return False
            
    print(f"[{out_path}] Successfully downloaded.")
    return True

def main():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models_cache"))
    
    failed = False
    missing_manual = []
    
    for name, info in MODELS_DB.items():
        out_path = os.path.join(base_dir, info["path"])
        if info["auto_download"]:
            print(f"==> Processing {name}")
            success = download_resumable(info["url"], out_path, expected_size=info.get("size"), expected_hash=info.get("hash"))
            if not success:
                failed = True
        else:
            # Check if exists
            if not os.path.exists(out_path):
                missing_manual.append((name, info))
                
    if missing_manual:
        print("\n" + "="*60)
        print("ACTION REQUIRED: Missing manual download models")
        print("="*60)
        for name, info in missing_manual:
            print(f"\nModel: {name}")
            print(f"Missing File: {info['path']}")
            print(f"Source URL: {info['url']}")
            print(f"Download Command:\n  {info['instructions']}")
        failed = True
        
    if failed:
        print("\n[ERROR] Not all models are available. Please download missing checkpoints.")
        sys.exit(1)
    else:
        print("\n[SUCCESS] All checked models are present.")
        sys.exit(0)

if __name__ == "__main__":
    main()
