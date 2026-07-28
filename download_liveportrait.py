import os
import urllib.request

files_to_download = {
    "appearance_feature_extractor.pth": "https://huggingface.co/KlingTeam/LivePortrait/resolve/main/liveportrait/base_models/appearance_feature_extractor.pth",
    "motion_extractor.pth": "https://huggingface.co/KlingTeam/LivePortrait/resolve/main/liveportrait/base_models/motion_extractor.pth",
    "warping_module.pth": "https://huggingface.co/KlingTeam/LivePortrait/resolve/main/liveportrait/base_models/warping_module.pth",
    "spade_generator.pth": "https://huggingface.co/KlingTeam/LivePortrait/resolve/main/liveportrait/base_models/spade_generator.pth",
    "stitching_retargeting_module.pth": "https://huggingface.co/KlingTeam/LivePortrait/resolve/main/liveportrait/retargeting_models/stitching_retargeting_module.pth",
}

base_dir = r"d:\Navtej\TalkingHeadAvatar\models_cache\liveportrait\weights"
os.makedirs(base_dir, exist_ok=True)

for filename, url in files_to_download.items():
    filepath = os.path.join(base_dir, filename)
    print(f"Downloading {filename} from HF mirror (KlingTeam)...")
    try:
        urllib.request.urlretrieve(url, filepath)
        size = os.path.getsize(filepath)
        print(f"Downloaded {filename}, size: {size} bytes")
    except Exception as e:
        print(f"Failed to download {filename}: {e}")
