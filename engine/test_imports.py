import os
import sys

vendor_dir = os.path.abspath("vendor")

print("--- Testing CodeFormer ---")
cf_path = os.path.join(vendor_dir, "CodeFormer")
if cf_path not in sys.path:
    sys.path.insert(0, cf_path)
try:
    from basicsr.utils.download_util import load_file_from_url
    from facelib.utils.face_restoration_helper import FaceRestoreHelper
    from facelib.utils.misc import is_gray
    from basicsr.utils.registry import ARCH_REGISTRY
    print("CodeFormer imports: SUCCESS")
except Exception as e:
    print(f"CodeFormer imports: FAILED ({e})")
sys.path.remove(cf_path)

print("--- Testing RIFE ---")
rife_path = os.path.join(vendor_dir, "arXiv2020-RIFE")
if rife_path not in sys.path:
    sys.path.insert(0, rife_path)
try:
    from model.RIFE_HDv3 import Model
    print("RIFE imports: SUCCESS")
except Exception as e:
    print(f"RIFE imports: FAILED ({e})")
sys.path.remove(rife_path)

print("--- Testing SyncNet ---")
wav2lip_eval_path = os.path.join(vendor_dir, "Wav2Lip", "evaluation", "scores_LSE")
if wav2lip_eval_path not in sys.path:
    sys.path.insert(0, wav2lip_eval_path)
try:
    from SyncNetInstance_calc_scores import SyncNetInstance
    print("SyncNet imports: SUCCESS")
except Exception as e:
    print(f"SyncNet imports: FAILED ({e})")
sys.path.remove(wav2lip_eval_path)
