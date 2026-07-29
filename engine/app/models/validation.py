import os
import torch
import logging

logger = logging.getLogger(__name__)

class ModelValidationError(Exception):
    pass

def validate_checkpoint(path: str, expected_keys: list = None, loader: callable = None):
    """
    Attempts to load a checkpoint to verify it's not corrupt, empty, or a dummy file.
    If expected_keys is provided, verifies those keys exist in the loaded object/state_dict.
    """
    if not os.path.exists(path):
        raise ModelValidationError(f"Missing file: {path}")
        
    try:
        if loader:
            obj = loader(path)
        else:
            obj = torch.load(path, map_location='cpu', weights_only=True)
            
        if expected_keys:
            # If it's a dict or state_dict
            keys_to_check = obj.keys() if isinstance(obj, dict) else []
            if hasattr(obj, 'state_dict'):
                keys_to_check = obj.state_dict().keys()
            elif 'state_dict' in obj:
                keys_to_check = obj['state_dict'].keys()
                
            missing_keys = [k for k in expected_keys if k not in keys_to_check and not any(str(k) in str(tk) for tk in keys_to_check)]
            if missing_keys:
                raise ModelValidationError(f"Checkpoint at {path} is missing expected keys: {missing_keys}")
                
    except Exception as e:
        if isinstance(e, ModelValidationError):
            raise
        raise ModelValidationError(f"Failed to load checkpoint at {path}. File may be corrupt or invalid. Error: {e}")

def validate_all_checkpoints(stages_to_run=None):
    """
    Validates all required checkpoints for the active profile at startup.
    Raises a single aggregated exception if any are missing or invalid.
    """
    from app.config import get_active_profile, MODELS_CACHE_DIR
    profile = get_active_profile()
    
    stage_names = [s.name for s in stages_to_run] if stages_to_run is not None else None
    
    errors = []
    
    def check(rel_path, keys=None):
        full_path = os.path.join(MODELS_CACHE_DIR, rel_path)
        try:
            validate_checkpoint(full_path, expected_keys=keys)
        except ModelValidationError as e:
            errors.append(str(e))

    # We validate based on the stages we will run.
    if stage_names is None or "face_processing" in stage_names or "identity_encoding" in stage_names:
        # Insightface handles its own load errors pretty well, but we can check if Buffalo_L exists
        insight_dir = os.path.expanduser("~/.insightface/models/buffalo_l")
        if not os.path.exists(insight_dir):
            pass
        
    if stage_names is None or "generation" in stage_names:
        # LivePortrait (Generation)
        check("liveportrait/weights/appearance_feature_extractor.pth")
        check("liveportrait/weights/motion_extractor.pth")
        check("liveportrait/weights/warping_module.pth")
        check("liveportrait/weights/spade_generator.pth")
        check("liveportrait/weights/stitching_retargeting_module.pth")
    
    if stage_names is None or "lip_refinement" in stage_names:
        # Lip Refinement
        if profile.enable_lip_refinement:
            if profile.name == "cpu_dev":
                # Wav2Lip
                check("wav2lip/weights/wav2lip.pth", keys=["face_encoder"])
            else:
                # MuseTalk
                check("musetalk/weights/unet.pth")
                check("musetalk/weights/vae.pth")
                check("musetalk/weights/audio_projector.pth")
                
            # SyncNet (always needed for evaluation)
            check("syncnet/weights/syncnet_v2.model", keys=["netcnn"])
            
    if stage_names is None or "face_restoration" in stage_names:
        # Face Restoration
        if profile.enable_restoration:
            check("CodeFormer/weights/CodeFormer/codeformer.pth")
        
    if stage_names is None or "frame_interpolation" in stage_names:
        # Frame Interpolation
        if profile.enable_interpolation:
            check("rife/weights/flownet.pth")

    if errors:
        error_msg = "Cannot start — missing/invalid checkpoints:\n"
        for err in errors:
            error_msg += f"  - {err}\n"
        error_msg += "See download_models.py for manual download instructions."
        raise RuntimeError(error_msg)
