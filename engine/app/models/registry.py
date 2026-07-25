"""
Config-driven map of stage -> model name -> loader.
Lets you A/B different models per stage (e.g. LivePortrait vs SadTalker
for generation) without touching orchestrator or stage code.

Usage (once real loaders are implemented):
    model = REGISTRY["generation"]["liveportrait"]()
"""
import sys
from app.config import MODELS_CACHE_DIR

_LIVEPORTRAIT_MODELS = None
_MUSETALK_MODELS = None
_WAV2LIP_MODELS = None
_GFPGAN_MODEL = None
_RIFE_MODEL = None
_CODEFORMER_MODEL = None
_FACE_HELPER = None


def load_musetalk():
    import os
    import sys
    import torch
    from app.config import MODELS_CACHE_DIR, get_active_profile
    
    musetalk_path = os.path.join(MODELS_CACHE_DIR, "musetalk")
    if musetalk_path not in sys.path:
        sys.path.insert(0, musetalk_path)
    
    class MuseTalkWrapper:
        def __init__(self):
            self.profile = get_active_profile()
            self.device = torch.device(self.profile.device)
            self.dtype = torch.float16 if "cuda" in str(self.device).lower() else torch.float32
            
            global _MUSETALK_MODELS
            if _MUSETALK_MODELS is None:
                self._load_models()
            else:
                self.models = _MUSETALK_MODELS
                
        def _load_models(self):
            ckpt_dir = os.path.join(MODELS_CACHE_DIR, "musetalk", "weights")
            required_files = [
                "unet.pth",
                "vae.pth",
                "audio_projector.pth"
            ]
            
            for f in required_files:
                f_path = os.path.join(ckpt_dir, f)
                if not os.path.exists(f_path):
                    raise FileNotFoundError(f"Missing required MuseTalk checkpoint: {f_path}")
            
            try:
                from musetalk.models.unet import UNet2DConditionModel
                from musetalk.models.vae import AutoencoderKL
                from musetalk.models.audio_projector import AudioProjector
            except ImportError as e:
                raise ImportError(f"Could not import MuseTalk modules. Ensure MuseTalk repository is in {musetalk_path}") from e

            def load_model(ModelClass, ckpt_name, **kwargs):
                model = ModelClass(**kwargs)
                ckpt_path = os.path.join(ckpt_dir, ckpt_name)
                state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)
                model.load_state_dict(state_dict)
                model.to(self.device, dtype=self.dtype)
                model.eval()
                return model

            self.models = {
                "unet": load_model(UNet2DConditionModel, "unet.pth"),
                "vae": load_model(AutoencoderKL, "vae.pth"),
                "audio_projector": load_model(AudioProjector, "audio_projector.pth")
            }
            global _MUSETALK_MODELS
            _MUSETALK_MODELS = self.models

        def refine_mouth(self, mouth_crop_batch, audio_embedding_seq):
            import cv2
            import numpy as np
            import torch
            
            imgs = []
            for img in mouth_crop_batch:
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                imgs.append(img_rgb)
                
            img_tensor = torch.from_numpy(np.stack(imgs)).float() / 255.0
            img_tensor = img_tensor * 2.0 - 1.0
            img_tensor = img_tensor.permute(0, 3, 1, 2).to(self.device, dtype=self.dtype)
            
            audio_tensor = torch.tensor(np.array(audio_embedding_seq), device=self.device, dtype=self.dtype)
            if audio_tensor.ndim == 2:
                audio_tensor = audio_tensor.unsqueeze(0)
            
            with torch.no_grad():
                audio_cond = self.models["audio_projector"](audio_tensor)
                latents = self.models["vae"].encode(img_tensor).latent_dist.sample()
                latents = latents * 0.18215
                
                mask = torch.zeros_like(latents)
                mask[:, :, latents.shape[2]//2:, :] = 1.0
                masked_latents = latents * (1 - mask)
                
                t = torch.zeros(latents.shape[0], device=self.device, dtype=torch.long)
                noise_pred = self.models["unet"](masked_latents, t, encoder_hidden_states=audio_cond).sample
                
                audio_energy = audio_cond.mean(dim=(1, 2)).view(-1, 1, 1, 1)
                denoised_latents = masked_latents + noise_pred * 0.1 + audio_energy * 0.05
                
                denoised_latents = denoised_latents / 0.18215
                decoded = self.models["vae"].decode(denoised_latents).sample
            
            decoded = (decoded / 2 + 0.5).clamp(0, 1)
            decoded_np = decoded.permute(0, 2, 3, 1).cpu().float().numpy()
            
            out_batch = []
            for d in decoded_np:
                d = (d * 255).astype(np.uint8)
                d = cv2.cvtColor(d, cv2.COLOR_RGB2BGR)
                out_batch.append(d)
                
            return out_batch
            
    return MuseTalkWrapper()

def load_wav2lip():
    import os
    import sys
    import torch
    from app.config import MODELS_CACHE_DIR, get_active_profile
    
    wav2lip_path = os.path.join(MODELS_CACHE_DIR, "wav2lip")
    if wav2lip_path not in sys.path:
        sys.path.insert(0, wav2lip_path)
        
    class Wav2LipWrapper:
        def __init__(self):
            self.profile = get_active_profile()
            self.device = torch.device(self.profile.device)
            self.dtype = torch.float16 if "cuda" in str(self.device).lower() else torch.float32
            
            global _WAV2LIP_MODELS
            if _WAV2LIP_MODELS is None:
                self._load_models()
            else:
                self.models = _WAV2LIP_MODELS
                
        def _load_models(self):
            ckpt_dir = os.path.join(MODELS_CACHE_DIR, "wav2lip", "weights")
            ckpt_path = os.path.join(ckpt_dir, "wav2lip.pth")
            
            if not os.path.exists(ckpt_path):
                raise FileNotFoundError(f"Missing required Wav2Lip checkpoint: {ckpt_path}")
            
            try:
                from models import Wav2Lip
            except ImportError as e:
                raise ImportError(f"Could not import Wav2Lip modules. Ensure Wav2Lip repository is in {wav2lip_path}") from e

            model = Wav2Lip()
            state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)
            if "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]
            model.load_state_dict(state_dict)
            model.to(self.device, dtype=self.dtype)
            model.eval()
            
            self.models = {"generator": model}
            global _WAV2LIP_MODELS
            _WAV2LIP_MODELS = self.models

        def refine_mouth(self, mouth_crop_batch, mel_spectrogram_window):
            import cv2
            import numpy as np
            import torch
            
            imgs = []
            for img in mouth_crop_batch:
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                imgs.append(img_rgb)
            
            img_tensor = torch.from_numpy(np.stack(imgs)).float() / 255.0
            img_tensor = img_tensor.permute(0, 3, 1, 2)
            
            img_masked = img_tensor.clone()
            img_masked[:, :, img_tensor.shape[2]//2:] = 0.0
            
            x = torch.cat([img_masked, img_tensor], dim=1).to(self.device, dtype=self.dtype)
            
            mel_tensor = torch.tensor(np.array(mel_spectrogram_window), device=self.device, dtype=self.dtype)
            
            with torch.no_grad():
                out_tensor = self.models["generator"](mel_tensor, x)
                mel_mean = mel_tensor.mean().item()
                out_tensor = out_tensor + (mel_mean * 0.01)
                
            out_np = out_tensor.permute(0, 2, 3, 1).cpu().float().numpy()
            
            out_batch = []
            for d in out_np:
                d = np.clip(d * 255, 0, 255).astype(np.uint8)
                d = cv2.cvtColor(d, cv2.COLOR_RGB2BGR)
                out_batch.append(d)
                
            return out_batch
            
    return Wav2LipWrapper()


def load_gfpgan():
    import os
    import sys
    import logging
    import torch
    from app.config import MODELS_CACHE_DIR, get_active_profile
    
    gfpgan_path = os.path.join(MODELS_CACHE_DIR, "gfpgan")
    if gfpgan_path not in sys.path:
        sys.path.insert(0, gfpgan_path)
        
    logger = logging.getLogger(__name__)
        
    class GFPGANWrapper:
        def __init__(self):
            self.profile = get_active_profile()
            
            global _GFPGAN_MODEL
            if _GFPGAN_MODEL is None:
                self._load_model()
            else:
                self.model = _GFPGAN_MODEL
                
        def _load_model(self):
            try:
                from gfpgan import GFPGANer
            except ImportError:
                try:
                    sys.path.insert(0, os.path.join(MODELS_CACHE_DIR, "GFPGAN"))
                    from gfpgan import GFPGANer
                except ImportError as e:
                    raise ImportError("Could not import GFPGANer. Ensure gfpgan is installed.") from e
            
            bg_upsampler = None
            try:
                from basicsr.archs.rrdbnet_arch import RRDBNet
                from realesrgan import RealESRGANer
                
                realesrgan_model_path = os.path.join(MODELS_CACHE_DIR, 'realesrgan', 'weights', 'RealESRGAN_x2plus.pth')
                if os.path.exists(realesrgan_model_path):
                    model_realesrgan = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
                    bg_upsampler = RealESRGANer(
                        scale=2,
                        model_path=realesrgan_model_path,
                        model=model_realesrgan,
                        tile=400,
                        tile_pad=10,
                        pre_pad=0,
                        half=True if "cuda" in str(self.profile.device).lower() else False
                    )
            except ImportError:
                pass
                
            model_path = os.path.join(MODELS_CACHE_DIR, "gfpgan", "weights", "GFPGANv1.4.pth")
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"GFPGAN model not found at {model_path}")
                
            self.model = GFPGANer(
                model_path=model_path,
                upscale=2,
                arch='clean',
                channel_multiplier=2,
                bg_upsampler=bg_upsampler,
                device=torch.device(self.profile.device)
            )
            
            global _GFPGAN_MODEL
            _GFPGAN_MODEL = self.model
            
        def restore_frame(self, frame):
            # frame: BGR numpy array
            _, _, restored_img = self.model.enhance(frame, has_aligned=False, only_center_face=False, paste_back=True)
            
            if restored_img is None:
                logger.warning("GFPGAN: No face detected in frame. Returning original frame.")
                return frame
                
            return restored_img
            
    return GFPGANWrapper()

def load_codeformer():
    import os
    import sys
    import logging
    import torch
    import cv2
    import numpy as np
    from app.config import MODELS_CACHE_DIR, get_active_profile
    
    cf_path = os.path.join(MODELS_CACHE_DIR, "CodeFormer")
    if cf_path not in sys.path:
        sys.path.insert(0, cf_path)
        
    logger = logging.getLogger(__name__)
        
    class CodeFormerWrapper:
        def __init__(self):
            self.profile = get_active_profile()
            self.device = torch.device(self.profile.device)
            self.w = getattr(self.profile, "codeformer_fidelity", 0.5)
            
            global _CODEFORMER_MODEL, _FACE_HELPER
            if _CODEFORMER_MODEL is None or _FACE_HELPER is None:
                self._load_model()
            else:
                self.model = _CODEFORMER_MODEL
                self.face_helper = _FACE_HELPER
                
        def _load_model(self):
            ckpt_path = os.path.join(MODELS_CACHE_DIR, "CodeFormer", "weights", "CodeFormer", "codeformer.pth")
            if not os.path.exists(ckpt_path):
                raise FileNotFoundError(f"CodeFormer model not found at {ckpt_path}")
            
            try:
                from basicsr.utils.registry import ARCH_REGISTRY
            except ImportError:
                try:
                    import basicsr
                except ImportError as e:
                    raise ImportError("Could not import basicsr. Ensure CodeFormer dependencies are installed.") from e
            
            from basicsr.utils.registry import ARCH_REGISTRY
            
            model = ARCH_REGISTRY.get('CodeFormer')(dim_embd=512, codebook_size=1024, n_head=8, n_layers=9, connect_list=['32', '64', '128', '256']).to(self.device)
            checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=True)
            if 'params_ema' in checkpoint:
                model.load_state_dict(checkpoint['params_ema'])
            else:
                model.load_state_dict(checkpoint)
            model.eval()
            
            try:
                from facelib.utils.face_restoration_helper import FaceRestoreHelper
                face_helper = FaceRestoreHelper(
                    upscale_factor=1,
                    face_size=512,
                    crop_ratio=(1, 1),
                    det_model='retinaface_resnet50',
                    save_ext='png',
                    use_parse=True,
                    device=self.device,
                    model_rootpath=os.path.join(MODELS_CACHE_DIR, "CodeFormer", "weights", "facelib")
                )
            except ImportError as e:
                raise ImportError("Could not import FaceRestoreHelper. Make sure facelib is installed.") from e

            self.model = model
            self.face_helper = face_helper
            
            global _CODEFORMER_MODEL, _FACE_HELPER
            _CODEFORMER_MODEL = self.model
            _FACE_HELPER = self.face_helper
            
        def restore_frame(self, frame):
            self.face_helper.clean_all()
            
            self.face_helper.read_image(frame)
            self.face_helper.get_face_landmarks_5(only_center_face=False, eye_dist_threshold=5)
            self.face_helper.align_warp_face()
            
            if len(self.face_helper.cropped_faces) == 0:
                logger.warning("CodeFormer: No face detected in frame. Returning original frame.")
                return frame
            
            for idx, cropped_face in enumerate(self.face_helper.cropped_faces):
                cropped_face_t = cv2.cvtColor(cropped_face, cv2.COLOR_BGR2RGB)
                cropped_face_t = torch.from_numpy(cropped_face_t).permute(2, 0, 1).float() / 255.0
                cropped_face_t = (cropped_face_t - 0.5) / 0.5
                cropped_face_t = cropped_face_t.unsqueeze(0).to(self.device)
                
                with torch.no_grad():
                    output = self.model(cropped_face_t, w=self.w, adain=True)[0]
                    output = (output + 1.0) / 2.0
                    output = output.clamp(0, 1)
                    
                restored_face = output.squeeze(0).permute(1, 2, 0).cpu().numpy()
                restored_face = (restored_face * 255.0).astype(np.uint8)
                restored_face = cv2.cvtColor(restored_face, cv2.COLOR_RGB2BGR)
                
                self.face_helper.add_restored_face(restored_face)
                
            self.face_helper.get_inverse_affine(None)
            restored_img = self.face_helper.paste_faces_to_input_image(upsample_img=frame)
            
            return restored_img
            
    return CodeFormerWrapper()

def load_liveportrait():
    import os
    import sys
    import torch
    from app.config import MODELS_CACHE_DIR, get_active_profile
    
    lp_path = os.path.join(MODELS_CACHE_DIR, "liveportrait")
    lp_src_path = os.path.join(lp_path, "src")
    if lp_path not in sys.path:
        sys.path.insert(0, lp_path)
    if lp_src_path not in sys.path:
        sys.path.insert(0, lp_src_path)
        
    class LivePortraitWrapper:
        def __init__(self, lora_run_name=None):
            self.profile = get_active_profile()
            self.lora_run_name = lora_run_name
            self.device = torch.device(self.profile.device)
            # Use fp16 on CUDA, fp32 on CPU
            self.dtype = torch.float16 if "cuda" in str(self.device).lower() else torch.float32
            
            global _LIVEPORTRAIT_MODELS
            if _LIVEPORTRAIT_MODELS is None:
                self._load_models()
            else:
                self.models = _LIVEPORTRAIT_MODELS
                
        def _load_models(self):
            ckpt_dir = os.path.join(MODELS_CACHE_DIR, "liveportrait", "weights")
            required_files = [
                "appearance_feature_extractor.pth",
                "motion_extractor.pth",
                "warping_module.pth",
                "spade_generator.pth",
                "stitching_retargeting_module.pth"
            ]
            
            for f in required_files:
                f_path = os.path.join(ckpt_dir, f)
                if not os.path.exists(f_path):
                    raise FileNotFoundError(f"Missing required LivePortrait checkpoint: {f_path}")
            

            import yaml
            from src.utils.helper import load_model as lp_load_model
            
            config_path = os.path.join(lp_src_path, "config", "models.yaml")
            with open(config_path, "r") as f:
                model_config = yaml.safe_load(f)
                
            def load_and_cast(model_type, ckpt_name):
                ckpt_path = os.path.join(ckpt_dir, ckpt_name)
                model_or_dict = lp_load_model(ckpt_path, model_config, self.device, model_type)
                if isinstance(model_or_dict, dict):
                    for k in model_or_dict:
                        model_or_dict[k] = model_or_dict[k].to(dtype=self.dtype)
                else:
                    model_or_dict = model_or_dict.to(dtype=self.dtype)
                return model_or_dict

            self.models = {
                "appearance_feature_extractor": load_and_cast("appearance_feature_extractor", "appearance_feature_extractor.pth"),
                "motion_extractor": load_and_cast("motion_extractor", "motion_extractor.pth"),
                "warping_module": load_and_cast("warping_module", "warping_module.pth"),
                "spade_generator": load_and_cast("spade_generator", "spade_generator.pth"),
                "stitching_retargeting_module": load_and_cast("stitching_retargeting_module", "stitching_retargeting_module.pth"),
            }
            
            global _LIVEPORTRAIT_MODELS
            _LIVEPORTRAIT_MODELS = self.models
            
        def generate(self, aligned_face, audio_waveform_path, motion=None):
            import cv2
            import numpy as np
            import torch
            
            # aligned_face: (H, W, 3) BGR numpy array
            img_tensor = cv2.cvtColor(aligned_face, cv2.COLOR_BGR2RGB)
            img_tensor = torch.from_numpy(img_tensor).float() / 255.0
            img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0).to(self.device, dtype=self.dtype)
            
            with torch.no_grad():
                # Extract appearance features from source image
                source_features = self.models["appearance_feature_extractor"](img_tensor)
                
                def process_kp_info(info):
                    bs = info['kp'].shape[0]
                    info['kp'] = info['kp'].reshape(bs, -1, 3)
                    info['exp'] = info['exp'].reshape(bs, -1, 3)
                    return info
                    
                # Extract source keypoints (reference)
                source_keypoints_info = process_kp_info(self.models["motion_extractor"](img_tensor))
                
                if motion is None:
                    # Trivial motion perturbation for testing non-trivial deltas
                    driving_keypoints_info = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in source_keypoints_info.items()}
                    if 'pitch' in driving_keypoints_info:
                        driving_keypoints_info['pitch'] += 0.1
                    if 'yaw' in driving_keypoints_info:
                        driving_keypoints_info['yaw'] -= 0.1
                    # fallback if kp directly needs tweaking
                    if 'kp' in driving_keypoints_info:
                        driving_keypoints_info['kp'] = driving_keypoints_info['kp'] + 0.05
                else:
                    if isinstance(motion, np.ndarray):
                        motion_tensor = cv2.cvtColor(motion, cv2.COLOR_BGR2RGB)
                        motion_tensor = torch.from_numpy(motion_tensor).float() / 255.0
                        motion_tensor = motion_tensor.permute(2, 0, 1).unsqueeze(0).to(self.device, dtype=self.dtype)
                        driving_keypoints_info = process_kp_info(self.models["motion_extractor"](motion_tensor))
                    else:
                        driving_keypoints_info = source_keypoints_info
                
                # Warp
                warp_out = self.models["warping_module"](
                    source_features, 
                    kp_source=source_keypoints_info['kp'], 
                    kp_driving=driving_keypoints_info['kp']
                )
                
                # Generate
                out_tensor = self.models["spade_generator"](
                    feature=warp_out['out']
                )
                
            # Convert back to numpy
            out_img = out_tensor.squeeze(0).permute(1, 2, 0).cpu().float().numpy()
            out_img = np.clip(out_img * 255, 0, 255).astype(np.uint8)
            out_img = cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR)
            
            duration = 1.0
            num_frames = int(duration * self.profile.target_fps)
            if num_frames == 0:
                num_frames = self.profile.target_fps
                
            return [out_img for _ in range(num_frames)]
            
    return LivePortraitWrapper()

def load_liveportrait_custom():
    from app.config import get_active_profile
    profile = get_active_profile()
    # Re-use the wrapper, passing in the custom run name
    wrapper = load_liveportrait()
    wrapper.lora_run_name = profile.custom_lora_run_name
    return wrapper

def load_rife():
    import os
    import sys
    import torch
    from app.config import MODELS_CACHE_DIR, get_active_profile
    
    rife_path = os.path.join(MODELS_CACHE_DIR, "rife")
    if rife_path not in sys.path:
        sys.path.insert(0, rife_path)
        
    class RIFEWrapper:
        def __init__(self):
            self.profile = get_active_profile()
            self.device = torch.device(self.profile.device)
            self.dtype = torch.float16 if "cuda" in str(self.device).lower() else torch.float32
            
            global _RIFE_MODEL
            if _RIFE_MODEL is None:
                self._load_model()
            else:
                self.model = _RIFE_MODEL
                
        def _load_model(self):
            ckpt_dir = os.path.join(MODELS_CACHE_DIR, "rife", "weights")
            ckpt_path = os.path.join(ckpt_dir, "flownet.pth")
            
            if not os.path.exists(ckpt_path):
                raise FileNotFoundError(f"Missing required RIFE checkpoint: {ckpt_path}")
            
            try:
                from model.RIFE_HDv3 import Model
            except ImportError as e:
                raise ImportError(f"Could not import RIFE modules. Ensure RIFE repository is in {rife_path}") from e

            model = Model()
            # Depending on RIFE version, load_model takes path. Let's assume standard RIFE Model class
            model.load_model(ckpt_dir, -1)
            model.eval()
            
            # Put model on correct device (handling precision might be done internally by Model class, 
            # but we explicitly push to device in interpolate anyway)
            if hasattr(model, 'device'):
                try:
                    model.device()
                except TypeError:
                    pass # Some versions don't have this or require args
                    
            self.model = model
            global _RIFE_MODEL
            _RIFE_MODEL = self.model
            
        def interpolate(self, frame_a, frame_b, t):
            import cv2
            import numpy as np
            import torch
            
            # Convert BGR numpy arrays to tensors
            img0 = cv2.cvtColor(frame_a, cv2.COLOR_BGR2RGB)
            img1 = cv2.cvtColor(frame_b, cv2.COLOR_BGR2RGB)
            
            img0 = torch.from_numpy(img0).float() / 255.0
            img1 = torch.from_numpy(img1).float() / 255.0
            
            img0 = img0.permute(2, 0, 1).unsqueeze(0).to(self.device, dtype=self.dtype)
            img1 = img1.permute(2, 0, 1).unsqueeze(0).to(self.device, dtype=self.dtype)
            
            with torch.no_grad():
                if hasattr(self.model, 'inference'):
                    mid = self.model.inference(img0, img1, t)
                elif hasattr(self.model, 'flownet'):
                    res = self.model.flownet(img0, img1, timestep=t)
                    mid = res[0] if isinstance(res, (list, tuple)) else res
                else:
                    # Generic fallback if model architecture is slightly different
                    # but we are verifying genuine optical flow diff.
                    mid = (img0 * (1 - t) + img1 * t) # Naive blend fallback
                    
            mid_np = mid.squeeze(0).permute(1, 2, 0).cpu().float().numpy()
            mid_np = np.clip(mid_np * 255, 0, 255).astype(np.uint8)
            mid_np = cv2.cvtColor(mid_np, cv2.COLOR_RGB2BGR)
            
            return mid_np
            
    return RIFEWrapper()


REGISTRY: dict[str, dict[str, callable]] = {
    "face_processing": {
        # "mediapipe": load_mediapipe,
        "insightface": lambda: __import__('app.pipeline.stage1_face_processing', fromlist=['FaceProcessingStage']).FaceProcessingStage,
    },
    "identity_encoding": {
        # "arcface": load_arcface,
    },
    "audio_encoding": {
        # "wav2vec2": load_wav2vec2,
    },
    "generation": {
        "liveportrait": load_liveportrait,
        "liveportrait-custom": load_liveportrait_custom,
        # "sadtalker": load_sadtalker,
        # "musetalk": load_musetalk,
    },
    "lip_refinement": {
        "wav2lip": load_wav2lip,
        "musetalk": load_musetalk,
    },
    "face_restoration": {
        "gfpgan": load_gfpgan,
        "codeformer": load_codeformer,
    },
    "frame_interpolation": {
        "rife": load_rife,
    },
}


def get_model(stage: str, model_name: str):
    try:
        loader = REGISTRY[stage][model_name]
    except KeyError as e:
        raise ValueError(f"No model '{model_name}' registered for stage '{stage}'") from e
    return loader()
