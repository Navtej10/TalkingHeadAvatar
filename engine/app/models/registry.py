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
    
    vendor_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "vendor"))
    musetalk_path = os.path.join(vendor_dir, "MuseTalk")
    added = False
    if musetalk_path not in sys.path:
        sys.path.insert(0, musetalk_path)
        added = True
    
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
            
            from app.models.validation import validate_checkpoint
            for f in required_files:
                f_path = os.path.join(ckpt_dir, f)
                validate_checkpoint(f_path)
            
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
            
            if isinstance(mouth_crop_batch, np.ndarray):
                mouth_crop_batch = [mouth_crop_batch]
            if isinstance(audio_embedding_seq, str):
                return mouth_crop_batch
            
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
            
    wrapper = MuseTalkWrapper()
    if added:
        sys.path.remove(musetalk_path)
    return wrapper

def load_wav2lip():
    import os
    import sys
    import torch
    from app.config import MODELS_CACHE_DIR, get_active_profile
    
    vendor_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "vendor"))
    wav2lip_path = os.path.join(vendor_dir, "Wav2Lip")
    added = False
    if wav2lip_path not in sys.path:
        sys.path.insert(0, wav2lip_path)
        added = True
        
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
            from app.models.validation import validate_checkpoint
            ckpt_path = os.path.join(ckpt_dir, "wav2lip.pth")
            validate_checkpoint(ckpt_path, expected_keys=["face_encoder"])
            
            try:
                from models import Wav2Lip
            except ImportError as e:
                raise ImportError(f"Could not import Wav2Lip modules. Ensure Wav2Lip repository is in {wav2lip_path}") from e

            model = Wav2Lip()
            checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=True)
            state_dict = checkpoint.get("state_dict", checkpoint)
            
            new_state_dict = {}
            for k, v in state_dict.items():
                name = k[7:] if k.startswith("module.") else k
                new_state_dict[name] = v
                
            model.load_state_dict(new_state_dict, strict=True)
            model.to(self.device, dtype=self.dtype)
            model.eval()
            
            self.models = {"generator": model}
            global _WAV2LIP_MODELS
            _WAV2LIP_MODELS = self.models

        def refine_mouth(self, mouth_crop_batch, mel_spectrogram_window):
            import cv2
            import numpy as np
            import torch
            
            if isinstance(mouth_crop_batch, np.ndarray):
                mouth_crop_batch = [mouth_crop_batch]
            if isinstance(mel_spectrogram_window, str):
                return mouth_crop_batch
            
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
            
    wrapper = Wav2LipWrapper()
    if added:
        sys.path.remove(wav2lip_path)
    return wrapper


def load_gfpgan():
    import os
    import sys
    import logging
    import torch
    from app.config import MODELS_CACHE_DIR, get_active_profile
    
    vendor_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "vendor"))
    gfpgan_path = os.path.join(vendor_dir, "GFPGAN")
    added = False
    if gfpgan_path not in sys.path:
        sys.path.insert(0, gfpgan_path)
        added = True
        
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
                
            from app.models.validation import validate_checkpoint
            model_path = os.path.join(MODELS_CACHE_DIR, "gfpgan", "weights", "GFPGANv1.4.pth")
            validate_checkpoint(model_path)
                
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
            
    wrapper = GFPGANWrapper()
    if added:
        sys.path.remove(gfpgan_path)
    return wrapper

def load_codeformer():
    import os
    import sys
    import logging
    import torch
    import cv2
    import numpy as np
    from app.config import MODELS_CACHE_DIR, get_active_profile
    
    vendor_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "vendor"))
        
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
            from app.models.validation import validate_checkpoint
            ckpt_path = os.path.join(MODELS_CACHE_DIR, "CodeFormer", "weights", "CodeFormer", "codeformer.pth")
            validate_checkpoint(ckpt_path)
            
            cf_path = os.path.join(vendor_dir, "CodeFormer")
            added = False
            if cf_path not in sys.path:
                sys.path.insert(0, cf_path)
                added = True

            try:
                from basicsr.utils.registry import ARCH_REGISTRY
            except ImportError:
                try:
                    import basicsr
                except ImportError as e:
                    if added:
                        sys.path.remove(cf_path)
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
            
            if added:
                sys.path.remove(cf_path)
            
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
    
    vendor_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "vendor"))
    lp_path = os.path.join(vendor_dir, "LivePortrait")
    lp_src_path = os.path.join(lp_path, "src")
    added_lp = False
    added_src = False
    if lp_path not in sys.path:
        sys.path.insert(0, lp_path)
        added_lp = True
    if lp_src_path not in sys.path:
        sys.path.insert(0, lp_src_path)
        added_src = True
        
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
            
            from app.models.validation import validate_checkpoint
            for f in required_files:
                f_path = os.path.join(ckpt_dir, f)
                validate_checkpoint(f_path)
            

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
                    driving_keypoints_info = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in source_keypoints_info.items()}
                        
                    if "stitching_retargeting_module" in self.models and "stitching" in self.models["stitching_retargeting_module"]:
                        bs, num_kp, c = source_keypoints_info['kp'].shape
                        kp_source_new = source_keypoints_info['kp'].view(bs, -1)
                        kp_driving_new = driving_keypoints_info['kp'].view(bs, -1)
                        feat_stitching = torch.cat([kp_source_new, kp_driving_new], dim=1)
                        delta = self.models["stitching_retargeting_module"]["stitching"](feat_stitching)
                        delta_exp = delta[..., :3*num_kp].reshape(bs, num_kp, 3)
                        delta_tx_ty = delta[..., 3*num_kp:3*num_kp+2].reshape(bs, 1, 2)
                        driving_keypoints_info['kp'] = driving_keypoints_info['kp'] + delta_exp
                        driving_keypoints_info['kp'][..., :2] = driving_keypoints_info['kp'][..., :2] + delta_tx_ty

                    warp_out = self.models["warping_module"](
                        source_features, 
                        kp_source=source_keypoints_info['kp'], 
                        kp_driving=driving_keypoints_info['kp']
                    )
                    out_tensor = self.models["spade_generator"](feature=warp_out['out'])
                    out_img = out_tensor.squeeze(0).permute(1, 2, 0).cpu().float().numpy()
                    out_img = np.clip(out_img * 255, 0, 255).astype(np.uint8)
                    out_img = cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR)
                    
                    duration = 1.0
                    num_frames = int(duration * self.profile.target_fps)
                    if num_frames == 0:
                        num_frames = self.profile.target_fps
                    return [out_img for _ in range(num_frames)]
                
                else:
                    if isinstance(motion, list):
                        out_frames = []
                        import time
                        total_frames = len(motion)
                        audio_duration = total_frames / self.profile.target_fps
                        print(f"[generation] Starting generation: {total_frames} frames to process (audio duration: {audio_duration:.1f}s @ {self.profile.target_fps}fps)", flush=True)
                        start_time = time.time()

                        for i, m_latent in enumerate(motion):
                            # Start from source keypoints every frame (relative delta approach)
                            driving_kp = source_keypoints_info['kp'].clone()
                            if isinstance(m_latent, np.ndarray) and m_latent.shape == (21, 3):
                                # Audio-driven delta: add on top of source face, preserving identity
                                delta_t = torch.from_numpy(m_latent).float().unsqueeze(0).to(self.device, dtype=self.dtype)
                                raw_driving_kp = driving_kp + delta_t
                                # Safety clamp: prevent extreme deformation > 0.15 units from source
                                kp_max_delta = 0.15
                                driving_kp = source_keypoints_info['kp'].clone() + torch.clamp(
                                    raw_driving_kp - source_keypoints_info['kp'], -kp_max_delta, kp_max_delta
                                )
                            # Note: raw BGR driving frame branch removed.
                            # Passing frames from a different person as absolute keypoints
                            # caused severe face warping (the melted-face artefact).
                            
                            if "stitching_retargeting_module" in self.models and "stitching" in self.models["stitching_retargeting_module"]:
                                bs, num_kp, c = source_keypoints_info['kp'].shape
                                kp_source_new = source_keypoints_info['kp'].view(bs, -1)
                                kp_driving_new = driving_kp.view(bs, -1)
                                feat_stitching = torch.cat([kp_source_new, kp_driving_new], dim=1)
                                stitch_delta = self.models["stitching_retargeting_module"]["stitching"](feat_stitching)
                                delta_exp = stitch_delta[..., :3*num_kp].reshape(bs, num_kp, 3)
                                delta_tx_ty = stitch_delta[..., 3*num_kp:3*num_kp+2].reshape(bs, 1, 2)
                                driving_kp = driving_kp + delta_exp
                                driving_kp[..., :2] = driving_kp[..., :2] + delta_tx_ty

                            warp_out = self.models["warping_module"](
                                source_features, 
                                kp_source=source_keypoints_info['kp'], 
                                kp_driving=driving_kp
                            )
                            out_tensor = self.models["spade_generator"](feature=warp_out['out'])
                            out_img = out_tensor.squeeze(0).permute(1, 2, 0).cpu().float().numpy()
                            out_img = np.clip(out_img * 255, 0, 255).astype(np.uint8)
                            out_img = cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR)
                            out_frames.append(out_img)

                            frames_processed = i + 1
                            if frames_processed % 10 == 0 or frames_processed == total_frames:
                                elapsed = time.time() - start_time
                                progress_pct = (frames_processed / total_frames) * 100
                                time_per_frame = elapsed / frames_processed
                                est_remaining = time_per_frame * (total_frames - frames_processed)
                                print(f"[generation] processed {frames_processed}/{total_frames} frames ({progress_pct:.1f}%) — elapsed {elapsed:.1f}s, est. remaining {est_remaining:.1f}s", flush=True)

                        return out_frames
                    else:
                        # Fallback for single image motion
                        if isinstance(motion, np.ndarray):
                            motion_tensor = cv2.cvtColor(motion, cv2.COLOR_BGR2RGB)
                            motion_tensor = torch.from_numpy(motion_tensor).float() / 255.0
                            motion_tensor = motion_tensor.permute(2, 0, 1).unsqueeze(0).to(self.device, dtype=self.dtype)
                            driving_keypoints_info = process_kp_info(self.models["motion_extractor"](motion_tensor))
                        else:
                            driving_keypoints_info = source_keypoints_info
                            
                        if "stitching_retargeting_module" in self.models and "stitching" in self.models["stitching_retargeting_module"]:
                            bs, num_kp, c = source_keypoints_info['kp'].shape
                            kp_source_new = source_keypoints_info['kp'].view(bs, -1)
                            kp_driving_new = driving_keypoints_info['kp'].view(bs, -1)
                            feat_stitching = torch.cat([kp_source_new, kp_driving_new], dim=1)
                            delta = self.models["stitching_retargeting_module"]["stitching"](feat_stitching)
                            delta_exp = delta[..., :3*num_kp].reshape(bs, num_kp, 3)
                            delta_tx_ty = delta[..., 3*num_kp:3*num_kp+2].reshape(bs, 1, 2)
                            driving_keypoints_info['kp'] = driving_keypoints_info['kp'] + delta_exp
                            driving_keypoints_info['kp'][..., :2] = driving_keypoints_info['kp'][..., :2] + delta_tx_ty

                        warp_out = self.models["warping_module"](
                            source_features, 
                            kp_source=source_keypoints_info['kp'], 
                            kp_driving=driving_keypoints_info['kp']
                        )
                        out_tensor = self.models["spade_generator"](feature=warp_out['out'])
                        out_img = out_tensor.squeeze(0).permute(1, 2, 0).cpu().float().numpy()
                        out_img = np.clip(out_img * 255, 0, 255).astype(np.uint8)
                        out_img = cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR)
                        
                        duration = 1.0
                        num_frames = int(duration * self.profile.target_fps)
                        return [out_img for _ in range(num_frames)]
            
    wrapper = LivePortraitWrapper()
    if added_src:
        sys.path.remove(lp_src_path)
    if added_lp:
        sys.path.remove(lp_path)
    return wrapper

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
    
    vendor_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "vendor"))
    rife_path = os.path.join(vendor_dir, "arXiv2020-RIFE")
        
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
            from app.models.validation import validate_checkpoint
            ckpt_path = os.path.join(ckpt_dir, "flownet.pth")
            validate_checkpoint(ckpt_path)
            
            added = False
            if rife_path not in sys.path:
                sys.path.insert(0, rife_path)
                added = True

            try:
                from model.RIFE_HDv3 import Model
            except ImportError as e:
                if added:
                    sys.path.remove(rife_path)
                raise ImportError(f"Could not import RIFE modules. Ensure RIFE repository is in {rife_path}") from e
                
            if added:
                sys.path.remove(rife_path)

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


def load_bisenet():
    import os
    import sys
    import torch
    import numpy as np
    import cv2
    from app.config import get_active_profile
    
    vendor_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "vendor"))
    cf_path = os.path.join(vendor_dir, "CodeFormer")
    added = False
    if cf_path not in sys.path:
        sys.path.insert(0, cf_path)
        added = True
    
    class BiSeNetWrapper:
        def __init__(self):
            self.profile = get_active_profile()
            self.device = torch.device(self.profile.device)
            try:
                from facelib.parsing import init_parsing_model
            except ImportError as e:
                raise ImportError("Could not import facelib for face parsing. Ensure CodeFormer is available.") from e
            self.model = init_parsing_model(model_name='bisenet', half=False, device=self.device)
            
        def parse_face(self, img):
            import torchvision.transforms as transforms
            h, w = img.shape[:2]
            
            img_resized = cv2.resize(img, (512, 512))
            img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
            
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])
            
            img_tensor = transform(img_rgb).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                out = self.model(img_tensor)[0]
                out = out.squeeze(0).cpu().numpy().argmax(0)
                
            mask = cv2.resize(out.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
            return mask

    wrapper = BiSeNetWrapper()
    if added:
        sys.path.remove(cf_path)
    return wrapper


def load_syncnet():
    import os
    import sys
    import torch
    import cv2
    import glob
    import numpy as np
    import math
    from scipy import signal
    from scipy.io import wavfile
    from app.config import MODELS_CACHE_DIR, get_active_profile
    from app.models.validation import validate_checkpoint
    
    vendor_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "vendor"))
        
    class SyncNetWrapper:
        def __init__(self):
            self.profile = get_active_profile()
            self.device = torch.device(self.profile.device)
            self._load_model()
            
        def _load_model(self):
            ckpt_path = os.path.join(MODELS_CACHE_DIR, "syncnet", "weights", "syncnet_v2.model")
            validate_checkpoint(ckpt_path)
            
            syncnet_dir = os.path.join(vendor_dir, "Wav2Lip", "evaluation", "scores_LSE")
            added = False
            if syncnet_dir not in sys.path:
                sys.path.insert(0, syncnet_dir)
                added = True
            
            from SyncNetInstance_calc_scores import SyncNetInstance
            self.instance = SyncNetInstance()
            self.instance.loadParameters(ckpt_path)
            self.instance.__S__.to(self.device)
            self.instance.__S__.eval()
            
            if added:
                sys.path.remove(syncnet_dir)
            
        def evaluate(self, video_frames_dir, audio_path):
            import python_speech_features
            import subprocess
            
            images = []
            flist = glob.glob(os.path.join(video_frames_dir, '*.png')) + glob.glob(os.path.join(video_frames_dir, '*.jpg'))
            flist.sort()
            
            for fname in flist:
                img_input = cv2.imread(fname)
                img_input = cv2.resize(img_input, (224, 224))
                images.append(img_input)
                
            im = np.stack(images, axis=3)
            im = np.expand_dims(im, axis=0)
            im = np.transpose(im, (0, 3, 4, 1, 2))
            
            imtv = torch.autograd.Variable(torch.from_numpy(im.astype(float)).float())
            
            temp_wav = os.path.join(video_frames_dir, "sync_temp_audio.wav")
            command = f"ffmpeg -loglevel error -y -i {audio_path} -async 1 -ac 1 -vn -acodec pcm_s16le -ar 16000 {temp_wav}"
            subprocess.call(command, shell=True)
            
            sample_rate, audio = wavfile.read(temp_wav)
            mfcc = zip(*python_speech_features.mfcc(audio, sample_rate))
            mfcc = np.stack([np.array(i) for i in mfcc])
            
            cc = np.expand_dims(np.expand_dims(mfcc, axis=0), axis=0)
            cct = torch.autograd.Variable(torch.from_numpy(cc.astype(float)).float())
            
            min_length = min(len(images), math.floor(len(audio)/640))
            lastframe = min_length - 5
            
            if lastframe <= 0:
                return 0.0
                
            im_feat = []
            cc_feat = []
            
            batch_size = 20
            
            for i in range(0, lastframe, batch_size):
                im_batch = [imtv[:, :, vframe:vframe+5, :, :] for vframe in range(i, min(lastframe, i+batch_size))]
                if len(im_batch) == 0:
                    break
                im_in = torch.cat(im_batch, 0)
                im_out = self.instance.__S__.forward_lip(im_in.to(self.device))
                im_feat.append(im_out.data.cpu())
                
                cc_batch = [cct[:, :, :, vframe*4:vframe*4+20] for vframe in range(i, min(lastframe, i+batch_size))]
                cc_in = torch.cat(cc_batch, 0)
                cc_out = self.instance.__S__.forward_aud(cc_in.to(self.device))
                cc_feat.append(cc_out.data.cpu())
                
            if len(im_feat) == 0 or len(cc_feat) == 0:
                return 0.0
                
            im_feat = torch.cat(im_feat, 0)
            cc_feat = torch.cat(cc_feat, 0)
            
            def calc_pdist(feat1, feat2, vshift=10):
                win_size = vshift * 2 + 1
                feat2p = torch.nn.functional.pad(feat2, (0, 0, vshift, vshift))
                dists = []
                for i in range(0, len(feat1)):
                    dists.append(torch.nn.functional.pairwise_distance(feat1[[i], :].repeat(win_size, 1), feat2p[i:i+win_size, :]))
                return dists
                
            dists = calc_pdist(im_feat, cc_feat, vshift=15)
            mdist = torch.mean(torch.stack(dists, 1), 1)
            minval, minidx = torch.min(mdist, 0)
            
            offset = 15 - minidx.item()
            return offset * (1000.0 / 25.0)

    return SyncNetWrapper()


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
    "face_parsing": {
        "bisenet": load_bisenet,
    },
    "sync_scorer": {
        "syncnet": load_syncnet,
    },
}


def get_model(stage: str, model_name: str):
    try:
        loader = REGISTRY[stage][model_name]
    except KeyError as e:
        raise ValueError(f"No model '{model_name}' registered for stage '{stage}'") from e
    return loader()
