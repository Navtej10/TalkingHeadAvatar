"""
Stage 2 - Identity Encoding
Build a compact identity embedding (geometry, proportions, hair, etc.)
instead of re-using raw pixels for every downstream stage. This is what
prevents identity drift across frames.

Reference model: InsightFace ArcFace embeddings
"""
from app.pipeline.base import PipelineStage
from app.config import get_active_profile
import numpy as np
import logging

logger = logging.getLogger(__name__)

class IdentityExtractionError(Exception):
    pass


class IdentityEncodingStage(PipelineStage):
    name = "identity_encoding"

    def __init__(self):
        super().__init__()
        self.profile = get_active_profile()
        self.rec_model = None

    def _init_detector(self):
        if self.rec_model is None:
            provider = 'CPUExecutionProvider' if self.profile.name == 'cpu_dev' else 'CUDAExecutionProvider'
            try:
                from insightface.app import FaceAnalysis
                app = FaceAnalysis(name='buffalo_l', providers=[provider])
                app.prepare(ctx_id=-1 if provider == 'CPUExecutionProvider' else 0, det_size=(640, 640))
                self.rec_model = app.models.get('recognition', None)
                if self.rec_model is None:
                    raise IdentityExtractionError("InsightFace loaded, but 'recognition' model is missing.")
            except Exception as e:
                logger.error(f"Failed to initialize InsightFace ArcFace model: {e}")
                raise IdentityExtractionError(f"Startup validation failed for IdentityEncodingStage: {e}")

    def run(self, context: dict) -> dict:
        if "identity" in context and "embedding" in context["identity"]:
            # Already loaded from identity store, skip
            return context

        job_id = context.get("job_id", "unknown")
        image_path = context.get("image_path")
        
        # Check cache if image_path is available
        from app.core.cache import get_file_hash, get_cache, set_cache
        cache_key = None
        if image_path:
            try:
                cache_key = f"identity_{get_file_hash(image_path)}"
                cached_embedding = get_cache(cache_key)
                if cached_embedding is not None:
                    if "identity" not in context:
                        context["identity"] = {}
                    context["identity"]["embedding"] = cached_embedding
                    return context
            except Exception as e:
                print(f"Warning: Failed to read image for cache hash: {e}")

        face_info = context.get("face")
        if not face_info or "aligned_face" not in face_info:
            raise ValueError(f"Job {job_id}: No aligned face found for identity encoding.")

        aligned_face = face_info["aligned_face"]

        # Compute embedding
        self._init_detector()
        if self.rec_model:
            try:
                embedding = self.rec_model.get_feat(aligned_face).flatten()
                if embedding is None or len(embedding) == 0:
                    raise ValueError("Empty feature returned from get_feat.")
            except Exception as e:
                logger.error(f"Runtime extraction failed for aligned face: {e}")
                raise IdentityExtractionError("Could not extract identity features from the aligned face.") from e
        else:
            raise IdentityExtractionError("Recognition model is not loaded. Cannot extract identity features.")

        if "identity" not in context:
            context["identity"] = {}
            
        context["identity"]["embedding"] = embedding
        
        if cache_key:
            set_cache(cache_key, embedding)
            
        return context
