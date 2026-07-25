"""
Stage 1 - Face Processing
Detect, align, crop, segment the source face; extract landmarks + pose.

Reference models: MediaPipe FaceMesh, InsightFace
"""
from app.pipeline.base import PipelineStage
from app.config import get_active_profile
import cv2
import numpy as np
from insightface.app import FaceAnalysis

class FaceProcessingStage(PipelineStage):
    name = "face_processing"

    def __init__(self):
        super().__init__()
        self.profile = get_active_profile()
        provider = 'CPUExecutionProvider' if self.profile.name == 'cpu_dev' else 'CUDAExecutionProvider'
        
        # Initialize detector once at class-init time
        self.detector = FaceAnalysis(name='buffalo_l', providers=[provider])
        self.detector.prepare(ctx_id=-1 if provider == 'CPUExecutionProvider' else 0, det_size=(320, 320))

    def run(self, context: dict) -> dict:
        image_path = context["image_path"]
        job_id = context.get("job_id", "unknown")

        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Job {job_id}: Could not read image at {image_path}")

        faces = self.detector.get(img)
        if len(faces) == 0:
            raise ValueError(f"Job {job_id}: No face detected in the input image.")

        # Use the first face found
        face = faces[0]
        
        # bounding box in original image coords
        crop_box = face.bbox.astype(int).tolist() 
        landmarks = face.kps.tolist() if face.kps is not None else None
        pose = face.pose.tolist() if face.pose is not None else None
        
        # crop and resize the face
        x1, y1, x2, y2 = crop_box
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(img.shape[1], x2), min(img.shape[0], y2)
        
        cropped = img[y1:y2, x1:x2]
        
        max_res = self.profile.max_resolution
        h, w = cropped.shape[:2]
        # resize maintaining aspect ratio and pad to square or just resize to square?
        # "square-cropped and resized to profile.max_resolution"
        aligned_face = cv2.resize(cropped, (max_res, max_res))
        
        mask = None  # Face parsing mask if cheap to get, else None

        context["face"] = {
            "aligned_face": aligned_face,
            "mask": mask,
            "landmarks": landmarks,
            "pose": pose,
            "crop_box": crop_box,
        }
        return context
