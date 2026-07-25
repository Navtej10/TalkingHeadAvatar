import os
import cv2
import uuid
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from app.core.identity_store import save_identity, list_identities
from app.pipeline.stage1_face_processing import FaceProcessingStage
from app.pipeline.stage2_identity_encoding import IdentityEncodingStage
from app.config import TMP_DIR

router = APIRouter()

@router.post("")
def create_identity(name: str = Form(...), image: UploadFile = File(...)):
    """
    Uploads an image, processes the face, encodes the identity, 
    and saves it to the local store for future reuse.
    """
    job_id = f"ident-{uuid.uuid4().hex[:8]}"
    os.makedirs(f"{TMP_DIR}/{job_id}", exist_ok=True)
    
    image_path = f"{TMP_DIR}/{job_id}/source_img.jpg"
    with open(image_path, "wb") as f:
        f.write(image.file.read())
        
    context = {
        "job_id": job_id,
        "image_path": image_path
    }
    
    # Run Stage 1 to align face
    try:
        stage1 = FaceProcessingStage()
        context = stage1.run(context)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Face detection failed: {e}")
        
    # Write aligned_face to disk so it can be stored as thumbnail
    aligned_face = context["face"]["aligned_face"]
    thumbnail_path = f"{TMP_DIR}/{job_id}/thumbnail.jpg"
    cv2.imwrite(thumbnail_path, aligned_face)
    
    # Run Stage 2 to get embedding
    try:
        stage2 = IdentityEncodingStage()
        context = stage2.run(context)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Identity encoding failed: {e}")
        
    embedding = context["identity"]["embedding"]
    
    # Save to store
    save_identity(name, embedding, thumbnail_path)
    
    return JSONResponse({"status": "success", "name": name})

@router.get("")
def get_identities():
    """
    Lists all available saved identities.
    """
    identities = list_identities()
    return {"identities": identities}
