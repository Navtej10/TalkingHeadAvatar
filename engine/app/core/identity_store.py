import os
import json
import shutil
import numpy as np
from app.config import DATA_DIR


IDENTITIES_DIR = os.path.join(DATA_DIR, "identities")
INDEX_FILE = os.path.join(IDENTITIES_DIR, "index.json")

def _ensure_store_exists():
    os.makedirs(IDENTITIES_DIR, exist_ok=True)
    if not os.path.exists(INDEX_FILE):
        with open(INDEX_FILE, "w") as f:
            json.dump({}, f)

def save_identity(name: str, embedding: np.ndarray, thumbnail_path: str):
    _ensure_store_exists()
    
    emb_path = os.path.join(IDENTITIES_DIR, f"{name}.npy")
    np.save(emb_path, embedding)
    
    # Store thumbnail as .jpg
    new_thumbnail_path = os.path.join(IDENTITIES_DIR, f"{name}.jpg")
    shutil.copy(thumbnail_path, new_thumbnail_path)
    
    with open(INDEX_FILE, "r") as f:
        index = json.load(f)
        
    index[name] = {
        "name": name,
        "embedding_file": f"{name}.npy",
        "thumbnail_file": f"{name}.jpg"
    }
    
    with open(INDEX_FILE, "w") as f:
        json.dump(index, f, indent=4)

def list_identities() -> list[dict]:
    _ensure_store_exists()
    with open(INDEX_FILE, "r") as f:
        index = json.load(f)
    return list(index.values())

def load_identity(name: str) -> tuple[np.ndarray, str]:
    _ensure_store_exists()
    with open(INDEX_FILE, "r") as f:
        index = json.load(f)
        
    if name not in index:
        raise ValueError(f"Identity '{name}' not found.")
        
    emb_path = os.path.join(IDENTITIES_DIR, index[name]["embedding_file"])
    thumbnail_path = os.path.join(IDENTITIES_DIR, index[name]["thumbnail_file"])
    
    embedding = np.load(emb_path)
    return embedding, thumbnail_path
