import os
import hashlib
import pickle
from app.config import DATA_DIR

CACHE_DIR = os.path.join(DATA_DIR, "cache")

def get_file_hash(filepath: str) -> str:
    """Returns SHA-256 hash of a file."""
    hasher = hashlib.sha256()
    with open(filepath, 'rb') as f:
        while chunk := f.read(8192):
            hasher.update(chunk)
    return hasher.hexdigest()

def get_text_hash(text: str) -> str:
    """Returns SHA-256 hash of a text string."""
    return hashlib.sha256(text.encode('utf-8')).hexdigest()

def get_cache(key: str):
    """Retrieves data from the disk cache."""
    cache_path = os.path.join(CACHE_DIR, f"{key}.pkl")
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            print(f"Warning: Failed to load cache key {key}: {e}")
    return None

def set_cache(key: str, data):
    """Saves data to the disk cache."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_path = os.path.join(CACHE_DIR, f"{key}.pkl")
    try:
        with open(cache_path, "wb") as f:
            pickle.dump(data, f)
    except Exception as e:
        print(f"Warning: Failed to save cache key {key}: {e}")
