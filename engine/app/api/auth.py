from fastapi import Security, HTTPException, Depends, Request
from fastapi.security import APIKeyHeader
from sqlalchemy.orm import Session
from app.db.session import get_db
from app.db.models import ApiKey

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

def verify_api_key(request: Request, api_key: str = Security(api_key_header), db: Session = Depends(get_db)):
    if not api_key:
        raise HTTPException(status_code=401, detail="Missing X-API-Key header")

    # Rate limiting via Redis is disabled (synchronous/no-Redis mode).
    # Restore a Redis token-bucket here if switching to RQ mode.

    # Check DB for valid key & credits
    db_key = db.query(ApiKey).filter(ApiKey.key == api_key).first()
    if not db_key:
        raise HTTPException(status_code=403, detail="Invalid API Key")

    if db_key.credits_remaining <= 0:
        raise HTTPException(status_code=402, detail="Insufficient credits")

    return api_key
