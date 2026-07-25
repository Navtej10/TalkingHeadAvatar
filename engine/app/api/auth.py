from fastapi import Security, HTTPException, Depends, Request
from fastapi.security import APIKeyHeader
from sqlalchemy.orm import Session
from app.db.session import get_db
from app.db.models import ApiKey
from app.jobs.queue import redis_conn
import time

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

def verify_api_key(request: Request, api_key: str = Security(api_key_header), db: Session = Depends(get_db)):
    if not api_key:
        raise HTTPException(status_code=401, detail="Missing X-API-Key header")
        
    # Rate Limiting (Token Bucket per IP/Key)
    # 5 requests per minute
    if redis_conn:
        current_time = int(time.time())
        window = current_time // 60
        rate_limit_key = f"rate_limit:{api_key}:{window}"
        
        requests_in_window = redis_conn.incr(rate_limit_key)
        if requests_in_window == 1:
            redis_conn.expire(rate_limit_key, 60)
            
        if requests_in_window > 5:
            raise HTTPException(status_code=429, detail="Too many requests. Limit 5 per minute.")
        
    # Check DB for valid key & credits
    db_key = db.query(ApiKey).filter(ApiKey.key == api_key).first()
    if not db_key:
        raise HTTPException(status_code=403, detail="Invalid API Key")
        
    if db_key.credits_remaining <= 0:
        raise HTTPException(status_code=402, detail="Insufficient credits")
        
    return api_key
