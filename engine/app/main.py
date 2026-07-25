from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router as jobs_router
from app.api.identities import router as identities_router
from app.api.streaming import router as streaming_router

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    from app.db.session import engine, SessionLocal
    from app.db.models import Base, ApiKey
    
    # Create DB tables
    Base.metadata.create_all(bind=engine)
    
    # Insert test key
    db = SessionLocal()
    test_key = "test_dev_key"
    if not db.query(ApiKey).filter(ApiKey.key == test_key).first():
        db.add(ApiKey(key=test_key, credits_remaining=100))
        db.commit()
    db.close()
    
    yield

app = FastAPI(
    title="Talking-Head Engine",
    description="Modular AI avatar / talking-head video generation service.",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten before production
    allow_methods=["*"],
    allow_headers=["*"],
)

from fastapi.staticfiles import StaticFiles
import os
from app.config import OUTPUT_DIR

app.include_router(jobs_router, prefix="/jobs", tags=["jobs"])
app.include_router(identities_router, prefix="/identities", tags=["identities"])
app.include_router(streaming_router)

os.makedirs(OUTPUT_DIR, exist_ok=True)
app.mount("/output", StaticFiles(directory=OUTPUT_DIR), name="output")



@app.get("/health")
def health():
    from app.config import get_active_profile

    profile = get_active_profile()
    return {"status": "ok", "profile": profile.name, "device": profile.device}
