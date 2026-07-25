# Talking-Head Engine

A modular AI talking-head / avatar video generation engine.
Built to run standalone (as a sellable product / API) and to be embeddable
into InterviewAI's avatar interview pipeline.

## Why this architecture

Commercial talking-head systems (JoyPix, LipSync.video, HeyGen, Synthesia)
are not one model — they're a **pipeline of 8-10 specialized models**
chained together with real engineering around them (queues, GPU scheduling,
caching). This repo mirrors that pipeline shape, but built on strong
pretrained open-source components rather than training from scratch.

```
Image/Video + Audio/Text
        │
        ▼
1. Face Processing        (MediaPipe / InsightFace)
2. Identity Encoding       (ArcFace embeddings)
3. Audio Encoding          (Wav2Vec2 / HuBERT + edge-tts)
4-5. Motion + Generation   (LivePortrait — core generator)
6. Lip Refinement          (MuseTalk / Wav2Lip post-pass)
7. Face Restoration        (GFPGAN / CodeFormer)
8. Frame Interpolation     (RIFE)
9. Temporal Stabilization  (EMA smoothing / latent smoothing)
10. Assembly                (ffmpeg)
        │
        ▼
      MP4 out
```

See `docs/ARCHITECTURE.md` for the full stage-by-stage design and
`docs/ROADMAP.md` for the phased build plan.

## Repo layout

```
engine/     FastAPI service — the actual ML pipeline + job orchestration
web/        React + Vite + TS frontend (upload, preview, job status)
infra/      Dockerfiles, model-download scripts, deployment config
docs/       Architecture and roadmap docs
models_cache/  Downloaded pretrained weights (gitignored)
data/       Local input/output/tmp working dirs (gitignored)
```

## Quickstart (local dev, CPU-only)

```bash
cd engine
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

```bash
cd web
npm install
npm run dev
```

## Quickstart (GPU inference via Docker)

```bash
docker compose up --build
```

See `infra/scripts/download_models.sh` to pull pretrained weights before
first run.
