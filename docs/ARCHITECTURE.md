# Architecture

## Design principle

Each pipeline stage is an isolated module behind a common interface
(`app/pipeline/stageN_*.py`), orchestrated by `app/pipeline/orchestrator.py`.
This means:

- Any stage can be swapped for a better model later without touching the rest
- Stages can be benchmarked/replaced independently once you know which one
  is your actual quality bottleneck
- The same orchestrator can run on CPU (dev/test, small resolution) or GPU
  (production) via config, not code changes

## Stage map

| # | Stage | Responsibility | Reference model(s) |
|---|-------|-----------------|---------------------|
| 1 | Face processing | detect, align, crop, segment, landmarks, pose | MediaPipe FaceMesh, InsightFace |
| 2 | Identity encoding | compact identity embedding (prevents identity drift) | InsightFace ArcFace |
| 3 | Audio encoding | phoneme/prosody/timing features from speech | Wav2Vec2 / HuBERT; edge-tts for TTS input |
| 4 | Motion prediction | latent motion representation (blink, gaze, head pose, emotion*) | LivePortrait motion module |
| 5 | Talking-head generation | identity + audio + motion → frames | LivePortrait (primary), SadTalker/MuseTalk (fallback/compare) |

*> Note on Emotion (Stage 4)*: Native emotion conditioning via LivePortrait's expression space is currently a known gap in Phase 1. The pipeline accepts an `emotion` flag (neutral, happy, serious, surprised) and bakes a static offset into the dummy latents for now, but a true implementation requires wiring up the full LivePortrait expression control branch (Phase 2+).
| 6 | Lip refinement | fix mouth/teeth/lip-sync artifacts | MuseTalk refiner, Wav2Lip |
| 7 | Face restoration | restore skin/eye/hair detail lost in generation | GFPGAN, CodeFormer |
| 8 | Frame interpolation | 12-25fps → 30/60fps | RIFE |
| 9 | Temporal stabilization | reduce frame-to-frame flicker/jitter | EMA smoothing on landmarks/latents |
| 10 | Assembly | mux audio+frames, encode | ffmpeg |

## Job flow (Batch / Asynchronous)

```
POST /jobs  {image, audio|text, options}
        │
        ▼
  Job queued (app/jobs/queue.py)
        │
        ▼
  Orchestrator runs stages 1-10 sequentially,
  writing intermediate artifacts to data/tmp/{job_id}/
        │
        ▼
  Final MP4 → data/output/{job_id}.mp4
        │
        ▼
  GET /jobs/{id}  → status + result URL
```

## Real-time Streaming Flow (WebSockets)

For real-time conversational agents (e.g., InterviewAI), the engine exposes an explicitly separated architectural fork that optimizes for lowest-possible latency rather than distributed throughput.

```
WS /stream
        │
        ▼
  (Init) Client sends Identity. Server runs Stages 1 & 2 once.
  Server caches the persistent Identity `context`.
        │
        ▼
  (Loop) Client streams audio binary chunks.
        │
        ▼
  Server clones base context, runs Stages 3-9 ONLY per chunk.
        │
        ▼
  Server encodes resulting raw frames to Base64 JPEGs.
        │
        ▼
  Server yields frames back over WebSocket immediately.
```
*Note: We purposefully do not merge the Batch API and Streaming API flows. Batch processing is designed to be distributed over Redis/RQ across multiple worker nodes, whereas Streaming requires stateful, synchronous in-memory persistence of the generator models.*

## Swappable model registry

`app/models/registry.py` holds a config-driven map of stage → model name →
loader function, so a stage's implementation can be selected at runtime
(useful for A/B quality testing, or falling back to a lighter model on
CPU-only hardware).

## Config profiles

`app/config.py` defines named hardware profiles:

- `cpu_dev` — smallest models, low res, for local development/testing
- `gpu_single` — single consumer GPU (e.g. RTX 4090), production-quality inference
- `gpu_cloud` — rented multi-GPU, for batch/fine-tuning workloads

## Notes on training vs. inference

This scaffold is inference-first. Nothing here trains a model from scratch.
When a stage's off-the-shelf quality isn't good enough, the fix is almost
always: (a) better preprocessing (stage 1), or (b) fine-tuning an existing
open-source checkpoint on a small custom dataset — not training a new
architecture. See ROADMAP.md Phase 3 for where fine-tuning fits.
