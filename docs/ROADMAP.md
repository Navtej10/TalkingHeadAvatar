# Roadmap

## Phase 0 — Decide & de-risk (this week)
- [ ] Pick the core generator: LivePortrait (already used in InterviewAI) vs. SadTalker vs. MuseTalk
      — run the same 5 test clips through each, compare quality/speed
- [ ] Rent a single cloud GPU (RunPod/Vast.ai, RTX 4090 class) for benchmarking; don't buy hardware yet
- [ ] Confirm licensing terms of chosen models for commercial use (some LivePortrait/SadTalker
      checkpoints have non-commercial clauses — check before selling this as a product)

## Phase 1 — Baseline pipeline (weeks 1-3)
- [ ] Stage 1: face detect/align/crop (MediaPipe or InsightFace)
- [ ] Stage 3: audio encoding wired to existing edge-tts output
- [ ] Stage 5: single generator (LivePortrait) producing raw frames
- [ ] Stage 10: ffmpeg assembly to MP4
- [ ] FastAPI job endpoint: submit image+audio → poll → download MP4
- **Goal: working end-to-end system, CPU or single-GPU, ugly but functional**

## Phase 2 — Quality pass (weeks 3-6)
- [ ] Stage 4: proper motion latent (not just raw generator output)
- [ ] Stage 6: lip refinement pass
- [ ] Stage 7: face restoration (GFPGAN/CodeFormer)
- [ ] Stage 9: temporal stabilization (EMA smoothing)
- [ ] Stage 8: frame interpolation (RIFE) to 30fps
- **Goal: quality approaching commercial demos; this is what you'd show to design partners**

## Phase 3 — Differentiation (weeks 6-10+)
- [ ] Multiple avatar identities, saved/reusable
- [ ] Emotion control parameter
- [ ] Head-pose / eye-contact control (relevant for InterviewAI's interview use case)
- [ ] If off-the-shelf quality plateaus: fine-tune chosen generator on a small
      custom dataset (rented A100/H100, days not months)
- **Goal: a product with a defensible edge, not a wrapper around public models**

## Phase 4 — Productionization (parallel, ongoing)
- [ ] Job queue + GPU scheduling (Redis/RQ or Celery)
- [ ] Caching identity/audio embeddings across jobs
- [ ] Streaming inference for InterviewAI's real-time interview use case
- [ ] Auth, rate limiting, credits/billing (if selling standalone)
- [ ] Multi-GPU autoscaling once there's real load
- [ ] Integrate as a service InterviewAI's avatar module calls, replacing
      the current direct LivePortrait/edge-tts calls

## Explicit non-goals for v1
- No training a generator from scratch
- No massive proprietary dataset collection
- No real-time (<1s latency) generation until Phase 4 — batch/near-real-time is fine to start
