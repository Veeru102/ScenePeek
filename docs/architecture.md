# Architecture notes

## Data model

| table | role |
|---|---|
| `videos` | upload + media metadata, status, `first_searchable_at`, `completed_at` |
| `video_chunks` | 60 s processing units with per-stage timestamps (`extracted_at`, `transcribed_at`, …) so retries skip finished work |
| `utterances` | transcript lines with word timings for the transcript panel |
| `segments` | ~10 s retrieval units: `text`, `context_text` (± neighbours, what gets embedded), `text_embedding vector(384)`, `ocr_text`, generated `tsvector`s |
| `frames` | keyframes with `visual_embedding vector(768)`, OCR lines + boxes, phash |
| `topics` | semantic timeline spans with labels, keyphrases, centroid embedding |
| `jobs` | durable queue rows: `idempotency_key`, `priority`, `attempts`, `run_after`, `locked_by`, `heartbeat_at` |
| `metric_samples` | append-only timing samples (inference, job, search) for exact percentiles in the System page |

Indexes: HNSW (cosine) on both embedding columns, GIN on `text_tsv` / `ocr_tsv`, trigram GIN on `ocr_text`, btree `(video_id, start_s)`.

## Job queue

```sql
UPDATE jobs SET status='running', locked_by=:w, heartbeat_at=now(), attempts=attempts+1
WHERE id = (SELECT id FROM jobs
            WHERE status='queued' AND queue = ANY(:queues) AND run_after <= now()
            ORDER BY priority DESC, created_at
            FOR UPDATE SKIP LOCKED LIMIT 1)
RETURNING ...
```

- `enqueue` is `INSERT … ON CONFLICT (idempotency_key) DO NOTHING`, so re-enqueuing is safe.
- A heartbeat thread touches `heartbeat_at` every 10 s. Each worker also runs a **reaper** every 30 s that requeues jobs whose heartbeat is older than the lease timeout (90 s) — or marks them `dead` when `attempts ≥ max_attempts`.
- Failures: retryable → `queued` with `run_after = now + 5s·2^attempts + jitter`; `NonRetryableError` (corrupt media) → `dead` immediately. Chunk rows record the error; the video flips to `failed` only once every chunk is terminal, and "Retry" in the UI resets attempts.
- Queues: `cpu` (probe, extract) and `ml` (index, timeline). `scenepeek worker --queues ml` lets a GPU box take only inference work.
- Priority `-chunk_index`: chunk 0 of every video outranks chunk 1 of any video, which minimises time-to-first-searchable when several uploads are in flight.

## Pipeline stages

1. **probe_video** — ffprobe; remux or transcode to h264/aac `web.mp4` with `+faststart` (browser seeking); poster; 16 kHz mono `audio.wav`; create chunk rows; enqueue `extract_chunk` per chunk.
2. **extract_chunk** — sample frames at 1 fps (≤480 px) from the web rendition; keep a frame when its perceptual hash differs from the last kept frame by ≥8 bits or 8 s have elapsed; upload keyframes and a JSON manifest. Object keys are deterministic (`videos/{id}/chunks/{i}/frames/{t_ms}.jpg`) so re-runs overwrite.
3. **index_chunk** — slice `[start−1 s, end+1 s]` of audio, transcribe with word timestamps, keep words starting inside the chunk; plan contiguous ~10 s windows snapped to utterance boundaries (silent spans still get windows so visual-only content is indexed); pick up to 3 mutually distinct keyframes per window; batch-embed `context_text` (bge) and keyframes (SigLIP); OCR keyframes (skipping identical phashes); **delete + insert the chunk's rows in one transaction** and mark the chunk done. Because rows only appear on commit, search is incremental by construction.
4. **build_timeline** — order segment embeddings; forced boundaries where the on-screen heading (tallest OCR line, forward-filled) changes and persists; TextTiling-style depth scores over ±3-segment windows add boundaries for topic shifts without a slide change; labels = slide heading, else KeyBERT-style MMR keyphrases (optionally rewritten by Ollama).

Workers cache `web.mp4` / `audio.wav` per video under `~/.cache/scenepeek` behind a file lock, so several worker processes on one host download once.

## Retrieval

```
query ─► planner ─► {speech_q, visual_q, ocr_q, terms, phrases, weights}
             │
             ├─ bge(speech_q)     ⟶ segments.text_embedding   (HNSW)      ┐
             ├─ websearch_to_tsquery(terms OR …) ⟶ text_tsv (ts_rank_cd)  │ parallel,
             ├─ SigLIP-text(visual_q) ⟶ frames.visual_embedding, max/segment │ own sessions;
             ├─ bge(visual_q)      ⟶ segments.caption_embedding (HNSW)    │ a lane with
             └─ ocr FTS ∪ word_similarity(ocr_q, ocr_text)                ┘ weight 0 is skipped
                         ▼
            weighted RRF  (score = Σ w_m / (60 + rank_m))   or min-max weighted sum
                         ▼
            cross-encoder rerank of top-30 on "transcript + [on screen] ocr + [visual] caption"
            final = 0.5·rerank + 0.3·fused/max + 0.2·w_visual·visual_norm
                         ▼
            temporal NMS (±12 s per video, ≤3 hits/video) ─► hits + per-modality signals
```

**Captions** — every distinct keyframe (phash-deduplicated) is captioned with BLIP-base at index time; a segment's captions are merged and embedded with the *text* model, so a silent scene has a natural-language representation that the query can match directly ("a rabbit sniffing a flower"). This exists because zero-shot SigLIP retrieval on homogeneous footage collapses onto a few "hub" frames — see the README's ablations. Captions are a separate column and lane (not folded into `context_text`) so they can be switched off cleanly (`weight_caption: 0`) for measurement.

Planner cues: leading "find/show me where…" is stripped; "the professor explains X" isolates X as the speech query; "X while showing Y" splits speech/visual; visual nouns, "slide/diagram/on screen" and quoted phrases bump the visual, OCR and lexical weights respectively. A local Ollama server is auto-detected (`ollama_auto`, one health ping per process) and, when present, performs the decomposition instead; every failure mode (down, model not pulled, timeout, malformed JSON) falls back to the heuristics.

**Debug mode** — `SearchOptions(debug=True)` returns each lane's ranked list alongside the fused hits; the eval harness uses it to attribute every correct answer to the lane(s) that found it, without re-running the lanes.

## Why these choices

- **Postgres for everything (pgvector + FTS + queue)** — one transactional store means chunk state, vectors and job leases commit together; metadata filters are plain `WHERE` clauses; there is no vector-store/DB sync problem. At the scale one machine can index (tens of thousands of hours) HNSW in Postgres is more than fast enough.
- **Hybrid runtime** — Docker can't reach the Apple GPU, so infra runs in Compose and the worker runs natively with MPS (5–10× faster Whisper/SigLIP). The same package has a CPU Dockerfile target for Linux/GPU hosts.
- **API loads only text-side encoders** — bge, SigLIP's text tower and the reranker (≈1.5 GB) are enough to encode queries; all audio/image inference stays in workers, so the API is horizontally cheap and the heavy models are separable.
- **RRF over score fusion by default** — modality scores live on incompatible scales (SigLIP logits, ts_rank, cosine); rank fusion is robust without calibration. Weighted min-max fusion is implemented for the eval harness to compare.
- **Chunks as the unit of work** — independent, idempotent, retryable, and small enough that a crash loses at most ~20 s of compute. Within a chunk the Whisper transcript is checkpointed to object storage (tagged with the model name), so a retry after an embedding/OCR/caption failure skips ASR entirely.
- **Clocks come from Postgres** — `run_after`, leases and backoff all use the database's `now()`. Mixing the worker's clock with the DB's produced a subtle flake: a job enqueued and leased within the clock skew was invisible to `lease()`.

## Metrics

Prometheus (`/metrics` on the API and each worker): `scenepeek_jobs_total{type,status}`, `scenepeek_job_duration_seconds`, `scenepeek_queue_depth`, `scenepeek_model_inference_seconds{model}`, `scenepeek_indexed_video_seconds_total`, `scenepeek_time_to_first_searchable_seconds`, `scenepeek_video_index_wall_seconds`, `scenepeek_search_latency_seconds{stage}`, `scenepeek_job_retries_total`.

`GET /api/metrics/summary` derives the same numbers from Postgres (chunk timestamps, job rows, `metric_samples`) so the System page works without Prometheus.

## Scaling path

- More throughput: `make worker N=4` on one host, or run the worker image on more machines pointing at the same Postgres/MinIO; `--queues ml` on GPU nodes.
- Bigger library: pgvector HNSW handles millions of rows; partition `segments`/`frames` by `video_id` range or move vectors to a dedicated store behind the same `candidates.py` interface.
- Better ranking: swap models via env (`TEXT_EMBED_MODEL`, `VISUAL_EMBED_MODEL`, `RERANKER_MODEL`), re-index into a fresh database, and compare with the eval harness.
