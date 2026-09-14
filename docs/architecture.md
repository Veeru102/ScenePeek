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
| `embeddings` | **versioned vector store**: `(video, segment, chunk, kind, model_key, start_s, end_s, embedding)` with a dimension-less `vector` column. Holds the temporal (X-CLIP) windows today; any new text/frame encoder version goes here too, next to the old one |
| `jobs` | durable queue rows: `idempotency_key`, `priority`, `attempts`, `run_after`, `locked_by`, `heartbeat_at` |
| `datasets` / `dataset_videos` / `dataset_queries` | benchmark registry: clips per split (with the library `video_id` once fetched + indexed) and queries with `[[start, end], …]` ground truth |
| `experiments` / `experiment_results` | one row per evaluation run (full `SearchConfig`, code SHA, model versions, aggregate metrics) + one row per query (metrics, lane attribution, top hits with segment ids) |
| `index_versions` | which model produced which stored artifact (`kind`, `model_key`, `dim`, `status`, HNSW index name); routers and fine-tuned rerankers register here too |
| `training_examples` | mined (query, positive segment, hard-negative segment) triples with the rank/score/lanes that produced the mistake |
| `metric_samples` | append-only timing samples (inference, job, search) for exact percentiles in the System page |

Indexes: HNSW (cosine) on `segments.text_embedding` / `caption_embedding` / `frames.visual_embedding`; on `embeddings`, one **partial expression HNSW index per (kind, model_key)** — `USING hnsw ((embedding::vector(512)) vector_cosine_ops) WHERE kind='temporal' AND model_key='xclip-base-patch32'` — created when a version becomes ready, so versions of different widths coexist in one table and each lane query hits exactly its version's index; GIN on `text_tsv` / `ocr_tsv`, trigram GIN on `ocr_text`, btree `(video_id, start_s)`.

Every artifact row carries its producer: `video_chunks.asr_model`, `segments.embedding_model` / `ocr_model`, `frames.visual_model`, `embeddings.model_key` (`ml/versions.py` derives the keys from settings). A model change is therefore a **backfill** (`scenepeek backfill temporal --model …`, lowest priority) followed by an experiment, never an in-place overwrite; `scenepeek versions --retire kind:key` retires the loser.

## Job queue

```sql
UPDATE jobs SET status='running', locked_by=:w, heartbeat_at=now(), attempts=attempts+1
WHERE id = (SELECT id FROM jobs
            WHERE status='queued' AND queue = ANY(:queues) AND run_after <= now()
              AND priority >= :min_priority                       -- reserved interactive worker
              AND (priority >= 0 OR :offline_budget IS NULL       -- offline concurrency budget
                   OR (SELECT count(*) FROM jobs WHERE status='running' AND priority < 0) < :offline_budget)
            ORDER BY priority DESC, created_at
            FOR UPDATE SKIP LOCKED LIMIT 1)
RETURNING ...
```

- `enqueue` is `INSERT … ON CONFLICT (idempotency_key) DO NOTHING`, so re-enqueuing is safe.
- A heartbeat thread touches `heartbeat_at` every 10 s. Each worker also runs a **reaper** every 30 s that requeues jobs whose heartbeat is older than the lease timeout (90 s) — or marks them `dead` when `attempts ≥ max_attempts`.
- Failures: retryable → `queued` with `run_after = now + 5s·2^attempts + jitter`; `NonRetryableError` (corrupt media) → `dead` immediately. Chunk rows record the error; the video flips to `failed` only once every chunk is terminal, and "Retry" in the UI resets attempts.
- Queues are workload classes: `cpu` (probe, extract, dataset fetch), `ml` (index, timeline — ASR + embeddings + OCR stay in one job so the chunk commit stays atomic), `vision` (temporal encoding, safe to run on a GPU box only: `scenepeek worker --queues vision`).
- **Priority bands** (`jobs/priority.py`): interactive uploads `+100`, benchmark imports `−100`, model backfills `−200`, each minus the chunk index, with a `+50` bonus for chunk 0 of interactive work — so chunk 0 of any upload outranks everything else and time-to-first-searchable stays low when several uploads are in flight. Temporal encoding runs at chunk priority − 10, after the chunk is already searchable.
- **Isolation without a second queue**: `OFFLINE_MAX_RUNNING=k` caps how many priority < 0 jobs run cluster-wide (one subquery in the lease), and `scenepeek worker --min-priority 0` is a worker that never takes offline work. `scripts/bench_isolation.sh` measures an interactive upload's time-to-first-searchable under a full-library backfill with shared / budgeted / reserved layouts.

## Pipeline stages

1. **probe_video** — ffprobe; remux or transcode to h264/aac `web.mp4` with `+faststart` (browser seeking); poster; 16 kHz mono `audio.wav`; create chunk rows; enqueue `extract_chunk` per chunk.
2. **extract_chunk** — sample frames at 1 fps (≤480 px) from the web rendition; keep a frame when its perceptual hash differs from the last kept frame by ≥8 bits or 8 s have elapsed; upload keyframes and a JSON manifest. Object keys are deterministic (`videos/{id}/chunks/{i}/frames/{t_ms}.jpg`) so re-runs overwrite.
3. **index_chunk** — slice `[start−1 s, end+1 s]` of audio, transcribe with word timestamps, keep words starting inside the chunk; plan contiguous ~10 s windows snapped to utterance boundaries (silent spans still get windows so visual-only content is indexed); pick up to 3 mutually distinct keyframes per window; batch-embed `context_text` (bge) and keyframes (SigLIP); OCR keyframes (skipping identical phashes); **delete + insert the chunk's rows in one transaction** and mark the chunk done. Because rows only appear on commit, search is incremental by construction.
4. **encode_temporal** (vision queue, after the chunk is searchable) — re-sample the chunk at 1 fps from the cached web rendition, slide 8 s windows with a 4 s stride (8 frames each), embed with the configured `VideoEncoder` (X-CLIP base by default; `siglip-meanpool` = mean of per-frame SigLIP vectors, the no-new-model control), assign each window to the segment containing its centre and write `embeddings(kind='temporal', model_key=…)` rows for the chunk in one transaction (delete + insert per chunk + model, so re-runs are idempotent). The same handler backfills old chunks for a new encoder.
5. **fetch_dataset_video** (cpu queue) — for benchmark clips: the dataset adapter downloads the clip (yt-dlp with `--download-sections`), a `Video` row is created with `source=<dataset>` and `priority_band=BENCHMARK`, and `probe_video` is enqueued — from there a benchmark clip is indistinguishable from an upload.
6. **build_timeline** — order segment embeddings; forced boundaries where the on-screen heading (tallest OCR line, forward-filled) changes and persists; TextTiling-style depth scores over ±3-segment windows add boundaries for topic shifts without a slide change; labels = slide heading, else KeyBERT-style MMR keyphrases (optionally rewritten by Ollama).

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
- **Weighted min-max fusion by default, RRF kept** — RRF gives a lane's rank-1 candidate nearly full credit even when that lane has nothing relevant; per-lane min-max normalisation was measured better once the caption lane was in the mix (README). Both are one `fusion:` switch in an experiment.
- **Benchmarks go through the pipeline** — QVHighlights clips are fetched and indexed exactly like uploads (at benchmark priority), never loaded as precomputed features; the benchmark exercises the ASR/frame/OCR/temporal path users hit, and its ingestion doubles as the load test for workload isolation.
- **Segments stay the retrieval unit** — lanes, fusion, NMS and the UI all work on ~10 s segments; the temporal lane attaches its best window to a segment (`spans`) instead of introducing a second unit, and `span_mode: window` is where IoU-level precision comes from when a query needs it.
- **Chunks as the unit of work** — independent, idempotent, retryable, and small enough that a crash loses at most ~20 s of compute. Within a chunk the Whisper transcript is checkpointed to object storage (tagged with the model name), so a retry after an embedding/OCR/caption failure skips ASR entirely.
- **Clocks come from Postgres** — `run_after`, leases and backoff all use the database's `now()`. Mixing the worker's clock with the DB's produced a subtle flake: a job enqueued and leased within the clock skew was invisible to `lease()`.

## Metrics

Prometheus (`/metrics` on the API and each worker): `scenepeek_jobs_total{type,status}`, `scenepeek_job_duration_seconds`, `scenepeek_queue_depth`, `scenepeek_model_inference_seconds{model}`, `scenepeek_indexed_video_seconds_total`, `scenepeek_time_to_first_searchable_seconds`, `scenepeek_video_index_wall_seconds`, `scenepeek_search_latency_seconds{stage}`, `scenepeek_job_retries_total`.

`GET /api/metrics/summary` derives the same numbers from Postgres (chunk timestamps, job rows, `metric_samples`) so the System page works without Prometheus.

## Scaling path

- More throughput: `make worker N=4` on one host, or run the worker image on more machines pointing at the same Postgres/MinIO; `--queues ml` on GPU nodes.
- Bigger library: pgvector HNSW handles millions of rows; partition `segments`/`frames` by `video_id` range or move vectors to a dedicated store behind the same `candidates.py` interface.
- Better ranking: a new temporal encoder is a `VideoEncoder` implementation + `scenepeek backfill temporal --model <key>` (rows land next to the old version) + an experiment with `models.temporal: <key>`; a new text/frame encoder follows the same path into `embeddings` once its lane reads from there. Rerankers and routers are paths/versions in `SearchConfig`.
- Bigger training sets: `dataset import qvhighlights --split train --limit N` + `eval run qvh_train.yaml` produce the lane-attribution labels for the router and the hits for hard-negative mining at whatever scale the workers can index.
