"""encode_temporal (vision queue): embed sliding multi-frame windows of a finished chunk with the
configured video encoder and store them as versioned `embeddings` rows.

Runs after `index_chunk` so it never delays first-searchable; the same handler backfills old
chunks for a new encoder version (rows for the old version stay until it is retired)."""

import shutil
from datetime import UTC, datetime

from PIL import Image
from sqlalchemy import delete, func, select, text
from sqlalchemy.engine import Engine

from scenepeek.core import storage
from scenepeek.core.config import get_settings
from scenepeek.jobs import priority
from scenepeek.jobs import queue as q
from scenepeek.jobs.registry import JobContext
from scenepeek.ml.video import get_encoder, sample_windows
from scenepeek.models import Embedding, IndexVersion, Segment, Video, VideoChunk
from scenepeek.models.chunk import ChunkStatus
from scenepeek.models.embedding import hnsw_index_name, hnsw_index_sql
from scenepeek.models.index_version import IndexVersionStatus
from scenepeek.pipeline import media

KIND = "temporal"


def encode_temporal(ctx: JobContext, payload: dict) -> None:
    settings = get_settings()
    video_id, idx = payload["video_id"], int(payload["chunk_index"])
    enc = get_encoder(payload.get("model"))
    force = bool(payload.get("force"))
    with ctx.session() as s:
        chunk = s.scalar(select(VideoChunk).where(VideoChunk.video_id == video_id, VideoChunk.index == idx))
        if chunk is None:
            raise q.NonRetryableError(f"chunk {idx} missing")
        if chunk.status != ChunkStatus.DONE:
            raise q.NonRetryableError(f"chunk {idx} is {chunk.status}, not done")
        start, end, chunk_id = chunk.start_s, chunk.end_s, chunk.id
        have = s.scalar(
            select(func.count()).where(
                Embedding.chunk_id == chunk_id, Embedding.kind == KIND, Embedding.model_key == enc.key
            )
        )
        if have and not force:
            return
        segs = [
            (sg.id, sg.start_s, sg.end_s)
            for sg in s.scalars(select(Segment).where(Segment.chunk_id == chunk_id).order_by(Segment.start_s))
        ]

    src = media.cached(video_id, storage.web_key(video_id), "web.mp4")
    out_dir = media.workdir(video_id, f"chunk_{idx}", "dense")
    paths = media.sample_frames(src, out_dir, start, end, 1.0, settings.frame_max_width)
    times = [start + i for i in range(len(paths))]
    windows = sample_windows(times, settings.temporal_window_s, settings.temporal_stride_s, enc.n_frames)
    if not windows:
        shutil.rmtree(out_dir, ignore_errors=True)
        return

    images: dict[int, Image.Image] = {}
    clips = []
    for _, _, idxs in windows:
        for i in idxs:
            if i not in images:
                images[i] = Image.open(paths[i]).convert("RGB")
        clips.append([images[i] for i in idxs])
    vecs = enc.embed_clips(clips)
    for im in images.values():
        im.close()
    shutil.rmtree(out_dir, ignore_errors=True)

    rows = []
    for (ws, we, _), v in zip(windows, vecs, strict=True):
        centre = (ws + we) / 2
        seg_id = next((sid for sid, a, b in segs if a <= centre < b), None)
        if seg_id is None and segs:
            seg_id = min(segs, key=lambda t: abs((t[1] + t[2]) / 2 - centre))[0]
        rows.append(
            Embedding(
                video_id=video_id,
                segment_id=seg_id,
                chunk_id=chunk_id,
                kind=KIND,
                model_key=enc.key,
                start_s=ws,
                end_s=we,
                embedding=v.tolist(),
            )
        )
    with ctx.session() as s:
        s.execute(
            delete(Embedding).where(
                Embedding.chunk_id == chunk_id, Embedding.kind == KIND, Embedding.model_key == enc.key
            )
        )
        s.add_all(rows)
        s.get(VideoChunk, chunk_id).temporal_at = datetime.now(UTC)
        s.commit()
    ensure_index(ctx.engine, enc.key, enc.dim)
    _maybe_drop_cache(ctx, video_id)
    ctx.log.info("temporal windows encoded", windows=len(rows), model=enc.key)


def _maybe_drop_cache(ctx: JobContext, video_id: str) -> None:
    """Benchmark clips are numerous: once every chunk of a non-upload video has its windows, the
    per-host media cache (web.mp4, audio, frames) has nothing left to serve."""
    with ctx.session() as s:
        source = s.scalar(select(Video.source).where(Video.id == video_id))
        pending = s.scalar(
            select(func.count()).where(VideoChunk.video_id == video_id, VideoChunk.temporal_at.is_(None))
        )
    if source != "upload" and pending == 0:
        media.drop_cache(video_id)


def ensure_index(engine: Engine, model_key: str, dim: int) -> None:
    """Register the version and build its partial HNSW index once (cheap while the table is small;
    `CREATE INDEX IF NOT EXISTS` makes concurrent workers harmless)."""
    with engine.begin() as conn:
        row = conn.execute(
            text("SELECT status FROM index_versions WHERE kind=:k AND model_key=:m"),
            {"k": KIND, "m": model_key},
        ).first()
        if row and row[0] == IndexVersionStatus.READY:
            return
    with engine.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
        conn.execute(text(hnsw_index_sql(KIND, model_key, dim)))
    with engine.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO index_versions (id, kind, model_key, dim, status, index_name) "
                "VALUES (gen_random_uuid(), :k, :m, :d, :s, :i) "
                "ON CONFLICT ON CONSTRAINT uq_index_version DO UPDATE SET status=:s, index_name=:i, dim=:d"
            ),
            {
                "k": KIND,
                "m": model_key,
                "d": dim,
                "s": IndexVersionStatus.READY,
                "i": hnsw_index_name(KIND, model_key),
            },
        )


def enqueue_for_chunk(
    conn, video_id: str, idx: int, band: int, *, model: str | None = None, force: bool = False
):
    payload = {"video_id": video_id, "chunk_index": idx}
    key = f"temporal:{video_id}:{idx}"
    if model:
        payload["model"] = model
        key += f":{model}"
    if force:
        payload["force"] = True
        key += f":{int(datetime.now(UTC).timestamp())}"
    return q.enqueue_sync(
        conn,
        "encode_temporal",
        payload,
        idempotency_key=key,
        queue="vision",
        priority=priority.chunk_priority(band, idx) - 10,
        video_id=video_id,
    )


def backfill(
    engine: Engine, *, model: str | None = None, video_id: str | None = None, force: bool = False
) -> int:
    """Enqueue encode_temporal at backfill priority for every finished chunk that has no rows for
    the encoder. Old versions are untouched; `retire` them once the new one measures better."""
    enc = get_encoder(model)
    n = 0
    with engine.begin() as conn:
        stmt = (
            select(VideoChunk.video_id, VideoChunk.index)
            .where(VideoChunk.status == ChunkStatus.DONE)
            .order_by(VideoChunk.video_id, VideoChunk.index)
        )
        if video_id:
            stmt = stmt.where(VideoChunk.video_id == video_id)
        if not force:
            done = select(Embedding.chunk_id).where(Embedding.kind == KIND, Embedding.model_key == enc.key)
            stmt = stmt.where(VideoChunk.id.not_in(done))
        for vid, idx in conn.execute(stmt):
            if enqueue_for_chunk(conn, str(vid), idx, priority.BACKFILL, model=model, force=force):
                n += 1
    return n


def retire(engine: Engine, model_key: str) -> None:
    with engine.begin() as conn:
        conn.execute(
            text("UPDATE index_versions SET status=:s WHERE kind=:k AND model_key=:m"),
            {"s": IndexVersionStatus.RETIRED, "k": KIND, "m": model_key},
        )


def versions(engine: Engine) -> list[IndexVersion]:
    from sqlalchemy.orm import Session

    with Session(engine) as s:
        return list(s.scalars(select(IndexVersion).order_by(IndexVersion.kind, IndexVersion.created_at)))
