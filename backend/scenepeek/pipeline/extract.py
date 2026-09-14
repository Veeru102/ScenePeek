"""extract_chunk (cpu queue): sample frames, pick keyframes, upload them + a manifest."""

from datetime import UTC, datetime

from sqlalchemy import select

from scenepeek.core import storage
from scenepeek.core.config import get_settings
from scenepeek.jobs import priority
from scenepeek.jobs import queue as q
from scenepeek.jobs.registry import JobContext
from scenepeek.models import VideoChunk
from scenepeek.models.chunk import ChunkStatus
from scenepeek.pipeline import media
from scenepeek.pipeline.frames import FrameRef, dump_manifest, select_keyframes


def manifest_key(video_id: str, chunk_index: int) -> str:
    return f"videos/{video_id}/chunks/{chunk_index}/frames.json"


def extract_chunk(ctx: JobContext, payload: dict) -> None:
    settings = get_settings()
    video_id, idx = payload["video_id"], int(payload["chunk_index"])
    with ctx.session() as s:
        chunk = s.scalar(select(VideoChunk).where(VideoChunk.video_id == video_id, VideoChunk.index == idx))
        if chunk is None:
            raise q.NonRetryableError(f"chunk {idx} missing")
        if chunk.status == ChunkStatus.DONE:
            return
        chunk.status, chunk.stage, chunk.attempts = ChunkStatus.RUNNING, "extract", chunk.attempts + 1
        s.commit()
        start, end, chunk_id = chunk.start_s, chunk.end_s, chunk.id
        band = chunk.video.priority_band

    try:
        if chunk.extracted_at is None:
            src = media.cached(video_id, storage.web_key(video_id), "web.mp4")
            out_dir = media.workdir(video_id, f"chunk_{idx}", "frames")
            sampled_paths = media.sample_frames(
                src, out_dir, start, end, settings.frame_sample_fps, settings.frame_max_width
            )
            sampled = [(start + i / settings.frame_sample_fps, p) for i, p in enumerate(sampled_paths)]
            kept = select_keyframes(sampled)
            refs: list[FrameRef] = []
            for t, p, h in kept:
                t_ms = int(round(t * 1000))
                key = storage.frame_key(video_id, idx, t_ms)
                storage.upload_file(p, key, "image/jpeg")
                refs.append(FrameRef(t_ms=t_ms, key=key, phash=h))
            storage.upload_bytes(dump_manifest(refs), manifest_key(video_id, idx), "application/json")
            ctx.log.info("keyframes", sampled=len(sampled), kept=len(refs))
            with ctx.session() as s:
                c = s.get(VideoChunk, chunk_id)
                c.extracted_at = datetime.now(UTC)
                c.stage = "queued:index"
                s.commit()
    except Exception as e:
        _record_chunk_error(ctx, chunk_id, e)
        raise

    with ctx.session() as s:
        q.enqueue_sync(
            s.connection(),
            "index_chunk",
            {"video_id": video_id, "chunk_index": idx},
            idempotency_key=f"index:{video_id}:{idx}",
            queue="ml",
            priority=priority.chunk_priority(band, idx),
            video_id=video_id,
        )
        s.commit()


def _record_chunk_error(ctx: JobContext, chunk_id, e: Exception) -> None:
    from scenepeek.pipeline.index import record_chunk_error

    record_chunk_error(ctx, chunk_id, e)
