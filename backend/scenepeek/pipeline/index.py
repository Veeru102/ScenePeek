"""index_chunk (ml queue): transcribe, segment, embed, OCR, and commit the chunk atomically."""

from datetime import UTC, datetime
from pathlib import Path

from PIL import Image
from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session

from scenepeek.core import storage
from scenepeek.core.config import get_settings
from scenepeek.core.metrics import INDEXED_VIDEO_SECONDS, TIME_TO_FIRST_SEARCHABLE, VIDEO_INDEX_WALL
from scenepeek.jobs import queue as q
from scenepeek.jobs.registry import JobContext
from scenepeek.ml import ocr as ocr_mod
from scenepeek.ml import siglip, text_embed, whisper
from scenepeek.models import Frame, Segment, Utterance, Video, VideoChunk
from scenepeek.models.chunk import ChunkStatus
from scenepeek.models.video import VideoStatus
from scenepeek.pipeline import media
from scenepeek.pipeline.extract import manifest_key
from scenepeek.pipeline.frames import FrameRef, load_manifest, pick_distinct
from scenepeek.pipeline.segment import plan_segments

AUDIO_PAD_S = 1.0


def index_chunk(ctx: JobContext, payload: dict) -> None:
    settings = get_settings()
    video_id, idx = payload["video_id"], int(payload["chunk_index"])
    with ctx.session() as s:
        chunk = s.scalar(select(VideoChunk).where(VideoChunk.video_id == video_id, VideoChunk.index == idx))
        if chunk is None:
            raise q.NonRetryableError(f"chunk {idx} missing")
        if chunk.status == ChunkStatus.DONE:
            return
        if chunk.extracted_at is None:
            raise RuntimeError("chunk not extracted yet")
        chunk.status, chunk.stage = ChunkStatus.RUNNING, "transcribe"
        s.commit()
        start, end, chunk_id = chunk.start_s, chunk.end_s, chunk.id

    try:
        _index(ctx, video_id, idx, chunk_id, start, end, settings)
    except Exception as e:
        record_chunk_error(ctx, chunk_id, e)
        raise


def _set_stage(ctx: JobContext, chunk_id, stage: str) -> None:
    with ctx.session() as s:
        s.get(VideoChunk, chunk_id).stage = stage
        s.commit()


def _index(ctx, video_id: str, idx: int, chunk_id, start: float, end: float, settings) -> None:
    # 1. transcribe a padded slice, keep words that start inside the chunk
    audio = media.cached(video_id, storage.audio_key(video_id), "audio.wav")
    slice_path = media.workdir(video_id, f"chunk_{idx}") / "audio.wav"
    slice_start = max(0.0, start - AUDIO_PAD_S)
    media.slice_audio(audio, slice_path, slice_start, end + AUDIO_PAD_S)
    utterances = whisper.transcribe(slice_path, offset_s=slice_start)
    for u in utterances:
        u.words = [w for w in u.words if start <= w.s < end]
    utterances = [u for u in utterances if u.words]
    for u in utterances:
        u.start, u.end = u.words[0].s, max(u.words[-1].e, u.words[0].s + 0.2)
        u.text = " ".join(w.w for w in u.words)
    ctx.log.info("transcribed", utterances=len(utterances), words=sum(len(u.words) for u in utterances))

    # 2. segments
    _set_stage(ctx, chunk_id, "segment")
    drafts = plan_segments(
        start,
        end,
        utterances,
        target_s=settings.segment_target_s,
        min_s=settings.segment_min_s,
        max_s=settings.segment_max_s,
    )
    manifest = load_manifest(
        storage.internal_client()
        .get_object(Bucket=storage.bucket(), Key=manifest_key(video_id, idx))["Body"]
        .read()
    )
    seg_frames: list[list[FrameRef]] = []
    for d in drafts:
        inside = [f for f in manifest if d.start_s <= f.t_s < d.end_s]
        if not inside and manifest:
            inside = [min(manifest, key=lambda f: abs(f.t_s - (d.start_s + d.end_s) / 2))]
        seg_frames.append(pick_distinct(inside, settings.keyframes_per_segment))

    # 3. embeddings (batched across the whole chunk)
    _set_stage(ctx, chunk_id, "embed")
    texts = [d.context_text for d in drafts]
    text_vecs = text_embed.embed_passages([t if t else " " for t in texts])
    flat = [(si, f) for si, fs in enumerate(seg_frames) for f in fs]
    images, paths = [], []
    for _, f in flat:
        p = _frame_path(video_id, idx, f)
        paths.append(p)
        images.append(Image.open(p).convert("RGB"))
    vis_vecs = siglip.embed_images(images) if images else []
    for im in images:
        im.close()

    # 4. OCR
    frame_ocr: list[str] = [""] * len(flat)
    frame_boxes: list[list | None] = [None] * len(flat)
    if settings.ocr_enabled and flat:
        _set_stage(ctx, chunk_id, "ocr")
        seen: dict[str, tuple[str, list]] = {}
        for k, ((_, f), p) in enumerate(zip(flat, paths, strict=True)):
            if f.phash in seen:  # identical slide sampled twice
                frame_ocr[k], frame_boxes[k] = seen[f.phash]
                continue
            lines = ocr_mod.ocr_image(str(p))
            frame_ocr[k] = ocr_mod.lines_to_text(lines)
            frame_boxes[k] = [{"t": ln.text, "c": round(ln.conf, 3), "b": ln.box} for ln in lines] or None
            seen[f.phash] = (frame_ocr[k], frame_boxes[k])

    # 5. atomic write
    _set_stage(ctx, chunk_id, "write")
    with ctx.session() as s:
        s.execute(delete(Segment).where(Segment.chunk_id == chunk_id))
        s.execute(delete(Utterance).where(Utterance.chunk_id == chunk_id))
        for u in utterances:
            s.add(
                Utterance(
                    video_id=video_id,
                    chunk_id=chunk_id,
                    start_s=u.start,
                    end_s=u.end,
                    text=u.text,
                    words=[{"w": w.w, "s": round(w.s, 2), "e": round(w.e, 2)} for w in u.words],
                )
            )
        seg_rows: list[Segment] = []
        for si, d in enumerate(drafts):
            ocr_text = _merge_ocr([frame_ocr[k] for k, (sj, _) in enumerate(flat) if sj == si])
            has_text = bool(d.context_text.strip())
            seg = Segment(
                video_id=video_id,
                chunk_id=chunk_id,
                index=idx * 10_000 + d.index,
                start_s=d.start_s,
                end_s=d.end_s,
                text=d.text,
                context_text=d.context_text,
                ocr_text=ocr_text,
                text_embedding=text_vecs[si].tolist() if has_text else None,
                embedding_model=settings.text_embed_model if has_text else None,
                keyframe_key=seg_frames[si][0].key if seg_frames[si] else None,
            )
            s.add(seg)
            seg_rows.append(seg)
        s.flush()
        for k, (si, f) in enumerate(flat):
            s.add(
                Frame(
                    video_id=video_id,
                    segment_id=seg_rows[si].id,
                    t_s=f.t_s,
                    image_key=f.key,
                    visual_embedding=vis_vecs[k].tolist(),
                    ocr_text=frame_ocr[k],
                    ocr_boxes=frame_boxes[k],
                    phash=f.phash,
                )
            )
        now = datetime.now(UTC)
        c = s.get(VideoChunk, chunk_id)
        c.status, c.stage, c.error, c.done_at = ChunkStatus.DONE, None, None, now
        c.transcribed_at = c.embedded_at = now
        c.ocr_at = now if settings.ocr_enabled else None
        video = s.get(Video, video_id)
        if video.first_searchable_at is None:
            video.first_searchable_at = now
            if video.upload_completed_at:
                TIME_TO_FIRST_SEARCHABLE.observe((now - video.upload_completed_at).total_seconds())
        INDEXED_VIDEO_SECONDS.inc(end - start)
        s.commit()
        finalize_if_complete(s, video)
        s.commit()
    ctx.log.info("chunk indexed", segments=len(drafts), frames=len(flat))


def _frame_path(video_id: str, idx: int, f: FrameRef) -> Path:
    """Reuse the local sample if this worker extracted the chunk; otherwise fetch from storage."""
    local = media.workdir(video_id, f"chunk_{idx}", "keyframes") / f"{f.t_ms}.jpg"
    if not local.exists():
        storage.download_file(f.key, local)
    return local


def _merge_ocr(texts: list[str]) -> str:
    seen: set[str] = set()
    out: list[str] = []
    for t in texts:
        for line in t.splitlines():
            key = line.strip().lower()
            if key and key not in seen:
                seen.add(key)
                out.append(line.strip())
    return "\n".join(out)


def finalize_if_complete(s: Session, video: Video) -> None:
    """Once every chunk is terminal, flip the video to ready/failed and queue the timeline."""
    counts = dict(
        s.execute(
            select(VideoChunk.status, func.count())
            .where(VideoChunk.video_id == video.id)
            .group_by(VideoChunk.status)
        ).all()
    )
    total = sum(counts.values())
    terminal = counts.get(ChunkStatus.DONE, 0) + counts.get(ChunkStatus.FAILED, 0)
    if total == 0 or terminal < total or video.status in (VideoStatus.READY, VideoStatus.FAILED):
        return
    now = datetime.now(UTC)
    video.completed_at = now
    if counts.get(ChunkStatus.FAILED, 0):
        video.status = VideoStatus.FAILED
        video.error = f"{counts[ChunkStatus.FAILED]} of {total} chunks failed"
    else:
        video.status = VideoStatus.READY
        video.error = None
        if video.upload_completed_at:
            VIDEO_INDEX_WALL.observe((now - video.upload_completed_at).total_seconds())
    if counts.get(ChunkStatus.DONE, 0):
        q.enqueue_sync(
            s.connection(),
            "build_timeline",
            {"video_id": str(video.id)},
            idempotency_key=f"timeline:{video.id}:{int(now.timestamp())}",
            queue="ml",
            priority=5,
            video_id=video.id,
        )


def record_chunk_error(ctx: JobContext, chunk_id, e: Exception) -> None:
    """Persist the error on the chunk; mark it failed only when the job has no retries left."""
    job = ctx.job
    final = isinstance(e, q.NonRetryableError) or job["attempts"] >= job["max_attempts"]
    with ctx.session() as s:
        c = s.get(VideoChunk, chunk_id)
        c.error = f"{type(e).__name__}: {e}"[:2000]
        c.status = ChunkStatus.FAILED if final else ChunkStatus.PENDING
        if final:
            video = s.get(Video, c.video_id)
            s.commit()
            finalize_if_complete(s, video)
        s.commit()
