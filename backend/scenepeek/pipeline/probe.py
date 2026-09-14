"""probe_video: inspect the upload, produce a web rendition + poster + audio, plan chunks."""

from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy import select

from scenepeek.core import storage
from scenepeek.core.config import get_settings
from scenepeek.jobs import priority
from scenepeek.jobs import queue as q
from scenepeek.jobs.registry import JobContext
from scenepeek.models import Video, VideoChunk
from scenepeek.models.video import VideoStatus
from scenepeek.pipeline import media
from scenepeek.pipeline.chunking import plan_chunks


def probe_video(ctx: JobContext, payload: dict) -> None:
    settings = get_settings()
    video_id = payload["video_id"]
    with ctx.session() as s:
        video = s.get(Video, video_id)
        if video is None:
            raise q.NonRetryableError("video row missing")
        if video.status == VideoStatus.READY:
            return
        video.status = VideoStatus.PROCESSING
        video.processing_started_at = video.processing_started_at or datetime.now(UTC)
        video.error = None
        s.commit()
        original_key = video.original_key

    ext = Path(original_key).suffix or ".bin"
    src = media.cached(video_id, original_key, f"original{ext}")
    info = media.probe(src)
    ctx.log.info("probed", duration=round(info.duration_s, 1), codec=info.vcodec, audio=info.has_audio)

    # Web rendition (skip if a previous attempt already uploaded it)
    web_path = settings.cache_dir / video_id / "web.mp4"
    exists, _ = storage.object_exists(storage.web_key(video_id))
    if not exists:
        if info.web_ready:
            media.remux_faststart(src, web_path)
        else:
            media.transcode_web(src, web_path)
        storage.upload_file(web_path, storage.web_key(video_id), "video/mp4")
    elif not web_path.exists():
        media.cached(video_id, storage.web_key(video_id), "web.mp4")

    audio_path = settings.cache_dir / video_id / "audio.wav"
    exists, _ = storage.object_exists(storage.audio_key(video_id))
    if not exists:
        if info.has_audio:
            media.extract_audio(web_path, audio_path)
        else:
            media.silent_audio(audio_path, info.duration_s)
        storage.upload_file(audio_path, storage.audio_key(video_id), "audio/wav")

    poster_path = settings.cache_dir / video_id / "poster.jpg"
    media.poster(web_path, poster_path, min(5.0, info.duration_s / 2))
    storage.upload_file(poster_path, storage.poster_key(video_id), "image/jpeg")

    windows = plan_chunks(info.duration_s, settings.chunk_seconds)
    with ctx.session() as s:
        video = s.get(Video, video_id)
        video.duration_s = info.duration_s
        video.width, video.height, video.fps, video.codec = info.width, info.height, info.fps, info.vcodec
        video.web_key = storage.web_key(video_id)
        video.poster_key = storage.poster_key(video_id)
        video.chunk_count = len(windows)
        existing = {c.index: c for c in s.scalars(select(VideoChunk).where(VideoChunk.video_id == video.id))}
        for w in windows:
            if w.index not in existing:
                s.add(VideoChunk(video_id=video.id, index=w.index, start_s=w.start_s, end_s=w.end_s))
        s.commit()

        conn = s.connection()
        for w in windows:
            q.enqueue_sync(
                conn,
                "extract_chunk",
                {"video_id": video_id, "chunk_index": w.index},
                idempotency_key=f"extract:{video_id}:{w.index}",
                queue="cpu",
                priority=priority.chunk_priority(video.priority_band, w.index),
                video_id=video_id,
            )
        s.commit()
    ctx.log.info("chunks planned", n=len(windows))
