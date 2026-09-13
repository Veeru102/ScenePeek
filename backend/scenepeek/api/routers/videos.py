import uuid
from datetime import UTC, datetime
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import func, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from scenepeek.api.schemas import (
    ChunkOut,
    TopicOut,
    UploadTarget,
    UtteranceOut,
    VideoCreate,
    VideoDetail,
    VideoOut,
)
from scenepeek.core import storage
from scenepeek.core.db import get_session
from scenepeek.jobs import queue as q
from scenepeek.models import Topic, Utterance, Video, VideoChunk
from scenepeek.models.chunk import ChunkStatus
from scenepeek.models.video import VideoStatus

router = APIRouter()

ALLOWED_EXT = {".mp4", ".mov", ".mkv", ".webm", ".m4v", ".avi"}


def _video_out(v: Video, counts: dict[str, int], topic_count: int = 0) -> dict:
    done = counts.get(ChunkStatus.DONE, 0)
    failed = counts.get(ChunkStatus.FAILED, 0)
    return {
        "id": v.id,
        "title": v.title,
        "status": v.status,
        "error": v.error,
        "duration_s": v.duration_s,
        "width": v.width,
        "height": v.height,
        "size_bytes": v.size_bytes,
        "chunk_count": v.chunk_count,
        "chunks_done": done,
        "chunks_failed": failed,
        "progress": (done / v.chunk_count) if v.chunk_count else 0.0,
        "poster_url": storage.presigned_get(v.poster_key) if v.poster_key else None,
        "playback_url": storage.presigned_get(v.web_key) if v.web_key else None,
        "created_at": v.created_at,
        "upload_completed_at": v.upload_completed_at,
        "first_searchable_at": v.first_searchable_at,
        "completed_at": v.completed_at,
        "topic_count": topic_count,
    }


async def _chunk_counts(s: AsyncSession, video_ids: list[uuid.UUID]) -> dict[uuid.UUID, dict[str, int]]:
    if not video_ids:
        return {}
    rows = await s.execute(
        select(VideoChunk.video_id, VideoChunk.status, func.count())
        .where(VideoChunk.video_id.in_(video_ids))
        .group_by(VideoChunk.video_id, VideoChunk.status)
    )
    out: dict[uuid.UUID, dict[str, int]] = {}
    for vid, status, n in rows:
        out.setdefault(vid, {})[status] = n
    return out


@router.post("", response_model=UploadTarget, status_code=201)
async def create_video(body: VideoCreate, s: AsyncSession = Depends(get_session)):
    ext = Path(body.filename).suffix.lower()
    if ext not in ALLOWED_EXT:
        raise HTTPException(400, f"unsupported extension {ext!r}")
    video = Video(
        title=body.title or Path(body.filename).stem,
        original_key="",
        content_type=body.content_type,
        size_bytes=body.size_bytes,
        status=VideoStatus.UPLOADING,
    )
    s.add(video)
    await s.flush()
    video.original_key = storage.original_key(str(video.id), ext)
    await s.commit()
    return UploadTarget(
        video_id=video.id,
        upload_url=storage.presigned_put(video.original_key, body.content_type),
        key=video.original_key,
    )


@router.post("/{video_id}/complete", response_model=VideoOut)
async def complete_upload(video_id: uuid.UUID, s: AsyncSession = Depends(get_session)):
    video = await s.get(Video, video_id)
    if video is None:
        raise HTTPException(404, "video not found")
    exists, size = storage.object_exists(video.original_key)
    if not exists:
        raise HTTPException(400, "upload not found in storage")
    video.size_bytes = size
    video.status = VideoStatus.UPLOADED
    video.upload_completed_at = datetime.now(UTC)
    await q.enqueue(
        s,
        "probe_video",
        {"video_id": str(video.id)},
        idempotency_key=f"probe:{video.id}",
        queue="cpu",
        priority=10,
        video_id=video.id,
    )
    await s.commit()
    return _video_out(video, {})


@router.get("", response_model=list[VideoOut])
async def list_videos(s: AsyncSession = Depends(get_session)):
    videos = list(await s.scalars(select(Video).order_by(Video.created_at.desc())))
    counts = await _chunk_counts(s, [v.id for v in videos])
    topics = dict((await s.execute(select(Topic.video_id, func.count()).group_by(Topic.video_id))).all())
    return [_video_out(v, counts.get(v.id, {}), topics.get(v.id, 0)) for v in videos]


@router.get("/{video_id}", response_model=VideoDetail)
async def get_video(video_id: uuid.UUID, s: AsyncSession = Depends(get_session)):
    video = await s.get(Video, video_id)
    if video is None:
        raise HTTPException(404, "video not found")
    chunks = list(
        await s.scalars(select(VideoChunk).where(VideoChunk.video_id == video_id).order_by(VideoChunk.index))
    )
    counts: dict[str, int] = {}
    for c in chunks:
        counts[c.status] = counts.get(c.status, 0) + 1
    topic_count = await s.scalar(select(func.count()).select_from(Topic).where(Topic.video_id == video_id))
    out = _video_out(video, counts, topic_count or 0)
    out["chunks"] = [
        ChunkOut(
            index=c.index, start_s=c.start_s, end_s=c.end_s, status=c.status, stage=c.stage, error=c.error
        )
        for c in chunks
    ]
    return out


@router.get("/{video_id}/transcript", response_model=list[UtteranceOut])
async def get_transcript(video_id: uuid.UUID, s: AsyncSession = Depends(get_session)):
    rows = await s.scalars(
        select(Utterance).where(Utterance.video_id == video_id).order_by(Utterance.start_s)
    )
    return [UtteranceOut(start_s=u.start_s, end_s=u.end_s, text=u.text, words=u.words) for u in rows]


@router.get("/{video_id}/timeline", response_model=list[TopicOut])
async def get_timeline(video_id: uuid.UUID, s: AsyncSession = Depends(get_session)):
    rows = await s.scalars(select(Topic).where(Topic.video_id == video_id).order_by(Topic.index))
    return [
        TopicOut(index=t.index, start_s=t.start_s, end_s=t.end_s, label=t.label, keyphrases=t.keyphrases)
        for t in rows
    ]


@router.post("/{video_id}/retry", response_model=VideoOut)
async def retry_video(video_id: uuid.UUID, s: AsyncSession = Depends(get_session)):
    """Re-queue dead jobs for this video and reset failed chunks."""
    video = await s.get(Video, video_id)
    if video is None:
        raise HTTPException(404, "video not found")
    await s.execute(
        text(
            "UPDATE jobs SET status='queued', attempts=0, run_after=now(), last_error=NULL, "
            "locked_by=NULL, heartbeat_at=NULL WHERE video_id=:vid AND status IN ('dead','failed')"
        ),
        {"vid": video_id},
    )
    await s.execute(
        text("UPDATE video_chunks SET status='pending', error=NULL WHERE video_id=:vid AND status='failed'"),
        {"vid": video_id},
    )
    if video.chunk_count == 0:
        await q.enqueue(
            s,
            "probe_video",
            {"video_id": str(video.id)},
            idempotency_key=f"probe:{video.id}:retry:{datetime.now(UTC).timestamp():.0f}",
            queue="cpu",
            priority=10,
            video_id=video.id,
        )
    video.status = VideoStatus.PROCESSING
    video.error = None
    await s.commit()
    counts = await _chunk_counts(s, [video.id])
    return _video_out(video, counts.get(video.id, {}))


@router.delete("/{video_id}", status_code=204)
async def delete_video(video_id: uuid.UUID, s: AsyncSession = Depends(get_session)):
    video = await s.get(Video, video_id)
    if video is None:
        raise HTTPException(404, "video not found")
    await s.delete(video)
    await s.commit()
    storage.delete_prefix(f"videos/{video_id}/")
