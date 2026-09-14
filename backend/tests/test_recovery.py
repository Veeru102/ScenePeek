"""Chaos test: a worker dies mid-job, the reaper requeues it, and the retry leaves no duplicates."""

import uuid

import pytest
from sqlalchemy import select, text
from sqlalchemy.orm import Session

from scenepeek.jobs import queue as q
from scenepeek.jobs.registry import job
from scenepeek.jobs.worker import Worker
from scenepeek.models import Utterance, Video, VideoChunk
from scenepeek.models.chunk import ChunkStatus
from scenepeek.models.video import VideoStatus


class _WorkerDied(BaseException):
    """Escapes the worker's `except Exception` — the process is gone, nobody marks the job failed."""


_calls: dict[str, int] = {}


@job("chaos_index", queue="ml")
def _chaos_index(ctx, payload):
    """Mirrors index_chunk's atomic delete+insert; attempt 1 dies after its writes are staged."""
    _calls["n"] = _calls.get("n", 0) + 1
    with ctx.session() as s:
        s.execute(text("DELETE FROM utterances WHERE chunk_id = :c"), {"c": payload["chunk_id"]})
        for i in range(3):
            s.add(
                Utterance(
                    video_id=payload["video_id"],
                    chunk_id=payload["chunk_id"],
                    start_s=i * 10.0,
                    end_s=i * 10.0 + 5,
                    text=f"utt {i}",
                    words=[],
                )
            )
        if _calls["n"] == 1:
            raise _WorkerDied()  # rolls back the open transaction, exactly like a SIGKILL would
        s.commit()


@pytest.fixture
def worker(engine, monkeypatch):
    w = Worker(queues=["ml"], worker_id="chaos-1")
    w.engine.dispose()
    w.engine = engine  # point the worker at the throwaway test database
    monkeypatch.setattr(w, "_maybe_reap", lambda: None)  # reap explicitly in the test
    return w


def test_worker_crash_is_reaped_and_retry_has_no_duplicates(engine, worker):
    _calls.clear()
    with Session(engine) as s:
        video = Video(title="chaos", original_key="k", status=VideoStatus.PROCESSING, chunk_count=1)
        s.add(video)
        s.flush()
        chunk = VideoChunk(video_id=video.id, index=0, start_s=0, end_s=60, status=ChunkStatus.PENDING)
        s.add(chunk)
        s.commit()
        video_id, chunk_id = str(video.id), str(chunk.id)

    with engine.begin() as conn:
        q.enqueue_sync(
            conn,
            "chaos_index",
            {"video_id": video_id, "chunk_id": chunk_id},
            idempotency_key=f"chaos:{chunk_id}",
            queue="ml",
            max_attempts=3,
            video_id=uuid.UUID(video_id),
        )

    # attempt 1: the worker "dies" mid-job
    with pytest.raises(_WorkerDied):
        worker.run(once=True)
    with engine.connect() as conn:
        status, attempts = conn.execute(text("SELECT status, attempts FROM jobs")).one()
        assert (status, attempts) == ("running", 1)
        assert conn.execute(text("SELECT count(*) FROM utterances")).scalar() == 0

    # lease expires, another worker's reaper notices
    with engine.begin() as conn:
        conn.execute(text("UPDATE jobs SET heartbeat_at = now() - interval '10 minutes'"))
        reaped = q.reap(conn, timeout_s=90)
    assert [r["status"] for r in reaped] == ["queued"]

    # attempt 2 succeeds, and the chunk's rows exist exactly once
    worker.run(once=True)
    assert _calls["n"] == 2
    with engine.connect() as conn:
        assert conn.execute(text("SELECT status, attempts FROM jobs")).one() == ("succeeded", 2)
    with Session(engine) as s:
        rows = list(s.scalars(select(Utterance).order_by(Utterance.start_s)))
        assert [u.text for u in rows] == ["utt 0", "utt 1", "utt 2"]
