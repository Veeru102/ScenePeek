"""Postgres-backed durable job queue.

Leasing uses `FOR UPDATE SKIP LOCKED` so any number of workers can pull concurrently without
double-processing. Running jobs heartbeat; a reaper requeues jobs whose worker went silent.
"""

import random
import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy import text
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import AsyncSession

from scenepeek.core.metrics import JOB_RETRIES, JOBS_TOTAL


class NonRetryableError(Exception):
    """Raise from a handler to send the job straight to `dead`."""


_ENQUEUE_SQL = text(
    """
    INSERT INTO jobs (id, type, queue, payload, idempotency_key, status, priority, max_attempts,
                      run_after, video_id)
    VALUES (:id, :type, :queue, CAST(:payload AS jsonb), :key, 'queued', :priority, :max_attempts,
            :run_after, :video_id)
    ON CONFLICT (idempotency_key) DO NOTHING
    RETURNING id
    """
)

_LEASE_SQL = text(
    """
    UPDATE jobs SET status = 'running', locked_by = :worker, locked_at = now(),
                    heartbeat_at = now(), started_at = COALESCE(started_at, now()),
                    attempts = attempts + 1
    WHERE id = (
        SELECT id FROM jobs
        WHERE status = 'queued' AND queue = ANY(:queues) AND run_after <= now()
        ORDER BY priority DESC, created_at
        FOR UPDATE SKIP LOCKED
        LIMIT 1
    )
    RETURNING id, type, queue, payload, attempts, max_attempts, video_id
    """
)

_HEARTBEAT_SQL = text("UPDATE jobs SET heartbeat_at = now() WHERE id = :id AND status = 'running'")

_COMPLETE_SQL = text(
    "UPDATE jobs SET status = 'succeeded', finished_at = now(), locked_by = NULL WHERE id = :id"
)

_RETRY_SQL = text(
    """
    UPDATE jobs SET status = 'queued', run_after = :run_after, last_error = :error,
                    locked_by = NULL, locked_at = NULL, heartbeat_at = NULL
    WHERE id = :id
    """
)

_DEAD_SQL = text(
    """
    UPDATE jobs SET status = 'dead', finished_at = now(), last_error = :error,
                    locked_by = NULL WHERE id = :id
    """
)

_REAP_SQL = text(
    """
    UPDATE jobs SET status = CASE WHEN attempts >= max_attempts THEN 'dead' ELSE 'queued' END,
                    last_error = 'lease expired (worker ' || COALESCE(locked_by, '?') || ' stopped heartbeating)',
                    locked_by = NULL, locked_at = NULL, heartbeat_at = NULL
    WHERE status = 'running' AND heartbeat_at < now() - make_interval(secs => :timeout)
    RETURNING id, type, status
    """
)

_STATS_SQL = text("SELECT queue, status, count(*) FROM jobs GROUP BY queue, status")


def _params(
    type: str, payload: dict[str, Any], idempotency_key: str, queue: str, priority: int,
    max_attempts: int, run_after: datetime | None, video_id: uuid.UUID | str | None,
) -> dict[str, Any]:
    import json

    return {
        "id": uuid.uuid4(),
        "type": type,
        "queue": queue,
        "payload": json.dumps(payload),
        "key": idempotency_key,
        "priority": priority,
        "max_attempts": max_attempts,
        "run_after": run_after or datetime.now(UTC),
        "video_id": uuid.UUID(str(video_id)) if video_id else None,
    }


async def enqueue(
    session: AsyncSession, type: str, payload: dict[str, Any], *, idempotency_key: str,
    queue: str = "cpu", priority: int = 0, max_attempts: int = 3,
    run_after: datetime | None = None, video_id: uuid.UUID | str | None = None,
) -> uuid.UUID | None:
    """Async enqueue (API side). Returns the new job id or None if the key already existed."""
    r = await session.execute(
        _ENQUEUE_SQL,
        _params(type, payload, idempotency_key, queue, priority, max_attempts, run_after, video_id),
    )
    row = r.first()
    return row[0] if row else None


def enqueue_sync(
    conn: Connection, type: str, payload: dict[str, Any], *, idempotency_key: str,
    queue: str = "cpu", priority: int = 0, max_attempts: int = 3,
    run_after: datetime | None = None, video_id: uuid.UUID | str | None = None,
) -> uuid.UUID | None:
    r = conn.execute(
        _ENQUEUE_SQL,
        _params(type, payload, idempotency_key, queue, priority, max_attempts, run_after, video_id),
    )
    row = r.first()
    return row[0] if row else None


def lease(conn: Connection, worker_id: str, queues: list[str]) -> dict[str, Any] | None:
    row = conn.execute(_LEASE_SQL, {"worker": worker_id, "queues": queues}).mappings().first()
    return dict(row) if row else None


def heartbeat(conn: Connection, job_id: uuid.UUID) -> None:
    conn.execute(_HEARTBEAT_SQL, {"id": job_id})


def complete(conn: Connection, job: dict[str, Any]) -> None:
    conn.execute(_COMPLETE_SQL, {"id": job["id"]})
    JOBS_TOTAL.labels(type=job["type"], status="succeeded").inc()


def backoff_delay(attempts: int, base: float = 5.0, cap: float = 300.0) -> float:
    return min(cap, base * (2 ** max(0, attempts - 1))) + random.uniform(0, 2)


def fail(conn: Connection, job: dict[str, Any], error: str, *, retryable: bool = True) -> str:
    """Mark a job failed. Returns the resulting status ('queued' or 'dead')."""
    error = error[:4000]
    if retryable and job["attempts"] < job["max_attempts"]:
        delay = backoff_delay(job["attempts"])
        conn.execute(
            _RETRY_SQL,
            {"id": job["id"], "error": error,
             "run_after": datetime.now(UTC) + timedelta(seconds=delay)},
        )
        JOB_RETRIES.labels(type=job["type"]).inc()
        JOBS_TOTAL.labels(type=job["type"], status="retried").inc()
        return "queued"
    conn.execute(_DEAD_SQL, {"id": job["id"], "error": error})
    JOBS_TOTAL.labels(type=job["type"], status="dead").inc()
    return "dead"


def reap(conn: Connection, timeout_s: float) -> list[dict[str, Any]]:
    rows = conn.execute(_REAP_SQL, {"timeout": timeout_s}).mappings().all()
    return [dict(r) for r in rows]


def stats(conn: Connection) -> dict[tuple[str, str], int]:
    return {(q, s): n for q, s, n in conn.execute(_STATS_SQL)}
