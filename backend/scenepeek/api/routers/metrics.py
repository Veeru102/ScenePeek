"""System metrics summary computed from Postgres (no Prometheus needed for the in-app page)."""

from datetime import UTC, datetime, timedelta

from fastapi import APIRouter, Depends, Query
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from scenepeek.core.db import get_session
from scenepeek.core.metrics import flush_samples_async

router = APIRouter()


def _pct(values: list[float], p: float) -> float | None:
    if not values:
        return None
    v = sorted(values)
    return round(v[min(len(v) - 1, int(len(v) * p))], 1)


@router.get("/summary")
async def summary(window_min: int = Query(30, ge=1, le=1440), s: AsyncSession = Depends(get_session)):
    await flush_samples_async(s, "api")
    since = datetime.now(UTC) - timedelta(minutes=window_min)

    jobs = {
        f"{q}.{st}": n
        for q, st, n in await s.execute(
            text("SELECT queue, status, count(*) FROM jobs GROUP BY queue, status")
        )
    }
    job_totals: dict[str, int] = {}
    for k, n in jobs.items():
        st = k.split(".", 1)[1]
        job_totals[st] = job_totals.get(st, 0) + n

    videos = dict((await s.execute(text("SELECT status, count(*) FROM videos GROUP BY status"))).all())

    # throughput: video seconds indexed in the window vs. wall time the workers were busy
    row = (
        await s.execute(
            text(
                """
                SELECT COALESCE(SUM(end_s - start_s), 0), MIN(done_at), MAX(done_at), COUNT(*)
                FROM video_chunks WHERE status = 'done' AND done_at >= :since
                """
            ),
            {"since": since},
        )
    ).one()
    indexed_s, first_done, last_done, n_chunks = float(row[0]), row[1], row[2], int(row[3])
    busy_s = (
        await s.execute(
            text(
                "SELECT COALESCE(SUM(EXTRACT(EPOCH FROM (finished_at - started_at))), 0) FROM jobs "
                "WHERE status = 'succeeded' AND finished_at >= :since "
                "AND type IN ('extract_chunk', 'index_chunk')"
            ),
            {"since": since},
        )
    ).scalar()
    wall_s = (last_done - first_done).total_seconds() if first_done and last_done and n_chunks > 1 else None
    throughput = {
        "indexed_video_seconds": round(indexed_s, 1),
        "chunks": n_chunks,
        "worker_busy_seconds": round(float(busy_s or 0), 1),
        "realtime_factor": round(indexed_s / float(busy_s), 2) if busy_s and float(busy_s) > 0 else None,
        "wall_seconds": round(wall_s, 1) if wall_s else None,
        "seconds_to_index_one_hour": round(3600 / (indexed_s / float(busy_s)), 0)
        if busy_s and float(busy_s) > 0 and indexed_s > 0
        else None,
    }

    ttfs = [
        float(x)
        for (x,) in await s.execute(
            text(
                "SELECT EXTRACT(EPOCH FROM (first_searchable_at - upload_completed_at)) FROM videos "
                "WHERE first_searchable_at IS NOT NULL AND upload_completed_at IS NOT NULL"
            )
        )
    ]
    walls = [
        (float(w), float(d))
        for w, d in await s.execute(
            text(
                "SELECT EXTRACT(EPOCH FROM (completed_at - upload_completed_at)), duration_s FROM videos "
                "WHERE status = 'ready' AND completed_at IS NOT NULL AND upload_completed_at IS NOT NULL "
                "AND duration_s > 0"
            )
        )
    ]
    per_hour = [w / d * 3600 for w, d in walls]

    samples: dict[str, list[float]] = {}
    for name, value in await s.execute(
        text("SELECT name, value FROM metric_samples WHERE ts >= :since ORDER BY id DESC LIMIT 20000"),
        {"since": since},
    ):
        samples.setdefault(name, []).append(float(value))
    inference = {
        k.removeprefix("infer."): {"p50_ms": _pct(v, 0.5), "p95_ms": _pct(v, 0.95), "n": len(v)}
        for k, v in samples.items()
        if k.startswith("infer.")
    }
    job_lat = {
        k.removeprefix("job."): {"p50_ms": _pct(v, 0.5), "p95_ms": _pct(v, 0.95), "n": len(v)}
        for k, v in samples.items()
        if k.startswith("job.")
    }
    search_v = samples.get("search.total", [])

    workers = [
        {"id": w, "last_seen": last.isoformat(), "jobs": int(n), "running": int(running)}
        for w, last, n, running in await s.execute(
            text(
                """
                SELECT locked_by, MAX(COALESCE(finished_at, heartbeat_at, started_at)), COUNT(*),
                       COUNT(*) FILTER (WHERE status = 'running')
                FROM jobs
                WHERE locked_by IS NOT NULL AND COALESCE(finished_at, heartbeat_at, started_at) >= :since
                GROUP BY locked_by ORDER BY 2 DESC
                """
            ),
            {"since": since},
        )
    ]
    failures = [
        dict(r._mapping)
        for r in await s.execute(
            text(
                "SELECT id, type, status, attempts, max_attempts, video_id, last_error, created_at, "
                "finished_at FROM jobs "
                "WHERE last_error IS NOT NULL ORDER BY COALESCE(finished_at, created_at) DESC LIMIT 10"
            )
        )
    ]
    retries = (await s.execute(text("SELECT COALESCE(SUM(GREATEST(attempts - 1, 0)), 0) FROM jobs"))).scalar()

    return {
        "window_min": window_min,
        "jobs": {"by_status": job_totals, "by_queue_status": jobs, "retries_total": int(retries or 0)},
        "videos": videos,
        "throughput": throughput,
        "time_to_first_searchable_s": {"p50": _pct(ttfs, 0.5), "p95": _pct(ttfs, 0.95), "n": len(ttfs)},
        "index_wall_s": {
            "p50": _pct([w for w, _ in walls], 0.5),
            "p95": _pct([w for w, _ in walls], 0.95),
            "n": len(walls),
            "per_hour_s": _pct(per_hour, 0.5),
        },
        "search": {"p50_ms": _pct(search_v, 0.5), "p95_ms": _pct(search_v, 0.95), "n": len(search_v)},
        "inference": inference,
        "job_latency": job_lat,
        "workers": workers,
        "recent_failures": failures,
    }
