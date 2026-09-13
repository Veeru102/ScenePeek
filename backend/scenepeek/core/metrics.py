"""Prometheus metrics shared by API and worker, plus DB-backed timing samples."""

import threading as _threading

from prometheus_client import Counter, Gauge, Histogram

JOBS_TOTAL = Counter("scenepeek_jobs_total", "Jobs finished by type and outcome", ["type", "status"])
JOB_RETRIES = Counter("scenepeek_job_retries_total", "Job retries", ["type"])
JOB_DURATION = Histogram(
    "scenepeek_job_duration_seconds",
    "Job wall time",
    ["type"],
    buckets=(1, 2, 5, 10, 20, 30, 60, 120, 300, 600),
)
QUEUE_DEPTH = Gauge("scenepeek_queue_depth", "Jobs by queue and status", ["queue", "status"])

MODEL_INFERENCE = Histogram(
    "scenepeek_model_inference_seconds",
    "Model inference latency",
    ["model"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60),
)
INDEXED_VIDEO_SECONDS = Counter("scenepeek_indexed_video_seconds_total", "Seconds of video fully indexed")
TIME_TO_FIRST_SEARCHABLE = Histogram(
    "scenepeek_time_to_first_searchable_seconds",
    "Upload complete -> first chunk searchable",
    buckets=(5, 10, 20, 30, 60, 120, 300, 600),
)
VIDEO_INDEX_WALL = Histogram(
    "scenepeek_video_index_wall_seconds",
    "Upload complete -> fully indexed",
    buckets=(30, 60, 120, 300, 600, 1200, 1800, 3600),
)

SEARCH_LATENCY = Histogram(
    "scenepeek_search_latency_seconds",
    "Search latency by stage",
    ["stage"],
    buckets=(0.01, 0.025, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1, 2, 5),
)


# ---- DB-backed samples (exact percentiles for the in-app System page) ------------------

_SAMPLES: list[tuple[str, float]] = []
_SAMPLES_LOCK = _threading.Lock()


def record_sample(name: str, value: float) -> None:
    """Buffer a sample in-process; the owner flushes with `drain_samples()`."""
    with _SAMPLES_LOCK:
        _SAMPLES.append((name, value))
        if len(_SAMPLES) > 10_000:
            del _SAMPLES[:5000]


def drain_samples() -> list[tuple[str, float]]:
    with _SAMPLES_LOCK:
        out = list(_SAMPLES)
        _SAMPLES.clear()
    return out


def flush_samples_sync(conn, source: str) -> int:
    from sqlalchemy import text

    rows = drain_samples()
    if rows:
        conn.execute(
            text("INSERT INTO metric_samples (name, value, source) VALUES (:n, :v, :s)"),
            [{"n": n, "v": v, "s": source} for n, v in rows],
        )
    return len(rows)


async def flush_samples_async(session, source: str) -> int:
    from sqlalchemy import text

    rows = drain_samples()
    if rows:
        await session.execute(
            text("INSERT INTO metric_samples (name, value, source) VALUES (:n, :v, :s)"),
            [{"n": n, "v": v, "s": source} for n, v in rows],
        )
        await session.commit()
    return len(rows)
