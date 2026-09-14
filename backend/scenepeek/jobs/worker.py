"""Worker loop: lease -> run handler (with heartbeat thread) -> complete/fail. Also reaps."""

import os
import signal
import socket
import threading
import time
import traceback
import uuid

from prometheus_client import start_http_server
from sqlalchemy import create_engine

from scenepeek.core.config import get_settings
from scenepeek.core.logging import get_logger
from scenepeek.core.metrics import JOB_DURATION, QUEUE_DEPTH, flush_samples_sync, record_sample
from scenepeek.jobs import queue as q
from scenepeek.jobs.registry import JobContext, get_spec, load_handlers

log = get_logger("worker")


class Worker:
    def __init__(self, queues: list[str] | None = None, worker_id: str | None = None):
        self.settings = get_settings()
        self.queues = queues or self.settings.queue_list
        self.worker_id = worker_id or f"{socket.gethostname()}-{os.getpid()}-{uuid.uuid4().hex[:6]}"
        self.engine = create_engine(self.settings.sync_database_url, pool_pre_ping=True)
        self._stop = threading.Event()
        self._last_reap = 0.0

    # -- lifecycle -----------------------------------------------------------------

    def stop(self, *_):
        log.info("stop requested", worker=self.worker_id)
        self._stop.set()

    def run(self, once: bool = False) -> None:
        load_handlers()
        signal.signal(signal.SIGINT, self.stop)
        signal.signal(signal.SIGTERM, self.stop)
        log.info("worker started", worker=self.worker_id, queues=self.queues)
        while not self._stop.is_set():
            self._maybe_reap()
            job = self._lease()
            if job is None:
                if once:
                    return
                self._stop.wait(self.settings.worker_poll_interval_s)
                continue
            self._run_job(job)
            if once:
                return
        self.engine.dispose()

    # -- internals -----------------------------------------------------------------

    def _lease(self):
        with self.engine.begin() as conn:
            return q.lease(conn, self.worker_id, self.queues)

    def _maybe_reap(self):
        now = time.monotonic()
        if now - self._last_reap < 30:
            return
        self._last_reap = now
        with self.engine.begin() as conn:
            reaped = q.reap(conn, self.settings.worker_lease_timeout_s)
            for r in reaped:
                log.warning("reaped stale job", job=str(r["id"]), type=r["type"], now=r["status"])
            for (queue, status), n in q.stats(conn).items():
                QUEUE_DEPTH.labels(queue=queue, status=status).set(n)
        if any(r["status"] == "dead" for r in reaped):
            from scenepeek.pipeline.index import fail_dead_chunk_jobs

            fail_dead_chunk_jobs(self.engine, reaped)

    def _heartbeat_loop(self, job_id: uuid.UUID, stop: threading.Event):
        while not stop.wait(self.settings.worker_heartbeat_s):
            try:
                with self.engine.begin() as conn:
                    q.heartbeat(conn, job_id)
            except Exception as e:  # pragma: no cover
                log.warning("heartbeat failed", error=str(e))

    def _run_job(self, job: dict) -> None:
        jlog = log.bind(job=str(job["id"])[:8], type=job["type"], attempt=job["attempts"])
        jlog.info("job start", payload=job["payload"])
        hb_stop = threading.Event()
        hb = threading.Thread(target=self._heartbeat_loop, args=(job["id"], hb_stop), daemon=True)
        hb.start()
        t0 = time.perf_counter()
        try:
            spec = get_spec(job["type"])
            ctx = JobContext(engine=self.engine, job=job, worker_id=self.worker_id, log=jlog)
            spec.handler(ctx, job["payload"])
        except q.NonRetryableError as e:
            hb_stop.set()
            with self.engine.begin() as conn:
                status = q.fail(conn, job, f"{type(e).__name__}: {e}", retryable=False)
            jlog.error("job dead (non-retryable)", error=str(e), status=status)
        except Exception as e:
            hb_stop.set()
            err = f"{type(e).__name__}: {e}\n{traceback.format_exc()[-2000:]}"
            with self.engine.begin() as conn:
                status = q.fail(conn, job, err)
            jlog.error("job failed", error=str(e), status=status)
        else:
            hb_stop.set()
            with self.engine.begin() as conn:
                q.complete(conn, job)
            jlog.info("job done", seconds=round(time.perf_counter() - t0, 2))
        finally:
            hb_stop.set()
            dt = time.perf_counter() - t0
            JOB_DURATION.labels(type=job["type"]).observe(dt)
            record_sample(f"job.{job['type']}", dt * 1000)
            try:
                with self.engine.begin() as conn:
                    flush_samples_sync(conn, self.worker_id)
            except Exception as e:  # metrics must never fail a job
                log.warning("metric flush failed", error=str(e))


def run_worker(queues: list[str] | None = None, once: bool = False) -> None:
    settings = get_settings()
    port = settings.worker_metrics_port
    for _ in range(10):  # multiple workers on one host: take the next free port
        try:
            start_http_server(port)
            break
        except OSError:
            port += 1
    log.info("metrics listening", port=port)
    Worker(queues=queues).run(once=once)
