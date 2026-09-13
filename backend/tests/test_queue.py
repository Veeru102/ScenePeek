import threading
import uuid
from datetime import UTC, timedelta

from sqlalchemy import text

from scenepeek.jobs import queue as q


def _enqueue(engine, key, **kw):
    with engine.begin() as conn:
        return q.enqueue_sync(conn, "noop", {"k": key}, idempotency_key=key, **kw)


def test_enqueue_is_idempotent(engine):
    assert _enqueue(engine, "a") is not None
    assert _enqueue(engine, "a") is None
    with engine.connect() as conn:
        assert conn.execute(text("select count(*) from jobs")).scalar() == 1


def test_lease_respects_priority_and_queue(engine):
    _enqueue(engine, "low", priority=0, queue="cpu")
    _enqueue(engine, "high", priority=5, queue="cpu")
    _enqueue(engine, "ml", priority=9, queue="ml")
    with engine.begin() as conn:
        j = q.lease(conn, "w1", ["cpu"])
    assert j["payload"]["k"] == "high"
    assert j["attempts"] == 1
    with engine.begin() as conn:
        assert q.lease(conn, "w1", ["cpu"])["payload"]["k"] == "low"
        assert q.lease(conn, "w1", ["cpu"]) is None
        assert q.lease(conn, "w1", ["ml"])["payload"]["k"] == "ml"


def test_concurrent_lease_never_double_leases(engine):
    for i in range(20):
        _enqueue(engine, f"j{i}")
    seen: list[uuid.UUID] = []
    lock = threading.Lock()

    def drain(worker):
        while True:
            with engine.begin() as conn:
                j = q.lease(conn, worker, ["cpu"])
            if j is None:
                return
            with lock:
                seen.append(j["id"])

    threads = [threading.Thread(target=drain, args=(f"w{i}",)) for i in range(4)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert len(seen) == 20 and len(set(seen)) == 20


def test_fail_retries_with_backoff_then_dies(engine):
    _enqueue(engine, "flaky", max_attempts=2)
    with engine.begin() as conn:
        j = q.lease(conn, "w", ["cpu"])
        assert q.fail(conn, j, "boom") == "queued"
    with engine.connect() as conn:
        row = conn.execute(text("select status, run_after > now(), last_error from jobs")).one()
        assert row[0] == "queued" and row[1] is True and "boom" in row[2]
        # not leasable until run_after passes
    with engine.begin() as conn:
        assert q.lease(conn, "w", ["cpu"]) is None
        conn.execute(text("update jobs set run_after = now()"))
    with engine.begin() as conn:
        j = q.lease(conn, "w", ["cpu"])
        assert j["attempts"] == 2
        assert q.fail(conn, j, "boom again") == "dead"


def test_non_retryable_goes_dead_immediately(engine):
    _enqueue(engine, "corrupt")
    with engine.begin() as conn:
        j = q.lease(conn, "w", ["cpu"])
        assert q.fail(conn, j, "bad media", retryable=False) == "dead"


def test_reaper_requeues_stale_running_jobs(engine):
    _enqueue(engine, "stale")
    _enqueue(engine, "fresh")
    with engine.begin() as conn:
        a = q.lease(conn, "crashed", ["cpu"])
        q.lease(conn, "alive", ["cpu"])
        conn.execute(
            text("update jobs set heartbeat_at = now() - interval '10 minutes' where id = :id"),
            {"id": a["id"]},
        )
    with engine.begin() as conn:
        reaped = q.reap(conn, timeout_s=90)
    assert [r["status"] for r in reaped] == ["queued"]
    with engine.begin() as conn:
        j = q.lease(conn, "w2", ["cpu"])
        assert j["id"] == a["id"] and j["attempts"] == 2


def test_backoff_grows_and_caps():
    assert q.backoff_delay(1) < q.backoff_delay(3) < q.backoff_delay(10)
    assert q.backoff_delay(20) <= 302


def test_run_after_in_future_is_not_leased(engine):
    from datetime import datetime

    _enqueue(engine, "later", run_after=datetime.now(UTC) + timedelta(hours=1))
    with engine.begin() as conn:
        assert q.lease(conn, "w", ["cpu"]) is None
