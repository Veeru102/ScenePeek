"""Benchmark adapters + importer + priority bands (DB-backed, no ML)."""

import json
from pathlib import Path

from sqlalchemy import select, text
from sqlalchemy.orm import Session

from scenepeek.datasets.importer import enqueue_fetches, get_or_create_dataset, import_split, status
from scenepeek.datasets.qvhighlights import QVHighlights, parse_vid
from scenepeek.jobs import priority
from scenepeek.jobs import queue as q
from scenepeek.models import DatasetQuery, DatasetVideo, Video
from scenepeek.models.dataset import DatasetVideoStatus

ROWS = [
    {
        "qid": 1,
        "query": "a man walks a dog",
        "vid": "abc_DEF-1_210.0_360.0",
        "duration": 150,
        "relevant_windows": [[10, 20], [40, 48]],
        "saliency_scores": [[1, 2, 3]],
    },
    {
        "qid": 2,
        "query": "cooking pasta",
        "vid": "abc_DEF-1_210.0_360.0",
        "duration": 150,
        "relevant_windows": [[100, 120]],
    },
    {
        "qid": 3,
        "query": "a crowd cheers",
        "vid": "zzz9_0.0_150.0",
        "duration": 150,
        "relevant_windows": [[0, 6]],
    },
    {
        "qid": 4,
        "query": "skateboard trick",
        "vid": "qqq_60.0_210.0",
        "duration": 150,
        "relevant_windows": [[70, 90]],
    },
]


def _root(tmp_path: Path) -> Path:
    (tmp_path / "highlight_val_release.jsonl").write_text("\n".join(json.dumps(r) for r in ROWS) + "\n")
    return tmp_path


def test_parse_vid_keeps_underscores_in_youtube_id():
    assert parse_vid("abc_DEF-1_210.0_360.0") == ("abc_DEF-1", 210.0, 360.0)


def test_qvh_adapter_dedupes_videos_and_keeps_windows(tmp_path):
    a = QVHighlights()
    videos = list(a.iter_videos(_root(tmp_path), "val"))
    assert [v.external_id for v in videos] == ["abc_DEF-1_210.0_360.0", "zzz9_0.0_150.0", "qqq_60.0_210.0"]
    assert videos[0].meta == {"youtube_id": "abc_DEF-1", "clip_start_s": 210.0, "clip_end_s": 360.0}
    queries = list(a.iter_queries(tmp_path, "val"))
    assert queries[0].relevant == [(10.0, 20.0), (40.0, 48.0)]
    assert queries[0].meta["saliency_scores"] == [[1, 2, 3]]


def test_import_is_idempotent_and_subset_keeps_whole_videos(engine, tmp_path):
    root = _root(tmp_path)
    with Session(engine) as s:
        r = import_split(s, QVHighlights(), root, "val", limit=2, seed=0)
        assert r.videos == 2 and r.queries in (2, 3)
        r2 = import_split(s, QVHighlights(), root, "val", limit=2, seed=0)
        assert r2.videos == 2
        n_v = s.scalar(select(text("count(*)")).select_from(DatasetVideo))
        n_q = s.scalar(select(text("count(*)")).select_from(DatasetQuery))
        assert n_v == 2 and n_q == r.queries
        # every imported query points at an imported video (no dangling subset edges)
        vids = {v.id for v in s.scalars(select(DatasetVideo))}
        assert all(qq.dataset_video_id in vids for qq in s.scalars(select(DatasetQuery)))


def test_fetch_enqueue_uses_benchmark_priority_and_links_existing_videos(engine, tmp_path):
    root = _root(tmp_path)
    with Session(engine) as s:
        # a clip already in the library (fetched earlier) is linked instead of re-fetched
        v = Video(title="zzz9_0.0_150.0", original_key="k", status="ready", source="qvhighlights")
        s.add(v)
        s.commit()
        r = import_split(s, QVHighlights(), root, "val")
        assert r.linked == 1
        ds = get_or_create_dataset(s, QVHighlights())
        assert enqueue_fetches(s, ds) == 2  # the other two are pending
        st = status(s, ds)
        assert st["val"][DatasetVideoStatus.INDEXED] == 1 and st["val"]["queries"] == 4
    with engine.begin() as conn:
        job = q.lease(conn, "w", ["cpu"])
        assert job["type"] == "fetch_dataset_video" and job["payload"]["dataset"] == "qvhighlights"
        pri = conn.execute(text("select priority from jobs where id=:id"), {"id": job["id"]}).scalar()
        assert pri == priority.BENCHMARK


def test_interactive_chunks_lease_before_benchmark_chunks(engine):
    with engine.begin() as conn:
        for idx in range(3):
            q.enqueue_sync(
                conn,
                "noop",
                {"k": f"bench{idx}"},
                idempotency_key=f"b{idx}",
                priority=priority.chunk_priority(priority.BENCHMARK, idx),
            )
        for idx in range(3):
            q.enqueue_sync(
                conn,
                "noop",
                {"k": f"user{idx}"},
                idempotency_key=f"u{idx}",
                priority=priority.chunk_priority(priority.INTERACTIVE, idx),
            )
    order = []
    with engine.begin() as conn:
        while (j := q.lease(conn, "w", ["cpu"])) is not None:
            order.append(j["payload"]["k"])
    assert order == ["user0", "user1", "user2", "bench0", "bench1", "bench2"]


def test_reserved_worker_and_offline_budget(engine):
    with engine.begin() as conn:
        q.enqueue_sync(conn, "noop", {"k": "bench_a"}, idempotency_key="ba", priority=priority.BENCHMARK)
        q.enqueue_sync(conn, "noop", {"k": "bench_b"}, idempotency_key="bb", priority=priority.BENCHMARK)
        q.enqueue_sync(conn, "noop", {"k": "backfill"}, idempotency_key="bf", priority=priority.BACKFILL)
    with engine.begin() as conn:
        # a reserved interactive worker sees nothing to do
        assert q.lease(conn, "reserved", ["cpu"], min_priority=0) is None
        # with a budget of one offline job, the second offline lease waits until the first finishes
        first = q.lease(conn, "w1", ["cpu"], offline_budget=1)
        assert first["payload"]["k"] == "bench_a"
        assert q.lease(conn, "w2", ["cpu"], offline_budget=1) is None
        q.enqueue_sync(conn, "noop", {"k": "user"}, idempotency_key="u", priority=priority.INTERACTIVE)
        # interactive work is never held back by the offline budget
        assert q.lease(conn, "w2", ["cpu"], offline_budget=1)["payload"]["k"] == "user"
        q.complete(conn, first)
        assert q.lease(conn, "w2", ["cpu"], offline_budget=1)["payload"]["k"] == "bench_b"
