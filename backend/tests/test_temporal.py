"""Temporal lane: window sampling, versioned embeddings SQL, backfill enqueue (fake vectors, no ML)."""

import asyncio

import numpy as np
import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import Session

from scenepeek.jobs import priority
from scenepeek.jobs import queue as q
from scenepeek.ml.video import sample_windows
from scenepeek.models import Embedding, Segment, Video, VideoChunk
from scenepeek.models.embedding import hnsw_index_sql
from scenepeek.pipeline import temporal
from scenepeek.search.candidates import by_temporal


def test_sample_windows_cover_chunk_with_fixed_frame_count():
    times = [float(t) for t in range(60)]  # 1 fps over a 60 s chunk
    w = sample_windows(times, window_s=8, stride_s=4, n_frames=8)
    assert w[0][:2] == (0.0, 8.0) and len(w[0][2]) == 8 and w[0][2] == list(range(8))
    assert w[-1][1] == 60.0 and all(len(x[2]) == 8 for x in w)
    assert len(w) == 14
    # a chunk shorter than one window still yields one window with repeated frames
    short = sample_windows([0.0, 1.0, 2.0], 8, 4, 8)
    assert len(short) == 1 and len(short[0][2]) == 8 and set(short[0][2]) <= {0, 1, 2}
    assert sample_windows([], 8, 4, 8) == []


def _seed(engine):
    with Session(engine) as s:
        v = Video(title="t", original_key="k", status="ready")
        s.add(v)
        s.flush()
        c = VideoChunk(video_id=v.id, index=0, start_s=0, end_s=60, status="done")
        s.add(c)
        s.flush()
        segs = []
        for i in range(3):
            sg = Segment(video_id=v.id, chunk_id=c.id, index=i, start_s=i * 20.0, end_s=(i + 1) * 20.0)
            s.add(sg)
            segs.append(sg)
        s.flush()
        rows = []
        for i, sg in enumerate(segs):
            for j, vec in enumerate(([1, 0, 0, 0], [0, 1, 0, 0])):
                # cosine is scale-invariant, so the non-target segments point slightly off-axis
                v4 = np.array(vec, dtype=np.float32) + (0.0 if i == 1 else 0.3) + (0.01 * j)
                rows.append(
                    Embedding(
                        video_id=v.id,
                        segment_id=sg.id,
                        chunk_id=c.id,
                        kind="temporal",
                        model_key="fake",
                        start_s=sg.start_s + 4 * j,
                        end_s=sg.start_s + 4 * j + 8,
                        embedding=v4.tolist(),
                    )
                )
            rows.append(  # a different model version must be invisible to the 'fake' lane
                Embedding(
                    video_id=v.id,
                    segment_id=sg.id,
                    chunk_id=c.id,
                    kind="temporal",
                    model_key="other",
                    start_s=sg.start_s,
                    end_s=sg.start_s + 8,
                    embedding=[1, 0, 0, 0],
                )
            )
        s.add_all(rows)
        s.commit()
        return v.id, c.id, [sg.id for sg in segs]


def test_by_temporal_max_pools_windows_to_segments_and_filters_model(engine, test_db_url):
    vid, _, seg_ids = _seed(engine)
    with engine.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
        conn.execute(text(hnsw_index_sql("temporal", "fake", 4)))

    async def run():
        aeng = create_async_engine(test_db_url.replace("+psycopg", "+asyncpg"))
        async with AsyncSession(aeng) as s:
            out = await by_temporal(s, np.array([1, 0, 0, 0], dtype=np.float32), 10, "fake", [vid])
            none = await by_temporal(s, np.array([1, 0, 0, 0], dtype=np.float32), 10, "missing", None)
        await aeng.dispose()
        return out, none

    out, none = asyncio.run(run())
    assert none == []
    assert len(out) == 3 and out[0].segment_id == seg_ids[1]  # the unscaled segment wins
    assert out[0].span == (20.0, 28.0)  # the best window, not the segment bounds
    assert out[0].score == pytest.approx(1.0, abs=1e-2) and out[1].score < out[0].score


def test_backfill_enqueues_only_missing_chunks_at_backfill_priority(engine, monkeypatch):
    vid, chunk_id, _ = _seed(engine)
    with Session(engine) as s:
        c2 = VideoChunk(video_id=vid, index=1, start_s=60, end_s=120, status="done")
        c3 = VideoChunk(video_id=vid, index=2, start_s=120, end_s=180, status="failed")
        s.add_all([c2, c3])
        s.commit()

    class Fake:
        key, dim, n_frames = "fake", 4, 8

    monkeypatch.setattr(temporal, "get_encoder", lambda name=None: Fake())
    assert temporal.backfill(engine) == 1  # chunk 0 has rows, chunk 2 failed -> only chunk 1
    assert temporal.backfill(engine) == 0  # idempotent
    with engine.begin() as conn:
        job = q.lease(conn, "w", ["vision"])
        assert job["type"] == "encode_temporal" and job["payload"]["chunk_index"] == 1
        pri = conn.execute(text("select priority from jobs where id=:id"), {"id": job["id"]}).scalar()
        assert pri == priority.chunk_priority(priority.BACKFILL, 1) - 10
    assert temporal.backfill(engine, force=True) == 2


def test_hnsw_index_sql_is_partial_and_cast():
    sql = hnsw_index_sql("temporal", "xclip-base-patch32", 512)
    assert (
        "embedding::vector(512)" in sql
        and "WHERE kind = 'temporal' AND model_key = 'xclip-base-patch32'" in sql
    )
