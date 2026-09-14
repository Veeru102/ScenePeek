"""Import benchmark annotations into Postgres and drive fetching through the job queue."""

import random
from dataclasses import dataclass
from pathlib import Path

from sqlalchemy import func, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from scenepeek.datasets.base import BenchmarkAdapter
from scenepeek.jobs import priority
from scenepeek.jobs import queue as q
from scenepeek.models import Dataset, DatasetQuery, DatasetVideo, Video
from scenepeek.models.dataset import DatasetVideoStatus


@dataclass
class ImportSummary:
    videos: int
    queries: int
    linked: int


def get_or_create_dataset(s: Session, adapter: BenchmarkAdapter) -> Dataset:
    ds = s.scalar(select(Dataset).where(Dataset.name == adapter.name))
    if ds is None:
        ds = Dataset(name=adapter.name, version=adapter.version)
        s.add(ds)
        s.flush()
    return ds


def import_split(
    s: Session,
    adapter: BenchmarkAdapter,
    root: Path,
    split: str,
    *,
    limit: int | None = None,
    seed: int = 0,
    ids: list[str] | None = None,
) -> ImportSummary:
    """Idempotent: re-importing adds nothing. The subset is either an explicit, persisted list of
    video ids (`ids`, the reproducible path: eval/subsets/*.txt) or a seeded random sample of
    `limit` videos; either way every query of a selected video comes along."""
    ds = get_or_create_dataset(s, adapter)
    videos = list(adapter.iter_videos(root, split))
    if ids is not None:
        want = set(ids)
        videos = [v for v in videos if v.external_id in want]
    elif limit is not None and limit < len(videos):
        videos = random.Random(seed).sample(videos, limit)
    keep = {v.external_id for v in videos}
    for v in videos:
        s.execute(
            insert(DatasetVideo)
            .values(
                dataset_id=ds.id,
                external_id=v.external_id,
                split=v.split,
                duration_s=v.duration_s,
                source_url=v.source_url,
                meta=v.meta,
            )
            .on_conflict_do_nothing(constraint="uq_dataset_video")
        )
    s.flush()
    vid_ids = dict(
        s.execute(
            select(DatasetVideo.external_id, DatasetVideo.id).where(
                DatasetVideo.dataset_id == ds.id, DatasetVideo.external_id.in_(keep)
            )
        ).all()
    )
    n_q = 0
    for qs in adapter.iter_queries(root, split):
        if qs.video_external_id not in vid_ids:
            continue
        n_q += 1
        s.execute(
            insert(DatasetQuery)
            .values(
                dataset_id=ds.id,
                dataset_video_id=vid_ids[qs.video_external_id],
                external_id=qs.external_id,
                split=qs.split,
                text=qs.text,
                modality=qs.modality,
                relevant=[list(r) for r in qs.relevant],
                meta=qs.meta,
            )
            .on_conflict_do_nothing(constraint="uq_dataset_query")
        )
    linked = link_library_videos(s, ds)
    s.commit()
    return ImportSummary(videos=len(videos), queries=n_q, linked=linked)


def link_library_videos(s: Session, ds: Dataset) -> int:
    """Attach library videos that already exist: fetched benchmark clips carry the dataset name as
    `source` and the external id as title; local YAML sets match uploads by title."""
    n = 0
    pending = list(
        s.scalars(
            select(DatasetVideo).where(DatasetVideo.dataset_id == ds.id, DatasetVideo.video_id.is_(None))
        )
    )
    if not pending:
        return 0
    by_title: dict[tuple[str, str], Video] = {}
    for v in s.scalars(select(Video)):
        by_title[(v.source, v.title)] = v
    for dv in pending:
        v = by_title.get((ds.name, dv.external_id)) or by_title.get(("upload", dv.meta.get("title", "")))
        if v is None:
            continue
        dv.video_id = v.id
        dv.status = DatasetVideoStatus.INDEXED if v.status == "ready" else DatasetVideoStatus.FETCHED
        n += 1
    return n


def enqueue_fetches(s: Session, ds: Dataset, split: str | None = None, limit: int | None = None) -> int:
    stmt = select(DatasetVideo).where(
        DatasetVideo.dataset_id == ds.id, DatasetVideo.status == DatasetVideoStatus.PENDING
    )
    if split:
        stmt = stmt.where(DatasetVideo.split == split)
    stmt = stmt.order_by(DatasetVideo.created_at)
    if limit:
        stmt = stmt.limit(limit)
    n = 0
    conn = s.connection()
    for dv in s.scalars(stmt):
        if q.enqueue_sync(
            conn,
            "fetch_dataset_video",
            {"dataset": ds.name, "dataset_video_id": str(dv.id)},
            idempotency_key=f"fetch:{ds.name}:{dv.external_id}",
            queue="cpu",
            priority=priority.FETCH,
        ):
            n += 1
    s.commit()
    return n


def status(s: Session, ds: Dataset) -> dict[str, dict[str, int]]:
    rows = s.execute(
        select(DatasetVideo.split, DatasetVideo.status, func.count())
        .where(DatasetVideo.dataset_id == ds.id)
        .group_by(DatasetVideo.split, DatasetVideo.status)
    ).all()
    out: dict[str, dict[str, int]] = {}
    for split, st, n in rows:
        out.setdefault(split, {})[st] = n
    for split, st_n in out.items():
        st_n["queries"] = s.scalar(
            select(func.count())
            .select_from(DatasetQuery)
            .where(DatasetQuery.dataset_id == ds.id, DatasetQuery.split == split)
        )
    return out
