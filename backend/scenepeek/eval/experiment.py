"""Experiments: one reproducible evaluation run = dataset split + full SearchConfig + model versions.

The run is recorded in Postgres (`experiments` / `experiment_results`) and mirrored to
`eval/reports/<name>.json` in the shape the compare/README tooling already reads."""

import asyncio
import json
import random
import statistics
import subprocess
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import yaml
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from scenepeek.core.config import get_settings
from scenepeek.core.db import get_sessionmaker
from scenepeek.core.logging import get_logger
from scenepeek.datasets.importer import get_or_create_dataset
from scenepeek.datasets.registry import get_adapter
from scenepeek.eval.metrics import (
    Range,
    hit_at,
    lane_attribution,
    mrr,
    ndcg_at,
    recall_at,
    relevance,
    temporal_metrics,
)
from scenepeek.eval.runner import apply_overrides
from scenepeek.models import DatasetQuery, DatasetVideo, Experiment, ExperimentResult, IndexVersion, Video
from scenepeek.models.experiment import ExperimentStatus
from scenepeek.search.config import SearchConfig
from scenepeek.search.service import SearchOptions, search

log = get_logger("experiment")
KS = (1, 5, 10)
OVERLAP_KEYS = ["mrr", "ndcg@10"] + [f"recall@{k}" for k in KS] + [f"hit@{k}" for k in KS]
TEMPORAL_KEYS = ["r1@0.5", "r1@0.7", "r5@0.5", "r5@0.7", "map@0.5", "map@0.75", "map"]


@dataclass
class ExperimentSpec:
    name: str
    dataset: str
    split: str = "test"
    subset: dict = field(default_factory=dict)  # {limit, seed} (by video) or {ids: [...]}
    scope: str = "dataset"  # video: the query's own clip | dataset: the split's clips | corpus: everything
    limit: int = 10
    tolerance_s: float = 3.0
    search: dict = field(default_factory=dict)  # SearchConfig overrides
    settings: dict = field(default_factory=dict)  # process settings overrides (e.g. ollama_enabled)
    group_by: str = "modality"

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ExperimentSpec":
        p = Path(path)
        data = yaml.safe_load(p.read_text()) or {}
        data.setdefault("name", p.stem)
        return cls(**data)


@dataclass
class _Q:
    id: uuid.UUID
    key: str
    text: str
    modality: str
    video_key: str
    video_id: uuid.UUID
    relevant: list[Range]
    meta: dict


def _code_sha() -> str | None:
    try:
        return (
            subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=5
            ).stdout.strip()
            or None
        )
    except Exception:
        return None


def _sync_session() -> Session:
    return Session(create_engine(get_settings().sync_database_url), expire_on_commit=False)


def load_queries(s: Session, spec: ExperimentSpec) -> tuple[list[_Q], list[uuid.UUID], uuid.UUID]:
    ds = get_or_create_dataset(s, get_adapter(spec.dataset))
    rows = s.execute(
        select(DatasetQuery, DatasetVideo, Video)
        .join(DatasetVideo, DatasetQuery.dataset_video_id == DatasetVideo.id)
        .join(Video, DatasetVideo.video_id == Video.id)
        .where(DatasetQuery.dataset_id == ds.id, DatasetQuery.split == spec.split)
        .order_by(DatasetVideo.external_id, DatasetQuery.external_id)
    ).all()
    videos = sorted({dv.external_id for _, dv, _ in rows})
    if "ids" in spec.subset:
        keep_q = set(map(str, spec.subset["ids"]))
        rows = [r for r in rows if r[0].external_id in keep_q]
    elif spec.subset.get("limit") and spec.subset["limit"] < len(videos):
        keep_v = set(random.Random(int(spec.subset.get("seed", 0))).sample(videos, int(spec.subset["limit"])))
        rows = [r for r in rows if r[1].external_id in keep_v]
    skipped = sum(1 for _, _, v in rows if v.status != "ready")
    if skipped:
        log.warning("videos not fully indexed are excluded", n=skipped)
    qs = [
        _Q(
            id=q.id,
            key=q.external_id,
            text=q.text,
            modality=q.modality or "all",
            video_key=dv.external_id,
            video_id=v.id,
            relevant=[Range(dv.external_id, float(a), float(b)) for a, b in q.relevant],
            meta=q.meta or {},
        )
        for q, dv, v in rows
        if v.status == "ready"
    ]
    vids = sorted({q.video_id for q in qs})
    return qs, vids, ds.id


async def _evaluate(
    qs: list[_Q], vids: list[uuid.UUID], spec: ExperimentSpec, cfg: SearchConfig
) -> list[dict]:
    key_of = {}
    with _sync_session() as s:
        for dv in s.scalars(select(DatasetVideo).where(DatasetVideo.video_id.in_(vids))):
            key_of[str(dv.video_id)] = dv.external_id
    rows = []
    async with get_sessionmaker()() as s:
        await search(s, "warm up the models", SearchOptions(limit=1, config=cfg))
        for q in qs:
            if spec.scope == "video":
                restrict = [q.video_id]
            elif spec.scope == "dataset":
                restrict = vids
            else:
                restrict = None
            opts = SearchOptions(limit=spec.limit, video_ids=restrict, debug=True, config=cfg)
            t0 = time.perf_counter()
            res = await search(s, q.text, opts)
            ms = (time.perf_counter() - t0) * 1000
            hits = [Range(key_of.get(str(h.video.id), str(h.video.id)), h.start_s, h.end_s) for h in res.hits]
            rels = relevance(hits, q.relevant, spec.tolerance_s)
            n = len(q.relevant)
            lane_ranges = {
                lane: [Range(key_of.get(str(h.video_id), str(h.video_id)), h.start_s, h.end_s) for h in hs]
                for lane, hs in (res.lanes or {}).items()
            }
            attribution = lane_attribution(lane_ranges, q.relevant, spec.tolerance_s, k=10)
            row = {
                "id": q.key,
                "query_id": str(q.id),
                "text": q.text,
                "modality": q.modality,
                "author": q.meta.get("author"),
                "notes": q.meta.get("notes"),
                "n_relevant": n,
                "latency_ms": round(ms, 1),
                "lanes_run": res.lanes_run,
                "cues": res.plan.cues,
                "lanes": attribution["lanes"],
                "lane_ceiling": attribution["ceiling"],
                "unique_lane": attribution["unique"],
                "mrr": mrr(rels),
                "first_hit_rank": (rels.index(1) + 1) if 1 in rels else None,
                **{f"recall@{k}": recall_at(rels, n, k) for k in KS},
                **{f"hit@{k}": hit_at(rels, k) for k in KS},
                "ndcg@10": ndcg_at(rels, n, 10),
                **temporal_metrics(hits, q.relevant),
                "top": [{"video": h.video, "start_s": h.start_s, "end_s": h.end_s} for h in hits[:3]],
            }
            rows.append(row)
            log.info("query", id=q.key, mrr=round(row["mrr"], 3), r1_iou=row["r1@0.5"], ms=row["latency_ms"])
    return rows


def aggregate(rows: list[dict]) -> dict:
    if not rows:
        return {}
    agg = {
        k: round(statistics.mean(r[k] for r in rows), 4) for k in OVERLAP_KEYS + TEMPORAL_KEYS if k in rows[0]
    }
    lat = sorted(r["latency_ms"] for r in rows)
    agg["latency_p50_ms"] = round(lat[len(lat) // 2], 1)
    agg["latency_p95_ms"] = round(lat[min(len(lat) - 1, int(len(lat) * 0.95))], 1)
    agg["lanes_per_query"] = round(statistics.mean(len(r.get("lanes_run", [])) for r in rows), 2)
    agg["n"] = len(rows)
    lane_names = sorted({lane for r in rows for lane in (r.get("lanes") or {})})
    if lane_names:
        agg["lanes"] = {
            lane: {
                "hit@10": round(
                    statistics.mean(r["lanes"].get(lane, {}).get("hit@10", 0.0) for r in rows), 4
                ),
                "unique": sum(1 for r in rows if r.get("unique_lane") == lane),
            }
            for lane in lane_names
        }
        agg["lane_ceiling"] = round(statistics.mean(1.0 if r.get("lane_ceiling") else 0.0 for r in rows), 4)
    return agg


def _model_versions(s: Session, cfg: SearchConfig) -> dict:
    registry = {
        f"{iv.kind}:{iv.model_key}": iv.status
        for iv in s.scalars(select(IndexVersion).where(IndexVersion.status != "retired"))
    }
    return {"active": cfg.models, "registry": registry}


def run(spec: ExperimentSpec, output: str | None = None, reports_dir: Path | None = None) -> dict:
    cfg = SearchConfig.from_settings().with_overrides(spec.search)
    with _sync_session() as s:
        qs, vids, dataset_id = load_queries(s, spec)
        exp = Experiment(
            name=spec.name,
            dataset_id=dataset_id,
            split=spec.split,
            subset={**spec.subset, "scope": spec.scope},
            config={"spec": asdict(spec), "search": cfg.to_dict()},
            code_sha=_code_sha(),
            model_versions=_model_versions(s, cfg),
            n_queries=len(qs),
        )
        s.add(exp)
        s.commit()
        exp_id = exp.id
    if not qs:
        raise RuntimeError(f"no evaluable queries for {spec.dataset}/{spec.split} (are the videos indexed?)")

    prev = apply_overrides(spec.settings)
    try:
        rows = asyncio.run(_evaluate(qs, vids, spec, cfg))
    except BaseException:
        with _sync_session() as s:
            e = s.get(Experiment, exp_id)
            e.status, e.finished_at = ExperimentStatus.FAILED, datetime.now(UTC)
            s.commit()
        raise
    finally:
        apply_overrides(prev)

    overall = aggregate(rows)
    groups = {
        g: aggregate([r for r in rows if r[spec.group_by] == g])
        for g in sorted({r[spec.group_by] for r in rows})
    }
    with _sync_session() as s:
        s.add_all(
            ExperimentResult(
                experiment_id=exp_id,
                dataset_query_id=uuid.UUID(r["query_id"]),
                query_key=r["id"],
                metrics={k: r[k] for k in OVERLAP_KEYS + TEMPORAL_KEYS},
                lanes={"attribution": r["lanes"], "run": r["lanes_run"], "unique": r["unique_lane"]},
                hits=r["top"],
                latency_ms=r["latency_ms"],
            )
            for r in rows
        )
        e = s.get(Experiment, exp_id)
        e.metrics, e.by_group, e.n_queries = overall, groups, len(rows)
        e.status, e.finished_at = ExperimentStatus.DONE, datetime.now(UTC)
        s.commit()

    report = {
        "name": spec.name,
        "experiment_id": str(exp_id),
        "config": {**asdict(spec), "search_config": cfg.to_dict()},
        "dataset": f"{spec.dataset}/{spec.split}",
        "scope": spec.scope,
        "code_sha": _code_sha(),
        "ran_at": datetime.now(UTC).isoformat(),
        "overall": overall,
        "by_modality": groups,
        "queries": rows,
    }
    out = Path(output) if output else (reports_dir or Path("../eval/reports")) / f"{spec.name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    from scenepeek.eval.report import print_report

    print_report(report)
    log.info("report written", path=str(out), experiment=str(exp_id))
    return report
