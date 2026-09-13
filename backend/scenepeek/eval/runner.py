"""Run every dataset query through the in-process search service and score the results."""

import asyncio
import json
import statistics
import time
from datetime import UTC, datetime
from pathlib import Path

import yaml
from sqlalchemy import select

from scenepeek.core.config import get_settings
from scenepeek.core.db import get_sessionmaker
from scenepeek.core.logging import get_logger
from scenepeek.eval.dataset import Dataset, load_dataset
from scenepeek.eval.metrics import Range, hit_at, mrr, ndcg_at, recall_at, relevance
from scenepeek.models import Video
from scenepeek.search.service import SearchOptions, search

log = get_logger("eval")
KS = (1, 5, 10)


def apply_overrides(overrides: dict) -> dict:
    """Mutate the settings singleton for this process; returns the previous values."""
    s = get_settings()
    prev = {}
    for k, v in (overrides or {}).items():
        if not hasattr(s, k):
            raise KeyError(f"unknown setting {k!r}")
        prev[k] = getattr(s, k)
        setattr(s, k, v)
    return prev


async def _resolve_videos(ds: Dataset) -> dict[str, str]:
    """dataset key -> video id, matched by exact title, then case-insensitive prefix."""
    async with get_sessionmaker()() as s:
        rows = list(await s.execute(select(Video.id, Video.title, Video.status)))
    out: dict[str, str] = {}
    for v in ds.videos:
        match = next((r for r in rows if r.title == v.title), None) or next(
            (r for r in rows if r.title.lower().startswith(v.title.lower()[:40])), None
        )
        if match is None:
            log.warning("dataset video not in library", key=v.key, title=v.title)
            continue
        if match.status != "ready":
            log.warning("dataset video not fully indexed", key=v.key, status=match.status)
        out[v.key] = str(match.id)
    return out


async def _run(ds: Dataset, limit: int, tolerance_s: float, restrict: bool) -> dict:
    vids = await _resolve_videos(ds)
    id_to_key = {v: k for k, v in vids.items()}
    per_query = []
    async with get_sessionmaker()() as s:
        await search(s, "warm up the models", SearchOptions(limit=1))  # exclude cold-start from latency
        for q in ds.queries:
            relevant = [r for r in q.relevant if r.video in vids]
            if not relevant:
                log.warning("query skipped (videos missing)", id=q.id)
                continue
            opts = SearchOptions(
                limit=limit,
                video_ids=[__import__("uuid").UUID(v) for v in vids.values()] if restrict else None,
            )
            t0 = time.perf_counter()
            res = await search(s, q.text, opts)
            ms = (time.perf_counter() - t0) * 1000
            hits = [
                Range(id_to_key.get(str(h.video.id), str(h.video.id)), h.start_s, h.end_s) for h in res.hits
            ]
            rels = relevance(hits, relevant, tolerance_s)
            n = len(relevant)
            row = {
                "id": q.id,
                "text": q.text,
                "modality": q.modality,
                "n_relevant": n,
                "latency_ms": round(ms, 1),
                "mrr": mrr(rels),
                "first_hit_rank": (rels.index(1) + 1) if 1 in rels else None,
                **{f"recall@{k}": recall_at(rels, n, k) for k in KS},
                **{f"hit@{k}": hit_at(rels, k) for k in KS},
                "ndcg@10": ndcg_at(rels, n, 10),
                "top": [{"video": h.video, "start_s": h.start_s, "end_s": h.end_s} for h in hits[:3]],
            }
            per_query.append(row)
            log.info("query", id=q.id, mrr=round(row["mrr"], 3), r5=row["recall@5"], ms=row["latency_ms"])
    return {"queries": per_query, "videos": vids}


def _aggregate(rows: list[dict]) -> dict:
    if not rows:
        return {}
    keys = ["mrr", "ndcg@10"] + [f"recall@{k}" for k in KS] + [f"hit@{k}" for k in KS]
    agg = {k: round(statistics.mean(r[k] for r in rows), 4) for k in keys}
    lat = sorted(r["latency_ms"] for r in rows)
    agg["latency_p50_ms"] = round(lat[len(lat) // 2], 1)
    agg["latency_p95_ms"] = round(lat[min(len(lat) - 1, int(len(lat) * 0.95))], 1)
    agg["n"] = len(rows)
    return agg


def run_eval(config_path: str, output: str | None = None) -> dict:
    cfg_path = Path(config_path)
    cfg = yaml.safe_load(cfg_path.read_text()) if cfg_path.exists() else {}
    name = cfg.get("name", cfg_path.stem)
    ds_path = (
        (cfg_path.parent / cfg.get("dataset", "../dataset.yaml")).resolve()
        if cfg
        else Path("../eval/dataset.yaml")
    )
    ds = load_dataset(ds_path)
    prev = apply_overrides(cfg.get("overrides", {}))
    try:
        result = asyncio.run(
            _run(
                ds,
                int(cfg.get("limit", 10)),
                float(cfg.get("tolerance_s", 3.0)),
                bool(cfg.get("restrict_to_dataset", True)),
            )
        )
    finally:
        apply_overrides(prev)
    rows = result["queries"]
    report = {
        "name": name,
        "config": cfg,
        "dataset": str(ds_path),
        "ran_at": datetime.now(UTC).isoformat(),
        "overall": _aggregate(rows),
        "by_modality": {
            m: _aggregate([r for r in rows if r["modality"] == m])
            for m in sorted({r["modality"] for r in rows})
        },
        "queries": rows,
    }
    out = Path(output) if output else cfg_path.parent.parent / "reports" / f"{name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    from scenepeek.eval.report import print_report

    print_report(report)
    log.info("report written", path=str(out))
    return report
