"""Hard-negative mining from recorded experiments.

For each labelled query: the positive is the indexed segment that overlaps the ground truth best;
hard negatives are the highest-ranked hits whose IoU with every ground-truth window is at most
`max_iou`. They are persisted (`training_examples`) and exported as JSONL passages so a reranker
can be fine-tuned on exactly the mistakes the fused retriever makes."""

import json
import uuid
from dataclasses import dataclass
from pathlib import Path

from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from scenepeek.core.config import get_settings
from scenepeek.eval.metrics import Range, iou
from scenepeek.models import (
    DatasetQuery,
    DatasetVideo,
    Experiment,
    ExperimentResult,
    Segment,
    TrainingExample,
)
from scenepeek.search.service import _passage


@dataclass
class Mined:
    queries: int
    positives: int
    negatives: int
    skipped_no_positive: int


def _find_experiment(s: Session, experiment: str) -> Experiment:
    try:
        exp = s.get(Experiment, uuid.UUID(experiment))
    except ValueError:
        exp = s.scalar(
            select(Experiment).where(Experiment.name == experiment).order_by(Experiment.started_at.desc())
        )
    if exp is None:
        raise KeyError(f"experiment {experiment!r} not found")
    return exp


def _best_positive(s: Session, video_id: uuid.UUID, relevant: list[Range]) -> Segment | None:
    """The indexed segment with the highest IoU against any ground-truth window (may be outside the
    hit list — that is the point: it is what the retriever should have ranked)."""
    lo = min(r.start_s for r in relevant) - 30
    hi = max(r.end_s for r in relevant) + 30
    cands = list(
        s.scalars(
            select(Segment).where(Segment.video_id == video_id, Segment.end_s > lo, Segment.start_s < hi)
        )
    )
    best, best_iou = None, 0.0
    for seg in cands:
        v = max(iou(Range("v", seg.start_s, seg.end_s), Range("v", r.start_s, r.end_s)) for r in relevant)
        if v > best_iou:
            best, best_iou = seg, v
    return best


def mine(
    experiment: str,
    *,
    max_iou: float = 0.1,
    per_query: int = 5,
    out: Path | None = None,
    replace: bool = True,
) -> Mined:
    engine = create_engine(get_settings().sync_database_url)
    n_q = n_pos = n_neg = n_skip = 0
    records = []
    with Session(engine) as s:
        exp = _find_experiment(s, experiment)
        if replace:
            for row in s.scalars(select(TrainingExample).where(TrainingExample.experiment_id == exp.id)):
                s.delete(row)
        rows = s.execute(
            select(ExperimentResult, DatasetQuery, DatasetVideo)
            .join(DatasetQuery, ExperimentResult.dataset_query_id == DatasetQuery.id)
            .join(DatasetVideo, DatasetQuery.dataset_video_id == DatasetVideo.id)
            .where(ExperimentResult.experiment_id == exp.id)
        ).all()
        for r, q, dv in rows:
            n_q += 1
            relevant = [Range(dv.external_id, float(a), float(b)) for a, b in q.relevant]
            pos = _best_positive(s, dv.video_id, relevant)
            if pos is None:
                n_skip += 1
                continue
            n_pos += 1
            negs = []
            for rank, h in enumerate(r.hits or [], start=1):
                span = Range(h["video"], h["start_s"], h["end_s"])
                if h["segment_id"] == str(pos.id):
                    continue
                if max(iou(span, rel) for rel in relevant) > max_iou:
                    continue
                negs.append((rank, h))
                if len(negs) >= per_query:
                    break
            neg_segs = {
                str(seg.id): seg
                for seg in s.scalars(
                    select(Segment).where(Segment.id.in_([uuid.UUID(h["segment_id"]) for _, h in negs]))
                )
            }
            exported = []
            for rank, h in negs:
                seg = neg_segs.get(h["segment_id"])
                if seg is None:
                    continue
                s.add(
                    TrainingExample(
                        experiment_id=exp.id,
                        dataset_query_id=q.id,
                        positive_segment_id=pos.id,
                        negative_segment_id=seg.id,
                        rank=rank,
                        score=h.get("score"),
                        lanes=h.get("signals") or {},
                    )
                )
                exported.append({"rank": rank, "score": h.get("score"), "passage": _passage(seg, True)})
                n_neg += 1
            if exported:
                records.append(
                    {
                        "query_id": q.external_id,
                        "query": q.text,
                        "positive": _passage(pos, True),
                        "positive_span": [pos.start_s, pos.end_s],
                        "negatives": exported,
                    }
                )
        s.commit()
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w") as f:
            for rec in records:
                f.write(json.dumps(rec) + "\n")
    return Mined(queries=n_q, positives=n_pos, negatives=n_neg, skipped_no_positive=n_skip)
