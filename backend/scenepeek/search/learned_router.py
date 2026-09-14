"""Learned routing: predict, per lane, how likely it is to surface the answer for this query.

Labels come from measured retrieval outcomes (lane attribution recorded by an experiment against
ground-truth windows), never from hand-assigned "visual"/"speech" tags. The model is deliberately
small — one logistic regression per lane over the query embedding + cue flags — so it trains in
seconds on a laptop and its weights can be inspected."""

import json
import math
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from scenepeek.core.config import get_settings
from scenepeek.search.planner import QueryPlan, parse

CUES = ("split", "speech", "visual", "ocr", "exact")


def _features(p: QueryPlan, q_vec: np.ndarray) -> np.ndarray:
    cue = [1.0 if c in p.cues else 0.0 for c in CUES]
    n_tok = len(p.raw.split())
    extra = [math.log1p(n_tok), 1.0 if p.exact_phrases else 0.0, 1.0 if p.speech_q != p.visual_q else 0.0]
    return np.concatenate([q_vec.astype(np.float32), np.array(cue + extra, dtype=np.float32)])


def _embed(text: str) -> np.ndarray:
    from scenepeek.ml import text_embed

    return text_embed.embed_query(text)


@dataclass
class RouterModel:
    version: str
    lanes: list[str]
    models: dict  # lane -> fitted sklearn estimator (only lanes with both classes in training)
    priors: dict[str, float]  # lane -> training hit rate (used when a lane had no negatives/positives)
    min_p: float = 0.15
    meta: dict = field(default_factory=dict)

    def probs(self, p: QueryPlan, q_vec: np.ndarray | None = None) -> dict[str, float]:
        x = _features(p, q_vec if q_vec is not None else _embed(p.raw)).reshape(1, -1)
        out = {}
        for lane in self.lanes:
            m = self.models.get(lane)
            out[lane] = float(m.predict_proba(x)[0, 1]) if m is not None else self.priors.get(lane, 0.5)
        return out

    def save(self, path: Path) -> None:
        import joblib

        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: Path) -> "RouterModel":
        import joblib

        return joblib.load(path)


class LearnedRouter:
    name = "learned"

    def __init__(self, model: RouterModel):
        self.model = model
        self.name = f"learned:{model.version}"

    @classmethod
    def load(cls, version: str) -> "LearnedRouter":
        return cls(RouterModel.load(router_path(version)))

    def route(self, p: QueryPlan, base: dict[str, float]) -> dict[str, float]:
        """Scale each configured lane by its predicted usefulness; drop lanes below `min_p`. A lane
        the config already turned off (weight 0) stays off — the router chooses among what exists."""
        probs = self.model.probs(p)
        top = max(probs.values()) or 1.0
        w = {}
        for lane, base_w in base.items():
            pr = probs.get(lane)
            if base_w <= 0 or pr is None:
                w[lane] = base_w
            elif pr < self.model.min_p:
                w[lane] = 0.0
            else:
                w[lane] = base_w * pr / top
        p.probs = {lane: round(pr, 3) for lane, pr in probs.items()}
        return w


def router_path(version: str) -> Path:
    return get_settings().models_dir.expanduser() / "router" / f"{version}.joblib"


# -- training ------------------------------------------------------------------------------------


def training_rows(experiment: str) -> list[dict]:
    """(query text, lane -> hit@10) pairs from one recorded experiment."""
    from sqlalchemy import create_engine, select
    from sqlalchemy.orm import Session

    from scenepeek.models import DatasetQuery, Experiment, ExperimentResult

    with Session(create_engine(get_settings().sync_database_url)) as s:
        try:
            exp = s.get(Experiment, uuid.UUID(experiment))
        except ValueError:
            exp = s.scalar(
                select(Experiment).where(Experiment.name == experiment).order_by(Experiment.started_at.desc())
            )
        if exp is None:
            raise KeyError(f"experiment {experiment!r} not found")
        rows = s.execute(
            select(ExperimentResult, DatasetQuery)
            .join(DatasetQuery, ExperimentResult.dataset_query_id == DatasetQuery.id)
            .where(ExperimentResult.experiment_id == exp.id)
        ).all()
    out = []
    for r, q in rows:
        attribution = (r.lanes or {}).get("attribution") or {}
        ran = set((r.lanes or {}).get("run") or attribution)
        out.append(
            {
                "text": q.text,
                # a lane that was switched off in the experiment carries no evidence either way
                "labels": {
                    lane: float(v.get("hit@10", 0.0)) for lane, v in attribution.items() if lane in ran
                },
                "experiment_id": str(exp.id),
            }
        )
    return out


def train(
    experiment: str, version: str, *, holdout: float = 0.2, seed: int = 0, min_p: float = 0.15
) -> RouterModel:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    rows = training_rows(experiment)
    if len(rows) < 20:
        raise ValueError(f"only {len(rows)} labelled queries; need at least 20")
    lanes = sorted({lane for r in rows for lane in r["labels"]})
    X = np.stack([_features(parse(r["text"]), _embed(r["text"])) for r in rows])
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(rows))
    n_hold = int(len(rows) * holdout)
    hold, fit = order[:n_hold], order[n_hold:]
    models, priors, aucs = {}, {}, {}
    for lane in lanes:
        y = np.array([r["labels"].get(lane, 0.0) for r in rows])
        priors[lane] = float(y[fit].mean())
        if len(set(y[fit])) < 2:
            continue
        clf = LogisticRegression(C=0.5, max_iter=2000, class_weight="balanced")
        clf.fit(X[fit], y[fit])
        models[lane] = clf
        if n_hold and len(set(y[hold])) == 2:
            aucs[lane] = round(float(roc_auc_score(y[hold], clf.predict_proba(X[hold])[:, 1])), 3)
    model = RouterModel(
        version=version,
        lanes=lanes,
        models=models,
        priors=priors,
        min_p=min_p,
        meta={
            "trained_on": rows[0]["experiment_id"],
            "n_train": int(len(fit)),
            "n_holdout": int(n_hold),
            "holdout_auc": aucs,
            "priors": priors,
            "features": f"{X.shape[1]}d = bge query embedding + cues {list(CUES)} + log_tokens/exact/split",
        },
    )
    path = router_path(version)
    model.save(path)
    path.with_suffix(".json").write_text(json.dumps(model.meta, indent=2))
    _register(version, path)
    return model


def _register(version: str, path: Path) -> None:
    from sqlalchemy import create_engine, text

    with create_engine(get_settings().sync_database_url).begin() as conn:
        conn.execute(
            text(
                "INSERT INTO index_versions (id, kind, model_key, status, index_name) "
                "VALUES (gen_random_uuid(), 'router', :v, 'ready', :p) "
                "ON CONFLICT ON CONSTRAINT uq_index_version DO UPDATE SET index_name=:p, status='ready'"
            ),
            {"v": version, "p": str(path)[-128:]},
        )
