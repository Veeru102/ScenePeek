"""Experiment runner: dataset -> per-query rows -> experiments/experiment_results + JSON report.
Search itself is faked (no ML); everything around it is real."""

import json
import uuid
from contextlib import asynccontextmanager
from types import SimpleNamespace

from sqlalchemy import select
from sqlalchemy.orm import Session

from scenepeek.datasets.importer import import_split
from scenepeek.datasets.local_yaml import LocalYaml
from scenepeek.eval import experiment as ex
from scenepeek.models import Experiment, ExperimentResult, Video
from scenepeek.search.dedup import Hit

YAML = """
videos:
- key: v1
  title: Video One
- key: v2
  title: Video Two
queries:
- id: q1
  text: first thing
  modality: speech
  relevant: [{video: v1, start_s: 10, end_s: 20}]
- id: q2
  text: second thing
  modality: visual
  relevant: [{video: v2, start_s: 100, end_s: 110}]
"""


def _fake_search(video_ids_by_key):
    async def search(s, query, opts):
        # q1 is answered at rank 1 with the exact span; q2 at rank 2
        v1, v2 = video_ids_by_key["v1"], video_ids_by_key["v2"]
        vid = SimpleNamespace(id=v1)
        vid2 = SimpleNamespace(id=v2)
        if query == "first thing":
            hits = [SimpleNamespace(video=vid, start_s=10.0, end_s=20.0)]
        elif query == "second thing":
            hits = [
                SimpleNamespace(video=vid, start_s=0.0, end_s=10.0),
                SimpleNamespace(video=vid2, start_s=98.0, end_s=112.0),
            ]
        else:
            hits = []
        lanes = {"text": [Hit(uuid.uuid4(), v2, 98.0, 112.0, 0.9)], "visual": []}
        return SimpleNamespace(
            hits=hits,
            lanes=lanes,
            lanes_run=["text", "visual"],
            plan=SimpleNamespace(cues=[]),
            config=opts.config,
        )

    return search


def test_run_records_experiment_and_results(engine, tmp_path, monkeypatch):
    (tmp_path / "ds.yaml").write_text(YAML)
    adapter = LocalYaml("unit_yaml", "ds.yaml")
    monkeypatch.setitem(
        __import__("scenepeek.datasets.registry", fromlist=["ADAPTERS"]).ADAPTERS, "unit_yaml", adapter
    )
    with Session(engine) as s:
        ids = {}
        for key, title in (("v1", "Video One"), ("v2", "Video Two")):
            v = Video(title=title, original_key="k", status="ready")
            s.add(v)
            s.flush()
            ids[key] = v.id
        s.commit()
        import_split(s, adapter, tmp_path, "test")

    @asynccontextmanager
    async def _fake_session():
        yield None

    monkeypatch.setattr(ex, "_sync_session", lambda: Session(engine, expire_on_commit=False))
    monkeypatch.setattr(ex, "get_sessionmaker", lambda: _fake_session)
    monkeypatch.setattr(ex, "search", _fake_search(ids))

    spec = ex.ExperimentSpec(
        name="unit", dataset="unit_yaml", scope="dataset", search={"lanes": {"ocr": 0.0}}
    )
    report = ex.run(spec, output=str(tmp_path / "unit.json"))

    assert report["overall"]["n"] == 2
    assert report["overall"]["mrr"] == 0.75  # 1.0 and 0.5
    assert report["overall"]["r1@0.5"] == 0.5  # q1 exact at rank 1; q2 hit is rank 2
    assert report["overall"]["r5@0.5"] == 1.0  # IoU(98-112 vs 100-110) = 10/14 >= 0.5
    assert report["by_modality"]["visual"]["mrr"] == 0.5
    assert report["config"]["search_config"]["lanes"]["ocr"] == 0.0
    assert json.loads((tmp_path / "unit.json").read_text())["name"] == "unit"

    with Session(engine) as s:
        e = s.scalar(select(Experiment).where(Experiment.name == "unit"))
        assert e.status == "done" and e.n_queries == 2 and e.metrics["mrr"] == 0.75
        assert e.config["search"]["lanes"]["ocr"] == 0.0 and e.config["search"]["router"] == "heuristic"
        rows = list(s.scalars(select(ExperimentResult).where(ExperimentResult.experiment_id == e.id)))
        assert {r.query_key for r in rows} == {"q1", "q2"}
        assert all(r.dataset_query_id is not None for r in rows)
        assert rows[0].lanes["run"] == ["text", "visual"]
