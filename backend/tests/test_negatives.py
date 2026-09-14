"""Hard-negative mining over a recorded (fake) experiment."""

from sqlalchemy.orm import Session

from scenepeek.eval import negatives
from scenepeek.models import (
    Dataset,
    DatasetQuery,
    DatasetVideo,
    Experiment,
    ExperimentResult,
    Segment,
    TrainingExample,
    Video,
    VideoChunk,
)


def test_mine_picks_best_overlap_positive_and_non_overlapping_negatives(engine, tmp_path, monkeypatch):
    monkeypatch.setattr(negatives, "create_engine", lambda url: engine)
    with Session(engine) as s:
        v = Video(title="clip", original_key="k", status="ready", source="unit")
        ds = Dataset(name="unit_neg")
        s.add_all([v, ds])
        s.flush()
        c = VideoChunk(video_id=v.id, index=0, start_s=0, end_s=60, status="done")
        s.add(c)
        s.flush()
        segs = [
            Segment(
                video_id=v.id, chunk_id=c.id, index=i, start_s=i * 10.0, end_s=(i + 1) * 10.0, text=f"seg {i}"
            )
            for i in range(6)
        ]
        s.add_all(segs)
        dv = DatasetVideo(dataset_id=ds.id, external_id="clip", split="test", video_id=v.id, status="indexed")
        s.add(dv)
        s.flush()
        q = DatasetQuery(
            dataset_id=ds.id,
            dataset_video_id=dv.id,
            external_id="q1",
            split="test",
            text="x",
            relevant=[[22, 30]],
        )
        exp = Experiment(name="unit_exp", dataset_id=ds.id, split="test", n_queries=1, status="done")
        s.add_all([q, exp])
        s.flush()
        hits = [  # ranked: seg5 (wrong), seg2 (the answer), seg3 (touches truth: IoU 0), seg0 (wrong)
            {"segment_id": str(segs[5].id), "video": "clip", "start_s": 50.0, "end_s": 60.0, "score": 0.9},
            {"segment_id": str(segs[2].id), "video": "clip", "start_s": 20.0, "end_s": 30.0, "score": 0.8},
            {"segment_id": str(segs[3].id), "video": "clip", "start_s": 30.0, "end_s": 40.0, "score": 0.7},
            {"segment_id": str(segs[0].id), "video": "clip", "start_s": 0.0, "end_s": 10.0, "score": 0.6},
        ]
        s.add(ExperimentResult(experiment_id=exp.id, dataset_query_id=q.id, query_key="q1", hits=hits))
        s.commit()
        exp_id, seg2 = exp.id, segs[2].id

    out = tmp_path / "neg.jsonl"
    m = negatives.mine(str(exp_id), max_iou=0.1, per_query=2, out=out)
    assert (m.queries, m.positives, m.negatives, m.skipped_no_positive) == (1, 1, 2, 0)
    with Session(engine) as s:
        rows = sorted(s.query(TrainingExample).all(), key=lambda r: r.rank)
        assert [r.rank for r in rows] == [1, 3] and all(r.positive_segment_id == seg2 for r in rows)
        # mining again replaces rather than duplicates
        negatives.mine(str(exp_id), per_query=2)
        assert s.query(TrainingExample).count() == 2
    rec = __import__("json").loads(out.read_text().splitlines()[0])
    assert rec["positive"] == "seg 2" and [n["passage"] for n in rec["negatives"]] == ["seg 5", "seg 3"]
