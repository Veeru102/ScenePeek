import uuid

from scenepeek.search.candidates import Cand
from scenepeek.search.dedup import Hit, suppress
from scenepeek.search.fusion import fuse
from scenepeek.search.planner import plan, terms
from scenepeek.search.service import highlight


def test_planner_strips_lead_and_detects_speaker_cue():
    p = plan("Find where the professor explains B+ tree leaf nodes")
    assert p.speech_q == "B+ tree leaf nodes"
    assert "speech" in p.cues
    assert p.lexical_terms == ["b+", "tree", "leaf", "nodes"]
    assert p.weights["visual"] < p.weights["text"]


def test_planner_splits_speech_and_visual_parts():
    p = plan("when the speaker mentions latency while showing an architecture diagram")
    assert p.speech_q == "latency"
    assert "architecture diagram" in p.visual_q
    assert "split" in p.cues


def test_planner_visual_only_query_boosts_visual():
    p = plan("someone holding a red umbrella")
    assert "visual" in p.cues
    assert p.weights["visual"] > p.weights["text"]


def test_planner_quoted_phrase_is_exact():
    p = plan('the slide titled "Transactions and WAL"')
    assert p.exact_phrases == ["Transactions and WAL"]
    assert "ocr" in p.cues and p.weights["ocr"] > 0.6


def test_planner_overrides_win():
    p = plan("anything", overrides={"visual": 0.0})
    assert p.weights["visual"] == 0.0


def test_terms_drop_stopwords():
    assert terms("find the moment where the cat sat") == ["cat", "sat"]


def test_rrf_fusion_prefers_multi_modal_agreement():
    a, b, c = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
    lists = {"text": [Cand(a, 0.9), Cand(b, 0.8)], "visual": [Cand(b, 0.5), Cand(c, 0.4)]}
    fused = fuse(lists, {"text": 1.0, "visual": 1.0}, "rrf", 60)
    assert fused[0].segment_id == b  # present in both lists
    assert fused[0].signals == {"text": 0.8, "visual": 0.5}


def test_weighted_fusion_normalises_per_modality():
    a, b = uuid.uuid4(), uuid.uuid4()
    lists = {"text": [Cand(a, 0.9), Cand(b, 0.1)], "visual": [Cand(b, 0.05), Cand(a, 0.01)]}
    fused = fuse(lists, {"text": 1.0, "visual": 1.0}, "weighted", 60)
    assert fused[0].segment_id == a and abs(fused[0].fused - 1.0) < 1e-9


def test_suppress_drops_neighbours_and_caps_per_video():
    v1, v2 = uuid.uuid4(), uuid.uuid4()
    hits = [
        Hit(uuid.uuid4(), v1, 20, 30, 0.9),
        Hit(uuid.uuid4(), v1, 30, 40, 0.8),  # within window of the first
        Hit(uuid.uuid4(), v1, 100, 110, 0.7),
        Hit(uuid.uuid4(), v1, 200, 210, 0.6),
        Hit(uuid.uuid4(), v1, 300, 310, 0.5),  # exceeds the per-video cap
        Hit(uuid.uuid4(), v2, 0, 10, 0.4),
    ]
    kept = suppress(hits, window_s=12, max_per_video=3)
    assert [(h.video_id, h.start_s) for h in kept] == [(v1, 20), (v1, 100), (v1, 200), (v2, 0)]
    assert kept[0].end_s == 30  # never widened


def test_highlight_escapes_and_marks():
    out = highlight("B+ trees <are> balanced", ["b+", "balanced"])
    assert "<mark>B+</mark>" in out and "&lt;are&gt;" in out and "<mark>balanced</mark>" in out
