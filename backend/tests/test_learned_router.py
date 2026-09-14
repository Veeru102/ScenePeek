"""Learned router: trains from lane-attribution labels and routes accordingly (fake embeddings)."""

import numpy as np

from scenepeek.search import learned_router as lr
from scenepeek.search.planner import parse
from scenepeek.search.routing import get_router


def _fake_embed(text: str) -> np.ndarray:
    # a 4-d "embedding" that separates on-screen-text queries from action queries
    v = np.zeros(4, dtype=np.float32)
    v[0] = 1.0 if "slide" in text else 0.0
    v[1] = 1.0 if ("running" in text or "jumps" in text) else 0.0
    v[2] = len(text) / 50.0
    return v


def _rows():
    rows = []
    for i in range(30):
        rows.append(
            {"text": f"the slide about topic {i}", "labels": {"ocr": 1.0, "visual": 0.0, "text": 1.0}}
        )
        rows.append(
            {"text": f"a person running {i} jumps", "labels": {"ocr": 0.0, "visual": 1.0, "text": 0.0}}
        )
    for r in rows:
        r["experiment_id"] = "exp"
    return rows


def test_train_and_route_prefers_lanes_that_found_answers(tmp_path, monkeypatch):
    monkeypatch.setattr(lr, "_embed", _fake_embed)
    monkeypatch.setattr(lr, "training_rows", lambda exp: _rows())
    monkeypatch.setattr(lr, "_register", lambda v, p: None)
    monkeypatch.setattr(lr, "router_path", lambda v: tmp_path / f"{v}.joblib")

    m = lr.train("exp", "t1", holdout=0.2, seed=0)
    assert set(m.lanes) == {"ocr", "visual", "text"}
    assert all(auc >= 0.9 for auc in m.meta["holdout_auc"].values())
    assert (tmp_path / "t1.json").exists()

    router = get_router("learned:t1")
    base = {"text": 1.0, "lexical": 0.8, "visual": 0.8, "ocr": 0.6, "caption": 0.0}
    w_slide = router.route(parse("the slide about b-trees"), base)
    w_action = router.route(parse("a person running and jumps"), base)
    assert w_slide["ocr"] > w_action["ocr"] and w_action["visual"] > w_slide["visual"]
    assert w_slide["visual"] == 0.0  # below min_p -> lane skipped (selective retrieval)
    assert w_slide["lexical"] == 0.8  # no label for this lane -> untouched
    assert w_slide["caption"] == 0.0  # disabled in config stays disabled


def test_fixed_and_heuristic_routers_are_baselines():
    base = {"text": 1.0, "lexical": 0.8, "visual": 0.8, "ocr": 0.6, "caption": 0.0}
    p = parse('the slide titled "Transactions"')
    assert get_router("fixed").route(p, base) == base
    h = get_router("heuristic").route(p, base)
    assert h["ocr"] > base["ocr"] and h["lexical"] > base["lexical"]
