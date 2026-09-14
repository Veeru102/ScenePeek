import pytest

from scenepeek.eval.metrics import Range, lane_attribution
from scenepeek.eval.report import paired_bootstrap
from scenepeek.eval.runner import apply_overrides


def test_paired_bootstrap_is_deterministic_and_brackets_mean():
    deltas = [0.1, 0.2, 0.05, 0.3, 0.15]
    a = paired_bootstrap(deltas, n=500, seed=7)
    b = paired_bootstrap(deltas, n=500, seed=7)
    assert a == b
    mean, lo, hi = a
    assert abs(mean - 0.16) < 1e-9
    assert lo <= mean <= hi
    assert lo > 0  # every delta is positive, so the CI must exclude zero


def test_paired_bootstrap_mixed_signs_straddles_zero():
    _, lo, hi = paired_bootstrap([0.5, -0.5, 0.4, -0.4, 0.0], n=500)
    assert lo < 0 < hi


def test_paired_bootstrap_empty():
    assert paired_bootstrap([]) == (0.0, 0.0, 0.0)


def test_lane_attribution_from_lanes():
    rel = [Range("v", 100.0, 110.0)]
    lanes = {
        "text": [Range("v", 0, 10), Range("v", 102, 112)],  # found at rank 2
        "visual": [Range("v", 500, 510)],  # missed
        "ocr": [Range("v", 100, 110)],  # found at rank 1
    }
    out = lane_attribution(lanes, rel, tolerance_s=3.0, k=10)
    assert out["lanes"]["text"] == {"first_rank": 2, "hit@10": 1.0}
    assert out["lanes"]["visual"] == {"first_rank": None, "hit@10": 0.0}
    assert out["lanes"]["ocr"]["first_rank"] == 1
    assert out["ceiling"] is True
    assert out["unique"] is None  # two lanes found it

    only_ocr = lane_attribution({"text": [Range("v", 0, 10)], "ocr": lanes["ocr"]}, rel)
    assert only_ocr["unique"] == "ocr"

    nothing = lane_attribution({"text": [Range("v", 0, 10)]}, rel)
    assert nothing["ceiling"] is False and nothing["unique"] is None


def test_lane_attribution_respects_k():
    rel = [Range("v", 100.0, 110.0)]
    deep = {"text": [Range("v", 1000 + i * 20, 1005 + i * 20) for i in range(20)] + [Range("v", 100, 110)]}
    assert lane_attribution(deep, rel, k=10)["lanes"]["text"]["first_rank"] is None


def test_apply_overrides_coerces_and_rejects_bad_values():
    prev = apply_overrides({"weight_ocr": 0, "rrf_k": "70"})  # int -> float, str -> int
    try:
        from scenepeek.core.config import get_settings

        s = get_settings()
        assert s.weight_ocr == 0.0 and isinstance(s.weight_ocr, float)
        assert s.rrf_k == 70 and isinstance(s.rrf_k, int)
    finally:
        apply_overrides(prev)
    with pytest.raises(KeyError):
        apply_overrides({"weight_ocrr": 0.5})
    with pytest.raises(ValueError):
        apply_overrides({"weight_ocr": None})
    with pytest.raises(TypeError):
        apply_overrides({"rerank_enabled": "no"})
    with pytest.raises(TypeError):
        apply_overrides({"rrf_k": "seventy"})


def test_search_config_overrides_are_validated():
    import pytest

    from scenepeek.search.config import SearchConfig

    cfg = SearchConfig(lanes={"text": 1.0, "visual": 0.8}, rerank=True)
    out = cfg.with_overrides({"lanes": {"visual": 0}, "rerank": False, "rrf_k": "70"})
    assert out.lanes == {"text": 1.0, "visual": 0.0} and out.rerank is False and out.rrf_k == 70
    assert cfg.lanes["visual"] == 0.8  # frozen: the original is untouched
    with pytest.raises(KeyError):
        cfg.with_overrides({"weight_visual": 0.0})
    with pytest.raises(KeyError):
        cfg.with_overrides({"lanes": {"vision": 0.0}})
    with pytest.raises(TypeError):
        cfg.with_overrides({"rerank": "no"})
    with pytest.raises(ValueError):
        cfg.with_overrides({"fusion": None})
