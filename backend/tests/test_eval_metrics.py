from scenepeek.eval.metrics import Range, mrr, ndcg_at, recall_at, relevance


def test_relevance_matches_overlap_or_tolerance_once():
    rel = [Range("v", 20, 30)]
    hits = [Range("v", 0, 10), Range("v", 28, 38), Range("v", 25, 29), Range("v", 32, 40)]
    assert relevance(hits, rel) == [
        0,
        1,
        0,
        0,
    ]  # second overlaps; third is a duplicate credit; fourth within 3 s tolerance but already credited
    assert relevance([Range("v", 32, 40)], rel) == [1]
    assert relevance([Range("other", 22, 26)], rel) == [0]


def test_metrics():
    rels = [0, 1, 0, 0, 1]
    assert mrr(rels) == 0.5
    assert recall_at(rels, 2, 5) == 1.0 and recall_at(rels, 2, 1) == 0.0
    assert 0 < ndcg_at(rels, 2, 5) < 1
    assert ndcg_at([1, 1], 2, 5) == 1.0
