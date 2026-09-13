import numpy as np

from scenepeek.pipeline.frames import FrameRef, hamming, pick_distinct
from scenepeek.pipeline.timeline import heading_cuts, segment_topics


def test_heading_cuts_forward_fill_and_persistence():
    hs = ["A", "A", None, "B", "B", "C", "B", "B"]
    # B at index 3 persists -> cut; C at 5 is a one-off -> no cut; B at 6 persists -> cut
    assert heading_cuts(hs) == [3, 6]


def test_segment_topics_uses_forced_cuts_and_embedding_shift():
    n = 12
    starts = [i * 10.0 for i in range(n)]
    ends = [(i + 1) * 10.0 for i in range(n)]
    emb = np.zeros((n, 4))
    emb[:6, 0] = 1  # first topic
    emb[6:, 1] = 1  # second topic (embedding shift at 6)
    spans = segment_topics(starts, ends, emb, min_topic_s=20, forced=[3])
    bounds = [s.start_i for s in spans]
    assert 3 in bounds and 6 in bounds


def test_pick_distinct_frames_by_phash():
    f = [
        FrameRef(0, "a", "0000000000000000"),
        FrameRef(1, "b", "0000000000000001"),
        FrameRef(2, "c", "ffffffffffffffff"),
    ]
    assert hamming(f[0].phash, f[2].phash) == 64
    chosen = pick_distinct(f, k=3, min_distance=6)
    assert [x.key for x in chosen] == ["a", "c"]
