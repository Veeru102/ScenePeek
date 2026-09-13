from scenepeek.ml.whisper import UtteranceOut, Word
from scenepeek.pipeline.chunking import plan_chunks
from scenepeek.pipeline.segment import plan_segments


def _utt(start: float, end: float, text: str) -> UtteranceOut:
    words = text.split()
    step = (end - start) / max(1, len(words))
    return UtteranceOut(
        start, end, text, [Word(w, start + i * step, start + (i + 1) * step) for i, w in enumerate(words)]
    )


def test_chunks_cover_duration_and_merge_short_tail():
    ws = plan_chunks(185, 60)
    assert [(w.start_s, w.end_s) for w in ws] == [(0, 60), (60, 120), (120, 185)]
    assert plan_chunks(130, 60)[-1].end_s == 130 and len(plan_chunks(130, 60)) == 2
    assert plan_chunks(0, 60) == []


def test_segments_are_contiguous_and_snap_to_utterance_boundaries():
    utts = [
        _utt(0, 4, "one two three"),
        _utt(4, 9.5, "four five six seven"),
        _utt(9.5, 18, "eight nine ten"),
        _utt(18, 30, "eleven twelve"),
    ]
    segs = plan_segments(0, 30, utts, target_s=10, min_s=6, max_s=15)
    assert segs[0].start_s == 0 and segs[-1].end_s == 30
    for a, b in zip(segs, segs[1:], strict=False):
        assert a.end_s == b.start_s
    assert segs[0].end_s == 9.5  # nearest utterance boundary to the 10 s target within [6, 15]
    assert "one" in segs[0].text and "eight" in segs[1].text


def test_silent_chunk_still_gets_windows():
    segs = plan_segments(60, 120, [], target_s=10, min_s=6, max_s=15)
    assert len(segs) == 6 and all(s.text == "" for s in segs)
    assert segs[-1].end_s == 120


def test_context_text_includes_neighbours():
    utts = [_utt(0, 10, "alpha beta"), _utt(10, 20, "gamma delta"), _utt(20, 30, "epsilon zeta")]
    segs = plan_segments(0, 30, utts, target_s=10, min_s=6, max_s=15)
    assert "alpha beta" in segs[1].context_text and "epsilon" in segs[1].context_text
    assert segs[1].text == "gamma delta"
