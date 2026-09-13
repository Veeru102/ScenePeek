"""Split a chunk into retrieval segments aligned to speech boundaries."""

from dataclasses import dataclass, field

from scenepeek.ml.whisper import UtteranceOut, Word


@dataclass
class SegmentDraft:
    index: int
    start_s: float
    end_s: float
    words: list[Word] = field(default_factory=list)
    context_text: str = ""

    @property
    def text(self) -> str:
        return " ".join(w.w for w in self.words).strip()


def _boundaries(utterances: list[UtteranceOut]) -> list[float]:
    ends = sorted({round(u.end, 3) for u in utterances})
    return ends


def plan_segments(
    chunk_start: float,
    chunk_end: float,
    utterances: list[UtteranceOut],
    *,
    target_s: float = 10.0,
    min_s: float = 6.0,
    max_s: float = 15.0,
) -> list[SegmentDraft]:
    """Contiguous windows covering the chunk. Each window ends at the utterance boundary nearest
    `target_s` within [min_s, max_s]; silent spans fall back to fixed windows so visual-only
    content is still indexed."""
    bounds = _boundaries(utterances)
    words = sorted((w for u in utterances for w in u.words), key=lambda w: w.s)
    segs: list[SegmentDraft] = []
    cur = chunk_start
    i = 0
    while cur < chunk_end - 0.5:
        lo, hi = cur + min_s, cur + max_s
        candidates = [b for b in bounds if lo <= b <= hi]
        if candidates:
            end = min(candidates, key=lambda b: abs(b - (cur + target_s)))
        else:
            end = cur + target_s
        end = min(end, chunk_end)
        if chunk_end - end < min_s:  # absorb a short tail
            end = chunk_end
        segs.append(SegmentDraft(index=i, start_s=round(cur, 3), end_s=round(end, 3)))
        cur = end
        i += 1
    if not segs:
        segs = [SegmentDraft(index=0, start_s=chunk_start, end_s=chunk_end)]

    # assign words by start time
    si = 0
    for w in words:
        while si < len(segs) - 1 and w.s >= segs[si].end_s:
            si += 1
        if segs[si].start_s <= w.s or si == 0:
            segs[si].words.append(w)

    # context = previous tail + own text + next head (helps embeddings for short segments)
    for k, s in enumerate(segs):
        prev_tail = " ".join(w.w for w in segs[k - 1].words[-15:]) if k > 0 else ""
        next_head = " ".join(w.w for w in segs[k + 1].words[:15]) if k + 1 < len(segs) else ""
        s.context_text = " ".join(x for x in (prev_tail, s.text, next_head) if x).strip()
    return segs
