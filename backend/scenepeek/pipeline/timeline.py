"""build_timeline: segment a video into topics from the flow of segment embeddings, then label them."""

from dataclasses import dataclass

import numpy as np
from sqlalchemy import delete, select

from scenepeek.core.config import get_settings
from scenepeek.jobs.registry import JobContext
from scenepeek.ml import keyphrases, text_embed
from scenepeek.models import Frame, Segment, Topic

MIN_TOPIC_S = 45.0
MAX_TOPICS = 24
WINDOW = 3  # segments on each side when comparing adjacent blocks


@dataclass
class Span:
    start_i: int
    end_i: int  # exclusive


def _depth_scores(emb: np.ndarray) -> np.ndarray:
    """TextTiling-style: for each gap, cosine similarity between the mean of the previous `WINDOW`
    segments and the mean of the next `WINDOW`; low similarity = likely topic boundary."""
    n = len(emb)
    gaps = np.ones(max(0, n - 1))
    for g in range(n - 1):
        left = emb[max(0, g - WINDOW + 1) : g + 1].mean(axis=0)
        right = emb[g + 1 : g + 1 + WINDOW].mean(axis=0)
        ln, rn = np.linalg.norm(left), np.linalg.norm(right)
        gaps[g] = float(left @ right / (ln * rn)) if ln and rn else 1.0
    # depth = how much lower this gap's similarity is than the peaks around it
    depth = np.zeros_like(gaps)
    for g in range(len(gaps)):
        lpeak = max(gaps[: g + 1]) if g > 0 else gaps[g]
        rpeak = max(gaps[g:]) if g < len(gaps) - 1 else gaps[g]
        depth[g] = (lpeak - gaps[g]) + (rpeak - gaps[g])
    return depth


def segment_topics(
    starts: list[float], ends: list[float], emb: np.ndarray, min_topic_s: float = MIN_TOPIC_S
) -> list[Span]:
    n = len(starts)
    if n == 0:
        return []
    if n <= 2:
        return [Span(0, n)]
    depth = _depth_scores(emb)
    thresh = depth.mean() + 0.5 * depth.std()
    cut_candidates = sorted(range(len(depth)), key=lambda g: -depth[g])
    cuts: list[int] = []
    for g in cut_candidates:
        if depth[g] < thresh or len(cuts) >= MAX_TOPICS - 1:
            break
        b = g + 1  # boundary before segment b
        # enforce minimum topic length relative to existing cuts
        neighbours = sorted(cuts + [0, n])
        prev = max(c for c in neighbours if c <= b)
        nxt = min(c for c in neighbours if c > b) if any(c > b for c in neighbours) else n
        if ends[b - 1] - starts[prev] < min_topic_s or ends[nxt - 1] - starts[b] < min_topic_s:
            continue
        cuts.append(b)
    bounds = [0] + sorted(cuts) + [n]
    return [Span(bounds[i], bounds[i + 1]) for i in range(len(bounds) - 1)]


def slide_heading(frames: list, start_s: float, end_s: float) -> str | None:
    """The tallest OCR line seen most often in a span is almost always the slide title."""
    votes: dict[str, float] = {}
    for t, boxes in frames:
        if not (start_s <= t < end_s) or not boxes:
            continue
        tallest = max(boxes, key=lambda b: _box_h(b["b"]))
        h = _box_h(tallest["b"])
        text = tallest["t"].strip()
        if 1 <= len(text.split()) <= 8 and len(text) >= 3 and h > 0:
            votes[text] = votes.get(text, 0.0) + h
    if not votes:
        return None
    return max(votes.items(), key=lambda kv: kv[1])[0][:60]


def _box_h(box: list) -> float:
    ys = [p[1] for p in box]
    return max(ys) - min(ys)


def build_timeline(ctx: JobContext, payload: dict) -> None:
    settings = get_settings()
    video_id = payload["video_id"]
    with ctx.session() as s:
        segs = list(
            s.scalars(
                select(Segment)
                .where(Segment.video_id == video_id, Segment.text_embedding.isnot(None))
                .order_by(Segment.start_s)
            )
        )
        if not segs:
            ctx.log.info("no text segments; skipping timeline")
            return
        starts = [x.start_s for x in segs]
        ends = [x.end_s for x in segs]
        emb = np.array([np.asarray(x.text_embedding, dtype=np.float32) for x in segs])
        texts = [x.text for x in segs]
        ocrs = [x.ocr_text for x in segs]
        frames = list(
            s.execute(
                select(Frame.t_s, Frame.ocr_boxes).where(
                    Frame.video_id == video_id, Frame.ocr_boxes.isnot(None)
                )
            ).all()
        )

    # short clips still deserve a few topics; long lectures get the full minimum
    spans = segment_topics(
        starts, ends, emb, min_topic_s=min(MIN_TOPIC_S, max(10.0, (ends[-1] - starts[0]) / 6))
    )
    topics = []
    use_llm = settings.ollama_enabled
    for i, sp in enumerate(spans):
        body = " ".join(texts[sp.start_i : sp.end_i])
        screen = " ".join(dict.fromkeys(" ".join(ocrs[sp.start_i : sp.end_i]).split("\n")))
        kp = keyphrases.extract((body + " " + screen).strip() or body, top_k=6)
        label = slide_heading(frames, starts[sp.start_i], ends[sp.end_i - 1])
        if label is None and use_llm:
            from scenepeek.ml.llm import title_for

            label = title_for(kp, body)
        source = "heading" if label else ("llm" if use_llm else "extractive")
        label = label or keyphrases.title_from_keyphrases(kp)
        centroid = emb[sp.start_i : sp.end_i].mean(axis=0)
        centroid = centroid / (np.linalg.norm(centroid) or 1.0)
        topics.append(
            Topic(
                video_id=video_id,
                index=i,
                start_s=starts[sp.start_i],
                end_s=ends[sp.end_i - 1],
                label=label,
                keyphrases=kp,
                embedding=centroid.tolist(),
                source=source,
            )
        )
    # make the timeline contiguous from 0 to the last segment
    if topics:
        topics[0].start_s = 0.0
        for a, b in zip(topics, topics[1:], strict=False):
            b.start_s = a.end_s

    with ctx.session() as s:
        s.execute(delete(Topic).where(Topic.video_id == video_id))
        s.add_all(topics)
        s.commit()
    _ = text_embed  # keep import explicit: label embeddings share the text model
    ctx.log.info("timeline built", topics=len(topics), labels=[t.label for t in topics])
