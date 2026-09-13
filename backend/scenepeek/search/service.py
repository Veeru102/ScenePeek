"""Multi-stage search: plan -> encode -> candidates (parallel) -> fuse -> rerank -> dedup."""

import asyncio
import html
import re
import time
import uuid
from dataclasses import dataclass, field

import numpy as np
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from scenepeek.core.config import get_settings
from scenepeek.core.db import get_sessionmaker
from scenepeek.core.metrics import SEARCH_LATENCY, record_sample
from scenepeek.models import Segment, Video
from scenepeek.search import candidates as cand
from scenepeek.search.dedup import Hit, suppress
from scenepeek.search.fusion import Fused, fuse
from scenepeek.search.planner import QueryPlan, plan


@dataclass
class SearchOptions:
    video_ids: list[uuid.UUID] | None = None
    limit: int = 20
    weights: dict[str, float] | None = None
    rerank: bool | None = None
    fusion: str | None = None
    candidates: int | None = None


@dataclass
class ResultHit:
    segment: Segment
    video: Video
    start_s: float
    end_s: float
    score: float
    signals: dict[str, float] = field(default_factory=dict)


@dataclass
class SearchResult:
    plan: QueryPlan
    hits: list[ResultHit]
    timings_ms: dict[str, float]
    total_candidates: int


async def _own(fn, *args):
    async with get_sessionmaker()() as cs:
        return await fn(cs, *args)


def _encode(p: QueryPlan) -> tuple[np.ndarray, np.ndarray]:
    from scenepeek.ml import siglip, text_embed

    return text_embed.embed_query(p.speech_q), siglip.embed_text(p.visual_q)


async def search(s: AsyncSession, query: str, opts: SearchOptions | None = None) -> SearchResult:
    opts = opts or SearchOptions()
    settings = get_settings()
    timings: dict[str, float] = {}
    t_all = time.perf_counter()

    t0 = time.perf_counter()
    p = plan(query, opts.weights)
    timings["plan"] = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    q_text, q_vis = await asyncio.to_thread(_encode, p)
    timings["encode"] = (time.perf_counter() - t0) * 1000

    k = opts.candidates or settings.search_candidates
    vids = opts.video_ids
    t0 = time.perf_counter()
    # each lane gets its own session so the four queries truly run in parallel
    text_c, vis_c, lex_c, ocr_c = await asyncio.gather(
        _own(cand.by_text_vector, q_text, k, vids),
        _own(cand.by_visual_vector, q_vis, k, vids),
        _own(cand.by_lexical, p.lexical_terms, p.exact_phrases, k, vids),
        _own(cand.by_ocr, p.ocr_q, p.lexical_terms, p.exact_phrases, k, vids),
    )
    timings["candidates"] = (time.perf_counter() - t0) * 1000
    lists = {"text": text_c, "visual": vis_c, "lexical": lex_c, "ocr": ocr_c}

    t0 = time.perf_counter()
    fused = fuse(lists, p.weights, opts.fusion or settings.fusion_method, settings.rrf_k)
    timings["fuse"] = (time.perf_counter() - t0) * 1000
    total = len(fused)
    if not fused:
        return SearchResult(p, [], _finish(timings, t_all), 0)

    # hydrate top candidates (enough for rerank + dedup headroom)
    top_n = max(settings.rerank_top_k, opts.limit * 4)
    head = fused[:top_n]
    seg_rows = {
        seg.id: seg
        for seg in await s.scalars(select(Segment).where(Segment.id.in_([f.segment_id for f in head])))
    }
    videos = {
        v.id: v
        for v in await s.scalars(
            select(Video).where(Video.id.in_({seg.video_id for seg in seg_rows.values()}))
        )
    }

    use_rerank = settings.rerank_enabled if opts.rerank is None else opts.rerank
    scored = _final_scores(head, seg_rows, p, use_rerank, settings.rerank_top_k)
    if use_rerank:
        timings["rerank"] = scored.pop("_ms")
    else:
        scored.pop("_ms", None)

    t0 = time.perf_counter()
    hits = [
        Hit(
            f.segment_id,
            seg_rows[f.segment_id].video_id,
            seg_rows[f.segment_id].start_s,
            seg_rows[f.segment_id].end_s,
            scored[f.segment_id],
        )
        for f in head
        if f.segment_id in seg_rows
    ]
    hits.sort(key=lambda h: h.score, reverse=True)
    cap = None if (vids and len(vids) == 1) else settings.max_hits_per_video
    kept = suppress(hits, settings.dedup_window_s, cap)[: opts.limit]
    timings["dedup"] = (time.perf_counter() - t0) * 1000

    out = []
    sig_by_id = {f.segment_id: f for f in head}
    for h in kept:
        seg = seg_rows[h.segment_id]
        f = sig_by_id[h.segment_id]
        signals = {m.lstrip("_"): round(v, 4) for m, v in f.signals.items()}
        signals["fused"] = round(f.fused, 4)
        out.append(ResultHit(seg, videos[seg.video_id], h.start_s, h.end_s, round(h.score, 4), signals))
    return SearchResult(p, out, _finish(timings, t_all), total)


def _final_scores(head: list[Fused], segs: dict, p: QueryPlan, use_rerank: bool, top_k: int) -> dict:
    """final = 0.5*rerank + 0.3*fused_norm + 0.2*visual_norm (rerank only for the top_k)."""
    max_f = max(f.fused for f in head) or 1.0
    vis = [f.signals.get("visual") for f in head if "visual" in f.signals]
    v_lo, v_hi = (min(vis), max(vis)) if vis else (0.0, 1.0)

    def vnorm(f: Fused) -> float:
        v = f.signals.get("visual")
        if v is None:
            return 0.0
        return (v - v_lo) / (v_hi - v_lo) if v_hi > v_lo else 1.0

    scores: dict = {}
    ms = 0.0
    rr: dict[uuid.UUID, float] = {}
    if use_rerank:
        from scenepeek.ml.reranker import rerank_scores

        t0 = time.perf_counter()
        cands = [f for f in head[:top_k] if f.segment_id in segs]
        passages = [_passage(segs[f.segment_id]) for f in cands]
        for f, r in zip(cands, rerank_scores(p.speech_q, passages), strict=True):
            rr[f.segment_id] = float(r)
            f.signals["_rerank"] = float(r)
        ms = (time.perf_counter() - t0) * 1000
    for f in head:
        base = 0.3 * (f.fused / max_f) + 0.2 * vnorm(f)
        if f.segment_id in rr:
            scores[f.segment_id] = 0.5 * rr[f.segment_id] + base
        else:
            scores[f.segment_id] = base * (1.0 if not use_rerank else 0.9)
    scores["_ms"] = ms
    return scores


def _passage(seg: Segment) -> str:
    t = seg.text or ""
    if seg.ocr_text:
        t += "\n[on screen] " + seg.ocr_text.replace("\n", " ")
    return t[:1500]


def _finish(timings: dict[str, float], t_all: float) -> dict[str, float]:
    timings["total"] = (time.perf_counter() - t_all) * 1000
    for stage, ms in timings.items():
        SEARCH_LATENCY.labels(stage=stage).observe(ms / 1000)
    record_sample("search.total", timings["total"])
    return {k: round(v, 1) for k, v in timings.items()}


def highlight(text: str, terms: list[str], max_len: int = 320) -> str:
    """HTML-escaped snippet with query terms wrapped in <mark>, centred on the first match."""
    if not text:
        return ""
    pat = (
        re.compile("|".join(re.escape(t) for t in sorted(terms, key=len, reverse=True)), re.I)
        if terms
        else None
    )
    if pat and len(text) > max_len:
        m = pat.search(text)
        if m:
            start = max(0, m.start() - max_len // 3)
            text = (
                ("…" if start else "")
                + text[start : start + max_len]
                + ("…" if start + max_len < len(text) else "")
            )
        else:
            text = text[:max_len] + "…"
    elif len(text) > max_len:
        text = text[:max_len] + "…"
    esc = html.escape(text)
    if pat:
        esc = pat.sub(lambda m: f"<mark>{m.group(0)}</mark>", esc)
    return esc
