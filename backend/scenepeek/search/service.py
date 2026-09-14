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

from scenepeek.core.db import get_sessionmaker
from scenepeek.core.metrics import SEARCH_LATENCY, record_sample
from scenepeek.models import Segment, Video
from scenepeek.search import candidates as cand
from scenepeek.search.config import SearchConfig
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
    router: str | None = None
    debug: bool = False  # also return each lane's ranked list (used by the eval harness)
    config: SearchConfig | None = None  # full ranking configuration (experiments); settings otherwise

    def resolve(self) -> SearchConfig:
        cfg = self.config or SearchConfig.from_settings()
        over: dict = {}
        if self.rerank is not None:
            over["rerank"] = self.rerank
        if self.fusion:
            over["fusion"] = self.fusion
        if self.candidates:
            over["candidates"] = self.candidates
        if self.router:
            over["router"] = self.router
        return cfg.with_overrides(over)


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
    lanes: dict[str, list[Hit]] | None = None  # per-lane ranked candidates, only when debug=True
    config: SearchConfig | None = None
    lanes_run: list[str] = field(default_factory=list)


async def _own(fn, *args):
    async with get_sessionmaker()() as cs:
        return await fn(cs, *args)


def _encode(p: QueryPlan, on: dict[str, bool], temporal_model: str | None = None) -> dict[str, np.ndarray]:
    from scenepeek.ml import siglip, text_embed

    vecs: dict[str, np.ndarray] = {}
    if on["text"]:
        vecs["text"] = text_embed.embed_query(p.speech_q)
    if on["visual"]:
        vecs["visual"] = siglip.embed_text(p.visual_q)
    if on.get("temporal"):
        from scenepeek.ml.video import get_encoder

        vecs["temporal"] = get_encoder(temporal_model).embed_text(p.visual_q)
    if on["caption"]:
        # captions are indexed with the text model, so the caption query uses it too
        if p.visual_q == p.speech_q and "text" in vecs:
            vecs["caption"] = vecs["text"]
        else:
            vecs["caption"] = text_embed.embed_query(p.visual_q)
    return vecs


async def _empty():
    return []


async def search(s: AsyncSession, query: str, opts: SearchOptions | None = None) -> SearchResult:
    opts = opts or SearchOptions()
    cfg = opts.resolve()
    timings: dict[str, float] = {}
    t_all = time.perf_counter()

    t0 = time.perf_counter()
    p = plan(query, opts.weights, cfg)
    timings["plan"] = (time.perf_counter() - t0) * 1000

    # a lane with weight 0 is skipped entirely: no encode, no query, no signal leaking into rerank
    on = {m: p.weights.get(m, 0.0) > 0 for m in ("text", "visual", "lexical", "ocr", "caption", "temporal")}

    t0 = time.perf_counter()
    qv = await asyncio.to_thread(_encode, p, on, cfg.models.get("temporal"))
    timings["encode"] = (time.perf_counter() - t0) * 1000

    k = cfg.candidates
    vids = opts.video_ids
    t0 = time.perf_counter()
    # each lane gets its own session so the queries truly run in parallel
    from scenepeek.ml.versions import short

    temporal_key = short(cfg.models.get("temporal", ""))
    text_c, vis_c, lex_c, ocr_c, cap_c, tmp_c = await asyncio.gather(
        _own(cand.by_text_vector, qv["text"], k, vids) if on["text"] else _empty(),
        _own(cand.by_visual_vector, qv["visual"], k, vids) if on["visual"] else _empty(),
        _own(cand.by_lexical, p.lexical_terms, p.exact_phrases, k, vids) if on["lexical"] else _empty(),
        _own(cand.by_ocr, p.ocr_q, p.lexical_terms, p.exact_phrases, k, vids, cfg.ocr_trgm_threshold)
        if on["ocr"]
        else _empty(),
        _own(cand.by_caption, qv["caption"], k, vids) if on["caption"] else _empty(),
        _own(cand.by_temporal, qv["temporal"], k, temporal_key, vids) if on["temporal"] else _empty(),
    )
    timings["candidates"] = (time.perf_counter() - t0) * 1000
    lists = {
        "text": text_c,
        "visual": vis_c,
        "lexical": lex_c,
        "ocr": ocr_c,
        "caption": cap_c,
        "temporal": tmp_c,
    }
    lanes_run = [m for m, flag in on.items() if flag]

    t0 = time.perf_counter()
    fused = fuse(lists, p.weights, cfg.fusion, cfg.rrf_k)
    timings["fuse"] = (time.perf_counter() - t0) * 1000
    total = len(fused)
    if not fused:
        return SearchResult(p, [], _finish(timings, t_all), 0, config=cfg, lanes_run=lanes_run)

    # hydrate top candidates (enough for rerank + dedup headroom)
    top_n = max(cfg.rerank_top_k, opts.limit * 4)
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

    use_rerank = cfg.rerank
    scored = _final_scores(head, seg_rows, p, cfg)
    if use_rerank:
        timings["rerank"] = scored.pop("_ms")
    else:
        scored.pop("_ms", None)

    t0 = time.perf_counter()
    hits = []
    for f in head:
        if f.segment_id not in seg_rows:
            continue
        seg = seg_rows[f.segment_id]
        span = (seg.start_s, seg.end_s)
        if cfg.span_mode == "window" and "temporal" in f.spans:
            span = f.spans["temporal"]
        hits.append(Hit(f.segment_id, seg.video_id, span[0], span[1], scored[f.segment_id]))
    hits.sort(key=lambda h: h.score, reverse=True)
    cap = None if (vids and len(vids) == 1) else cfg.max_hits_per_video
    kept = suppress(hits, cfg.dedup_window_s, cap)[: opts.limit]
    timings["dedup"] = (time.perf_counter() - t0) * 1000

    out = []
    sig_by_id = {f.segment_id: f for f in head}
    for h in kept:
        seg = seg_rows[h.segment_id]
        f = sig_by_id[h.segment_id]
        signals = {m.lstrip("_"): round(v, 4) for m, v in f.signals.items()}
        signals["fused"] = round(f.fused, 4)
        out.append(ResultHit(seg, videos[seg.video_id], h.start_s, h.end_s, round(h.score, 4), signals))
    lanes = await _lane_lists(s, fused, lists.keys()) if opts.debug else None
    return SearchResult(p, out, _finish(timings, t_all), total, lanes, config=cfg, lanes_run=lanes_run)


async def _lane_lists(s: AsyncSession, fused: list, lane_names) -> dict[str, list[Hit]]:
    """Per-lane ranked Hit lists rebuilt from the ranks fusion already recorded (no lane re-runs)."""
    ids = [f.segment_id for f in fused]
    rows = await s.execute(
        select(Segment.id, Segment.video_id, Segment.start_s, Segment.end_s).where(Segment.id.in_(ids))
    )
    span = {r.id: (r.video_id, r.start_s, r.end_s) for r in rows}
    lanes: dict[str, list[Hit]] = {}
    for lane in lane_names:
        ranked = sorted((f for f in fused if lane in f.ranks), key=lambda f: f.ranks[lane])
        lanes[lane] = [
            Hit(f.segment_id, *span[f.segment_id], f.signals.get(lane, 0.0))
            for f in ranked
            if f.segment_id in span
        ]
    return lanes


def _final_scores(head: list[Fused], segs: dict, p: QueryPlan, cfg: SearchConfig) -> dict:
    """final = mix.rerank*rerank + mix.fused*fused_norm + mix.visual*w_visual*visual_norm
    (rerank only for the top_k).

    The visual term is scaled by the plan's visual weight so a weight of 0 (e.g. a
    text-only ablation) contributes nothing; otherwise a "text only" run would still
    be quietly ranked by SigLIP.
    """
    use_rerank, top_k, mix = cfg.rerank, cfg.rerank_top_k, cfg.final_mix
    max_f = max(f.fused for f in head) or 1.0
    w_vis = max(p.weights.get("visual", 0.0), p.weights.get("temporal", 0.0))
    vis_coef = mix["visual"] * min(1.0, max(0.0, w_vis))
    norms = {}
    for lane in ("visual", "temporal"):
        if p.weights.get(lane, 0.0) <= 0:
            continue
        vals = [f.signals[lane] for f in head if lane in f.signals]
        if vals:
            norms[lane] = (min(vals), max(vals))

    def vnorm(f: Fused) -> float:
        """Best min-max-normalised visual evidence across the frame and temporal lanes."""
        best = 0.0
        for lane, (lo, hi) in norms.items():
            v = f.signals.get(lane)
            if v is None:
                continue
            best = max(best, (v - lo) / (hi - lo) if hi > lo else 1.0)
        return best

    scores: dict = {}
    ms = 0.0
    rr: dict[uuid.UUID, float] = {}
    if use_rerank:
        from scenepeek.ml.reranker import rerank_scores

        t0 = time.perf_counter()
        cands = [f for f in head[:top_k] if f.segment_id in segs]
        passages = [_passage(segs[f.segment_id], cfg.caption_in_rerank_passage) for f in cands]
        for f, r in zip(cands, rerank_scores(p.speech_q, passages, cfg.reranker), strict=True):
            rr[f.segment_id] = float(r)
            f.signals["_rerank"] = float(r)
        ms = (time.perf_counter() - t0) * 1000
    for f in head:
        base = mix["fused"] * (f.fused / max_f) + vis_coef * vnorm(f)
        if f.segment_id in rr:
            scores[f.segment_id] = mix["rerank"] * rr[f.segment_id] + base
        else:
            scores[f.segment_id] = base * (1.0 if not use_rerank else 0.9)
    scores["_ms"] = ms
    return scores


def _passage(seg: Segment, with_caption: bool = False) -> str:
    t = seg.text or ""
    if seg.ocr_text:
        t += "\n[on screen] " + seg.ocr_text.replace("\n", " ")
    if seg.caption_text and with_caption:
        t += "\n[visual] " + seg.caption_text
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
