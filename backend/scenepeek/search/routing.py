"""Routing: decide how much each retrieval lane matters for this query (0 = don't run it).

query -> parse -> router -> lane weights -> selective retrieval -> fusion -> rerank

`fixed` and `heuristic` are the baselines every learned router has to beat."""

from typing import Protocol

from scenepeek.search.planner import QueryPlan


class Router(Protocol):
    name: str

    def route(self, p: QueryPlan, base: dict[str, float]) -> dict[str, float]: ...


class FixedRouter:
    """The configured weights, whatever the query says."""

    name = "fixed"

    def route(self, p: QueryPlan, base: dict[str, float]) -> dict[str, float]:
        return dict(base)


class HeuristicRouter:
    """Regex cues ("the slide that says", "someone holding", quoted phrases) nudge the weights."""

    name = "heuristic"

    def __init__(self, ocr_no_cue_factor: float = 1.0):
        self.ocr_no_cue_factor = ocr_no_cue_factor

    def route(self, p: QueryPlan, base: dict[str, float]) -> dict[str, float]:
        w = dict(base)
        cues = set(p.cues)
        visual_like = [lane for lane in ("visual", "caption", "temporal") if lane in w]
        if "split" in cues:
            for lane in visual_like:
                w[lane] *= 1.4
            w["ocr"] *= 1.3
        elif "visual" in cues and "speech" not in cues:
            for lane in visual_like:
                w[lane] *= 1.5
        if "speech" in cues and "visual" not in cues:
            w["text"] *= 1.2
            w["lexical"] *= 1.1
            for lane in visual_like:
                w[lane] *= 0.6
        if "ocr" in cues:
            w["ocr"] *= 1.6
        else:
            w["ocr"] *= self.ocr_no_cue_factor
        if "exact" in cues:
            w["lexical"] *= 1.5
            w["ocr"] *= 1.3
        return w


def get_router(name: str, *, ocr_no_cue_factor: float = 1.0) -> Router:
    if name == "fixed":
        return FixedRouter()
    if name == "heuristic":
        return HeuristicRouter(ocr_no_cue_factor)
    if name.startswith("learned:"):
        from scenepeek.search.learned_router import LearnedRouter

        return LearnedRouter.load(name.split(":", 1)[1])
    raise ValueError(f"unknown router {name!r} (fixed | heuristic | learned:<version>)")
