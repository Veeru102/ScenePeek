"""Turn a natural-language query into per-modality sub-queries and weights.

Heuristic cue detection by default; an optional local LLM (Ollama) can replace the split
with a structured decomposition. The system never depends on the LLM being present.
"""

import re
from dataclasses import asdict, dataclass, field

_LEAD = re.compile(
    r"^\s*(?:please\s+)?(?:can you\s+)?(?:find|show|search(?: for)?|locate|get|give)\s*(?:me)?\s*"
    r"(?:the\s+)?(?:all\s+)?(?:parts?|moments?|clips?|scenes?|places?|points?|segments?|times?|videos?)?\s*"
    r"(?:where|when|in which|of|that|with)?\s*",
    re.I,
)
_SPEAKER = re.compile(
    r"\b(?:the\s+)?(?:professor|lecturer|speaker|narrator|teacher|instructor|presenter|host|he|she|they|someone)\s+"
    r"(?:explains?|talks?\s+about|mentions?|says?|discuss(?:es)?|describes?|covers?|introduces?|defines?|goes\s+over)\s+",
    re.I,
)
_SPLIT = re.compile(
    r"\s+(?:while|when|as|and)\s+(?:(?:he|she|they|it)\s+)?(?:showing|displaying|presenting|there\s+is|there's|"
    r"the\s+(?:slide|screen|diagram|chart|graph|video|camera)\s+(?:shows?|displays?|has|is|reads?|says?))\s+",
    re.I,
)
_VISUAL_CUES = re.compile(
    r"\b(?:showing|shows|displayed|on\s+screen|on\s+the\s+screen|slide|diagram|chart|graph|whiteboard|"
    r"someone\s+(?:holding|wearing|carrying|riding|playing)|person|people|wearing|holding|scene|footage|"
    r"camera|visible|picture|image|looks\s+like|appears|color|red|blue|green|yellow|black|white|"
    r"umbrella|car|dog|cat|ball|basketball|court|crowd|building|street|screenshot|code|terminal)\b",
    re.I,
)
_OCR_CUES = re.compile(
    r"\b(?:slide\s+(?:says|reads|titled|title)|title|caption|text\s+(?:says|reads)|written|reads|labell?ed|"
    r"the\s+words?|heading|bullet|code|terminal|screenshot|equation|formula)\b",
    re.I,
)
_QUOTED = re.compile(r'"([^"]{2,})"')
_STOP = {
    "the",
    "a",
    "an",
    "of",
    "to",
    "in",
    "on",
    "at",
    "for",
    "and",
    "or",
    "is",
    "are",
    "was",
    "were",
    "be",
    "it",
    "its",
    "this",
    "that",
    "these",
    "those",
    "with",
    "about",
    "from",
    "by",
    "as",
    "into",
    "there",
    "where",
    "when",
    "which",
    "what",
    "who",
    "how",
    "some",
    "someone",
    "something",
    "video",
    "clip",
    "moment",
    "part",
    "scene",
    "find",
    "show",
    "me",
    "all",
    "any",
    "does",
    "do",
    "did",
    "has",
    "have",
}


@dataclass
class QueryPlan:
    raw: str
    speech_q: str
    visual_q: str
    ocr_q: str
    lexical_terms: list[str] = field(default_factory=list)
    exact_phrases: list[str] = field(default_factory=list)
    weights: dict[str, float] = field(default_factory=dict)
    cues: list[str] = field(default_factory=list)
    source: str = "heuristic"  # who produced the sub-queries: heuristic | llm
    router: str = ""  # who produced the weights
    probs: dict[str, float] = field(default_factory=dict)  # learned router: P(lane finds it)

    def to_dict(self) -> dict:
        return asdict(self)


def _clean(q: str) -> str:
    q = _QUOTED.sub(lambda m: m.group(1), q)
    q = _LEAD.sub("", q, count=1)
    return re.sub(r"\s+", " ", q).strip(" .?!,")


def terms(text: str) -> list[str]:
    toks = re.findall(r"[A-Za-z0-9][A-Za-z0-9+#.'-]*", text.lower())
    out, seen = [], set()
    for t in toks:
        t = t.strip(".'-")
        if len(t) < 2 or t in _STOP or t in seen:
            continue
        seen.add(t)
        out.append(t)
    return out


def parse(query: str) -> QueryPlan:
    """Split the query into per-lane sub-queries and detect cues. Weights are the router's job."""
    exact = _QUOTED.findall(query)
    cleaned = _clean(query)
    cues: list[str] = []

    speech_q, visual_q = cleaned, cleaned
    m = _SPLIT.search(cleaned)
    if m:
        speech_q, visual_q = cleaned[: m.start()].strip(), cleaned[m.end() :].strip()
        cues.append("split")
    sm = _SPEAKER.search(speech_q)
    if sm:
        speech_q = speech_q[sm.end() :].strip() or speech_q
        cues.append("speech")
        if not m:
            visual_q = speech_q
    ocr_q = visual_q
    if _VISUAL_CUES.search(visual_q):
        cues.append("visual")
    if _OCR_CUES.search(cleaned):
        cues.append("ocr")
        ocr_q = re.sub(r"^\s*(?:the|a|an)\s+", "", _OCR_CUES.sub("", ocr_q)).strip(" :-") or ocr_q
    if exact:
        cues.append("exact")

    p = QueryPlan(
        raw=query,
        speech_q=speech_q or cleaned,
        visual_q=visual_q or cleaned,
        ocr_q=ocr_q or cleaned,
        lexical_terms=terms(speech_q or cleaned),
        exact_phrases=exact,
        cues=cues,
    )
    from scenepeek.ml import llm as llm_mod

    if llm_mod.enabled():
        llm = llm_mod.decompose(query)
        if llm:
            p.speech_q = llm.get("speech") or p.speech_q
            p.visual_q = llm.get("visual") or p.visual_q
            p.ocr_q = llm.get("ocr") or p.ocr_q
            p.lexical_terms = terms(p.speech_q)
            p.source = "llm"
    return p


def plan(query: str, overrides: dict[str, float] | None = None, config=None) -> QueryPlan:
    """parse + route with the configured router; `overrides` pin individual lane weights."""
    from scenepeek.search.config import SearchConfig
    from scenepeek.search.routing import get_router

    cfg = config or SearchConfig.from_settings()
    p = parse(query)
    router = get_router(cfg.router, ocr_no_cue_factor=cfg.ocr_no_cue_factor)
    w = router.route(p, cfg.lanes)
    if overrides:
        for k, v in overrides.items():
            if v is not None and k in w:
                w[k] = float(v)
    p.weights = w
    p.router = router.name
    return p
