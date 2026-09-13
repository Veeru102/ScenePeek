"""KeyBERT-style keyphrase extraction using the shared text embedding model (no extra deps)."""

import re

import numpy as np

from scenepeek.ml import text_embed

_STOP = set(
    """a an the and or but if of to in on at for with about from by as into than then so is are was were be
    been being have has had do does did will would can could should may might must this that these those it
    its
    we you they he she i me my our your their them his her us what which who whom whose where when why how
    all any both each few more most other some such no nor not only own same too very just also let us now
    here there very really okay ok right yeah um uh like get got going go one two three want thing things way
    kind sort lot bit talk talking say said see look going""".split()
)


def _candidates(text: str, max_n: int = 3, min_count: int = 1) -> list[str]:
    toks = [t for t in re.findall(r"[a-z][a-z0-9+#'-]*", text.lower())]
    counts: dict[str, int] = {}
    for n in range(1, max_n + 1):
        for i in range(len(toks) - n + 1):
            gram = toks[i : i + n]
            if gram[0] in _STOP or gram[-1] in _STOP or any(len(g) < 3 for g in gram):
                continue
            if len(set(gram)) != len(gram):  # "latency latency"
                continue
            if len(gram) == 1 and len(gram[0]) < 4:
                continue
            key = " ".join(gram)
            counts[key] = counts.get(key, 0) + 1
    cands = [k for k, c in counts.items() if c >= min_count]
    # prefer frequent, then longer phrases; cap for speed
    cands.sort(key=lambda k: (-counts[k], -len(k)))
    return cands[:200]


def extract(text: str, top_k: int = 5, diversity: float = 0.6) -> list[str]:
    """Rank candidate n-grams by cosine similarity to the document embedding, with MMR to keep
    the set diverse."""
    cands = _candidates(text)
    if not cands:
        return []
    doc = text_embed.embed_passages([text[:2000]])[0]
    cv = text_embed.embed_passages(cands)
    sims = cv @ doc
    chosen: list[int] = [int(np.argmax(sims))]
    while len(chosen) < min(top_k, len(cands)):
        rest = [i for i in range(len(cands)) if i not in chosen]
        red = np.max(cv[rest] @ cv[chosen].T, axis=1)
        mmr = (1 - diversity) * sims[rest] - diversity * red
        chosen.append(rest[int(np.argmax(mmr))])
    return [cands[i] for i in chosen]


def title_from_keyphrases(phrases: list[str]) -> str:
    if not phrases:
        return "Untitled"
    best = phrases[0]
    return " ".join(w if w in ("and", "of", "the", "in", "for") else w.capitalize() for w in best.split())[
        :60
    ]
