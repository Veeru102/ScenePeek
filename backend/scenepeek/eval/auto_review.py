"""Automatic, non-interactive review of eval/real candidates.

Cleans up the "[edit] ..." templates into real query text, filters out
candidates with no genuine content signal (garbage OCR, empty templates),
and writes everything else straight into dataset.yaml. No human review.
"""

import re
from pathlib import Path

from scenepeek.eval.real import CANDIDATES_PATH, REPO_ROOT, SOURCES_PATH, load_sources
from scenepeek.eval.review import _load_candidates, _save_candidates, _save_dataset

DATASET_PATH = REPO_ROOT / "eval/real/dataset.yaml"
DICT_PATH = Path("/usr/share/dict/words")

_EDIT_PREFIX = re.compile(r"^\[edit\]\s*")


def _load_dictionary() -> set[str]:
    if not DICT_PATH.exists():
        return set()
    return {w.strip().lower() for w in DICT_PATH.read_text(errors="ignore").splitlines() if w.strip()}


_WORDS = _load_dictionary()


def _has_real_word(text: str, min_len: int = 3) -> bool:
    """True if text contains at least one token that's a real dictionary word."""
    if not _WORDS:
        return True  # no dictionary available, don't block on it
    tokens = re.findall(r"[a-zA-Z]+", text)
    return any(len(t) >= min_len and t.lower() in _WORDS for t in tokens)


def _clean_text(text: str) -> str:
    text = _EDIT_PREFIX.sub("", text).strip()
    if text and text[0].islower():
        text = text[0].upper() + text[1:]
    return text


def _ocr_fragment(text: str) -> str | None:
    m = re.search(r"(?:on-screen text mentions|while showing)\s+(.*)$", text, re.IGNORECASE)
    return m.group(1) if m else None


def should_keep(candidate: dict) -> tuple[bool, str]:
    """Decide whether to approve a candidate. Returns (keep, cleaned_text)."""
    raw_text = candidate["text"]
    was_template = bool(_EDIT_PREFIX.match(raw_text))
    text = _clean_text(raw_text)

    if not text or len(text) < 5:
        return False, text

    ctype = candidate["type"]

    if ctype == "visual":
        # No vision-captioning model in this pipeline, so a still-templated
        # visual candidate carries zero real content signal — drop it.
        # Only keep visual candidates that already have real hand-written text.
        return (not was_template), text

    if ctype == "ocr":
        frag = _ocr_fragment(raw_text) or text
        return _has_real_word(frag), text

    if ctype == "multi":
        # Keep if either the speech or OCR half has real signal.
        frag = _ocr_fragment(raw_text)
        return True if not frag else (_has_real_word(frag) or _has_real_word(text)), text

    # speech / semantic: already grounded in real transcript text
    return True, text


def auto_review() -> dict:
    cand_data = _load_candidates(CANDIDATES_PATH)
    sources = {v["key"]: v for v in load_sources(SOURCES_PATH)}

    dataset: dict = {"videos": [], "queries": []}
    approved = rejected = 0
    by_type: dict[str, int] = {}

    for c in cand_data["candidates"]:
        keep, cleaned_text = should_keep(c)
        c["text"] = cleaned_text

        if keep:
            c["status"] = "approved"
            approved += 1
            by_type[c["type"]] = by_type.get(c["type"], 0) + 1

            if c["video"] not in {v["key"] for v in dataset["videos"]}:
                sv = sources[c["video"]]
                dataset["videos"].append(
                    {"key": sv["key"], "title": sv["title"], "file": sv["file"], "license": sv.get("license")}
                )
            dataset["queries"].append(
                {
                    "id": c["id"],
                    "text": c["text"],
                    "modality": c["type"],
                    "relevant": [{"video": c["video"], "start_s": c["start_s"], "end_s": c["end_s"]}],
                }
            )
        else:
            c["status"] = "rejected"
            rejected += 1

    _save_candidates(CANDIDATES_PATH, cand_data)
    _save_dataset(DATASET_PATH, dataset)

    total = approved + rejected
    print(f"\n✅ Auto-review complete: {approved} approved, {rejected} rejected (of {total})")
    print(f"   By type: {by_type}")
    print(f"   Dataset: {len(dataset['videos'])} videos, {len(dataset['queries'])} queries -> {DATASET_PATH}")

    return {"approved": approved, "rejected": rejected, "by_type": by_type}


if __name__ == "__main__":
    auto_review()
