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
_GENERIC_VISUAL_TEMPLATE = "describe what is shown on screen"


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


def _window_key(candidate_id: str) -> str:
    """ "mit_streams.w0.ocr" -> "mit_streams.w0" — groups candidates from the same window."""
    return ".".join(candidate_id.split(".")[:-1])


def _build_ocr_window_index(candidates: list[dict]) -> dict[str, dict[str, set[str]]]:
    """video -> token -> set of window keys whose OCR text contains that token.

    Used to catch OCR fragments that recur throughout a video (e.g. "method",
    "(define") — technically real words, but not unique to the window they
    were pulled from, so a poor search target for that specific timestamp.
    """
    index: dict[str, dict[str, set[str]]] = {}
    for c in candidates:
        if c["type"] not in ("ocr", "multi"):
            continue
        ctx = c.get("context") or ""
        m = re.search(r"ocr:\s*(.*)$", ctx, re.IGNORECASE | re.DOTALL)
        ocr_blob = m.group(1) if m else ""
        tokens = {t.lower() for t in re.findall(r"[a-zA-Z]+", ocr_blob) if len(t) >= 3}
        video_index = index.setdefault(c["video"], {})
        wkey = _window_key(c["id"])
        for t in tokens:
            video_index.setdefault(t, set()).add(wkey)
    return index


def _is_discriminative(fragment: str, video: str, window_key: str, ocr_index: dict) -> bool:
    """True if at least one real-word token in fragment is unique to this window in its video."""
    tokens = [t.lower() for t in re.findall(r"[a-zA-Z]+", fragment) if len(t) >= 3]
    video_index = ocr_index.get(video, {})
    return any(len(video_index.get(t, set())) <= 1 for t in tokens)


def should_keep(candidate: dict, ocr_index: dict) -> tuple[bool, str]:
    """Decide whether to approve a candidate. Returns (keep, cleaned_text)."""
    raw_text = candidate["text"]
    text = _clean_text(raw_text)

    if not text or len(text) < 5:
        return False, text

    ctype = candidate["type"]

    if ctype == "visual":
        # No vision-captioning model in this pipeline, so a still-templated
        # visual candidate carries zero real content signal — drop it.
        # Only keep visual candidates that already have real hand-written text.
        # (Checked against the template phrase itself, not an "[edit]" prefix,
        # since that prefix is stripped by _clean_text and won't survive a
        # second pass over already-cleaned candidates.)
        return (text.strip().lower() != _GENERIC_VISUAL_TEMPLATE), text

    if ctype == "ocr":
        frag = _ocr_fragment(raw_text) or text
        wkey = _window_key(candidate["id"])
        keep = _has_real_word(frag) and _is_discriminative(frag, candidate["video"], wkey, ocr_index)
        return keep, text

    if ctype == "multi":
        # Keep if either the speech or OCR half has real signal.
        frag = _ocr_fragment(raw_text)
        return True if not frag else (_has_real_word(frag) or _has_real_word(text)), text

    # speech / semantic: already grounded in real transcript text
    return True, text


def auto_review() -> dict:
    cand_data = _load_candidates(CANDIDATES_PATH)
    sources = {v["key"]: v for v in load_sources(SOURCES_PATH)}
    ocr_index = _build_ocr_window_index(cand_data["candidates"])

    dataset: dict = {"videos": [], "queries": []}
    approved = rejected = 0
    by_type: dict[str, int] = {}

    for c in cand_data["candidates"]:
        keep, cleaned_text = should_keep(c, ocr_index)
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
