"""Optional local LLM helpers (Ollama). Every function degrades to None on any failure."""

import json

import httpx

from scenepeek.core.config import get_settings

_DECOMPOSE_PROMPT = """You split video search queries into what is SAID (speech), what is SEEN (visual),
and what TEXT appears on screen (ocr). Return JSON with keys speech, visual, ocr; use an empty string when
a modality is irrelevant. Keep each value a short noun phrase. Query: {q}"""

_LABEL_PROMPT = """Give a 2-5 word title for a section of a video whose transcript keyphrases are: {kp}.
Transcript excerpt: {excerpt}
Answer with only the title."""


_available: bool | None = None


def is_available() -> bool:
    """One cheap health ping per process: is an Ollama server answering, and is the model pulled?"""
    global _available
    if _available is not None:
        return _available
    s = get_settings()
    try:
        r = httpx.get(f"{s.ollama_url}/api/tags", timeout=1.0)
        r.raise_for_status()
        names = {m.get("name", "") for m in r.json().get("models", [])}
        want = s.ollama_model
        _available = any(n == want or n.split(":")[0] == want.split(":")[0] for n in names)
    except Exception:
        _available = False
    return _available


def enabled() -> bool:
    """LLM features run when explicitly enabled, or when `ollama_auto` finds a live server."""
    s = get_settings()
    if s.ollama_enabled:
        return True
    return bool(s.ollama_auto and is_available())


def reset_cache() -> None:
    global _available
    _available = None


def _chat(prompt: str, json_mode: bool = False, timeout: float = 8.0) -> str | None:
    s = get_settings()
    try:
        r = httpx.post(
            f"{s.ollama_url}/api/generate",
            json={
                "model": s.ollama_model,
                "prompt": prompt,
                "stream": False,
                **({"format": "json"} if json_mode else {}),
            },
            timeout=timeout,
        )
        r.raise_for_status()
        return r.json().get("response")
    except Exception:
        return None


def decompose(query: str) -> dict | None:
    out = _chat(_DECOMPOSE_PROMPT.format(q=query), json_mode=True)
    if not out:
        return None
    try:
        d = json.loads(out)
        return {k: str(d.get(k, "")).strip() for k in ("speech", "visual", "ocr")}
    except Exception:
        return None


def title_for(keyphrases: list[str], excerpt: str) -> str | None:
    out = _chat(_LABEL_PROMPT.format(kp=", ".join(keyphrases), excerpt=excerpt[:600]), timeout=15.0)
    if not out:
        return None
    t = out.strip().strip('"').splitlines()[0].strip()
    return t[:80] if 2 <= len(t) <= 80 else None
