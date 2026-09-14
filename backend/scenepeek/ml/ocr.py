"""OCR on keyframes via RapidOCR (ONNX runtime, CPU)."""

import re
from dataclasses import dataclass

from scenepeek.core.config import get_settings
from scenepeek.ml.registry import singleton, timed


@dataclass
class OcrLine:
    text: str
    conf: float
    box: list[list[float]]


def _load():
    from rapidocr_onnxruntime import RapidOCR

    return RapidOCR()


def ocr_image(path: str, min_conf: float | None = None) -> list[OcrLine]:
    if min_conf is None:
        min_conf = get_settings().ocr_min_conf
    engine = singleton("rapidocr", _load)
    with timed("rapidocr"):
        result, _elapse = engine(path)
    lines = []
    for box, text, conf in result or []:
        if float(conf) >= min_conf and len(text.strip()) >= 2:
            lines.append(
                OcrLine(
                    text=respace(text.strip()), conf=float(conf), box=[[float(x), float(y)] for x, y in box]
                )
            )
    return lines


def lines_to_text(lines: list[OcrLine]) -> str:
    """Order lines top-to-bottom, left-to-right, and join with newlines per row."""
    if not lines:
        return ""
    lines = sorted(lines, key=lambda ln: (round(ln.box[0][1] / 12), ln.box[0][0]))
    return "\n".join(ln.text for ln in lines)


_ALPHA_RUN = re.compile(r"[A-Za-z]{7,}")
_SHORT_OK = {
    "a",
    "i",
    "of",
    "on",
    "in",
    "to",
    "at",
    "by",
    "is",
    "as",
    "or",
    "an",
    "if",
    "it",
    "be",
    "we",
    "do",
    "no",
    "up",
    "so",
}


def respace(text: str) -> str:
    """RapidOCR often drops spaces between English words ("Storageand"). Re-segment long
    alphabetic runs with a unigram word model; short tokens and mixed tokens are left alone."""
    import wordninja

    def split(m: re.Match) -> str:
        parts = wordninja.split(m.group(0))
        if len(parts) <= 1 or any(len(p) < 3 and p.lower() not in _SHORT_OK for p in parts):
            return m.group(0)
        return " ".join(parts)

    return _ALPHA_RUN.sub(split, text)
