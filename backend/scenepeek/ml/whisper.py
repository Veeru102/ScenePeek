"""Speech-to-text with word timestamps. Backends: faster_whisper (default), mlx_whisper (Apple)."""

from dataclasses import dataclass, field
from pathlib import Path

from scenepeek.core.config import get_settings
from scenepeek.ml.registry import singleton, timed


@dataclass
class Word:
    w: str
    s: float
    e: float


@dataclass
class UtteranceOut:
    start: float
    end: float
    text: str
    words: list[Word] = field(default_factory=list)


def whisper_device() -> str:
    """CTranslate2 runs on CPU or CUDA only (no MPS), so this is independent of registry.device()."""
    s = get_settings()
    if s.whisper_device != "auto":
        return s.whisper_device
    try:
        import ctranslate2

        return "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
    except Exception:
        return "cpu"


def _load_faster_whisper():
    from faster_whisper import WhisperModel

    s = get_settings()
    dev = whisper_device()
    # int8 is the CPU default; on a GPU float16 is both faster and more accurate
    compute = s.whisper_compute_type if (dev == "cpu" or s.whisper_compute_type != "int8") else "float16"
    return WhisperModel(s.whisper_model, device=dev, compute_type=compute)


def transcribe(audio_path: Path, offset_s: float = 0.0) -> list[UtteranceOut]:
    """Transcribe a wav file. Timestamps are shifted by `offset_s` to absolute video time."""
    backend = get_settings().whisper_backend
    with timed(f"whisper_{backend}"):
        if backend == "mlx_whisper":
            return _transcribe_mlx(audio_path, offset_s)
        return _transcribe_faster(audio_path, offset_s)


def _transcribe_faster(audio_path: Path, offset_s: float) -> list[UtteranceOut]:
    model = singleton("whisper", _load_faster_whisper)
    segments, _info = model.transcribe(
        str(audio_path),
        word_timestamps=True,
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 500},
        beam_size=3,
        condition_on_previous_text=False,
    )
    out = []
    for seg in segments:
        text = seg.text.strip()
        if not text:
            continue
        words = [Word(w.word.strip(), w.start + offset_s, w.end + offset_s) for w in (seg.words or [])]
        out.append(UtteranceOut(seg.start + offset_s, seg.end + offset_s, text, words))
    return out


def _transcribe_mlx(audio_path: Path, offset_s: float) -> list[UtteranceOut]:
    import mlx_whisper  # type: ignore

    s = get_settings()
    repo = s.whisper_model if "/" in s.whisper_model else f"mlx-community/whisper-{s.whisper_model}-mlx"
    res = mlx_whisper.transcribe(str(audio_path), path_or_hf_repo=repo, word_timestamps=True)
    out = []
    for seg in res.get("segments", []):
        text = seg.get("text", "").strip()
        if not text:
            continue
        words = [
            Word(w["word"].strip(), w["start"] + offset_s, w["end"] + offset_s) for w in seg.get("words", [])
        ]
        out.append(UtteranceOut(seg["start"] + offset_s, seg["end"] + offset_s, text, words))
    return out
