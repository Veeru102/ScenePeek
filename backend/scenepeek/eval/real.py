"""Lightweight candidate-query generation for the real-world eval set.

Runs the same local tooling as the indexing pipeline (probe, audio extraction, faster-whisper,
keyframe selection, RapidOCR, keyphrase extraction) directly on files in `eval/real/sources.yaml`
— no upload, no job queue, no database. Output is a list of *candidates* in
`eval/real/candidates.yaml` for a human to approve/reject/edit via `scenepeek eval real-review`;
nothing here writes to the final dataset directly.
"""

import tempfile
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import yaml

from scenepeek.core.logging import get_logger
from scenepeek.ml import keyphrases
from scenepeek.ml import ocr as ocr_ml
from scenepeek.ml import whisper as whisper_ml
from scenepeek.pipeline import media
from scenepeek.pipeline.frames import select_keyframes

log = get_logger("eval.real")

REPO_ROOT = Path(__file__).resolve().parents[3]
SOURCES_PATH = REPO_ROOT / "eval/real/sources.yaml"
CANDIDATES_PATH = REPO_ROOT / "eval/real/candidates.yaml"

WINDOW_S = 60.0
MIN_WINDOWS, MAX_WINDOWS = 4, 10
TARGET_WINDOW_STRIDE_S = 240.0  # aim for roughly one window per 4 minutes of video
SCAN_SPAN = (0.05, 0.95)  # skip opening titles / closing credits
FRAME_FPS = 1.0 / 3.0
MAX_KEYFRAMES_PER_WINDOW = 3
MIN_SPEECH_WORDS = 8
MIN_OCR_CHARS = 6
MAX_EMIT_WINDOWS_PER_VIDEO = 6  # keep total candidate volume manageable for a human reviewer


def load_sources(path: Path = SOURCES_PATH) -> list[dict]:
    data = yaml.safe_load(path.read_text()) or {}
    return data.get("videos", [])


def _load_candidates(path: Path) -> dict:
    if not path.exists():
        return {"generated_at": None, "candidates": []}
    return yaml.safe_load(path.read_text()) or {"generated_at": None, "candidates": []}


def _save_candidates(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.dump(data, sort_keys=False, allow_unicode=True, width=100))


def plan_windows(duration_s: float) -> list[tuple[float, float]]:
    """Evenly spaced ~60 s windows across the middle 90% of the video, bounded in count so a
    long lecture doesn't take as long to scan as it does to watch."""
    n = max(MIN_WINDOWS, min(MAX_WINDOWS, round(duration_s / TARGET_WINDOW_STRIDE_S)))
    lo, hi = duration_s * SCAN_SPAN[0], duration_s * SCAN_SPAN[1]
    step = max(WINDOW_S, (hi - lo) / n)
    windows = []
    for i in range(n):
        start = lo + i * step
        end = min(start + WINDOW_S, duration_s - 1)
        if end - start >= 15:
            windows.append((round(start, 2), round(end, 2)))
    return windows


@dataclass
class WindowResult:
    start_s: float
    end_s: float
    speech_text: str = ""
    ocr_texts: list[tuple[float, str]] = field(default_factory=list)
    n_keyframes: int = 0
    phrases: list[str] = field(default_factory=list)


def analyze_window(src: Path, full_audio: Path, start: float, end: float, tmp: Path) -> WindowResult:
    slice_wav = tmp / "slice.wav"
    media.slice_audio(full_audio, slice_wav, start, end)
    utterances = whisper_ml.transcribe(slice_wav, offset_s=start)
    speech_text = " ".join(u.text for u in utterances).strip()

    frame_dir = tmp / "frames"
    paths = media.sample_frames(src, frame_dir, start, end, FRAME_FPS, 480)
    sampled = [(start + i / FRAME_FPS, p) for i, p in enumerate(paths)]
    kept = select_keyframes(sampled, min_distance=8, max_gap_s=15.0)
    if len(kept) > MAX_KEYFRAMES_PER_WINDOW:
        step = len(kept) / MAX_KEYFRAMES_PER_WINDOW
        kept = [kept[int(i * step)] for i in range(MAX_KEYFRAMES_PER_WINDOW)]

    ocr_texts = []
    for t, p, _h in kept:
        text = ocr_ml.lines_to_text(ocr_ml.ocr_image(str(p)))
        if len(text.strip()) >= MIN_OCR_CHARS:
            ocr_texts.append((t, text.strip()))

    phrases = keyphrases.extract(speech_text, top_k=3) if len(speech_text.split()) >= 15 else []
    return WindowResult(start, end, speech_text, ocr_texts, len(kept), phrases)


def _candidate(video_key: str, seq, ctype: str, start: float, end: float, text: str, context: str) -> dict:
    return {
        "id": f"{video_key}.w{seq}.{ctype}",
        "video": video_key,
        "type": ctype,
        "start_s": round(start, 2),
        "end_s": round(end, 2),
        "text": text,
        "context": context,
        "status": "pending",
    }


def _window_candidates(video_key: str, i: int, w: WindowResult) -> list[dict]:
    out = []
    n_words = len(w.speech_text.split())
    ocr_text = max(w.ocr_texts, key=lambda t: len(t[1]))[1] if w.ocr_texts else ""

    if n_words >= MIN_SPEECH_WORDS:
        phrase = w.phrases[0] if w.phrases else None
        text = (
            f"[edit] talks about {phrase}"
            if phrase
            else f'[edit] "{" ".join(w.speech_text.split()[:12])}..."'
        )
        out.append(
            _candidate(video_key, i, "speech", w.start_s, w.end_s, text, f"transcript: {w.speech_text[:200]}")
        )

    if ocr_text:
        out.append(
            _candidate(
                video_key,
                i,
                "ocr",
                w.start_s,
                w.end_s,
                f"[edit] on-screen text mentions {ocr_text.splitlines()[0][:60]}",
                f"ocr: {ocr_text[:200]}",
            )
        )

    if n_words < 5 and w.n_keyframes >= 2:
        out.append(
            _candidate(
                video_key,
                i,
                "visual",
                w.start_s,
                w.end_s,
                "[edit] describe what is shown on screen",
                f"{w.n_keyframes} distinct scene(s) sampled, little or no speech/on-screen text",
            )
        )

    if n_words >= MIN_SPEECH_WORDS and ocr_text:
        phrase = w.phrases[0] if w.phrases else "the topic"
        out.append(
            _candidate(
                video_key,
                i,
                "multi",
                w.start_s,
                w.end_s,
                f"[edit] explains {phrase} while showing {ocr_text.splitlines()[0][:40]}",
                f"transcript: {w.speech_text[:120]} | ocr: {ocr_text[:120]}",
            )
        )
    return out


def _semantic_candidates(video_key: str, windows: list[WindowResult]) -> list[dict]:
    """Two broader, topic-level candidates spanning the first/second half of the scanned windows."""
    halves = (windows[: len(windows) // 2 + 1], windows[len(windows) // 2 :])
    out = []
    for h, half in enumerate(halves):
        half = [w for w in half if w.speech_text]
        if not half:
            continue
        combined = " ".join(w.speech_text for w in half)
        phrases = keyphrases.extract(combined, top_k=2) if len(combined.split()) >= 15 else []
        if not phrases:
            continue
        start, end = half[0].start_s, half[-1].end_s
        out.append(
            _candidate(
                video_key,
                f"semantic{h}",
                "semantic",
                start,
                end,
                f"[edit] the section about {phrases[0]}",
                f"combined keyphrases: {', '.join(phrases)}",
            )
        )
    return out


def scan_video(v: dict) -> list[dict]:
    src = REPO_ROOT / v["file"]
    if not src.exists():
        log.warning("video file missing, skipped", key=v["key"], file=str(src))
        return []
    info = media.probe(src)
    windows_s = plan_windows(info.duration_s)
    log.info("scanning video", key=v["key"], duration_s=round(info.duration_s), windows=len(windows_s))

    with tempfile.TemporaryDirectory(prefix=f"scenepeek-real-{v['key']}-") as tmpdir:
        tmp = Path(tmpdir)
        full_audio = tmp / "audio.wav"
        media.extract_audio(src, full_audio)

        results = []
        for i, (start, end) in enumerate(windows_s):
            win_tmp = tmp / f"w{i}"
            win_tmp.mkdir()
            w = analyze_window(src, full_audio, start, end, win_tmp)
            results.append(w)
            log.info(
                "window scanned",
                key=v["key"],
                window=i,
                words=len(w.speech_text.split()),
                ocr_hits=len(w.ocr_texts),
                keyframes=w.n_keyframes,
            )

    emit_at = _evenly_spaced_indices(len(results), min(MAX_EMIT_WINDOWS_PER_VIDEO, len(results)))
    candidates = []
    for i in emit_at:
        candidates.extend(_window_candidates(v["key"], i, results[i]))
    candidates.extend(_semantic_candidates(v["key"], results))  # uses every scanned window
    return candidates


def _evenly_spaced_indices(n: int, k: int) -> list[int]:
    """k indices spread across range(n), e.g. to sample a subset of scanned windows for
    candidates while still using every window's transcript for the semantic pass."""
    if n <= k:
        return list(range(n))
    return sorted({round(i * (n - 1) / (k - 1)) for i in range(k)}) if k > 1 else [n // 2]


def scan_videos(
    video_keys: list[str] | None = None,
    force: bool = False,
    sources_path: Path = SOURCES_PATH,
    candidates_path: Path = CANDIDATES_PATH,
) -> None:
    sources = load_sources(sources_path)
    if video_keys:
        sources = [v for v in sources if v["key"] in video_keys]

    data = _load_candidates(candidates_path)
    already = {c["video"] for c in data["candidates"]}
    todo = [v for v in sources if force or v["key"] not in already]
    if not todo:
        log.info("nothing to scan (use --force to regenerate, or --videos to pick specific ones)")
        return

    if force:
        wipe = {v["key"] for v in todo}
        dropped = sum(1 for c in data["candidates"] if c["video"] in wipe)
        if dropped:
            log.warning("dropping existing candidates for rescanned videos", n=dropped)
        data["candidates"] = [c for c in data["candidates"] if c["video"] not in wipe]

    for v in todo:
        new = scan_video(v)
        data["candidates"].extend(new)
        data["generated_at"] = datetime.now(UTC).isoformat()
        _save_candidates(candidates_path, data)  # persist after every video: safe to interrupt
        log.info("candidates written", key=v["key"], n=len(new), total=len(data["candidates"]))
