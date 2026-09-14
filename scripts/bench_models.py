"""Time each ML stage on one 60 s chunk — no database, no MinIO, runs anywhere.

    cd backend && uv run python ../scripts/bench_models.py [video.mp4] [--json out.json]

Use it to compare hardware (M3 Pro CPU/MPS vs. a free Kaggle/Colab T4) for the README table.
"""

import argparse
import json
import platform
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from PIL import Image

from scenepeek.core.config import get_settings
from scenepeek.ml import captioner, ocr, registry, reranker, siglip, text_embed, whisper
from scenepeek.ml.whisper import whisper_device

DEFAULT_VIDEO = Path(__file__).resolve().parents[1] / "eval/videos/real/Lecture-2-Asymptotic-Notation.mp4"


def _run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True, capture_output=True)


def _prepare(video: Path, start: float, seconds: float, work: Path) -> tuple[Path, list[Path]]:
    wav = work / "chunk.wav"
    _run(
        [
            "ffmpeg",
            "-y",
            "-ss",
            str(start),
            "-t",
            str(seconds),
            "-i",
            str(video),
            "-ac",
            "1",
            "-ar",
            "16000",
            str(wav),
        ]
    )
    frames_dir = work / "frames"
    frames_dir.mkdir()
    _run(
        [
            "ffmpeg", "-y", "-ss", str(start), "-t", str(seconds), "-i", str(video),
            "-vf", "fps=1,scale=480:-2", str(frames_dir / "%03d.jpg"),
        ]
    )  # fmt: skip
    return wav, sorted(frames_dir.glob("*.jpg"))


def _timed(fn, *a, warmup: bool = True, **k) -> tuple[float, object]:
    if warmup:
        fn(*a, **k)
    t = time.perf_counter()
    out = fn(*a, **k)
    return time.perf_counter() - t, out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("video", nargs="?", default=str(DEFAULT_VIDEO))
    ap.add_argument("--start", type=float, default=300.0)
    ap.add_argument("--seconds", type=float, default=60.0)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    s = get_settings()
    hw = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "torch_device": registry.device(),
        "whisper_device": whisper_device(),
        "python": sys.version.split()[0],
    }
    try:
        import torch

        hw["torch"] = torch.__version__
        if torch.cuda.is_available():
            hw["gpu"] = torch.cuda.get_device_name(0)
    except Exception:
        pass

    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        wav, frame_paths = _prepare(Path(args.video), args.start, args.seconds, work)
        images = [Image.open(p).convert("RGB") for p in frame_paths]
        rows: dict[str, dict] = {}

        # ASR (the dominant cost): report realtime factor
        dt, utts = _timed(whisper.transcribe, wav, warmup=False)
        _ = whisper.transcribe(wav)  # warm second pass for a fair number
        dt, utts = _timed(whisper.transcribe, wav, warmup=False)
        rows["whisper"] = {"model": s.whisper_model, "seconds": dt, "realtime_x": args.seconds / dt}

        dt, _ = _timed(siglip.embed_images, images)
        rows["siglip_image"] = {
            "model": s.visual_embed_model,
            "seconds": dt,
            "ms_per_image": dt * 1000 / len(images),
        }

        texts = [u.text for u in utts] or ["silence"]
        dt, _ = _timed(text_embed.embed_passages, texts)
        rows["bge_passages"] = {
            "model": s.text_embed_model,
            "seconds": dt,
            "ms_per_text": dt * 1000 / len(texts),
        }

        sample = frame_paths[:: max(1, len(frame_paths) // 10)][:10]
        dt, _ = _timed(lambda: [ocr.ocr_image(str(p)) for p in sample])
        rows["rapidocr"] = {"model": "rapidocr-onnx", "seconds": dt, "ms_per_image": dt * 1000 / len(sample)}

        dt, caps = _timed(captioner.caption_images, images[:10])
        rows["blip_caption"] = {"model": s.caption_model, "seconds": dt, "ms_per_image": dt * 1000 / 10}

        passages = [u.text for u in utts][:30] or ["silence"] * 30
        dt, _ = _timed(reranker.rerank_scores, "what is asymptotic notation", passages)
        rows["reranker_top30"] = {"model": s.reranker_model, "seconds": dt, "ms": dt * 1000}

    print(f"\nHardware: {hw}\nChunk: {args.seconds:.0f}s of {Path(args.video).name}, {len(images)} frames\n")
    print("| stage | model | time | rate |")
    print("|---|---|---|---|")
    for name, r in rows.items():
        rate = next(
            (f"{v:.1f} {k.replace('_', ' ')}" for k, v in r.items() if k not in ("model", "seconds")), ""
        )
        print(f"| {name} | {r['model']} | {r['seconds']:.2f}s | {rate} |")
    if args.json:
        Path(args.json).write_text(json.dumps({"hardware": hw, "stages": rows}, indent=2))
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
