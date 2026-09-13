"""FFmpeg/ffprobe helpers and the per-worker local media cache."""

import fcntl
import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from scenepeek.core import storage
from scenepeek.core.config import get_settings
from scenepeek.jobs.queue import NonRetryableError


@dataclass
class MediaInfo:
    duration_s: float
    width: int | None
    height: int | None
    fps: float | None
    vcodec: str | None
    acodec: str | None
    container: str | None
    has_audio: bool

    @property
    def web_ready(self) -> bool:
        return (
            self.vcodec == "h264"
            and (self.acodec in ("aac", None))
            and (self.container or "").startswith(("mov", "mp4"))
        )


def _run(cmd: list[str], timeout: int = 3600) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=timeout)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"{cmd[0]} failed: {e.stderr[-800:]}") from e


def probe(path: Path) -> MediaInfo:
    try:
        out = _run(
            ["ffprobe", "-v", "error", "-print_format", "json", "-show_format", "-show_streams", str(path)],
            timeout=120,
        ).stdout
    except RuntimeError as e:
        raise NonRetryableError(f"ffprobe could not read media: {e}") from e
    data = json.loads(out)
    fmt = data.get("format", {})
    v = next((s for s in data.get("streams", []) if s.get("codec_type") == "video"), None)
    a = next((s for s in data.get("streams", []) if s.get("codec_type") == "audio"), None)
    if v is None:
        raise NonRetryableError("file has no video stream")
    duration = float(fmt.get("duration") or v.get("duration") or 0)
    if duration <= 0:
        raise NonRetryableError("could not determine duration")
    fps = None
    if v.get("avg_frame_rate") and v["avg_frame_rate"] != "0/0":
        num, den = v["avg_frame_rate"].split("/")
        fps = float(num) / float(den) if float(den) else None
    return MediaInfo(
        duration_s=duration,
        width=v.get("width"),
        height=v.get("height"),
        fps=fps,
        vcodec=v.get("codec_name"),
        acodec=a.get("codec_name") if a else None,
        container=fmt.get("format_name"),
        has_audio=a is not None,
    )


def transcode_web(src: Path, dst: Path, max_height: int = 720) -> None:
    """h264/aac mp4 with faststart so browsers can seek immediately."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    _run(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-i",
            str(src),
            "-vf",
            f"scale=-2:'min({max_height},ih)'",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "23",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-b:a",
            "128k",
            "-ac",
            "2",
            "-movflags",
            "+faststart",
            str(dst),
        ]
    )


def remux_faststart(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    _run(["ffmpeg", "-y", "-v", "error", "-i", str(src), "-c", "copy", "-movflags", "+faststart", str(dst)])


def extract_audio(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    _run(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-i",
            str(src),
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-c:a",
            "pcm_s16le",
            str(dst),
        ]
    )


def silent_audio(dst: Path, duration_s: float) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    _run(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "anullsrc=r=16000:cl=mono",
            "-t",
            f"{duration_s:.3f}",
            str(dst),
        ]
    )


def poster(src: Path, dst: Path, t_s: float, width: int = 640) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    _run(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-ss",
            f"{t_s:.3f}",
            "-i",
            str(src),
            "-frames:v",
            "1",
            "-vf",
            f"scale={width}:-2",
            "-q:v",
            "3",
            str(dst),
        ]
    )


def slice_audio(src: Path, dst: Path, start_s: float, end_s: float) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    _run(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-ss",
            f"{max(0.0, start_s):.3f}",
            "-to",
            f"{end_s:.3f}",
            "-i",
            str(src),
            "-c",
            "copy",
            str(dst),
        ]
    )


def sample_frames(
    src: Path, out_dir: Path, start_s: float, end_s: float, fps: float, max_width: int
) -> list[Path]:
    """Sample frames at `fps` between start and end. Returns paths in time order."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("*.jpg"):
        old.unlink()
    _run(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-ss",
            f"{start_s:.3f}",
            "-to",
            f"{end_s:.3f}",
            "-i",
            str(src),
            "-vf",
            f"fps={fps},scale='min({max_width},iw)':-2",
            "-q:v",
            "4",
            str(out_dir / "f_%05d.jpg"),
        ]
    )
    return sorted(out_dir.glob("f_*.jpg"))


# ---- local cache -----------------------------------------------------------------


def cached(video_id: str, key: str, filename: str) -> Path:
    """Download an object once per host; concurrent workers wait on a file lock."""
    d = get_settings().cache_dir / video_id
    d.mkdir(parents=True, exist_ok=True)
    path = d / filename
    if path.exists() and path.stat().st_size > 0:
        return path
    lock = d / f".{filename}.lock"
    with open(lock, "w") as lf:
        fcntl.flock(lf, fcntl.LOCK_EX)
        try:
            if not (path.exists() and path.stat().st_size > 0):
                tmp = path.with_suffix(path.suffix + ".part")
                storage.download_file(key, tmp)
                tmp.replace(path)
        finally:
            fcntl.flock(lf, fcntl.LOCK_UN)
    return path


def drop_cache(video_id: str) -> None:
    shutil.rmtree(get_settings().cache_dir / video_id, ignore_errors=True)


def workdir(video_id: str, *parts: str) -> Path:
    d = get_settings().cache_dir / video_id / "work"
    for p in parts:
        d = d / p
    d.mkdir(parents=True, exist_ok=True)
    return d
