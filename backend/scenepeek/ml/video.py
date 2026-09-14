"""Temporal (multi-frame) video encoders behind one interface, so the temporal lane and its
backfill never care which model produced the vectors.

A window is `n_frames` frames sampled evenly across `window_s` seconds; the encoder maps it into
the same space as `embed_text`, so query/clip similarity is a cosine."""

from typing import Protocol

import numpy as np

from scenepeek.core.config import get_settings
from scenepeek.ml.registry import device, singleton, timed
from scenepeek.ml.siglip import _tensor


class VideoEncoder(Protocol):
    key: str
    dim: int
    n_frames: int

    def embed_clips(self, clips: list[list]) -> np.ndarray:
        """clips: list of frame lists (PIL.Image, exactly n_frames each). Returns L2-normalised (n, dim)."""
        ...

    def embed_text(self, text: str) -> np.ndarray: ...


class XClipEncoder:
    """microsoft/xclip-*: CLIP frames + a cross-frame transformer; text and video share a space."""

    n_frames = 8

    def __init__(self, name: str):
        self.name = name
        self.key = name.rstrip("/").split("/")[-1]
        self.dim = 512

    def _load_model(self):
        import torch
        from transformers import XCLIPModel

        return XCLIPModel.from_pretrained(self.name, dtype=torch.float32).to(device()).eval()

    def _load_tokenizer(self):
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(self.name)

    def _load_processor(self):
        # needs pillow/torchvision: only the worker (embed_clips) ever loads it
        from transformers import XCLIPProcessor

        return XCLIPProcessor.from_pretrained(self.name)

    def embed_clips(self, clips: list[list], batch_size: int = 4) -> np.ndarray:
        import torch

        if not clips:
            return np.zeros((0, self.dim), dtype=np.float32)
        model = singleton(f"video:{self.key}", self._load_model)
        proc = singleton(f"video:{self.key}:processor", self._load_processor)
        out = []
        with timed("xclip_video"), torch.inference_mode():
            for i in range(0, len(clips), batch_size):
                batch = clips[i : i + batch_size]
                pv = proc(videos=[[im.convert("RGB") for im in c] for c in batch], return_tensors="pt")
                feats = _tensor(model.get_video_features(pixel_values=pv["pixel_values"].to(device())))
                feats = feats / feats.norm(dim=-1, keepdim=True)
                out.append(feats.float().cpu().numpy())
        return np.concatenate(out)

    def embed_text(self, text: str) -> np.ndarray:
        import torch

        model = singleton(f"video:{self.key}", self._load_model)
        tokenizer = singleton(f"video:{self.key}:tokenizer", self._load_tokenizer)
        with timed("xclip_text"), torch.inference_mode():
            tok = tokenizer([text], return_tensors="pt", padding=True, truncation=True, max_length=77)
            feats = _tensor(
                model.get_text_features(
                    input_ids=tok["input_ids"].to(device()), attention_mask=tok["attention_mask"].to(device())
                )
            )
            feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats[0].float().cpu().numpy()


class MeanPoolSiglipEncoder:
    """Control: the mean of per-frame SigLIP embeddings over the window. Same text tower as the
    frame lane, no new model — if X-CLIP can't beat this, temporal modelling isn't earning its cost."""

    key = "siglip-meanpool"
    n_frames = 8

    def __init__(self):
        self.dim = get_settings().visual_embed_dim

    def embed_clips(self, clips: list[list]) -> np.ndarray:
        from scenepeek.ml import siglip

        if not clips:
            return np.zeros((0, self.dim), dtype=np.float32)
        flat = [im for c in clips for im in c]
        vecs = siglip.embed_images(flat)
        out = []
        pos = 0
        for c in clips:
            v = vecs[pos : pos + len(c)].mean(axis=0)
            pos += len(c)
            out.append(v / (np.linalg.norm(v) or 1.0))
        return np.stack(out).astype(np.float32)

    def embed_text(self, text: str) -> np.ndarray:
        from scenepeek.ml import siglip

        return siglip.embed_text(text)


def get_encoder(name: str | None = None) -> VideoEncoder:
    name = name or get_settings().temporal_model
    if name in ("siglip-meanpool", "meanpool"):
        return MeanPoolSiglipEncoder()
    if "xclip" in name.lower():
        return XClipEncoder(name)
    raise ValueError(f"unknown temporal encoder {name!r}")


def sample_windows(
    frame_times: list[float], window_s: float, stride_s: float, n_frames: int
) -> list[tuple[float, float, list[int]]]:
    """Slide a window over 1 fps frame timestamps; returns (start, end, frame indices) with exactly
    n_frames indices per window (evenly spaced, repeating at the edges of short chunks)."""
    if not frame_times:
        return []
    t0, t1 = frame_times[0], frame_times[-1]
    out = []
    start = t0
    while True:
        end = min(start + window_s, t1 + 1.0)
        inside = [i for i, t in enumerate(frame_times) if start <= t < end]
        if not inside:
            break
        idx = [
            inside[min(len(inside) - 1, round(j * (len(inside) - 1) / max(1, n_frames - 1)))]
            for j in range(n_frames)
        ]
        out.append((start, end, idx))
        if end >= t1 + 1.0:
            break
        start += stride_s
    return out
