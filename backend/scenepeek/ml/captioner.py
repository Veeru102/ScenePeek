"""Keyframe captions (BLIP). Gives silent scenes a natural-language representation the text
lane can match, instead of relying only on SigLIP's zero-shot image/text similarity."""

import re

from scenepeek.core.config import get_settings
from scenepeek.ml.registry import device, singleton, timed

# BLIP was trained on alt-text and occasionally emits "arafed"/"araffe" artifacts.
_ARTIFACTS = re.compile(r"\b(arafed|araffe|arafes)\b\s*", re.I)


def _load():
    import torch
    from transformers import BlipForConditionalGeneration, BlipProcessor

    name = get_settings().caption_model
    model = BlipForConditionalGeneration.from_pretrained(name, dtype=torch.float32).to(device()).eval()
    return model, BlipProcessor.from_pretrained(name)


def _clean(text: str) -> str:
    text = _ARTIFACTS.sub("", text).strip()
    return text[:1].lower() + text[1:] if text else ""


def caption_images(images: list, batch_size: int = 8) -> list[str]:
    """images: list of PIL.Image. Returns one short caption per image."""
    import torch

    if not images:
        return []
    model, processor = singleton("captioner", _load)
    max_new = get_settings().caption_max_tokens
    out: list[str] = []
    with timed("captioner"), torch.inference_mode():
        for i in range(0, len(images), batch_size):
            batch = [im.convert("RGB") for im in images[i : i + batch_size]]
            inputs = processor(images=batch, return_tensors="pt").to(device())
            ids = model.generate(**inputs, max_new_tokens=max_new, num_beams=3, repetition_penalty=1.3)
            out.extend(_clean(t) for t in processor.batch_decode(ids, skip_special_tokens=True))
    return out


def merge_captions(caps: list[str]) -> str:
    """Join per-keyframe captions for one segment, dropping case-insensitive duplicates."""
    seen: set[str] = set()
    kept: list[str] = []
    for c in caps:
        k = c.strip().lower()
        if k and k not in seen:
            seen.add(k)
            kept.append(c.strip())
    return ". ".join(kept)
