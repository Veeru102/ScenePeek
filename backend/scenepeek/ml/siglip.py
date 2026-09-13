"""SigLIP vision-language embeddings. The worker uses the image tower; the API only the text tower."""

import numpy as np

from scenepeek.core.config import get_settings
from scenepeek.ml.registry import device, singleton, timed


def _load_full():
    import torch
    from transformers import AutoModel, AutoProcessor

    name = get_settings().visual_embed_model
    m = AutoModel.from_pretrained(name, dtype=torch.float32).to(device()).eval()
    return m, AutoProcessor.from_pretrained(name)


def _load_text():
    import torch
    from transformers import AutoTokenizer, SiglipTextModel

    name = get_settings().visual_embed_model
    m = SiglipTextModel.from_pretrained(name, dtype=torch.float32).to(device()).eval()
    return m, AutoTokenizer.from_pretrained(name)


def embed_images(images: list, batch_size: int = 16) -> np.ndarray:
    """images: list of PIL.Image. Returns L2-normalised (n, dim)."""
    import torch

    if not images:
        return np.zeros((0, get_settings().visual_embed_dim), dtype=np.float32)
    model, processor = singleton("siglip", _load_full)
    out = []
    with timed("siglip_image"), torch.inference_mode():
        for i in range(0, len(images), batch_size):
            batch = [im.convert("RGB") for im in images[i : i + batch_size]]
            inputs = processor(images=batch, return_tensors="pt").to(device())
            feats = _tensor(model.get_image_features(**inputs))
            feats = feats / feats.norm(dim=-1, keepdim=True)
            out.append(feats.float().cpu().numpy())
    return np.concatenate(out)


def embed_text(text: str) -> np.ndarray:
    """Query-side embedding using only the text tower (light enough for the API process)."""
    import torch

    if "siglip" in _loaded():
        model, processor = singleton("siglip", _load_full)
        tok = processor.tokenizer
        get = model.get_text_features
    else:
        model, tok = singleton("siglip_text", _load_text)
        get = lambda **kw: model(**kw).pooler_output  # noqa: E731
    with timed("siglip_text"), torch.inference_mode():
        inputs = tok(
            [text.lower()], padding="max_length", max_length=64, truncation=True, return_tensors="pt"
        ).to(device())
        feats = _tensor(get(**inputs))
        feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats[0].float().cpu().numpy()


def _tensor(out):
    # transformers >=5 returns an output object; older versions return the tensor directly
    return getattr(out, "pooler_output", out)


def _loaded() -> set[str]:
    from scenepeek.ml import registry

    return set(registry._instances)
