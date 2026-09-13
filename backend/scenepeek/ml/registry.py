"""Lazy model singletons + device selection + inference timing."""

import threading
import time
from collections.abc import Callable
from contextlib import contextmanager
from functools import lru_cache

from scenepeek.core.logging import get_logger
from scenepeek.core.metrics import MODEL_INFERENCE

log = get_logger("ml")

_lock = threading.Lock()
_instances: dict[str, object] = {}


@lru_cache
def device() -> str:
    import torch

    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def singleton[T](name: str, factory: Callable[[], T]) -> T:
    if name in _instances:
        return _instances[name]  # type: ignore[return-value]
    with _lock:
        if name not in _instances:
            t0 = time.perf_counter()
            log.info("loading model", model=name, device=device())
            _instances[name] = factory()
            log.info("model ready", model=name, seconds=round(time.perf_counter() - t0, 1))
    return _instances[name]  # type: ignore[return-value]


@contextmanager
def timed(model: str):
    t0 = time.perf_counter()
    try:
        yield
    finally:
        MODEL_INFERENCE.labels(model=model).observe(time.perf_counter() - t0)
