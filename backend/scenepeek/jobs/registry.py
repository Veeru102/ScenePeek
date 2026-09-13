"""Job handler registry. Handlers are plain functions: handler(ctx, payload)."""

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session


@dataclass
class JobContext:
    engine: Engine
    job: dict[str, Any]
    worker_id: str
    log: Any
    extra: dict[str, Any] = field(default_factory=dict)

    def session(self) -> Session:
        return Session(self.engine, expire_on_commit=False)


Handler = Callable[[JobContext, dict[str, Any]], None]


@dataclass
class JobSpec:
    type: str
    queue: str
    handler: Handler


_REGISTRY: dict[str, JobSpec] = {}


def job(type: str, *, queue: str = "cpu"):
    def deco(fn: Handler) -> Handler:
        _REGISTRY[type] = JobSpec(type=type, queue=queue, handler=fn)
        return fn

    return deco


def get_spec(type: str) -> JobSpec:
    try:
        return _REGISTRY[type]
    except KeyError:
        raise KeyError(f"no handler registered for job type {type!r}") from None


def queue_for(type: str) -> str:
    return _REGISTRY[type].queue


def load_handlers() -> None:
    """Import pipeline modules so their @job decorators register."""
    import scenepeek.pipeline.handlers  # noqa: F401
