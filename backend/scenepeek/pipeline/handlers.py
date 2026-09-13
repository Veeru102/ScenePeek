"""Registers every job type with the worker."""

from scenepeek.jobs.registry import job
from scenepeek.pipeline.probe import probe_video

job("probe_video", queue="cpu")(probe_video)
