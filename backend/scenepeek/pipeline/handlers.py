"""Registers every job type with the worker."""

from scenepeek.jobs.registry import job
from scenepeek.pipeline.extract import extract_chunk
from scenepeek.pipeline.index import index_chunk
from scenepeek.pipeline.probe import probe_video

job("probe_video", queue="cpu")(probe_video)
job("extract_chunk", queue="cpu")(extract_chunk)
job("index_chunk", queue="ml")(index_chunk)
