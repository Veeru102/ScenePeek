"""Registers every job type with the worker."""

from scenepeek.jobs.registry import job
from scenepeek.pipeline.extract import extract_chunk
from scenepeek.pipeline.fetch import fetch_dataset_video
from scenepeek.pipeline.index import index_chunk
from scenepeek.pipeline.probe import probe_video
from scenepeek.pipeline.temporal import encode_temporal

job("probe_video", queue="cpu")(probe_video)
job("fetch_dataset_video", queue="cpu")(fetch_dataset_video)
job("extract_chunk", queue="cpu")(extract_chunk)
job("index_chunk", queue="ml")(index_chunk)
job("encode_temporal", queue="vision")(encode_temporal)
from scenepeek.pipeline.timeline import build_timeline  # noqa: E402

job("build_timeline", queue="ml")(build_timeline)
