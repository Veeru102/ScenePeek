"""Priority bands. The lease query orders by priority DESC, so an interactive upload's first chunk
always beats a benchmark import, and a model backfill only runs when nothing else is waiting."""

INTERACTIVE = 100  # a person uploaded it and is waiting
NORMAL = 0
BENCHMARK = -100  # dataset imports: bulk, nobody is watching
FETCH = BENCHMARK - 50  # downloading the next benchmark clip: below indexing the ones already here
BACKFILL = -200  # re-encoding existing videos for a new model version

FIRST_CHUNK_BONUS = 50


def chunk_priority(band: int, index: int) -> int:
    """Earlier chunks first (search becomes useful sooner); the first chunk of interactive work gets
    an extra bump so time-to-first-searchable stays low under load."""
    bonus = FIRST_CHUNK_BONUS if (index == 0 and band >= NORMAL) else 0
    return band + bonus - index


def band_for_source(source: str) -> int:
    return INTERACTIVE if source == "upload" else BENCHMARK
