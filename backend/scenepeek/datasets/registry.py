from scenepeek.datasets.base import BenchmarkAdapter
from scenepeek.datasets.local_yaml import LocalYaml
from scenepeek.datasets.qvhighlights import QVHighlights

ADAPTERS: dict[str, BenchmarkAdapter] = {
    "qvhighlights": QVHighlights(),
    "scenepeek_human": LocalYaml("scenepeek_human", "eval/real/human.yaml"),
    "scenepeek_auto": LocalYaml("scenepeek_auto", "eval/real/dataset.yaml"),
    "scenepeek_synthetic": LocalYaml("scenepeek_synthetic", "eval/dataset.yaml"),
}


def get_adapter(name: str) -> BenchmarkAdapter:
    try:
        return ADAPTERS[name]
    except KeyError:
        raise KeyError(f"unknown dataset {name!r}; known: {', '.join(ADAPTERS)}") from None
