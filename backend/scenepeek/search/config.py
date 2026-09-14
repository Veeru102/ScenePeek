"""The complete, serialisable description of how one search is ranked.

Built from settings by default; experiments override fields and record the whole thing, so a
number in a report can always be traced back to the exact configuration that produced it."""

from dataclasses import asdict, dataclass, field, fields, replace
from typing import Any

from scenepeek.core.config import get_settings

LANES = ("text", "lexical", "visual", "ocr", "caption", "temporal")


@dataclass(frozen=True)
class SearchConfig:
    lanes: dict[str, float] = field(default_factory=dict)  # weight per lane; 0 = lane is skipped
    router: str = "heuristic"  # fixed | heuristic | learned:<version>
    fusion: str = "weighted"  # weighted | rrf
    rrf_k: int = 60
    candidates: int = 200
    rerank: bool = True
    reranker: str = "BAAI/bge-reranker-base"  # HF id or local checkpoint path
    rerank_top_k: int = 30
    final_mix: dict[str, float] = field(default_factory=lambda: {"rerank": 0.5, "fused": 0.3, "visual": 0.2})
    caption_in_rerank_passage: bool = False
    ocr_no_cue_factor: float = 1.0
    ocr_trgm_threshold: float = 0.45
    dedup_window_s: float = 12.0
    max_hits_per_video: int = 3
    span_mode: str = "segment"  # segment | window (report the temporal lane's best window as the span)
    models: dict[str, str] = field(default_factory=dict)  # kind -> model key the lanes read

    @classmethod
    def from_settings(cls) -> "SearchConfig":
        from scenepeek.ml.versions import model_keys

        s = get_settings()
        return cls(
            lanes={
                "text": s.weight_text,
                "lexical": s.weight_lexical,
                "visual": s.weight_visual,
                "ocr": s.weight_ocr,
                "caption": s.weight_caption,
                "temporal": s.weight_temporal,
            },
            router=s.router,
            fusion=s.fusion_method,
            rrf_k=s.rrf_k,
            candidates=s.search_candidates,
            rerank=s.rerank_enabled,
            reranker=s.reranker_model,
            rerank_top_k=s.rerank_top_k,
            caption_in_rerank_passage=s.caption_in_rerank_passage,
            ocr_no_cue_factor=s.ocr_no_cue_factor,
            ocr_trgm_threshold=s.ocr_trgm_threshold,
            dedup_window_s=s.dedup_window_s,
            max_hits_per_video=s.max_hits_per_video,
            span_mode=s.span_mode,
            models={**model_keys(), "temporal": s.temporal_model},
        )

    def with_overrides(self, overrides: dict[str, Any] | None) -> "SearchConfig":
        """Type-checked overrides. Dict fields (`lanes`, `final_mix`, `models`) merge key-wise."""
        if not overrides:
            return self
        known = {f.name: f for f in fields(self)}
        changes: dict[str, Any] = {}
        for key, val in overrides.items():
            if key not in known:
                raise KeyError(f"unknown search config field {key!r}; known: {', '.join(known)}")
            if val is None:
                raise ValueError(f"{key}: null is not a valid override")
            cur = getattr(self, key)
            if isinstance(cur, dict):
                if not isinstance(val, dict):
                    raise TypeError(f"{key}: expected a mapping")
                merged = dict(cur)
                for k, v in val.items():
                    if key == "lanes" and k not in LANES:
                        raise KeyError(f"lanes.{k}: unknown lane; known: {', '.join(LANES)}")
                    merged[k] = float(v) if isinstance(cur.get(k, 0.0), float) else v
                changes[key] = merged
            elif isinstance(cur, bool):
                if not isinstance(val, bool):
                    raise TypeError(f"{key}: expected a bool, got {val!r}")
                changes[key] = val
            elif isinstance(cur, int):
                changes[key] = int(val)
            elif isinstance(cur, float):
                changes[key] = float(val)
            else:
                changes[key] = str(val)
        return replace(self, **changes)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def lane_on(self, lane: str) -> bool:
        return self.lanes.get(lane, 0.0) > 0
