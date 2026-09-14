"""Render the README evaluation tables from eval/reports/*.json so the numbers are reproducible.

    python scripts/eval_tables.py            # prints markdown
    python scripts/eval_tables.py --write    # rewrites the block between the markers in README.md
"""

import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "eval/reports"
README = ROOT / "README.md"
BEGIN, END = "<!-- eval-tables:begin -->", "<!-- eval-tables:end -->"

ABLATIONS = [
    ("text_only", "dense text only (no fusion, no rerank)"),
    ("no_rerank", "− cross-encoder rerank"),
    ("no_lexical", "− keyword (FTS) lane"),
    ("no_visual", "− visual (SigLIP) lane"),
    ("no_ocr", "− OCR lane"),
    ("rrf", "RRF fusion instead of weighted"),
    ("with_caption", "+ BLIP caption lane (w=0.7)"),
    ("temporal", "+ temporal lane (X-CLIP, w=0.8)"),
    ("temporal_only", "temporal lane instead of SigLIP frames"),
    ("temporal_window", "+ temporal lane, window spans"),
    ("router_fixed", "fixed weights (no cue heuristics)"),
    ("router_learned", "learned router"),
    ("rerank_hn", "hard-negative-tuned reranker"),
]
SETS = [
    ("human", "Hand-written queries (41, the headline number)", "mrr"),
    ("auto", "Auto-generated queries (119)", "mrr"),
    ("qvh_val", "QVHighlights val (single-video grounding, R1@0.5 is the literature metric)", "r1@0.5"),
]


def _load(name: str) -> dict | None:
    p = REPORTS / f"{name}.json"
    return json.loads(p.read_text()) if p.exists() else None


def _ci(base: dict, other: dict, metric: str = "mrr") -> str:
    b = {q["id"]: q[metric] for q in base["queries"]}
    deltas = [q[metric] - b[q["id"]] for q in other["queries"] if q["id"] in b]
    if not deltas:
        return ""
    rng = random.Random(0)
    m = len(deltas)
    means = sorted(sum(rng.choices(deltas, k=m)) / m for _ in range(1000))
    lo, hi = means[int(0.025 * 999)], means[int(0.975 * 999)]
    star = " **\\***" if (lo > 0 or hi < 0) else ""
    return f"{sum(deltas) / m:+.3f} [{lo:+.3f}, {hi:+.3f}]{star}"


def ablation_table(prefix: str, delta_metric: str = "mrr") -> str:
    base = _load(prefix)
    if not base:
        return f"_no report for {prefix}_"
    o = base["overall"]
    temporal = prefix.startswith("qvh") and "r1@0.5" in o
    cols = ["mrr", "r1@0.5", "r1@0.7", "map"] if temporal else ["mrr", "recall@1", "recall@5", "recall@10"]
    head = ["MRR", "R1@0.5", "R1@0.7", "mAP"] if temporal else ["MRR", "R@1", "R@5", "R@10"]
    rows = [
        f"| configuration | {' | '.join(head)} | Δ{delta_metric} vs default [95% CI] |",
        "|---|" + "---|" * (len(cols) + 1),
        "| **default** (heuristic routing, weighted fusion of text + keyword + visual + OCR, reranked) | "
        + " | ".join(f"**{o[c]:.3f}**" if i == 0 else f"{o[c]:.3f}" for i, c in enumerate(cols))
        + " | — |",
    ]
    for key, label in ABLATIONS:
        r = _load(f"{prefix}_{key}")
        if not r:
            continue
        a = r["overall"]
        rows.append(f"| {label} | " + " | ".join(f"{a[c]:.3f}" for c in cols) + f" | {_ci(base, r, delta_metric)} |")
    return "\n".join(rows)


def modality_table(prefix: str) -> str:
    base = _load(prefix)
    if not base:
        return ""
    rows = ["| modality | n | MRR | R@5 | strongest single lane (hit@10 alone) |", "|---|---|---|---|---|"]
    for m, agg in base["by_modality"].items():
        lanes = agg.get("lanes", {})
        best = max(lanes.items(), key=lambda kv: kv[1]["hit@10"]) if lanes else None
        best_s = f"{best[0]} ({best[1]['hit@10']:.2f})" if best else "—"
        rows.append(f"| {m} | {agg['n']} | {agg['mrr']:.3f} | {agg['recall@5']:.3f} | {best_s} |")
    return "\n".join(rows)


def lane_table(prefix: str) -> str:
    base = _load(prefix)
    if not base or "lanes" not in base["overall"]:
        return ""
    o = base["overall"]
    rows = ["| lane | had the answer in its own top-10 | only lane that found it |", "|---|---|---|"]
    for lane, v in sorted(o["lanes"].items(), key=lambda kv: -kv[1]["hit@10"]):
        rows.append(f"| {lane} | {v['hit@10']:.2f} | {v['unique']} |")
    rows.append(f"| **any lane (ceiling)** | **{o['lane_ceiling']:.2f}** | fused system: {o['hit@10']:.2f} |")
    return "\n".join(rows)


def render() -> str:
    parts = []
    for prefix, title, metric in SETS:
        if not _load(prefix):
            continue
        parts.append(f"**{title}**\n\n{ablation_table(prefix, metric)}\n\n{modality_table(prefix)}\n\n{lane_table(prefix)}")
    return "\n\n".join(parts)


if __name__ == "__main__":
    md = render()
    if "--write" in sys.argv:
        text = README.read_text()
        i, j = text.index(BEGIN) + len(BEGIN), text.index(END)
        README.write_text(text[:i] + "\n" + md + "\n" + text[j:])
        print("README updated")
    else:
        print(md)
