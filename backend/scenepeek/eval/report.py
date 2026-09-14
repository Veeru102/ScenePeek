"""Pretty-print and compare eval reports as markdown tables, with paired-bootstrap CIs."""

import json
import random
from pathlib import Path

COLS = ["mrr", "recall@1", "recall@5", "recall@10", "ndcg@10", "latency_p50_ms", "latency_p95_ms", "n"]
CMP_COLS = ["mrr", "recall@1", "recall@5", "recall@10", "ndcg@10", "latency_p50_ms"]
TEMPORAL_COLS = ["r1@0.5", "r1@0.7", "r5@0.5", "map@0.5", "map"]
DELTA_METRICS = ("mrr", "recall@5")
TEMPORAL_DELTAS = ("r1@0.5", "map")


def _cols(report: dict, base: list[str]) -> list[str]:
    """Temporal-grounding columns appear when the report has them (benchmark runs)."""
    if "r1@0.5" in report.get("overall", {}):
        return base[:1] + TEMPORAL_COLS + base[1:]
    return base


def _deltas(report: dict) -> tuple[str, ...]:
    return DELTA_METRICS + TEMPORAL_DELTAS if "r1@0.5" in report.get("overall", {}) else DELTA_METRICS


def _fmt(v) -> str:
    if isinstance(v, float):
        return f"{v:.3f}" if v <= 1 else f"{v:.0f}"
    return str(v)


def paired_bootstrap(deltas: list[float], n: int = 1000, seed: int = 0) -> tuple[float, float, float]:
    """Mean of per-query deltas with a 95% percentile-bootstrap CI. Deterministic for a given seed."""
    if not deltas:
        return 0.0, 0.0, 0.0
    rng = random.Random(seed)
    m = len(deltas)
    means = sorted(sum(rng.choices(deltas, k=m)) / m for _ in range(n))
    lo = means[int(0.025 * (n - 1))]
    hi = means[int(0.975 * (n - 1))]
    return sum(deltas) / m, lo, hi


def _lane_table(agg: dict) -> list[str]:
    lanes = agg.get("lanes")
    if not lanes:
        return []
    lines = ["", "| lane | hit@10 (alone) | queries only this lane found |", "|---|---|---|"]
    for lane, v in lanes.items():
        lines.append(f"| {lane} | {v['hit@10']:.3f} | {v['unique']} |")
    lines.append(f"| any lane (ceiling) | {agg.get('lane_ceiling', 0.0):.3f} | |")
    return lines


def print_report(report: dict) -> None:
    cols = _cols(report, COLS)
    print(f"\n## {report['name']}  ({report['overall'].get('n', 0)} queries)\n")
    print("| scope | " + " | ".join(cols) + " |")
    print("|---|" + "---|" * len(cols))
    print("| overall | " + " | ".join(_fmt(report["overall"].get(c, "")) for c in cols) + " |")
    for m, agg in report.get("by_modality", {}).items():
        print(f"| {m} | " + " | ".join(_fmt(agg.get(c, "")) for c in cols) + " |")
    print("\n".join(_lane_table(report["overall"])))
    misses = [q for q in report["queries"] if q["first_hit_rank"] is None or q["first_hit_rank"] > 5]
    if misses:
        print("\nQueries with no hit in top 5:")
        for q in misses:
            found_by = [lane for lane, v in (q.get("lanes") or {}).items() if v.get("first_rank")]
            extra = f"  lanes that had it: {', '.join(found_by)}" if found_by else ""
            print(f"  - [{q['id']}] {q['text']}  (first hit: {q['first_hit_rank']}){extra}")


def _delta_line(name: str, metric: str, base_q: dict, q_rows: list[dict]) -> str | None:
    pairs = [
        (q[metric], base_q[q["id"]][metric])
        for q in q_rows
        if q["id"] in base_q and metric in q and metric in base_q[q["id"]]
    ]
    if not pairs:
        return None
    mean, lo, hi = paired_bootstrap([a - b for a, b in pairs])
    sig = " *" if (lo > 0 or hi < 0) else ""
    return f"  {name:>20}  Δ{metric} {mean:+.3f}  [95% CI {lo:+.3f}, {hi:+.3f}]  n={len(pairs)}{sig}"


def compare_reports(paths: list[str]) -> None:
    reports = [json.loads(Path(p).read_text()) for p in paths]
    cmp_cols = _cols(reports[0], CMP_COLS)
    print("\n| config | " + " | ".join(cmp_cols) + " |")
    print("|---|" + "---|" * len(cmp_cols))
    for r in reports:
        print(f"| {r['name']} | " + " | ".join(_fmt(r["overall"].get(c, "")) for c in cmp_cols) + " |")
    if len(reports) < 2:
        return

    base = reports[0]
    base_q = {q["id"]: q for q in base["queries"]}
    print(f"\nPaired bootstrap vs {base['name']} (1000 resamples, * = 95% CI excludes 0):")
    for r in reports[1:]:
        for metric in _deltas(base):
            line = _delta_line(r["name"], metric, base_q, r["queries"])
            if line:
                print(line)

    modalities = sorted({m for r in reports for m in r.get("by_modality", {})})
    for m in modalities:
        print(f"\n### {m}\n")
        print("| config | " + " | ".join(cmp_cols) + " |")
        print("|---|" + "---|" * len(cmp_cols))
        for r in reports:
            agg = r.get("by_modality", {}).get(m, {})
            print(f"| {r['name']} | " + " | ".join(_fmt(agg.get(c, "")) for c in cmp_cols) + " |")
        for r in reports[1:]:
            rows = [q for q in r["queries"] if q["modality"] == m]
            line = _delta_line(r["name"], "mrr", base_q, rows)
            if line:
                print(line)

    print(f"\nPer-query MRR changes vs {base['name']}:")
    for r in reports[1:]:
        for q in r["queries"]:
            b = base_q.get(q["id"])
            if b and abs(q["mrr"] - b["mrr"]) > 1e-9:
                print(f"  {r['name']:>16} [{q['id']}] {b['mrr']:.2f} -> {q['mrr']:.2f}  {q['text'][:60]}")
