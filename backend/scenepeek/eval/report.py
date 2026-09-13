"""Pretty-print and compare eval reports as markdown tables."""

import json
from pathlib import Path

COLS = ["mrr", "recall@1", "recall@5", "recall@10", "ndcg@10", "latency_p50_ms", "latency_p95_ms", "n"]


def _fmt(v) -> str:
    if isinstance(v, float):
        return f"{v:.3f}" if v <= 1 else f"{v:.0f}"
    return str(v)


def print_report(report: dict) -> None:
    print(f"\n## {report['name']}  ({report['overall'].get('n', 0)} queries)\n")
    print("| scope | " + " | ".join(COLS) + " |")
    print("|---|" + "---|" * len(COLS))
    print("| overall | " + " | ".join(_fmt(report["overall"].get(c, "")) for c in COLS) + " |")
    for m, agg in report.get("by_modality", {}).items():
        print(f"| {m} | " + " | ".join(_fmt(agg.get(c, "")) for c in COLS) + " |")
    misses = [q for q in report["queries"] if q["first_hit_rank"] is None or q["first_hit_rank"] > 5]
    if misses:
        print("\nQueries with no hit in top 5:")
        for q in misses:
            print(f"  - [{q['id']}] {q['text']}  (first hit: {q['first_hit_rank']})")


def compare_reports(paths: list[str]) -> None:
    reports = [json.loads(Path(p).read_text()) for p in paths]
    cols = ["mrr", "recall@1", "recall@5", "recall@10", "ndcg@10", "latency_p50_ms"]
    print("\n| config | " + " | ".join(cols) + " |")
    print("|---|" + "---|" * len(cols))
    for r in reports:
        print(f"| {r['name']} | " + " | ".join(_fmt(r["overall"].get(c, "")) for c in cols) + " |")
    # per-query deltas vs. the first report
    if len(reports) > 1:
        base = {q["id"]: q for q in reports[0]["queries"]}
        print(f"\nPer-query MRR vs {reports[0]['name']}:")
        for r in reports[1:]:
            for q in r["queries"]:
                b = base.get(q["id"])
                if b and abs(q["mrr"] - b["mrr"]) > 1e-9:
                    print(f"  {r['name']:>16} [{q['id']}] {b['mrr']:.2f} -> {q['mrr']:.2f}  {q['text'][:60]}")
