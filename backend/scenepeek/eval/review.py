"""Interactive terminal review of eval/real candidates: approve, reject, or edit before they
land in eval/real/dataset.yaml. Nothing here calls the search service — it just curates data."""

from pathlib import Path

import yaml

from scenepeek.eval.real import CANDIDATES_PATH, REPO_ROOT, SOURCES_PATH, load_sources

DATASET_PATH = REPO_ROOT / "eval/real/dataset.yaml"

HELP = "[a]pprove  [r]eject  [e]dit text  [t]edit times  [s]kip  [q]uit"


def _load_candidates(path: Path) -> dict:
    if not path.exists():
        return {"generated_at": None, "candidates": []}
    return yaml.safe_load(path.read_text()) or {"generated_at": None, "candidates": []}


def _save_candidates(path: Path, data: dict) -> None:
    path.write_text(yaml.dump(data, sort_keys=False, allow_unicode=True, width=100))


def _load_dataset(path: Path) -> dict:
    if not path.exists():
        return {"videos": [], "queries": []}
    return yaml.safe_load(path.read_text()) or {"videos": [], "queries": []}


def _save_dataset(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.dump(data, sort_keys=False, allow_unicode=True, width=100))


def _add_to_dataset(dataset: dict, sources: dict[str, dict], c: dict) -> None:
    if c["video"] not in {v["key"] for v in dataset["videos"]}:
        sv = sources[c["video"]]
        dataset["videos"].append(
            {"key": sv["key"], "title": sv["title"], "file": sv["file"], "license": sv.get("license")}
        )
    dataset["queries"].append(
        {
            "id": c["id"],
            "text": c["text"],
            "modality": c["type"],
            "relevant": [{"video": c["video"], "start_s": c["start_s"], "end_s": c["end_s"]}],
        }
    )


def review(
    candidates_path: Path = CANDIDATES_PATH,
    dataset_path: Path = DATASET_PATH,
    sources_path: Path = SOURCES_PATH,
    type_filter: str | None = None,
) -> None:
    cand_data = _load_candidates(candidates_path)
    if not cand_data["candidates"]:
        print(f"No candidates found at {candidates_path}. Run `scenepeek eval real-scan` first.")
        return
    sources = {v["key"]: v for v in load_sources(sources_path)}
    dataset = _load_dataset(dataset_path)

    pending = [
        c
        for c in cand_data["candidates"]
        if c.get("status", "pending") == "pending" and (type_filter is None or c["type"] == type_filter)
    ]
    if not pending:
        print("No pending candidates (everything reviewed, or --type filtered them all out).")
        return

    print(f"{len(pending)} pending candidates. {HELP}\n")
    for i, c in enumerate(pending):
        header = f"[{c['type']}] {c['video']} {c['start_s']:.1f}-{c['end_s']:.1f}s"
        print(f"--- {i + 1}/{len(pending)}  {header} ---")
        print(f"query : {c['text']}")
        if c.get("context"):
            print(f"context: {c['context']}")
        while True:
            choice = input(f"{HELP} > ").strip().lower()
            if choice in ("", "s"):
                break
            if choice == "q":
                _save_candidates(candidates_path, cand_data)
                _save_dataset(dataset_path, dataset)
                print("Saved progress. Exiting.")
                return
            if choice == "e":
                new_text = input(f"new text [{c['text']}]: ").strip()
                if new_text:
                    c["text"] = new_text
                continue
            if choice == "t":
                s_raw = input(f"start_s [{c['start_s']}]: ").strip()
                e_raw = input(f"end_s [{c['end_s']}]: ").strip()
                try:
                    if s_raw:
                        c["start_s"] = round(float(s_raw), 2)
                    if e_raw:
                        c["end_s"] = round(float(e_raw), 2)
                except ValueError:
                    print("not a number, unchanged")
                continue
            if choice == "r":
                c["status"] = "rejected"
                break
            if choice == "a":
                c["status"] = "approved"
                _add_to_dataset(dataset, sources, c)
                break
            print(f"unrecognized choice. {HELP}")
        _save_candidates(candidates_path, cand_data)  # persist after every decision
        _save_dataset(dataset_path, dataset)

    n_approved = sum(1 for c in cand_data["candidates"] if c["status"] == "approved")
    print(f"\nDone. {n_approved} approved total in {dataset_path}.")
