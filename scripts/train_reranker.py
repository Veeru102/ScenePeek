"""Fine-tune the cross-encoder reranker on mined hard negatives.

    uv run --directory backend python ../scripts/train_reranker.py \
        --data ../data/negatives/qvh_train.jsonl --version hn-v1 [--epochs 2] [--base BAAI/bge-reranker-base]

Input: JSONL from `scenepeek mine-negatives` (query, positive passage, hard-negative passages).
Output: models/reranker/<version>/ — point `SearchConfig.reranker` (or an experiment's
`search.reranker`) at that directory to compare it against the pretrained model. Runs on MPS/CUDA/CPU;
a few thousand triples train in minutes on a T4 and in well under an hour on an M-series Mac.
"""

import argparse
import json
import random
from pathlib import Path


def load_pairs(path: Path, max_neg: int, seed: int) -> tuple[list[dict], list[dict]]:
    rng = random.Random(seed)
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    rng.shuffle(records)
    n_dev = max(1, len(records) // 10)
    dev, train = records[:n_dev], records[n_dev:]

    def pairs(recs):
        out = []
        for r in recs:
            if not r["positive"].strip():
                continue  # silent scene: a text reranker cannot learn anything from an empty passage
            out.append({"query": r["query"], "passage": r["positive"], "label": 1.0})
            for n in r["negatives"][:max_neg]:
                if n["passage"].strip():
                    out.append({"query": r["query"], "passage": n["passage"], "label": 0.0})
        return out

    return pairs(train), pairs(dev)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--version", required=True)
    ap.add_argument("--base", default="BAAI/bge-reranker-base")
    ap.add_argument("--out-dir", default="../models/reranker")
    ap.add_argument("--epochs", type=float, default=2)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--max-neg", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from datasets import Dataset
    from sentence_transformers.cross_encoder import CrossEncoder, CrossEncoderTrainer, CrossEncoderTrainingArguments
    from sentence_transformers.cross_encoder.losses import BinaryCrossEntropyLoss

    train, dev = load_pairs(Path(args.data), args.max_neg, args.seed)
    print(f"train pairs={len(train)} dev pairs={len(dev)} (positives: {sum(p['label'] for p in train):.0f})")
    out = Path(args.out_dir) / args.version
    model = CrossEncoder(args.base, num_labels=1, max_length=512)
    trainer_args = CrossEncoderTrainingArguments(
        output_dir=str(out / "checkpoints"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch,
        learning_rate=args.lr,
        warmup_ratio=0.1,
        logging_steps=20,
        eval_strategy="epoch" if dev else "no",
        save_strategy="no",
        seed=args.seed,
        report_to="none",
    )
    trainer = CrossEncoderTrainer(
        model=model,
        args=trainer_args,
        train_dataset=Dataset.from_list(train),
        eval_dataset=Dataset.from_list(dev) if dev else None,
        loss=BinaryCrossEntropyLoss(model),
    )
    trainer.train()
    model.save_pretrained(str(out))
    meta = {
        "base": args.base,
        "data": str(args.data),
        "train_pairs": len(train),
        "dev_pairs": len(dev),
        "epochs": args.epochs,
        "lr": args.lr,
        "max_neg": args.max_neg,
    }
    (out / "scenepeek.json").write_text(json.dumps(meta, indent=2))
    print(f"saved {out}\nuse it with:  search:\n  reranker: {out.resolve()}")


if __name__ == "__main__":
    main()
