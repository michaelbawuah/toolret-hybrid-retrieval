import argparse
import json
import random
from pathlib import Path

from toolret_research.data import read_jsonl


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--queries", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    rows = read_jsonl(args.queries)

    print(f"Loaded {len(rows)} queries.")

    rng = random.Random(args.seed)
    rng.shuffle(rows)

    n = len(rows)

    train_end = int(n * 0.70)
    val_end = train_end + int(n * 0.15)

    train_rows = rows[:train_end]
    val_rows = rows[train_end:val_end]
    test_rows = rows[val_end:]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    splits = {
        "train": train_rows,
        "validation": val_rows,
        "test": test_rows,
    }

    for split_name, split_rows in splits.items():
        path = output_dir / f"{split_name}.jsonl"

        with path.open("w", encoding="utf-8") as f:
            for row in split_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

        print(f"{split_name}: {len(split_rows)} queries -> {path}")

    print("\nSplit complete.")
    print(f"Seed: {args.seed}")
    print(f"Train: {len(train_rows)}")
    print(f"Validation: {len(val_rows)}")
    print(f"Test: {len(test_rows)}")


if __name__ == "__main__":
    main()