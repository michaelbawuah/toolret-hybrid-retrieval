from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
from sentence_transformers import CrossEncoder, InputExample
from torch.utils.data import DataLoader


def read_jsonl(path):
    rows = []

    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if line:
                rows.append(json.loads(line))

    return rows


def choose_device():
    if torch.backends.mps.is_available():
        return "mps"

    if torch.cuda.is_available():
        return "cuda"

    return "cpu"


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--training-data",
        required=True,
    )

    parser.add_argument(
        "--corpus",
        required=True,
    )

    parser.add_argument(
        "--base-model",
        default="cross-encoder/ms-marco-MiniLM-L-6-v2",
    )

    parser.add_argument(
        "--output-dir",
        required=True,
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
    )

    args = parser.parse_args()

    print("Loading training data...")
    training_rows = read_jsonl(args.training_data)

    print("Loading corpus...")
    corpus_rows = read_jsonl(args.corpus)

    corpus = {
        str(row["id"]): str(row["text"])
        for row in corpus_rows
    }

    print(f"Loaded {len(training_rows)} hard-negative records.")
    print(f"Loaded {len(corpus)} tools.")

    training_examples = []

    for row in training_rows:
        query = str(row["query"])
        positive_id = str(row["positive_id"])

        if positive_id not in corpus:
            continue

        positive_text = corpus[positive_id]

        training_examples.append(
            InputExample(
                texts=[
                    query,
                    positive_text,
                ],
                label=1.0,
            )
        )

        for negative_id in row["negative_ids"]:
            negative_id = str(negative_id)

            if negative_id not in corpus:
                continue

            negative_text = corpus[negative_id]

            training_examples.append(
                InputExample(
                    texts=[
                        query,
                        negative_text,
                    ],
                    label=0.0,
                )
            )

    print(
        f"Created {len(training_examples)} "
        "query-tool training examples."
    )

    if not training_examples:
        raise ValueError(
            "No training examples were created."
        )

    device = choose_device()

    print(f"\nUsing device: {device}")

    print(
        f"Loading base reranker model: "
        f"{args.base_model}"
    )

    model = CrossEncoder(
        args.base_model,
        num_labels=1,
        device=device,
    )

    train_dataloader = DataLoader(
        training_examples,
        shuffle=True,
        batch_size=args.batch_size,
    )

    steps_per_epoch = math.ceil(
        len(training_examples) / args.batch_size
    )

    total_steps = steps_per_epoch * args.epochs

    warmup_steps = max(
        1,
        int(total_steps * 0.10),
    )

    print("\nTraining configuration:")
    print(f"  Training records: {len(training_rows)}")
    print(f"  Query-tool examples: {len(training_examples)}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Warmup steps: {warmup_steps}")

    print(
        "\nStarting ToolRet-specific "
        "reranker training..."
    )

    model.fit(
        train_dataloader=train_dataloader,
        epochs=args.epochs,
        warmup_steps=warmup_steps,
        show_progress_bar=True,
    )

    print("\nTraining finished.")

    # ---------------------------------------------------------
    # Explicitly save the trained reranker
    # ---------------------------------------------------------

    output_path = Path(args.output_dir)
    output_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        f"Saving trained reranker to: "
        f"{output_path.resolve()}"
    )

    model.save(
        str(output_path)
    )

    # Verify that files were actually written.
    saved_files = list(output_path.iterdir())

    if not saved_files:
        raise RuntimeError(
            "Model save failed: output directory is empty."
        )

    print(
        f"Saved {len(saved_files)} model files."
    )

    print("\nTraining complete.")
    print(
        f"ToolRet reranker saved to: "
        f"{output_path.resolve()}"
    )


if __name__ == "__main__":
    main()