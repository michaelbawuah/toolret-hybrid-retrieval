from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
from sentence_transformers import (
    InputExample,
    SentenceTransformer,
    losses,
)
from torch.utils.data import DataLoader

from toolret_research.data import load_corpus
from toolret_research.text import flatten_tool


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
        "--corpus",
        required=True,
    )

    parser.add_argument(
        "--training-data",
        required=True,
    )

    parser.add_argument(
        "--model",
        default="sentence-transformers/all-MiniLM-L6-v2",
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

    parser.add_argument(
        "--margin",
        type=float,
        default=0.2,
    )

    args = parser.parse_args()

    print("Loading ToolRet corpus...")
    corpus = load_corpus(args.corpus)

    print("Loading hard-negative training data...")
    rows = read_jsonl(args.training_data)

    print(f"Loaded {len(corpus)} tools.")
    print(f"Loaded {len(rows)} hard-negative records.")

    print("\nPreparing tool text...")

    tool_text = {
        tool_id: flatten_tool(tool)
        for tool_id, tool in corpus.items()
    }

    training_examples = []
    skipped = 0

    for row in rows:
        positive_id = str(row["positive_id"])
        negative_ids = row["negative_ids"]

        if positive_id not in tool_text:
            skipped += 1
            continue

        positive_text = tool_text[positive_id]
        query_text = str(row["query"])

        for negative_id in negative_ids:
            negative_id = str(negative_id)

            if negative_id not in tool_text:
                skipped += 1
                continue

            negative_text = tool_text[negative_id]

            training_examples.append(
                InputExample(
                    texts=[
                        query_text,
                        positive_text,
                        negative_text,
                    ]
                )
            )

    print(
        f"Created {len(training_examples)} training triplets."
    )

    if skipped:
        print(f"Skipped {skipped} missing tool references.")

    if not training_examples:
        raise ValueError("No training examples were created.")

    device = choose_device()

    print(f"\nUsing device: {device}")

    print(f"Loading base model: {args.model}")

    model = SentenceTransformer(
        args.model,
        device=device,
    )

    train_dataloader = DataLoader(
        training_examples,
        shuffle=True,
        batch_size=args.batch_size,
    )

    train_loss = losses.TripletLoss(
        model=model,
        triplet_margin=args.margin,
    )

    total_steps = (
        math.ceil(
            len(training_examples) / args.batch_size
        )
        * args.epochs
    )

    warmup_steps = max(
        1,
        int(total_steps * 0.10),
    )

    print("\nTraining configuration:")
    print(f"  Training records: {len(rows)}")
    print(f"  Triplets: {len(training_examples)}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Margin: {args.margin}")
    print(f"  Warmup steps: {warmup_steps}")

    print("\nStarting fine-tuning...")

    model.fit(
        train_objectives=[
            (
                train_dataloader,
                train_loss,
            )
        ],
        epochs=args.epochs,
        warmup_steps=warmup_steps,
        output_path=args.output_dir,
        show_progress_bar=True,
    )

    print("\nTraining complete.")
    print(
        f"Fine-tuned model saved to: "
        f"{args.output_dir}"
    )


if __name__ == "__main__":
    main()