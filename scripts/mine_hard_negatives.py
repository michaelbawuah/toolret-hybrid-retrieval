from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

from toolret_research.data import load_corpus, load_queries
from toolret_research.dense import DenseRetriever
from toolret_research.hard_negatives import mine_dataset_hard_negatives
from toolret_research.text import flatten_tool


def parse_args():
    parser = argparse.ArgumentParser(
        description="Mine hard negatives from dense ToolRet retrieval results."
    )

    parser.add_argument(
        "--corpus",
        required=True,
        help="Path to ToolRet corpus JSONL file.",
    )

    parser.add_argument(
        "--queries",
        required=True,
        help="Path to ToolRet queries JSONL file.",
    )

    parser.add_argument(
        "--model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Sentence Transformer model used for mining.",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=50,
        help="Number of retrieved candidates considered for each query.",
    )

    parser.add_argument(
        "--num-negatives",
        type=int,
        default=5,
        help="Number of hard negatives stored for each positive example.",
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Path to output JSONL file.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if args.top_k <= 0:
        raise ValueError("--top-k must be greater than 0.")

    if args.num_negatives <= 0:
        raise ValueError("--num-negatives must be greater than 0.")

    if args.top_k < args.num_negatives:
        raise ValueError("--top-k must be at least --num-negatives.")

    print("Loading corpus...")
    corpus = load_corpus(args.corpus)

    print("Loading queries...")
    queries = load_queries(args.queries)

    print(f"Loaded {len(corpus)} documents.")
    print(f"Loaded {len(queries)} queries.")

    doc_ids = list(corpus.keys())

    # Convert each structured ToolRet tool into searchable text.
    document_texts = [
        flatten_tool(corpus[doc_id])
        for doc_id in doc_ids
    ]

    query_texts = [query.text for query in queries]

    print()
    print(f"Loading dense model: {args.model}")
    retriever = DenseRetriever(model_name=args.model)

    print()
    print("Encoding documents...")
    start = perf_counter()
    document_embeddings = retriever.encode_documents(document_texts)
    document_seconds = perf_counter() - start

    print(
        f"Encoded {len(document_texts)} documents "
        f"in {document_seconds:.2f} seconds."
    )

    print()
    print("Encoding queries...")
    start = perf_counter()
    query_embeddings = retriever.encode_queries(query_texts)
    query_seconds = perf_counter() - start

    print(
        f"Encoded {len(query_texts)} queries "
        f"in {query_seconds:.2f} seconds."
    )

    print()
    print(
        f"Retrieving top {args.top_k} candidates "
        "for hard-negative mining..."
    )

    rankings = {}

    start = perf_counter()

    for i, query in enumerate(queries):
        query_embedding = query_embeddings[i : i + 1]

        results = retriever.search(
            query_embedding,
            document_embeddings,
            doc_ids,
            k=args.top_k,
        )

        # DenseRetriever.search returns one ranked list per query.
        # Because we pass one query at a time, take the first list.
        if (
            isinstance(results, list)
            and len(results) == 1
            and isinstance(results[0], list)
        ):
            results = results[0]

        rankings[query.id] = results

    retrieval_seconds = perf_counter() - start

    print(
        f"Retrieved candidates for {len(queries)} queries "
        f"in {retrieval_seconds:.2f} seconds."
    )

    print()
    print("Mining hard negatives...")

    examples = mine_dataset_hard_negatives(
        queries=queries,
        rankings=rankings,
        num_negatives=args.num_negatives,
    )

    print(f"Mined {len(examples)} training examples.")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        for example in examples:
            row = {
                "query_id": example.query_id,
                "query": example.query,
                "positive_id": example.positive_id,
                "negative_ids": list(example.negative_ids),
            }

            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print()
    print(f"Saved hard negatives to {output_path}")

    print()
    print("Mining summary:")
    print(f"  Documents: {len(corpus)}")
    print(f"  Queries: {len(queries)}")
    print(f"  Top-k retrieved: {args.top_k}")
    print(f"  Negatives per example: {args.num_negatives}")
    print(f"  Training examples: {len(examples)}")
    print(f"  Document encoding time: {document_seconds:.2f}s")
    print(f"  Query encoding time: {query_seconds:.2f}s")
    print(f"  Retrieval time: {retrieval_seconds:.2f}s")


if __name__ == "__main__":
    main()