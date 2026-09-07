from __future__ import annotations

import argparse
import json
import time

from toolret_research.data import load_corpus, load_queries
from toolret_research.dense import DenseRetriever
from toolret_research.metrics import evaluate_rankings
from toolret_research.text import flatten_tool


def evaluate_model(
    model_name,
    corpus,
    queries,
    ks,
):
    print("=" * 80)
    print(f"Evaluating model: {model_name}")
    print("=" * 80)

    document_ids = list(corpus.keys())

    document_texts = [
        flatten_tool(corpus[doc_id])
        for doc_id in document_ids
    ]

    query_texts = [
        query.text
        for query in queries
    ]

    retriever = DenseRetriever(
        model_name=model_name
    )

    print("\nEncoding documents...")

    start = time.perf_counter()

    document_embeddings = retriever.encode_documents(
        document_texts
    )

    document_encoding_seconds = (
        time.perf_counter() - start
    )

    print(
        f"Document encoding time: "
        f"{document_encoding_seconds:.2f}s"
    )

    print("\nEncoding queries...")

    start = time.perf_counter()

    query_embeddings = retriever.encode_queries(
        query_texts
    )

    query_encoding_seconds = (
        time.perf_counter() - start
    )

    print(
        f"Query encoding time: "
        f"{query_encoding_seconds:.2f}s"
    )

    max_k = max(ks)

    print("\nSearching...")

    start = time.perf_counter()

    results = retriever.search(
        query_embeddings,
        document_embeddings,
        document_ids,
        k=max_k,
    )

    search_seconds = time.perf_counter() - start

    rankings = {
        query.id: [
            doc_id
            for doc_id, _score in query_results
        ]
        for query, query_results in zip(
            queries,
            results,
        )
    }

    qrels = {
        query.id: query.relevant_ids
        for query in queries
    }

    metrics = evaluate_rankings(
        rankings,
        qrels,
        ks,
    )

    result = {
        "model": model_name,
        "num_documents": len(document_ids),
        "num_queries": len(queries),
        "document_encoding_seconds": document_encoding_seconds,
        "query_encoding_seconds": query_encoding_seconds,
        "mean_search_latency_ms": (
            search_seconds / len(queries)
        ) * 1000,
        "metrics": metrics,
    }

    print("\nMetrics:")
    print(json.dumps(metrics, indent=2))

    return result


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--corpus",
        required=True,
    )

    parser.add_argument(
        "--queries",
        required=True,
    )

    parser.add_argument(
        "--base-model",
        default="sentence-transformers/all-MiniLM-L6-v2",
    )

    parser.add_argument(
        "--fine-tuned-model",
        required=True,
    )

    parser.add_argument(
        "--k",
        nargs="+",
        type=int,
        default=[1, 3, 5, 10],
    )

    parser.add_argument(
        "--output",
        required=True,
    )

    args = parser.parse_args()

    print("Loading corpus...")
    corpus = load_corpus(args.corpus)

    print("Loading validation queries...")
    queries = load_queries(args.queries)

    print(f"Loaded {len(corpus)} documents.")
    print(f"Loaded {len(queries)} validation queries.")

    print("\nEvaluating BASE model...\n")

    base_result = evaluate_model(
        model_name=args.base_model,
        corpus=corpus,
        queries=queries,
        ks=args.k,
    )

    print("\n\nEvaluating FINE-TUNED model...\n")

    fine_tuned_result = evaluate_model(
        model_name=args.fine_tuned_model,
        corpus=corpus,
        queries=queries,
        ks=args.k,
    )

    comparison = {
        "split": "validation",
        "base": base_result,
        "fine_tuned": fine_tuned_result,
    }

    with open(
        args.output,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            comparison,
            f,
            indent=2,
        )

    print("\n" + "=" * 80)
    print("VALIDATION COMPARISON")
    print("=" * 80)

    print("\nBASE MODEL:")
    print(
        json.dumps(
            base_result["metrics"],
            indent=2,
        )
    )

    print("\nFINE-TUNED MODEL:")
    print(
        json.dumps(
            fine_tuned_result["metrics"],
            indent=2,
        )
    )

    print(
        f"\nSaved comparison to "
        f"{args.output}"
    )


if __name__ == "__main__":
    main()