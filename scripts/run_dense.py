import argparse
import json
import time

from toolret_research.data import load_corpus, load_queries
from toolret_research.dense import DenseRetriever
from toolret_research.metrics import evaluate_rankings


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--corpus", required=True)
    parser.add_argument("--queries", required=True)
    parser.add_argument("--k", nargs="+", type=int, default=[1, 3, 5, 10])
    parser.add_argument("--output")
    parser.add_argument(
        "--model",
        default="sentence-transformers/all-MiniLM-L6-v2",
    )

    args = parser.parse_args()

    print("Loading corpus...")
    corpus = load_corpus(args.corpus)

    print("Loading queries...")
    queries = load_queries(args.queries)

    document_ids = list(corpus.keys())

    document_texts = [
        str(corpus[doc_id]["text"])
        for doc_id in document_ids
    ]

    query_texts = [
        query.text
        for query in queries
    ]

    print(f"Loaded {len(document_ids)} documents.")
    print(f"Loaded {len(queries)} queries.")

    print(f"\nLoading dense model: {args.model}")

    retriever = DenseRetriever(args.model)

    print("\nEncoding documents...")

    build_start = time.perf_counter()

    document_embeddings = retriever.encode_documents(
        document_texts
    )

    build_seconds = time.perf_counter() - build_start

    print("\nEncoding queries...")

    query_embeddings = retriever.encode_queries(
        query_texts
    )

    max_k = max(args.k)

    print("\nSearching...")

    search_start = time.perf_counter()

    results = retriever.search(
        query_embeddings,
        document_embeddings,
        document_ids,
        k=max_k,
    )

    search_seconds = time.perf_counter() - search_start

    rankings = {
        query.id: [
            doc_id
            for doc_id, _score in query_results
        ]
        for query, query_results in zip(queries, results)
    }

    qrels = {
        query.id: query.relevant_ids
        for query in queries
    }

    metrics = evaluate_rankings(
        rankings,
        qrels,
        args.k,
    )

    result = {
        "retriever": "dense",
        "model": args.model,
        "num_documents": len(document_ids),
        "num_queries": len(queries),
        "build_seconds": build_seconds,
        "mean_query_latency_ms": (
            search_seconds / len(queries)
        ) * 1000,
        "metrics": metrics,
    }

    print("\nResults:")
    print(json.dumps(result, indent=2))

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)

        print(f"\nSaved results to {args.output}")


if __name__ == "__main__":
    main()