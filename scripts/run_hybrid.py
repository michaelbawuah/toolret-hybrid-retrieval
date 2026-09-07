import argparse
import json
import time

from toolret_research.bm25 import BM25Index
from toolret_research.data import load_corpus, load_queries
from toolret_research.dense import DenseRetriever
from toolret_research.hybrid import reciprocal_rank_fusion
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

    parser.add_argument(
        "--candidate-k",
        type=int,
        default=100,
        help="Number of candidates retrieved from each retriever before fusion.",
    )

    parser.add_argument(
        "--rrf-k",
        type=int,
        default=60,
        help="RRF smoothing constant.",
    )

    args = parser.parse_args()

    print("Loading corpus...")
    corpus = load_corpus(args.corpus)

    print("Loading queries...")
    queries = load_queries(args.queries)

    document_ids = list(corpus.keys())

    document_texts = {
        doc_id: str(corpus[doc_id]["text"])
        for doc_id in document_ids
    }

    print(f"Loaded {len(document_ids)} documents.")
    print(f"Loaded {len(queries)} queries.")

    print("\nBuilding BM25 index...")

    bm25_start = time.perf_counter()

    bm25 = BM25Index.build(
        document_texts
    )

    bm25_build_seconds = time.perf_counter() - bm25_start

    print(f"BM25 index built in {bm25_build_seconds:.2f} seconds.")

    print(f"\nLoading dense model: {args.model}")

    dense = DenseRetriever(args.model)

    print("\nEncoding documents...")

    dense_start = time.perf_counter()

    dense_document_embeddings = dense.encode_documents(
        [document_texts[doc_id] for doc_id in document_ids]
    )

    dense_build_seconds = time.perf_counter() - dense_start

    print(
        f"Dense document embeddings built in "
        f"{dense_build_seconds:.2f} seconds."
    )

    query_texts = [
        query.text
        for query in queries
    ]

    print("\nEncoding queries...")

    query_embeddings = dense.encode_queries(
        query_texts
    )

    print("\nRunning BM25 retrieval...")

    bm25_rankings = {}

    bm25_search_start = time.perf_counter()

    for query in queries:
        results = bm25.search(
            query.text,
            k=args.candidate_k,
        )

        bm25_rankings[query.id] = [
            doc_id
            for doc_id, _score in results
        ]

    bm25_search_seconds = (
        time.perf_counter() - bm25_search_start
    )

    print("\nRunning dense retrieval...")

    dense_search_start = time.perf_counter()

    dense_results = dense.search(
        query_embeddings,
        dense_document_embeddings,
        document_ids,
        k=args.candidate_k,
    )

    dense_search_seconds = (
        time.perf_counter() - dense_search_start
    )

    dense_rankings = {
        query.id: [
            doc_id
            for doc_id, _score in query_results
        ]
        for query, query_results in zip(
            queries,
            dense_results,
        )
    }

    print("\nApplying Reciprocal Rank Fusion...")

    hybrid_rankings = {}

    fusion_start = time.perf_counter()

    for query in queries:
        hybrid_rankings[query.id] = reciprocal_rank_fusion(
            [
                bm25_rankings[query.id],
                dense_rankings[query.id],
            ],
            k=args.rrf_k,
        )

    fusion_seconds = (
        time.perf_counter() - fusion_start
    )

    qrels = {
        query.id: query.relevant_ids
        for query in queries
    }

    metrics = evaluate_rankings(
        hybrid_rankings,
        qrels,
        args.k,
    )

    result = {
        "retriever": "hybrid_rrf",
        "dense_model": args.model,
        "num_documents": len(document_ids),
        "num_queries": len(queries),
        "candidate_k": args.candidate_k,
        "rrf_k": args.rrf_k,
        "bm25_build_seconds": bm25_build_seconds,
        "dense_build_seconds": dense_build_seconds,
        "bm25_mean_query_latency_ms": (
            bm25_search_seconds / len(queries)
        ) * 1000,
        "dense_mean_query_latency_ms": (
            dense_search_seconds / len(queries)
        ) * 1000,
        "fusion_mean_query_latency_ms": (
            fusion_seconds / len(queries)
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