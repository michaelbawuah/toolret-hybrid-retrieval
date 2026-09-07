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
    parser.add_argument("--output")

    parser.add_argument(
        "--model",
        default="sentence-transformers/all-MiniLM-L6-v2",
    )

    parser.add_argument(
        "--candidate-values",
        nargs="+",
        type=int,
        default=[25, 50, 100, 200, 500],
    )

    parser.add_argument(
        "--rrf-k",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--eval-k",
        nargs="+",
        type=int,
        default=[1, 3, 5, 10],
    )

    args = parser.parse_args()

    max_candidate_k = max(args.candidate_values)

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

    # ---------------------------------------------------------
    # Build BM25
    # ---------------------------------------------------------

    print("\nBuilding BM25 index...")

    bm25_start = time.perf_counter()

    bm25 = BM25Index.build(document_texts)

    bm25_build_seconds = time.perf_counter() - bm25_start

    print(
        f"BM25 index built in "
        f"{bm25_build_seconds:.2f} seconds."
    )

    # ---------------------------------------------------------
    # Build dense embeddings ONCE
    # ---------------------------------------------------------

    print(f"\nLoading dense model: {args.model}")

    dense = DenseRetriever(args.model)

    print("\nEncoding documents once...")

    dense_start = time.perf_counter()

    dense_document_embeddings = dense.encode_documents(
        [document_texts[doc_id] for doc_id in document_ids]
    )

    dense_build_seconds = time.perf_counter() - dense_start

    print(
        f"Dense document embeddings built in "
        f"{dense_build_seconds:.2f} seconds."
    )

    print("\nEncoding queries once...")

    query_embeddings = dense.encode_queries(
        [query.text for query in queries]
    )

    # ---------------------------------------------------------
    # Retrieve MAXIMUM candidate pool ONCE
    # ---------------------------------------------------------

    print(
        f"\nRunning BM25 retrieval once "
        f"with top {max_candidate_k}..."
    )

    bm25_max_rankings = {}

    for query in queries:
        results = bm25.search(
            query.text,
            k=max_candidate_k,
        )

        bm25_max_rankings[query.id] = [
            doc_id
            for doc_id, _score in results
        ]

    print(
        f"\nRunning dense retrieval once "
        f"with top {max_candidate_k}..."
    )

    dense_results = dense.search(
        query_embeddings,
        dense_document_embeddings,
        document_ids,
        k=max_candidate_k,
    )

    dense_max_rankings = {
        query.id: [
            doc_id
            for doc_id, _score in query_results
        ]
        for query, query_results in zip(
            queries,
            dense_results,
        )
    }

    qrels = {
        query.id: query.relevant_ids
        for query in queries
    }

    # ---------------------------------------------------------
    # Candidate-depth ablation
    # ---------------------------------------------------------

    all_results = {}

    print("\nRunning candidate-depth ablations...")

    for candidate_k in args.candidate_values:

        start = time.perf_counter()

        hybrid_rankings = {}

        for query in queries:

            bm25_candidates = (
                bm25_max_rankings[query.id][:candidate_k]
            )

            dense_candidates = (
                dense_max_rankings[query.id][:candidate_k]
            )

            hybrid_rankings[query.id] = reciprocal_rank_fusion(
                [
                    bm25_candidates,
                    dense_candidates,
                ],
                k=args.rrf_k,
            )

        fusion_seconds = time.perf_counter() - start

        metrics = evaluate_rankings(
            hybrid_rankings,
            qrels,
            args.eval_k,
        )

        all_results[str(candidate_k)] = {
            "metrics": metrics,
            "fusion_seconds": fusion_seconds,
        }

        print(f"\nCandidate k = {candidate_k}")
        print(json.dumps(metrics, indent=2))

        print(
            f"Fusion time: "
            f"{fusion_seconds:.6f} seconds"
        )

    # ---------------------------------------------------------
    # Save experiment
    # ---------------------------------------------------------

    result = {
        "experiment": "candidate_depth_ablation",
        "retriever": "hybrid_rrf",
        "dense_model": args.model,
        "num_documents": len(document_ids),
        "num_queries": len(queries),
        "rrf_k": args.rrf_k,
        "candidate_values": args.candidate_values,
        "bm25_build_seconds": bm25_build_seconds,
        "dense_build_seconds": dense_build_seconds,
        "results": all_results,
    }

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)

        print(
            f"\nSaved candidate ablation results "
            f"to {args.output}"
        )


if __name__ == "__main__":
    main()