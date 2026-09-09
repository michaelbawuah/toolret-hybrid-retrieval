from __future__ import annotations

import argparse
import json
import time

from toolret_research.bm25 import BM25Index
from toolret_research.data import load_corpus, load_queries
from toolret_research.dense import DenseRetriever
from toolret_research.fusion import weighted_reciprocal_rank_fusion
from toolret_research.metrics import evaluate_rankings
from toolret_research.reranker import CrossEncoderReranker
from toolret_research.text import flatten_tool


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--corpus", required=True)
    parser.add_argument("--queries", required=True)
    parser.add_argument("--fine-tuned-model", required=True)
    parser.add_argument("--output", required=True)

    parser.add_argument(
        "--reranker-model",
        default="cross-encoder/ms-marco-MiniLM-L-6-v2",
    )

    parser.add_argument(
        "--candidate-k",
        type=int,
        default=500,
        help="Candidates retrieved by each first-stage retriever.",
    )

    parser.add_argument(
        "--rerank-k",
        type=int,
        default=25,
        help="Number of fused candidates sent to the cross-encoder.",
    )

    parser.add_argument(
        "--rrf-k",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--bm25-weight",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--dense-weight",
        type=float,
        default=1.25,
    )

    parser.add_argument(
        "--eval-k",
        nargs="+",
        type=int,
        default=[1, 3, 5, 10],
    )

    args = parser.parse_args()

    if args.rerank_k <= 0:
        raise ValueError("--rerank-k must be greater than 0.")

    if args.rerank_k > args.candidate_k * 2:
        raise ValueError(
            "--rerank-k is larger than the possible fused candidate pool."
        )

    print("Loading corpus...")
    corpus = load_corpus(args.corpus)

    print("Loading validation queries...")
    queries = load_queries(args.queries)

    print(f"Loaded {len(corpus)} documents.")
    print(f"Loaded {len(queries)} validation queries.")

    document_ids = list(corpus.keys())

    print("\nPreparing document text...")

    document_texts = {
        doc_id: flatten_tool(corpus[doc_id])
        for doc_id in document_ids
    }

    # ---------------------------------------------------------
    # BM25
    # ---------------------------------------------------------

    print("\nBuilding BM25 index...")

    start = time.perf_counter()

    bm25 = BM25Index.build(document_texts)

    bm25_build_seconds = time.perf_counter() - start

    print(
        f"BM25 index built in "
        f"{bm25_build_seconds:.2f}s"
    )

    # ---------------------------------------------------------
    # Fine-tuned dense retriever
    # ---------------------------------------------------------

    print(
        f"\nLoading fine-tuned dense model: "
        f"{args.fine_tuned_model}"
    )

    dense = DenseRetriever(
        model_name=args.fine_tuned_model
    )

    print("\nEncoding dense documents...")

    start = time.perf_counter()

    dense_document_embeddings = dense.encode_documents(
        [
            document_texts[doc_id]
            for doc_id in document_ids
        ]
    )

    dense_build_seconds = time.perf_counter() - start

    print(
        f"Dense document encoding time: "
        f"{dense_build_seconds:.2f}s"
    )

    print("\nEncoding queries...")

    query_embeddings = dense.encode_queries(
        [
            query.text
            for query in queries
        ]
    )

    # ---------------------------------------------------------
    # First-stage retrieval
    # ---------------------------------------------------------

    print("\nRunning BM25 retrieval...")

    bm25_rankings = {}

    start = time.perf_counter()

    for query in queries:
        results = bm25.search(
            query.text,
            k=args.candidate_k,
        )

        bm25_rankings[query.id] = [
            doc_id
            for doc_id, _score in results
        ]

    bm25_search_seconds = time.perf_counter() - start

    print("\nRunning dense retrieval...")

    start = time.perf_counter()

    dense_results = dense.search(
        query_embeddings,
        dense_document_embeddings,
        document_ids,
        k=args.candidate_k,
    )

    dense_search_seconds = time.perf_counter() - start

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

    # ---------------------------------------------------------
    # Weighted RRF
    # ---------------------------------------------------------

    print("\nApplying weighted RRF...")

    fused_rankings = {}

    start = time.perf_counter()

    for query in queries:
        fused_rankings[query.id] = (
            weighted_reciprocal_rank_fusion(
                rankings=[
                    bm25_rankings[query.id],
                    dense_rankings[query.id],
                ],
                weights=[
                    args.bm25_weight,
                    args.dense_weight,
                ],
                k=args.rrf_k,
            )
        )

    fusion_seconds = time.perf_counter() - start

    # ---------------------------------------------------------
    # Cross-encoder reranking
    # ---------------------------------------------------------

    print(
        f"\nLoading cross-encoder reranker: "
        f"{args.reranker_model}"
    )

    reranker = CrossEncoderReranker(
        model_name=args.reranker_model
    )

    print(
        f"\nReranking top {args.rerank_k} "
        "fused candidates per query..."
    )

    final_rankings = {}

    rerank_start = time.perf_counter()

    for index, query in enumerate(queries, start=1):
        fused_ids = fused_rankings[query.id]

        rerank_ids = fused_ids[: args.rerank_k]

        candidates = [
            (
                doc_id,
                document_texts[doc_id],
            )
            for doc_id in rerank_ids
        ]

        reranked = reranker.rerank(
            query=query.text,
            candidates=candidates,
        )

        reranked_ids = [
            doc_id
            for doc_id, _score in reranked
        ]

        # Preserve the rest of the RRF ranking after the
        # cross-encoder's reranked candidate set.
        reranked_set = set(reranked_ids)

        remaining_ids = [
            doc_id
            for doc_id in fused_ids
            if doc_id not in reranked_set
        ]

        final_rankings[query.id] = (
            reranked_ids + remaining_ids
        )

        print(
            f"Reranked query "
            f"{index}/{len(queries)}"
        )

    rerank_seconds = (
        time.perf_counter() - rerank_start
    )

    # ---------------------------------------------------------
    # Evaluation
    # ---------------------------------------------------------

    qrels = {
        query.id: query.relevant_ids
        for query in queries
    }

    metrics = evaluate_rankings(
        final_rankings,
        qrels,
        args.eval_k,
    )

    result = {
        "split": "test" if "test" in str(args.queries).lower() else "validation",
        "retriever": (
            "bm25_plus_finetuned_dense_weighted_rrf"
            "_plus_cross_encoder"
        ),
        "fine_tuned_model": args.fine_tuned_model,
        "reranker_model": args.reranker_model,
        "num_documents": len(document_ids),
        "num_queries": len(queries),
        "candidate_k": args.candidate_k,
        "rerank_k": args.rerank_k,
        "rrf_k": args.rrf_k,
        "bm25_weight": args.bm25_weight,
        "dense_weight": args.dense_weight,
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
        "reranker_mean_query_latency_ms": (
            rerank_seconds / len(queries)
        ) * 1000,
        "metrics": metrics,
    }

    print("\n" + "=" * 80)
    print("HYBRID + RERANKER VALIDATION RESULTS")
    print("=" * 80)

    print(json.dumps(result, indent=2))

    with open(
        args.output,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            result,
            f,
            indent=2,
        )

    print(
        f"\nSaved results to "
        f"{args.output}"
    )


if __name__ == "__main__":
    main()