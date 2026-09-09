from __future__ import annotations

import argparse
import json
import time

from toolret_research.bm25 import BM25Index
from toolret_research.data import load_corpus, load_queries
from toolret_research.dense import DenseRetriever
from toolret_research.fusion import weighted_reciprocal_rank_fusion
from toolret_research.metrics import evaluate_rankings
from toolret_research.text import flatten_tool


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--corpus", required=True)
    parser.add_argument("--queries", required=True)
    parser.add_argument("--fine-tuned-model", required=True)
    parser.add_argument("--output", required=True)

    parser.add_argument(
        "--candidate-k",
        type=int,
        default=500,
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
        default=1.0,
    )

    parser.add_argument(
        "--eval-k",
        nargs="+",
        type=int,
        default=[1, 3, 5, 10],
    )

    args = parser.parse_args()

    split_name = "test" if "test" in str(args.queries).lower() else "validation"

    print("Loading corpus...")
    corpus = load_corpus(args.corpus)

    print(f"Loading {split_name} queries...")
    queries = load_queries(args.queries)

    print(f"Loaded {len(corpus)} documents.")
    print(f"Loaded {len(queries)} {split_name} queries.")

    document_ids = list(corpus.keys())

    document_texts = {
        doc_id: flatten_tool(corpus[doc_id])
        for doc_id in document_ids
    }

    print("\nBuilding BM25 index...")

    start = time.perf_counter()

    bm25 = BM25Index.build(document_texts)

    bm25_build_seconds = time.perf_counter() - start

    print(
        f"BM25 index built in "
        f"{bm25_build_seconds:.2f}s"
    )

    print(
        f"\nLoading fine-tuned dense model: "
        f"{args.fine_tuned_model}"
    )

    dense = DenseRetriever(
        model_name=args.fine_tuned_model
    )

    print("\nEncoding documents...")

    start = time.perf_counter()

    dense_document_embeddings = dense.encode_documents(
        [document_texts[doc_id] for doc_id in document_ids]
    )

    dense_build_seconds = time.perf_counter() - start

    print(
        f"Dense document encoding time: "
        f"{dense_build_seconds:.2f}s"
    )

    print("\nEncoding queries...")

    query_embeddings = dense.encode_queries(
        [query.text for query in queries]
    )

    print("\nRunning BM25 retrieval...")

    bm25_rankings = {}

    for query in queries:
        results = bm25.search(
            query.text,
            k=args.candidate_k,
        )

        bm25_rankings[query.id] = [
            doc_id
            for doc_id, _score in results
        ]

    print("\nRunning dense retrieval...")

    dense_results = dense.search(
        query_embeddings,
        dense_document_embeddings,
        document_ids,
        k=args.candidate_k,
    )

    dense_rankings = {
        query.id: [
            doc_id
            for doc_id, _score in results
        ]
        for query, results in zip(
            queries,
            dense_results,
        )
    }

    print("\nApplying weighted RRF...")

    hybrid_rankings = {}

    start = time.perf_counter()

    for query in queries:
        hybrid_rankings[query.id] = (
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

    qrels = {
        query.id: query.relevant_ids
        for query in queries
    }

    metrics = evaluate_rankings(
        hybrid_rankings,
        qrels,
        args.eval_k,
    )

    result = {
       "split": split_name,
        "retriever": "bm25_plus_finetuned_dense_weighted_rrf",
        "fine_tuned_model": args.fine_tuned_model,
        "num_documents": len(document_ids),
        "num_queries": len(queries),
        "candidate_k": args.candidate_k,
        "rrf_k": args.rrf_k,
        "bm25_weight": args.bm25_weight,
        "dense_weight": args.dense_weight,
        "bm25_build_seconds": bm25_build_seconds,
        "dense_build_seconds": dense_build_seconds,
        "fusion_seconds": fusion_seconds,
        "metrics": metrics,
    }

    print("\nResults:")
    print(json.dumps(result, indent=2))

    with open(
        args.output,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(result, f, indent=2)

    print(
        f"\nSaved hybrid {split_name} results "
        f"to {args.output}"
    )


if __name__ == "__main__":
    main()