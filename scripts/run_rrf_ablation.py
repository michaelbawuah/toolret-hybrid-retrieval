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
        "--candidate-k",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--rrf-values",
        nargs="+",
        type=int,
        default=[10, 20, 40, 60, 100],
    )

    parser.add_argument(
        "--eval-k",
        nargs="+",
        type=int,
        default=[1, 3, 5, 10],
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

    bm25 = BM25Index.build(document_texts)

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

    print("\nEncoding queries...")

    query_embeddings = dense.encode_queries(
        [query.text for query in queries]
    )

    print("\nRunning BM25 retrieval once...")

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

    print("\nRunning dense retrieval once...")

    dense_results = dense.search(
        query_embeddings,
        dense_document_embeddings,
        document_ids,
        k=args.candidate_k,
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

    qrels = {
        query.id: query.relevant_ids
        for query in queries
    }

    all_results = {}

    print("\nRunning RRF ablations...")

    for rrf_k in args.rrf_values:
        hybrid_rankings = {}

        for query in queries:
            hybrid_rankings[query.id] = reciprocal_rank_fusion(
                [
                    bm25_rankings[query.id],
                    dense_rankings[query.id],
                ],
                k=rrf_k,
            )

        metrics = evaluate_rankings(
            hybrid_rankings,
            qrels,
            args.eval_k,
        )

        all_results[str(rrf_k)] = metrics

        print(f"\nRRF k = {rrf_k}")
        print(json.dumps(metrics, indent=2))

    result = {
        "retriever": "hybrid_rrf_ablation",
        "dense_model": args.model,
        "num_documents": len(document_ids),
        "num_queries": len(queries),
        "candidate_k": args.candidate_k,
        "rrf_values": args.rrf_values,
        "dense_build_seconds": dense_build_seconds,
        "results": all_results,
    }

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)

        print(f"\nSaved ablation results to {args.output}")


if __name__ == "__main__":
    main()