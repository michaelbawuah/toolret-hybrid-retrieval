from __future__ import annotations

import csv
import json
import time
from pathlib import Path

from toolret_research.bm25 import BM25Index
from toolret_research.data import load_corpus, load_queries
from toolret_research.dense import DenseRetriever
from toolret_research.fusion import weighted_reciprocal_rank_fusion
from toolret_research.reranker import CrossEncoderReranker
from toolret_research.text import flatten_tool


# ---------------------------------------------------------------------------
# Frozen test configuration.
#
# These values were selected using validation experiments and MUST NOT be
# tuned using the test set.
# ---------------------------------------------------------------------------

CORPUS_PATH = "data/toolret/corpus.jsonl"
TEST_QUERIES_PATH = "data/toolret/splits/test.jsonl"

DENSE_MODEL = "models/minilm_toolret_hardneg"
RERANKER_MODEL = "models/toolret_reranker_hardneg"

CANDIDATE_K = 500
RRF_K = 10
BM25_WEIGHT = 1.0
DENSE_WEIGHT = 1.25
RERANK_K = 3

JSON_OUTPUT = Path("results/test_failure_analysis.json")
CSV_OUTPUT = Path("results/test_failure_analysis.csv")


def first_relevant_rank(
    ranking: list[str],
    relevant_ids: list[str],
) -> int | None:
    """Return the rank of the highest-ranked relevant tool."""

    relevant_set = set(relevant_ids)

    for rank, doc_id in enumerate(ranking, start=1):
        if doc_id in relevant_set:
            return rank

    return None


def reciprocal_rank(rank: int | None) -> float:
    """Convert a rank into reciprocal rank."""

    if rank is None:
        return 0.0

    return 1.0 / rank


def compare_ranks(
    before: int | None,
    after: int | None,
) -> str:
    """Describe whether a retrieval stage improved the relevant-tool rank."""

    if before is None and after is None:
        return "unchanged_missing"

    if before is None and after is not None:
        return "improved"

    if before is not None and after is None:
        return "hurt"

    if after < before:
        return "improved"

    if after > before:
        return "hurt"

    return "unchanged"


def main() -> None:
    print("=" * 80)
    print("TOOLRET FROZEN TEST FAILURE ANALYSIS")
    print("=" * 80)

    print("\nFrozen configuration:")
    print(f"  candidate_k   = {CANDIDATE_K}")
    print(f"  rrf_k         = {RRF_K}")
    print(f"  bm25_weight   = {BM25_WEIGHT}")
    print(f"  dense_weight  = {DENSE_WEIGHT}")
    print(f"  rerank_k      = {RERANK_K}")

    # -----------------------------------------------------------------------
    # Load corpus and frozen test queries
    # -----------------------------------------------------------------------

    print("\nLoading corpus...")
    corpus = load_corpus(CORPUS_PATH)

    print("Loading frozen test queries...")
    queries = load_queries(TEST_QUERIES_PATH)

    print(f"Loaded {len(corpus)} tools.")
    print(f"Loaded {len(queries)} frozen test queries.")

    document_ids = list(corpus.keys())

    document_texts = {
        doc_id: flatten_tool(corpus[doc_id])
        for doc_id in document_ids
    }

    # -----------------------------------------------------------------------
    # BM25
    # -----------------------------------------------------------------------

    print("\nBuilding BM25 index...")
    start = time.perf_counter()

    bm25 = BM25Index.build(document_texts)

    print(
        f"BM25 index built in "
        f"{time.perf_counter() - start:.2f}s"
    )

    bm25_rankings: dict[str, list[str]] = {}

    print("Running BM25 retrieval...")

    for query in queries:
        results = bm25.search(
            query.text,
            k=CANDIDATE_K,
        )

        bm25_rankings[query.id] = [
            doc_id
            for doc_id, _score in results
        ]

    # -----------------------------------------------------------------------
    # Fine-tuned dense retriever
    # -----------------------------------------------------------------------

    print(f"\nLoading dense model: {DENSE_MODEL}")

    dense = DenseRetriever(
        model_name=DENSE_MODEL
    )

    print("Encoding corpus...")

    dense_document_embeddings = dense.encode_documents(
        [
            document_texts[doc_id]
            for doc_id in document_ids
        ]
    )

    print("Encoding test queries...")

    query_embeddings = dense.encode_queries(
        [
            query.text
            for query in queries
        ]
    )

    print("Running dense retrieval...")

    dense_results = dense.search(
        query_embeddings,
        dense_document_embeddings,
        document_ids,
        k=CANDIDATE_K,
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

    # -----------------------------------------------------------------------
    # Weighted Reciprocal Rank Fusion
    # -----------------------------------------------------------------------

    print("\nApplying frozen weighted RRF...")

    hybrid_rankings: dict[str, list[str]] = {}

    for query in queries:
        hybrid_rankings[query.id] = (
            weighted_reciprocal_rank_fusion(
                rankings=[
                    bm25_rankings[query.id],
                    dense_rankings[query.id],
                ],
                weights=[
                    BM25_WEIGHT,
                    DENSE_WEIGHT,
                ],
                k=RRF_K,
            )
        )

    # -----------------------------------------------------------------------
    # Cross-encoder reranking
    # -----------------------------------------------------------------------

    print(f"\nLoading frozen reranker: {RERANKER_MODEL}")

    reranker = CrossEncoderReranker(
        model_name=RERANKER_MODEL
    )

    reranked_rankings: dict[str, list[str]] = {}

    print(
        f"Reranking only the top {RERANK_K} "
        "hybrid candidates per query..."
    )

    for index, query in enumerate(queries, start=1):
        fused_ids = hybrid_rankings[query.id]

        rerank_ids = fused_ids[:RERANK_K]

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

        reranked_set = set(reranked_ids)

        remaining_ids = [
            doc_id
            for doc_id in fused_ids
            if doc_id not in reranked_set
        ]

        reranked_rankings[query.id] = (
            reranked_ids + remaining_ids
        )

        print(
            f"Analyzed query "
            f"{index}/{len(queries)}"
        )

    # -----------------------------------------------------------------------
    # Per-query analysis
    # -----------------------------------------------------------------------

    print("\nComputing per-query rank changes...")

    records = []

    hybrid_improved_over_dense = 0
    hybrid_hurt_vs_dense = 0

    reranker_improved = 0
    reranker_hurt = 0
    reranker_unchanged = 0

    for query in queries:
        relevant_ids = list(query.relevant_ids)

        bm25_rank = first_relevant_rank(
            bm25_rankings[query.id],
            relevant_ids,
        )

        dense_rank = first_relevant_rank(
            dense_rankings[query.id],
            relevant_ids,
        )

        hybrid_rank = first_relevant_rank(
            hybrid_rankings[query.id],
            relevant_ids,
        )

        reranked_rank = first_relevant_rank(
            reranked_rankings[query.id],
            relevant_ids,
        )

        hybrid_effect = compare_ranks(
            dense_rank,
            hybrid_rank,
        )

        reranker_effect = compare_ranks(
            hybrid_rank,
            reranked_rank,
        )

        if hybrid_effect == "improved":
            hybrid_improved_over_dense += 1
        elif hybrid_effect == "hurt":
            hybrid_hurt_vs_dense += 1

        if reranker_effect == "improved":
            reranker_improved += 1
        elif reranker_effect == "hurt":
            reranker_hurt += 1
        else:
            reranker_unchanged += 1

        record = {
            "query_id": query.id,
            "query": query.text,
            "relevant_ids": relevant_ids,

            "bm25_rank": bm25_rank,
            "dense_rank": dense_rank,
            "hybrid_rank": hybrid_rank,
            "reranked_rank": reranked_rank,

            "bm25_rr": reciprocal_rank(bm25_rank),
            "dense_rr": reciprocal_rank(dense_rank),
            "hybrid_rr": reciprocal_rank(hybrid_rank),
            "reranked_rr": reciprocal_rank(reranked_rank),

            "hybrid_vs_dense": hybrid_effect,
            "reranker_effect": reranker_effect,

            "bm25_top3": bm25_rankings[query.id][:3],
            "dense_top3": dense_rankings[query.id][:3],
            "hybrid_top3": hybrid_rankings[query.id][:3],
            "reranked_top3": reranked_rankings[query.id][:3],
        }

        records.append(record)

    # Sort the human-readable analysis so the most interesting reranker
    # failures appear first.
    effect_order = {
        "hurt": 0,
        "improved": 1,
        "unchanged": 2,
        "unchanged_missing": 3,
    }

    records_sorted = sorted(
        records,
        key=lambda row: (
            effect_order.get(
                row["reranker_effect"],
                99,
            ),
            row["query_id"],
        ),
    )

    summary = {
        "split": "test",
        "num_queries": len(queries),
        "num_documents": len(corpus),

        "frozen_configuration": {
            "candidate_k": CANDIDATE_K,
            "rrf_k": RRF_K,
            "bm25_weight": BM25_WEIGHT,
            "dense_weight": DENSE_WEIGHT,
            "rerank_k": RERANK_K,
            "dense_model": DENSE_MODEL,
            "reranker_model": RERANKER_MODEL,
        },

        "hybrid_vs_dense": {
            "improved_queries": hybrid_improved_over_dense,
            "hurt_queries": hybrid_hurt_vs_dense,
        },

        "reranker_effect": {
            "improved_queries": reranker_improved,
            "hurt_queries": reranker_hurt,
            "unchanged_queries": reranker_unchanged,
        },

        "queries": records_sorted,
    }

    # -----------------------------------------------------------------------
    # Save JSON
    # -----------------------------------------------------------------------

    JSON_OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with JSON_OUTPUT.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            summary,
            f,
            indent=2,
        )

    # -----------------------------------------------------------------------
    # Save CSV
    # -----------------------------------------------------------------------

    csv_fields = [
        "query_id",
        "query",
        "relevant_ids",
        "bm25_rank",
        "dense_rank",
        "hybrid_rank",
        "reranked_rank",
        "bm25_rr",
        "dense_rr",
        "hybrid_rr",
        "reranked_rr",
        "hybrid_vs_dense",
        "reranker_effect",
    ]

    with CSV_OUTPUT.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=csv_fields,
        )

        writer.writeheader()

        for row in records_sorted:
            csv_row = {
                field: row[field]
                for field in csv_fields
            }

            csv_row["relevant_ids"] = ",".join(
                row["relevant_ids"]
            )

            writer.writerow(csv_row)

    # -----------------------------------------------------------------------
    # Console summary
    # -----------------------------------------------------------------------

    print("\n" + "=" * 80)
    print("FAILURE ANALYSIS SUMMARY")
    print("=" * 80)

    print(
        "\nHybrid vs fine-tuned dense:"
    )
    print(
        f"  Improved relevant-tool rank: "
        f"{hybrid_improved_over_dense}"
    )
    print(
        f"  Hurt relevant-tool rank:     "
        f"{hybrid_hurt_vs_dense}"
    )

    print(
        "\nReranker vs hybrid:"
    )
    print(
        f"  Improved queries: "
        f"{reranker_improved}"
    )
    print(
        f"  Hurt queries:     "
        f"{reranker_hurt}"
    )
    print(
        f"  Unchanged:        "
        f"{reranker_unchanged}"
    )

    print("\nQueries hurt by reranking:")

    hurt_records = [
        row
        for row in records_sorted
        if row["reranker_effect"] == "hurt"
    ]

    if not hurt_records:
        print("  None")
    else:
        for row in hurt_records:
            print(
                f"\n  {row['query_id']}: "
                f"{row['query']}"
            )
            print(
                f"    Relevant: "
                f"{row['relevant_ids']}"
            )
            print(
                f"    Hybrid rank:   "
                f"{row['hybrid_rank']}"
            )
            print(
                f"    Reranked rank: "
                f"{row['reranked_rank']}"
            )
            print(
                f"    Hybrid top 3:   "
                f"{row['hybrid_top3']}"
            )
            print(
                f"    Reranked top 3: "
                f"{row['reranked_top3']}"
            )

    print(
        f"\nSaved JSON analysis to "
        f"{JSON_OUTPUT}"
    )
    print(
        f"Saved CSV analysis to "
        f"{CSV_OUTPUT}"
    )


if __name__ == "__main__":
    main()