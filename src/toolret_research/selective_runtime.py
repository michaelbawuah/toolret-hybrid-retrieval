"""A small runtime adapter for the fixed disagreement-based reranking policy."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .selective import should_rerank


class Reranker(Protocol):
    def rerank(self, query: str, candidates: list[tuple[str, str]]) -> list[tuple[str, float]]: ...


@dataclass(frozen=True)
class RoutingResult:
    ranking: tuple[str, ...]
    reranked: bool
    scored_pairs: int


def route_and_rerank(
    query: str,
    bm25_ids: list[str],
    dense_ids: list[str],
    hybrid_ids: list[str],
    documents: dict[str, str],
    reranker: Reranker,
    *,
    depth: int = 20,
) -> RoutingResult:
    """Skip the reranker on top-1 agreement; otherwise reorder its prefix only.

    No labels, learned threshold, or benchmark metrics are visible to this code.
    A backend is injected so model invocation can be independently verified.
    """
    if isinstance(depth, bool) or not isinstance(depth, int) or depth < 1:
        raise ValueError("depth must be a positive integer")
    for name, values in (("bm25", bm25_ids), ("dense", dense_ids), ("hybrid", hybrid_ids)):
        if not values or len(values) != len(set(values)):
            raise ValueError(f"{name} must be a nonempty unique ranking")
        if any(doc_id not in documents for doc_id in values):
            raise ValueError(f"{name} references an unknown document")
    if not should_rerank({"rankings": {"bm25": bm25_ids, "dense": dense_ids}}):
        return RoutingResult(tuple(hybrid_ids), False, 0)
    candidates = hybrid_ids[:depth]
    output = reranker.rerank(query, [(doc_id, documents[doc_id]) for doc_id in candidates])
    reranked_ids = [doc_id for doc_id, _ in output]
    if len(reranked_ids) != len(candidates) or set(reranked_ids) != set(candidates):
        raise ValueError("Reranker must return exactly a permutation of its candidates")
    return RoutingResult(tuple(reranked_ids + hybrid_ids[len(candidates):]), True, len(candidates))
