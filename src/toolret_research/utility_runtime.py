"""Execute the frozen learned router without exposing qrels or CE outcomes."""
from __future__ import annotations
from dataclasses import dataclass
import math

from .selective_runtime import Reranker
from .utility_router import RidgeUtilityModel, features, predict, shouldroute


@dataclass(frozen=True)
class UtilityRoutingResult:
    ranking: tuple[str, ...]
    reranked: bool
    scored_pairs: int
    predicted_utility: float


def route_utility_and_rerank(
    query: str, bm25_ids: list[str], dense_ids: list[str], hybrid_ids: list[str],
    documents: dict[str, str], model: RidgeUtilityModel, threshold: float,
    reranker: Reranker, *, depth: int = 20,
) -> UtilityRoutingResult:
    """Predict from first-stage ranks, then score only the requested prefix."""
    if isinstance(depth, bool) or not isinstance(depth, int) or depth < 1:
        raise ValueError("depth must be a positive integer")
    for name, ids in (("bm25", bm25_ids), ("dense", dense_ids), ("hybrid", hybrid_ids)):
        if not ids or len(ids) != len(set(ids)) or any(doc not in documents for doc in ids):
            raise ValueError(f"{name} must contain unique known document IDs")
    score = float(predict(model, features(query, {"bm25": bm25_ids, "dense": dense_ids})))
    if not shouldroute(score, threshold):
        return UtilityRoutingResult(tuple(hybrid_ids), False, 0, score)
    prefix = hybrid_ids[:depth]
    scored = reranker.rerank(query, [(doc, documents[doc]) for doc in prefix])
    ids = [doc for doc, _ in scored]
    if len(ids) != len(prefix) or len(set(ids)) != len(prefix) or set(ids) != set(prefix):
        raise ValueError("backend must score each prefix candidate exactly once")
    if any(not math.isfinite(float(value)) for _, value in scored):
        raise ValueError("backend scores must be finite")
    ordered = [doc for doc, _ in sorted(scored, key=lambda pair: (-float(pair[1]), pair[0]))]
    return UtilityRoutingResult(tuple(ordered + hybrid_ids[len(prefix):]), True, len(prefix), score)
