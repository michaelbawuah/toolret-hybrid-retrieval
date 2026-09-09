from __future__ import annotations

from collections import defaultdict
from typing import Iterable


def weighted_reciprocal_rank_fusion(
    rankings: Iterable[Iterable[str]],
    weights: Iterable[float] | None = None,
    k: int = 60,
) -> list[str]:
    """
    Combine ranked lists using Weighted Reciprocal Rank Fusion.

    Each document receives:

        weight / (k + rank)

    from every retriever in which it appears.

    Parameters
    ----------
    rankings:
        Ranked document-id lists.

    weights:
        Weight assigned to each ranking.
        If omitted, all retrievers receive equal weight.

    k:
        RRF smoothing constant.

    Returns
    -------
    list[str]
        Fused document IDs ordered from highest to lowest score.
    """

    rankings = [list(ranking) for ranking in rankings]

    if not rankings:
        return []

    if weights is None:
        weights = [1.0] * len(rankings)
    else:
        weights = list(weights)

    if len(weights) != len(rankings):
        raise ValueError(
            "Number of weights must equal number of rankings."
        )

    if k < 0:
        raise ValueError("k must be non-negative.")

    scores = defaultdict(float)

    for ranking, weight in zip(rankings, weights):
        for rank, doc_id in enumerate(ranking, start=1):
            scores[str(doc_id)] += float(weight) / (k + rank)

    fused = sorted(
        scores.items(),
        key=lambda item: (-item[1], item[0]),
    )

    return [
        doc_id
        for doc_id, _score in fused
    ]