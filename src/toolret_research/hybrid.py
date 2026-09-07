from collections import defaultdict


def reciprocal_rank_fusion(rankings, k=60):
    """
    Combine multiple ranked lists using Reciprocal Rank Fusion.

    rankings:
        A list of ranked document-id lists.

        Example:
        [
            ["tool_a", "tool_b", "tool_c"],
            ["tool_b", "tool_d", "tool_a"],
        ]

    k:
        RRF smoothing constant. 60 is a common default.

    Returns:
        A single fused ranking of document IDs.
    """

    scores = defaultdict(float)

    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] += 1.0 / (k + rank)

    fused = sorted(
        scores.items(),
        key=lambda item: item[1],
        reverse=True,
    )

    return [doc_id for doc_id, _score in fused]