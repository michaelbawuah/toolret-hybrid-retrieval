"""Small, dependency-free statistics for paired retrieval observations.

The sampling unit is one query, not an individual candidate tool. Percentile
bootstrap intervals describe sampling uncertainty conditional on the observed
queries; they do not establish representative benchmark performance.
"""

from __future__ import annotations

import math
import random
from collections.abc import Mapping, Sequence
from typing import Any


HISTORICAL_SYSTEMS = ("bm25", "dense", "hybrid", "reranked")


def _paired_differences(
    before: Sequence[float], after: Sequence[float]
) -> list[float]:
    if len(before) != len(after) or not before:
        raise ValueError("Paired observations must be nonempty and equal in length.")
    differences = []
    for left, right in zip(before, after):
        left, right = float(left), float(right)
        if not math.isfinite(left) or not math.isfinite(right):
            raise ValueError("Paired observations must be finite.")
        difference = right - left
        if not math.isfinite(difference):
            raise ValueError("Paired differences must be finite.")
        differences.append(difference)
    return differences


def _quantile(sorted_values: Sequence[float], probability: float) -> float:
    position = probability * (len(sorted_values) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    return sorted_values[lower] + (position - lower) * (
        sorted_values[upper] - sorted_values[lower]
    )


def paired_bootstrap(
    before: Sequence[float],
    after: Sequence[float],
    *,
    seed: int = 20261003,
    resamples: int = 20000,
    confidence: float = 0.95,
) -> dict[str, Any]:
    """Estimate mean(after-before) with a paired percentile bootstrap interval.

    Each resample draws n query indices with replacement and uses the same
    indices for both systems. This preserves within-query pairing. The fixed
    seed and linear-interpolated percentiles make the result reproducible.
    """
    differences = _paired_differences(before, after)
    if isinstance(resamples, bool) or not isinstance(resamples, int) or resamples < 1:
        raise ValueError("resamples must be a positive integer.")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer.")
    if not math.isfinite(confidence) or not 0 < confidence < 1:
        raise ValueError("confidence must lie strictly between zero and one.")
    n = len(differences)
    rng = random.Random(seed)
    means = sorted(
        math.fsum(differences[rng.randrange(n)] for _ in range(n)) / n
        for _ in range(resamples)
    )
    tail = (1 - confidence) / 2
    return {
        "delta_mean": math.fsum(differences) / n,
        "low": _quantile(means, tail),
        "high": _quantile(means, 1 - tail),
        "confidence": confidence,
        "resamples": resamples,
        "seed": seed,
        "n": n,
        "sample_unit": "query",
        "method": "paired_percentile_bootstrap_linear_quantiles",
    }


def exact_sign_test(
    before: Sequence[float], after: Sequence[float]
) -> dict[str, int | float | str]:
    """Two-sided exact binomial sign test, dropping exact ties.

    This tests win/loss balance, not the mean effect magnitude. The p-value is
    twice the smaller binomial tail under p=0.5, capped at 1. No multiplicity
    correction is applied; callers must treat multiple comparisons accordingly.
    """
    differences = _paired_differences(before, after)
    wins = sum(value > 0 for value in differences)
    losses = sum(value < 0 for value in differences)
    ties = len(differences) - wins - losses
    nonzero = wins + losses
    numerator = 2 * sum(math.comb(nonzero, i) for i in range(min(wins, losses) + 1))
    pvalue = min(1.0, numerator / (2**nonzero)) if nonzero else 1.0
    return {
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "n_nonzero": nonzero,
        "two_sided_pvalue": pvalue,
        "method": "exact_two_sided_binomial_sign_test_ties_dropped",
    }


def reciprocal_rank_from_position(rank: int | None) -> float:
    if rank is None:
        return 0.0
    if isinstance(rank, bool) or not isinstance(rank, int) or rank < 1:
        raise ValueError("A rank must be a positive integer or None.")
    return 1.0 / rank


def rank_effect(before: int | None, after: int | None) -> str:
    before_rr = reciprocal_rank_from_position(before)
    after_rr = reciprocal_rank_from_position(after)
    if before is None and after is None:
        return "unchanged_missing"
    return "improved" if after_rr > before_rr else "hurt" if after_rr < before_rr else "unchanged"


def validate_rank_records(
    rows: Sequence[Mapping[str, str]],
) -> list[dict[str, Any]]:
    """Validate historical first-relevant rank rows without using full rankings.

    Blank rank means absent from the saved candidate ranking, not absent from
    the complete tool catalog. Duplicate query IDs, invalid positions, nonfinite
    reciprocal ranks, inconsistent RR values, and wrong effect labels fail.
    """
    if not rows:
        raise ValueError("The historical rank CSV has no records.")
    seen = set()
    validated = []
    for line, row in enumerate(rows, start=2):
        query_id = (row.get("query_id") or "").strip()
        if not query_id:
            raise ValueError(f"CSV line {line}: query_id is missing.")
        if query_id in seen:
            raise ValueError(f"CSV line {line}: duplicate query_id {query_id!r}.")
        seen.add(query_id)
        result: dict[str, Any] = dict(row)
        result["query_id"] = query_id
        for system in HISTORICAL_SYSTEMS:
            field = f"{system}_rank"
            if field not in row or f"{system}_rr" not in row:
                raise ValueError(f"CSV line {line}: missing {system} rank/RR field.")
            raw_rank = (row[field] or "").strip()
            try:
                rank = int(raw_rank) if raw_rank else None
                expected = reciprocal_rank_from_position(rank)
                actual = float(row[f"{system}_rr"])
            except (ValueError, TypeError) as exc:
                raise ValueError(f"CSV line {line}: invalid {system} rank/RR.") from exc
            if not math.isfinite(actual) or not math.isclose(
                actual, expected, rel_tol=0, abs_tol=1e-12
            ):
                raise ValueError(f"CSV line {line}: inconsistent {system} rank/RR.")
            result[field] = rank
            result[f"{system}_rr"] = expected
        for label, before, after in (
            ("hybrid_vs_dense", "dense", "hybrid"),
            ("reranker_effect", "hybrid", "reranked"),
        ):
            expected = rank_effect(result[f"{before}_rank"], result[f"{after}_rank"])
            if row.get(label) != expected:
                raise ValueError(f"CSV line {line}: inconsistent {label} label.")
        validated.append(result)
    return validated
