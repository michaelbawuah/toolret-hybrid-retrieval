"""Cheap, identity-invariant regression of the expected benefit of reranking.

Features consume query length and BM25/dense *ranks* only. They do not consume
relevance labels, source names, query IDs, tool descriptions, or CE scores. Tool
IDs are used solely to compare overlap and positions: a consistent renaming of
every tool leaves the feature vector unchanged.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math
from typing import Any

import numpy as np


FEATURE_VERSION = "cheap-first-stage-v1"
FEATURE_NAMES = (
    "top1_disagreement",
    "top10_jaccard",
    "top20_jaccard",
    "top20_reciprocal_weighted_jaccard",
    "top20_rank_coherence",
    "query_token_count_log1p",
    "rrf_top2_normalized_margin",
)
MODEL_SCHEMA = "toolret-utility-ridge-v1"


def _ranking(value: Any, name: str) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{name} must be a nonempty sequence of string IDs")
    result = tuple(value)
    if not result or any(not isinstance(item, str) or not item for item in result):
        raise ValueError(f"{name} must be a nonempty sequence of string IDs")
    if len(result) != len(set(result)):
        raise ValueError(f"{name} contains duplicate IDs")
    return result


def _jaccard(left: Sequence[str], right: Sequence[str]) -> float:
    a, b = set(left), set(right)
    return len(a & b) / len(a | b)


def features(query_text: str, rankings: Mapping[str, Sequence[str]]) -> np.ndarray:
    """Return seven finite, cheap features in the frozen FEATURE_NAMES order.

    Only ``bm25`` and ``dense`` mapping entries are read. Extra ranking entries
    such as ``reranked`` are ignored. RRF is reconstructed with equal weights
    and k=60; no score from an expensive model is inspected. Query whitespace
    tokens are counted, but their contents are never embedded or categorized.
    """
    if not isinstance(query_text, str) or not query_text.strip():
        raise ValueError("query_text must be a nonempty string")
    if not isinstance(rankings, Mapping):
        raise ValueError("rankings must contain bm25 and dense sequences")
    sparse = _ranking(rankings.get("bm25"), "bm25")
    dense = _ranking(rankings.get("dense"), "dense")
    sparse20, dense20 = sparse[:20], dense[:20]
    sparse_positions = {doc_id: rank for rank, doc_id in enumerate(sparse20, 1)}
    dense_positions = {doc_id: rank for rank, doc_id in enumerate(dense20, 1)}
    union = set(sparse_positions) | set(dense_positions)
    # math.fsum ensures traversal order (and hence arbitrary ID spelling) does
    # not create order-dependent accumulation error in these overlap features.
    weighted_intersection = math.fsum(
        min(1 / sparse_positions[doc_id], 1 / dense_positions[doc_id])
        for doc_id in union if doc_id in sparse_positions and doc_id in dense_positions
    )
    weighted_union = math.fsum(
        max(1 / sparse_positions[doc_id] if doc_id in sparse_positions else 0.,
            1 / dense_positions[doc_id] if doc_id in dense_positions else 0.)
        for doc_id in union
    )
    # Absence from a top-20 list receives position 21, and positional distances
    # are normalized by 20. Identical top-20 order therefore has coherence 1.
    coherence = 1 - math.fsum(
        abs(sparse_positions.get(doc_id, 21) - dense_positions.get(doc_id, 21)) / 20
        for doc_id in union
    ) / len(union)
    rrf_scores: dict[str, float] = {}
    for ranking in (sparse, dense):
        for rank, doc_id in enumerate(ranking, 1):
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.) + 1 / (60 + rank)
    largest = sorted(rrf_scores.values(), reverse=True)[:2]
    margin = (largest[0] - largest[1]) / largest[0] if len(largest) == 2 else 1.
    result = np.asarray([
        float(sparse[0] != dense[0]),
        _jaccard(sparse[:10], dense[:10]),
        _jaccard(sparse20, dense20),
        weighted_intersection / weighted_union,
        coherence,
        math.log1p(len(query_text.split())),
        margin,
    ], dtype=np.float64)
    if not np.isfinite(result).all():
        raise ValueError("feature vector must be finite")
    return result


def _matrix(values: Any, *, label: str, single: bool = False) -> tuple[np.ndarray, bool]:
    try:
        result = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a finite numeric feature matrix") from exc
    one_row = result.ndim == 1
    if single and one_row:
        result = result.reshape(1, -1)
    if result.ndim != 2 or result.shape[1] != len(FEATURE_NAMES) or not len(result):
        raise ValueError(f"{label} must have shape (n, {len(FEATURE_NAMES)}) with n > 0")
    if not np.isfinite(result).all():
        raise ValueError(f"{label} must be finite")
    return result, one_row


@dataclass(frozen=True)
class RidgeUtilityModel:
    """JSON-serializable model; the intercept is not ridge-penalized."""

    means: tuple[float, ...]
    scales: tuple[float, ...]
    coefficients: tuple[float, ...]
    intercept: float
    alpha: float
    training_rows: int

    def __post_init__(self) -> None:
        for name in ("means", "scales", "coefficients"):
            values = getattr(self, name)
            if len(values) != len(FEATURE_NAMES) or any(not math.isfinite(value) for value in values):
                raise ValueError(f"model.{name} must contain seven finite values")
        if any(value <= 0 for value in self.scales):
            raise ValueError("model scales must be positive")
        if not math.isfinite(self.intercept) or not math.isfinite(self.alpha) or self.alpha <= 0:
            raise ValueError("model intercept must be finite and alpha positive")
        if isinstance(self.training_rows, bool) or not isinstance(self.training_rows, int) or self.training_rows < 2:
            raise ValueError("model training_rows must be an integer >= 2")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": MODEL_SCHEMA,
            "feature_version": FEATURE_VERSION,
            "feature_names": list(FEATURE_NAMES),
            "standardization": "training_population_mean_and_std; constant_feature_scale=1",
            "means": list(self.means), "scales": list(self.scales),
            "coefficients": list(self.coefficients), "intercept": self.intercept,
            "alpha": self.alpha, "training_rows": self.training_rows,
            "target": "binary_nDCG@10(always_reranked)-binary_nDCG@10(hybrid)",
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "RidgeUtilityModel":
        if (value.get("schema_version") != MODEL_SCHEMA
                or value.get("feature_version") != FEATURE_VERSION
                or value.get("feature_names") != list(FEATURE_NAMES)):
            raise ValueError("model schema or frozen feature order does not match")
        try:
            return cls(*(tuple(float(x) for x in value[name]) for name in ("means", "scales", "coefficients")),
                       float(value["intercept"]), float(value["alpha"]), value["training_rows"])
        except (KeyError, TypeError, OverflowError) as exc:
            raise ValueError("invalid serialized model") from exc


def fitmodel(train_features: Any, target: Any, alpha: float = 10.) -> RidgeUtilityModel:
    """Fit standardized ridge regression using training observations only."""
    x, _ = _matrix(train_features, label="train_features")
    if len(x) < 2:
        raise ValueError("ridge fitting requires at least two training rows")
    if isinstance(alpha, bool) or not isinstance(alpha, (float, int)) or not math.isfinite(alpha) or alpha <= 0:
        raise ValueError("alpha must be finite and strictly positive")
    y = np.asarray(target, dtype=np.float64)
    if y.shape != (len(x),) or not np.isfinite(y).all():
        raise ValueError("target must be a finite vector aligned to training rows")
    if np.any(y < -1) or np.any(y > 1):
        raise ValueError("nDCG difference targets must lie in [-1, 1]")
    means = x.mean(axis=0)
    scales = x.std(axis=0, ddof=0)
    scales = np.where(scales > 1e-12, scales, 1.)
    standardized = (x - means) / scales
    intercept = float(y.mean())
    coefficients = np.linalg.solve(
        standardized.T @ standardized + float(alpha) * np.eye(len(FEATURE_NAMES)),
        standardized.T @ (y - intercept),
    )
    return RidgeUtilityModel(tuple(float(x) for x in means), tuple(float(x) for x in scales),
                             tuple(float(x) for x in coefficients), intercept, float(alpha), len(x))


def predict(model: RidgeUtilityModel, feature_values: Any) -> np.ndarray | float:
    """Predict benefit without clipping; out-of-range estimates remain auditable."""
    if not isinstance(model, RidgeUtilityModel):
        raise ValueError("model must be a RidgeUtilityModel")
    x, single = _matrix(feature_values, label="feature_values", single=True)
    result = ((x - np.asarray(model.means)) / np.asarray(model.scales)) @ np.asarray(model.coefficients) + model.intercept
    if not np.isfinite(result).all():
        raise ValueError("model predictions must be finite")
    return float(result[0]) if single else result


def calibratebudgetthreshold(predictions: Any, target_fraction: float) -> float:
    """Linear quantile at 1-budget, with an explicitly strict routing boundary.

    Repeated scores on the boundary are skipped rather than broken by query or
    tool ID. The observed invocation fraction can therefore be below the target
    on calibration and can vary in either direction on future observations.
    No labels are accepted by this function.
    """
    values = np.asarray(predictions, dtype=np.float64)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("calibration predictions must be a finite nonempty vector")
    if isinstance(target_fraction, bool) or not isinstance(target_fraction, (float, int)) or not 0 < target_fraction < 1:
        raise ValueError("target_fraction must lie strictly between zero and one")
    return float(np.quantile(values, 1 - target_fraction, method="linear"))


def shouldroute(prediction: float, threshold: float) -> bool:
    """A pointwise decision, independent of the other confirmation queries."""
    if not math.isfinite(prediction) or not math.isfinite(threshold):
        raise ValueError("prediction and threshold must be finite")
    return bool(prediction > threshold)
