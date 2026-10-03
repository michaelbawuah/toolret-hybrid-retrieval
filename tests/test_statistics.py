import csv
from pathlib import Path

import pytest

from toolret_research.statistics import (
    exact_sign_test,
    paired_bootstrap,
    rank_effect,
    reciprocal_rank_from_position,
    validate_rank_records,
)


def valid_row(query_id="q1"):
    return {
        "query_id": query_id,
        "bm25_rank": "",
        "bm25_rr": "0.0",
        "dense_rank": "2",
        "dense_rr": "0.5",
        "hybrid_rank": "1",
        "hybrid_rr": "1.0",
        "reranked_rank": "2",
        "reranked_rr": "0.5",
        "hybrid_vs_dense": "improved",
        "reranker_effect": "hurt",
    }


def test_bootstrap_preserves_pairing_and_exact_constant_difference():
    result = paired_bootstrap([0.0, 10.0, 20.0], [2.0, 12.0, 22.0], resamples=100)
    assert result["delta_mean"] == result["low"] == result["high"] == 2.0
    assert result["sample_unit"] == "query"
    assert result["n"] == 3


def test_bootstrap_reproducible_and_sign_reversible():
    before, after = [0.0, 1.0, 0.5], [1.0, 0.0, 0.25]
    first = paired_bootstrap(before, after, seed=15, resamples=2000)
    repeat = paired_bootstrap(before, after, seed=15, resamples=2000)
    reverse = paired_bootstrap(after, before, seed=15, resamples=2000)
    assert first == repeat
    assert first["delta_mean"] == pytest.approx(-1 / 12)
    assert reverse["delta_mean"] == pytest.approx(-first["delta_mean"])
    assert reverse["low"] == pytest.approx(-first["high"])
    assert reverse["high"] == pytest.approx(-first["low"])
    assert first["low"] <= first["delta_mean"] <= first["high"]


def test_bootstrap_single_query_and_ties():
    assert paired_bootstrap([0], [0.5], resamples=1)["low"] == 0.5
    tied = paired_bootstrap([0, 1], [0, 1], resamples=20)
    assert tied["low"] == tied["high"] == 0


@pytest.mark.parametrize("before,after", [([], []), ([1], []), ([float("nan")], [1]), ([1], [float("inf")])])
def test_statistics_reject_invalid_pairs(before, after):
    with pytest.raises(ValueError):
        paired_bootstrap(before, after)
    with pytest.raises(ValueError):
        exact_sign_test(before, after)


@pytest.mark.parametrize("kwargs", [{"resamples": 0}, {"resamples": True}, {"resamples": 1.5}, {"confidence": 0}, {"confidence": 1}, {"confidence": float("nan")}, {"seed": True}])
def test_bootstrap_reject_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        paired_bootstrap([0], [1], **kwargs)


def test_exact_sign_test_known_binomial_tails():
    all_wins = exact_sign_test([0] * 5, [1] * 5)
    assert all_wins["two_sided_pvalue"] == 1 / 16
    one_vs_four = exact_sign_test([0] * 6, [1, -1, -1, -1, -1, 0])
    assert one_vs_four["two_sided_pvalue"] == 0.375
    assert one_vs_four["ties"] == 1
    assert one_vs_four["n_nonzero"] == 5
    assert exact_sign_test([0, 1], [0, 1])["two_sided_pvalue"] == 1
    assert exact_sign_test([0, 0], [1, -1])["two_sided_pvalue"] == 1


def test_rank_positions_and_missing_effects():
    assert reciprocal_rank_from_position(None) == 0
    assert reciprocal_rank_from_position(4) == 0.25
    assert rank_effect(None, None) == "unchanged_missing"
    assert rank_effect(None, 20) == "improved"
    assert rank_effect(20, None) == "hurt"
    assert rank_effect(2, 1) == "improved"
    assert rank_effect(1, 2) == "hurt"
    assert rank_effect(3, 3) == "unchanged"
    for rank in (0, -1, 1.5, True):
        with pytest.raises(ValueError):
            reciprocal_rank_from_position(rank)


def test_valid_rank_csv_records_and_historical_effect_counts():
    source = Path(__file__).resolve().parents[1] / "results/test_failure_analysis.csv"
    with source.open(newline="", encoding="utf-8") as handle:
        rows = validate_rank_records(list(csv.DictReader(handle)))
    assert len(rows) == 16
    assert sum(row["hybrid_vs_dense"] == "improved" for row in rows) == 4
    assert sum(row["hybrid_vs_dense"] == "hurt" for row in rows) == 6
    assert sum(row["reranker_effect"] == "improved" for row in rows) == 2
    assert sum(row["reranker_effect"] == "hurt" for row in rows) == 3
    result = validate_rank_records([valid_row()])[0]
    assert result["bm25_rank"] is None
    assert result["hybrid_rank"] == 1


@pytest.mark.parametrize("field,value", [("query_id", ""), ("dense_rank", "0"), ("dense_rank", "-2"), ("dense_rank", "1.5"), ("dense_rank", "3"), ("bm25_rr", "0.1"), ("dense_rr", "nan"), ("dense_rr", "inf"), ("dense_rr", "-0.5"), ("hybrid_vs_dense", "hurt"), ("reranker_effect", "improved")])
def test_rank_records_reject_invalid_or_inconsistent_rows(field, value):
    row = valid_row()
    row[field] = value
    with pytest.raises(ValueError):
        validate_rank_records([row])


def test_rank_records_reject_duplicates_missing_columns_and_empty_input():
    with pytest.raises(ValueError, match="duplicate query_id"):
        validate_rank_records([valid_row(), valid_row()])
    row = valid_row()
    del row["dense_rank"]
    with pytest.raises(ValueError, match="missing dense"):
        validate_rank_records([row])
    with pytest.raises(ValueError, match="no records"):
        validate_rank_records([])
