from copy import deepcopy
import math

import numpy as np
import pytest

from toolret_research.confirmation import (
    DOMAINS, assert_split_disjointness, evaluate_confirmation, freeze_predictions,
    stratified_bootstrap_deltas, validate_confirmation_rows, validate_predictions,
)
from toolret_research.fusion import weighted_reciprocal_rank_fusion
from toolret_research.utility_router import RidgeUtilityModel


def sample_rows(per_source=2):
    rows = []
    for source in DOMAINS:
        for index in range(per_source):
            ranking = [f"{source}_q{index}_tool{j}" for j in range(20)]
            changed = list(ranking)
            changed[0], changed[1] = changed[1], changed[0]
            rows.append({"query_id": f"{source}_{index}", "query": f"{source} request {index}",
                         "source_domain": source, "component_id": f"component_{source}_{index}", "split": "confirmation",
                         "relevant_ids": [ranking[0]], "rankings": {"bm25": ranking, "dense": ranking,
                         "hybrid": ranking, "reranked": changed}, "rerank_k": 20})
    return sorted(rows, key=lambda row: row["query_id"])


def frozen_router(intercept=.2):
    model = RidgeUtilityModel((0.,) * 7, (1.,) * 7, (0.,) * 7, intercept, 10., 300)
    return {"schema_version": "toolret-frozen-utility-router-v1", "model": model.to_dict(),
            "thresholds": {str(value): {"threshold": threshold, "nominal_budget_fraction": value,
                            "primary": value == .75, "decision_rule": "predicted_ndcg_delta > threshold"}
                           for value, threshold in ((.25, .3), (.5, .2), (.75, .1))}}


def predictions(rows, *, utility75=None):
    records = freeze_predictions(rows, frozen_router())
    if utility75 is not None:
        for record in records:
            record["rerank_decisions"]["utility_75"] = record["query_id"] in utility75
    return records


def test_freeze_routes_strictly_and_never_reads_labels_or_expensive_rankings():
    rows = sample_rows()
    baseline = freeze_predictions(rows, frozen_router())
    for row in rows:
        row["relevant_ids"] = None
        row["rankings"]["reranked"] = object()
        row["rankings"]["hybrid"] = object()
    assert freeze_predictions(rows, frozen_router()) == baseline
    assert all(not row["rerank_decisions"]["utility_50"] for row in baseline)  # exactly on threshold
    assert all(row["rerank_decisions"]["utility_75"] for row in baseline)
    assert all(not row["rerank_decisions"]["fixed_disagreement"] for row in baseline)
    assert all(not row["rerank_decisions"]["cheap_jaccard"] for row in baseline)


def test_frozen_prediction_ids_and_boolean_decisions_are_validated():
    rows = sample_rows()
    records = predictions(rows)
    validate_predictions(rows, records)
    with pytest.raises(ValueError, match="unique"):
        validate_predictions(rows, records + records[:1])
    records[0]["rerank_decisions"]["utility_75"] = 1
    with pytest.raises(ValueError, match="decisions"):
        validate_predictions(rows, records)


def test_stratified_bootstrap_uses_paired_draws_and_query_macro_target():
    rows = sample_rows(per_source=3)
    for source in DOMAINS:
        group = [row for row in rows if row["source_domain"] == source]
        group[0]["component_id"] = group[1]["component_id"]
    deltas = [1. if row["query_id"].endswith("_2") else 0. for row in rows]
    result = stratified_bootstrap_deltas(rows, {"one": deltas, "two": [2 * value for value in deltas]}, resamples=400)
    for method in ("source_stratified_query_bootstrap", "source_stratified_component_bootstrap"):
        assert result["one"][method]["delta_mean"] == pytest.approx(1 / 3)
        for field in ("delta_mean", "low", "high"):
            assert result["two"][method][field] == pytest.approx(2 * result["one"][method][field])
    assert result["one"]["source_stratified_component_bootstrap"]["components_by_source"] == {source: 2 for source in DOMAINS}


def test_bootstrap_constant_differences_are_exact_and_deterministic():
    rows = sample_rows()
    first = stratified_bootstrap_deltas(rows, {"constant": [.125] * len(rows)}, resamples=101)
    assert first == stratified_bootstrap_deltas(rows, {"constant": [.125] * len(rows)}, resamples=101)
    for interval in first["constant"].values():
        assert interval["delta_mean"] == pytest.approx(.125)
        assert interval["low"] == pytest.approx(.125)
        assert interval["high"] == pytest.approx(.125)


@pytest.mark.parametrize("change,match", [
    ("crosssource", "cannot span"), ("nonfinite", "finite"), ("length", "align"), ("resamples", "positive"),
])
def test_bootstrap_rejects_invalid_dependencies_or_inputs(change, match):
    rows = sample_rows()
    values, count = [0.] * len(rows), 10
    if change == "crosssource":
        rows[-1]["component_id"] = rows[0]["component_id"]
    elif change == "nonfinite":
        values[0] = math.nan
    elif change == "length":
        values.pop()
    else:
        count = 0
    with pytest.raises(ValueError, match=match):
        stratified_bootstrap_deltas(rows, {"difference": values}, resamples=count)


def test_equal_budget_controls_match_total_and_source_counts_oracle_is_labeled():
    rows = sample_rows(per_source=3)
    chosen = {row["query_id"] for row in rows[::2]}
    summary, per_query, decisions = evaluate_confirmation(rows, predictions(rows, utility75=chosen), resamples=40)
    assert summary["policies"]["utility_75"]["reranker_invocations"] == len(chosen)
    assert len(summary["paired_nDCG_comparisons"]) == 8
    decision_lookup = {row["query_id"]: set(row["rerank_policies"]) for row in decisions}
    for name, policy in summary["policies"].items():
        if name.startswith("random_global_utility_75") or name.startswith("random_source_matched_utility_75"):
            assert policy["reranker_invocations"] == len(chosen)
            assert policy["reranker_candidate_pairs"] == 20 * len(chosen)
        if name.startswith("random_source_matched_utility_75"):
            for source in DOMAINS:
                assert sum(name in decision_lookup[row["query_id"]] for row in rows if row["source_domain"] == source) == sum(row["query_id"] in chosen for row in rows if row["source_domain"] == source)
    oracle = summary["oracle_matched_budget_ceiling"]["utility_75"]
    assert oracle["uses_confirmation_labels"] is True and oracle["deployable"] is False
    assert oracle["nDCG@10"] >= summary["policies"]["utility_75"]["metrics"]["nDCG@10"] - 1e-12
    assert all("random" not in row["policies"] for row in per_query)


def test_primary_noninferiority_requires_compute_reduction():
    rows = sample_rows()
    result = evaluate_confirmation(rows, predictions(rows), resamples=30)[0]["primary_noninferiority"]
    assert result["query_bootstrap_lower"] == pytest.approx(0.)
    assert result["component_bootstrap_lower"] == pytest.approx(0.)
    assert result["quality_and_required_call_criterion_met"] is False
    assert result["fewer_required_reranker_invocations"] is False


def test_primary_noninferiority_passes_beneficial_skips_and_rejects_harmful_skips():
    rows = sample_rows()
    result = evaluate_confirmation(rows, predictions(rows, utility75=set()), resamples=30)[0]
    assert result["primary_noninferiority"]["quality_and_required_call_criterion_met"] is True
    assert result["routing_effects"]["utility_75"]["beneficial_skips_count"] == len(rows)
    for row in rows:
        row["relevant_ids"] = [row["rankings"]["reranked"][0]]
    result = evaluate_confirmation(rows, predictions(rows, utility75=set()), resamples=30)[0]
    assert result["primary_noninferiority"]["quality_and_required_call_criterion_met"] is False
    assert result["routing_effects"]["utility_75"]["harmful_skips_count"] == len(rows)


def test_component_partition_leakage_and_text_duplicates_rejected():
    left = [{"id": "a", "component_id": "g1", "query": "hello", "relevant_ids": ["tool1"]}]
    right = [{"id": "b", "component_id": "g2", "query": "other", "relevant_ids": ["tool2"]}]
    assert assert_split_disjointness(left, right)["disjoint_positive_tools"] is True
    for key, value in (("id", "a"), ("component_id", "g1"), ("query", "  HELLO "), ("relevant_ids", ["tool1"])):
        changed = deepcopy(right)
        changed[0][key] = value
        with pytest.raises(ValueError, match="leakage"):
            assert_split_disjointness(left, changed)


def validation_fixture():
    rows = sample_rows(per_source=1)
    corpus = {doc_id for row in rows for doc_id in row["rankings"]["bm25"]}
    for row in rows:
        row["rankings"]["hybrid"] = weighted_reciprocal_rank_fusion([row["rankings"]["bm25"], row["rankings"]["dense"]], weights=[1, 1], k=60)
        row["rankings"]["reranked"] = list(row["rankings"]["hybrid"])
    queries = [{"id": row["query_id"], "query": row["query"], "domain": row["source_domain"],
                "relevant_ids": row["relevant_ids"], "component_id": row["component_id"], "split": "confirmation"} for row in rows]
    manifest = {"schema_version": "toolret-selective-v1", "scope": "confirmation_study", "corpus_sha256": "0" * 64,
                "dataset": {"name": "test", "version": "test", "source": "test"}, "models": {"dense": "test", "reranker": "test"},
                "split": {"name": "confirmation", "selection": "test"}, "cache_provenance": {"queries": 3, "partition": "confirmation"}, "rrf": {"k": 60, "weights": [1, 1]}}
    return rows, corpus, queries, manifest


def test_confirmation_validation_preserves_scope_and_exact_cohort():
    rows, corpus, queries, manifest = validation_fixture()
    validated = validate_confirmation_rows(rows, corpus, manifest, queries, expected_count=3, expected_per_domain=1)
    assert len(validated) == 3
    assert manifest["scope"] == "confirmation_study"
    queries[0]["relevant_ids"] = [rows[0]["rankings"]["bm25"][1]]
    with pytest.raises(ValueError, match="gold IDs differ"):
        validate_confirmation_rows(rows, corpus, manifest, queries, expected_count=3, expected_per_domain=1)


def test_confirmation_validation_rejects_crosssource_components_and_wrong_split():
    rows, corpus, queries, manifest = validation_fixture()
    queries[1]["component_id"] = queries[0]["component_id"]
    rows[1]["component_id"] = queries[0]["component_id"]
    with pytest.raises(ValueError, match="cannot span"):
        validate_confirmation_rows(rows, corpus, manifest, queries, expected_count=3, expected_per_domain=1)
    rows, corpus, queries, manifest = validation_fixture()
    queries[0]["split"] = "calibration"
    with pytest.raises(ValueError, match="split"):
        validate_confirmation_rows(rows, corpus, manifest, queries, expected_count=3, expected_per_domain=1)
