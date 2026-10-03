"""Fail-closed completion checks use synthetic evidence, never held-out scores."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path

import pytest


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts/finalize_confirmation_study.py"
SPEC = importlib.util.spec_from_file_location("finalize_confirmation_study", MODULE_PATH)
completion = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(completion)


def runtime_fixture():
    predictions = [{"query_id": f"q{i}", "rerank_decisions": {"utility_75": i % 2 == 0}} for i in range(6)]
    runtime = {"queries": 6, "primary_policy": "utility_75", "actual_backend_calls": 3, "actual_backend_pairs": 60,
               "skipped_calls": 3, "all_decisions_match_frozen_predictions": True, "all_rankings_match_evaluation": True,
               "per_query": [{"query_id": r["query_id"], "reranked": r["rerank_decisions"]["utility_75"],
                              "scored_pairs": 20 if r["rerank_decisions"]["utility_75"] else 0,
                              "matches_frozen_decision": True, "matches_evaluated_ranking": True} for r in predictions]}
    return runtime, predictions


def test_actual_runtime_requires_exact_cohort_decisions_pairs_and_rank_matches():
    runtime, predictions = runtime_fixture()
    assert completion.verify_runtime(runtime, predictions, expected_count=6)["calls"] == 3
    for change in ("cohort", "decision", "pairs", "rank", "calls"):
        altered = deepcopy(runtime)
        if change == "cohort":
            altered["per_query"][-1]["query_id"] = "different"
        elif change == "decision":
            altered["per_query"][0]["reranked"] = False
        elif change == "pairs":
            altered["per_query"][0]["scored_pairs"] = 19
        elif change == "rank":
            altered["per_query"][0]["matches_evaluated_ranking"] = False
        else:
            altered["actual_backend_calls"] = 4
        with pytest.raises(completion.StudyIncompleteError):
            completion.verify_runtime(altered, predictions, expected_count=6)


def primary_fixture(low_query=-.002, low_component=-.003, criterion=True):
    interval = {"low": low_query, "high": .01, "delta_mean": .001, "resamples": 10000, "seed": 20261003, "confidence": .95}
    return {"num_queries": 1500, "primary_noninferiority": {"policy": "utility_75", "reference": "always", "metric": "nDCG@10",
            "predeclared_engineering_margin": .01, "quality_and_required_call_criterion_met": criterion},
            "paired_nDCG_comparisons": {"utility_75_minus_always": {
                "source_stratified_query_bootstrap": interval,
                "source_stratified_component_bootstrap": dict(interval, low=low_component)}}}


def test_negative_primary_result_can_complete_and_boundary_is_strict():
    assert completion.primary_conclusion(primary_fixture(), actual_calls=1000) is True
    assert completion.primary_conclusion(primary_fixture(low_component=-.01, criterion=False), actual_calls=1000) is False
    assert completion.primary_conclusion(primary_fixture(criterion=False), actual_calls=1500) is False
    with pytest.raises(completion.StudyIncompleteError, match="disagrees"):
        completion.primary_conclusion(primary_fixture(low_component=-.02), actual_calls=1000)


def test_pending_or_stale_independent_audit_cannot_certify_completion(tmp_path):
    root = tmp_path
    (root / "scripts").mkdir()
    (root / "scripts/audit_confirmation_study.py").write_text("independent synthetic auditor")
    source = root / "evidence.json"
    source.write_text("frozen source bytes")
    checks = [{"name": name, "status": "pass"} for name in completion.EXPECTED_CHECKS]
    audit = {"schema_version": "toolret-independent-confirmation-audit-v1", "independent_implementation": True,
             "imports_study_metric_router_or_evaluator_code": False, "all_executed_checks_pass": True,
             "checks": checks, "counts": {"pass": len(checks)}, "input_evidence_sha256": {"evidence.json": completion.digest(source)},
             "auditor_source_sha256": completion.digest(root / "scripts/audit_confirmation_study.py")}
    assert completion.verify_full_audit(audit, {"protocol": source}, root) == len(checks)
    pending = deepcopy(audit)
    pending["checks"].append({"name": "remaining_runtime", "status": "pending"})
    with pytest.raises(completion.StudyIncompleteError, match="pending"):
        completion.verify_full_audit(pending, {"protocol": source}, root)
    source.write_text("changed after the independent audit")
    with pytest.raises(completion.StudyIncompleteError, match="stale"):
        completion.verify_full_audit(audit, {"protocol": source}, root)


def test_missing_required_evidence_never_produces_completed_manifest(tmp_path):
    with pytest.raises(completion.StudyIncompleteError, match="missing"):
        completion.finalize({"protocol": tmp_path / "missing.json"}, tmp_path)


def latency_fixture():
    cache, predictions, records = [], [], []
    for number, source in enumerate(completion.DOMAINS):
        qid = f"{source}_q"
        rank = [f"tool{i}" for i in range(20)]
        cache.append({"query_id": qid, "source_domain": source, "rankings": {"hybrid": rank, "reranked": rank[::-1]}})
        predictions.append({"query_id": qid, "rerank_decisions": {"utility_75": number == 0, "fixed_disagreement": number < 2}})
    for repetition in range(3):
        for row, prediction in zip(cache, predictions):
            for position, policy in enumerate(completion.LATENCY_POLICIES):
                routed = policy == "always" or policy not in ("always", "hybrid") and prediction["rerank_decisions"][policy]
                ranking = row["rankings"]["reranked" if routed else "hybrid"]
                records.append({"query_id": row["query_id"], "source_domain": row["source_domain"], "policy": policy,
                                "policy_position": position, "repetition": repetition, "reranked": routed,
                                "scored_pairs": 20 if routed else 0, "matches_cache": True, "total_ms": 10.,
                                "components_ms": {"bm25": 2., "dense": 5., "rrf": 1., "routing": 1. if routed else 2., "cross_encoder": 1. if routed else 0.},
                                "ranking_sha256": completion.hashlib.sha256(json.dumps(ranking).encode()).hexdigest()})
    selection = {"query_ids": [row["query_id"] for row in cache], "schedule": [{key: event[key] for key in ("query_id", "policy", "policy_position", "repetition")} for event in records]}
    per_query = [{"query_id": row["query_id"], "policies": {policy: {"mean_total_ms": 10.} for policy in completion.LATENCY_POLICIES}} for row in cache]
    summary = {"queries": 3, "repetitions": 3, "source_domain_counts": {source: 1 for source in completion.DOMAINS}, "all_actual_rankings_match_cache": True,
               "policies": {policy: {"requests": 9, "mean_ms": 10., "p50_query_mean_ms": 10., "p95_query_mean_ms": 10.,
                                     "actual_cross_encoder_calls": sum(event["reranked"] for event in records if event["policy"] == policy),
                                     "actual_cross_encoder_pairs": sum(event["scored_pairs"] for event in records if event["policy"] == policy)} for policy in completion.LATENCY_POLICIES}}
    return summary, selection, records, per_query, cache, predictions


def test_repeated_latency_events_are_recomputed_and_rank_fingerprints_checked():
    values = latency_fixture()
    completion.verify_latency(*values, expected_per_source=1)
    altered = deepcopy(values)
    altered[2][0]["ranking_sha256"] = "incorrect"
    with pytest.raises(completion.StudyIncompleteError, match="fingerprint"):
        completion.verify_latency(*altered, expected_per_source=1)
    altered = deepcopy(values)
    altered[2][0]["total_ms"] = 100.
    with pytest.raises(completion.StudyIncompleteError, match="boundary"):
        completion.verify_latency(*altered, expected_per_source=1)
    altered = deepcopy(values)
    altered[2][0]["repetition"] = 1
    altered[1]["schedule"][0]["repetition"] = 1
    with pytest.raises(completion.StudyIncompleteError, match="repeated"):
        completion.verify_latency(*altered, expected_per_source=1)
