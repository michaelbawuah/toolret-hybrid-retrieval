"""Frozen-artifact identity and descriptive routing-accounting regressions."""
import copy
import importlib.util
from pathlib import Path

import pytest


spec = importlib.util.spec_from_file_location("confirmation_failure_analysis", Path(__file__).parents[1] / "scripts/analyze_confirmation_failures.py")
analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)


def evidence(tmp_path):
    paths = {}
    for key in ("cache_sha256", "cache_manifest_sha256", "queries_sha256", "corpus_sha256", "predictions_sha256", "prediction_provenance_sha256", "evaluation_summary_sha256"):
        path = tmp_path / key
        path.write_text(key)
        paths[key] = path
    hashes = {key: analysis.pilot.digest(path) for key, path in paths.items()}
    frozen = {key: hashes[key] for key in ("cache_sha256", "cache_manifest_sha256", "queries_sha256", "corpus_sha256", "predictions_sha256")}
    frozen.update(created_before_quality_scoring=True, router_sha256="router", protocol_sha256="protocol",
                  data_manifest_sha256="data", protocol_frozen_commit="commit")
    frozen["evaluation_source_sha256"] = {relative: analysis.pilot.digest(analysis.REPO_ROOT / relative) for relative in (
        "scripts/evaluate_confirmation_study.py", "src/toolret_research/confirmation.py", "src/toolret_research/utility_router.py")}
    manifest = {"scope": "confirmation_study", "split": {"name": "confirmation"},
                "models": {"reranker": {"name": analysis.pilot.MODEL, "revision": analysis.pilot.REVISION, "fine_tuned_here": False}},
                "cache_provenance": {"protocol": {"rerank_k": 20, "max_sequence_length": 256}, "source_sha256": {
                    "src/toolret_research/text.py": analysis.pilot.digest(analysis.REPO_ROOT / "src/toolret_research/text.py")}},
                **{key: hashes[key] for key in ("cache_sha256", "queries_sha256", "corpus_sha256")}}
    evaluation = {"schema_version": "toolret-confirmation-evaluation-v1", "scope": "confirmation_study",
                  "manifest": copy.deepcopy(manifest), "input_integrity": copy.deepcopy(frozen)}
    return paths, manifest, evaluation, frozen


def test_links_exact_frozen_predictions_and_analysis_version(tmp_path):
    paths, manifest, evaluation, frozen = evidence(tmp_path)
    result = analysis.verify_integrity(paths, manifest, evaluation, frozen)
    assert result["created_after_frozen_predictions"] is True
    assert result["declared_predictions_created_before_quality_scoring"] is True
    assert result["labels_altered"] is False
    assert result["reused_analysis_script_sha256"] == analysis.pilot.digest(analysis.SCRIPT_ROOT / "analyze_tool_retrieval_failures.py")


@pytest.mark.parametrize("mutation,match", [
    ("decision_order", "before quality"), ("prediction_content", "identity mismatch"),
    ("provenance_content", "differs from evaluation"), ("manifest_content", "different confirmation"),
    ("source_hash", "source differs"), ("missing_source_hash", "Missing evaluator"),
])
def test_modified_or_unfrozen_evidence_fails(tmp_path, mutation, match):
    paths, manifest, evaluation, frozen = evidence(tmp_path)
    if mutation == "decision_order":
        frozen["created_before_quality_scoring"] = False
    elif mutation == "prediction_content":
        paths["predictions_sha256"].write_text("changed predictions")
    elif mutation == "provenance_content":
        frozen["router_sha256"] = "changed"
    elif mutation == "manifest_content":
        manifest["changed"] = True
    else:
        if mutation == "source_hash":
            frozen["evaluation_source_sha256"]["src/toolret_research/confirmation.py"] = "changed"
        else:
            frozen["evaluation_source_sha256"] = {}
        evaluation["input_integrity"] = copy.deepcopy(frozen)
    with pytest.raises(ValueError, match=match):
        analysis.verify_integrity(paths, manifest, evaluation, frozen)


def routed_fixture():
    records = [{"query_id": "a", "source_domain": "apigen", "hybrid_ndcg10": .8,
                "reranked_ndcg10": .2, "delta_ndcg10": -.6, "outcome": "loss", "gate_selected": False},
               {"query_id": "b", "source_domain": "toolbench", "hybrid_ndcg10": .1,
                "reranked_ndcg10": .9, "delta_ndcg10": .8, "outcome": "win", "gate_selected": True}]
    decisions = [dict(fixed_disagreement=False, cheap_jaccard=True, utility_25=False, utility_50=True, utility_75=False),
                 dict(fixed_disagreement=True, cheap_jaccard=True, utility_25=False, utility_50=False, utility_75=True)]
    predictions = [{"query_id": r["query_id"], "source_domain": r["source_domain"], "predicted_reranking_utility": 0.,
                    "rerank_decisions": choice} for r, choice in zip(records, decisions)]
    return records, predictions


def test_counterfactual_accounting_keeps_selected_and_bypassed_effects_distinct():
    records, predictions = routed_fixture()
    mapped = analysis.prediction_map(predictions, records)
    point = analysis.routing_summary(records, mapped, "utility_75")
    assert point["ndcg10"] == pytest.approx(.85)
    assert point["mean_delta_vs_always_ndcg10"] == pytest.approx(.3)
    assert point["total_harm_avoided_by_bypass"] == pytest.approx(.6)
    assert point["total_gain_forgone_by_bypass"] == 0
    assert point["selected_outcomes_of_always_vs_hybrid"] == {"win": 1}
    assert point["bypassed_outcomes_of_always_vs_hybrid"] == {"loss": 1}


def test_checks_independent_overall_and_source_policy_metrics_against_evaluator():
    records, predictions = routed_fixture()
    mapped = analysis.prediction_map(predictions, records)
    policy_points = {"fixed_disagreement": (.85, 1), "cheap_jaccard": (.55, 2), "utility_25": (.45, 0),
                     "utility_50": (.15, 1), "utility_75": (.85, 1)}
    source_points = {
        "apigen": {"fixed_disagreement": (.8, 0), "cheap_jaccard": (.2, 1), "utility_25": (.8, 0),
                   "utility_50": (.2, 1), "utility_75": (.8, 0)},
        "toolbench": {"fixed_disagreement": (.9, 1), "cheap_jaccard": (.9, 1), "utility_25": (.1, 0),
                      "utility_50": (.1, 0), "utility_75": (.9, 1)},
    }
    evaluation = {"num_queries": 2, "policies": {
        "hybrid": {"metrics": {"nDCG@10": .45}}, "always": {"metrics": {"nDCG@10": .55}},
        **{policy: {"metrics": {"nDCG@10": value}, "reranker_invocations": calls}
           for policy, (value, calls) in policy_points.items()}},
        "source_domain_breakdown": {source: {"policies": {
            policy: {"metrics": {"nDCG@10": value}, "reranker_invocations": calls}
            for policy, (value, calls) in values.items()}} for source, values in source_points.items()}}
    result = analysis.check_evaluation(records, mapped, evaluation)
    assert result["utility_75"]["ndcg10"] == pytest.approx(.85)
    evaluation["source_domain_breakdown"]["apigen"]["policies"]["utility_75"]["reranker_invocations"] = 1
    with pytest.raises(ValueError, match="Frozen source policy mismatch"):
        analysis.check_evaluation(records, mapped, evaluation)


@pytest.mark.parametrize("mutation,match", [("nonboolean", "five boolean"), ("gate", "disagreement"), ("nonfinite", "finite"), ("missing", "query IDs")])
def test_frozen_decision_contract_is_strict(mutation, match):
    records, predictions = routed_fixture()
    if mutation == "nonboolean":
        predictions[0]["rerank_decisions"]["utility_75"] = 1
    elif mutation == "gate":
        predictions[0]["rerank_decisions"]["fixed_disagreement"] = True
    elif mutation == "nonfinite":
        predictions[0]["predicted_reranking_utility"] = float("nan")
    else:
        predictions.pop()
    with pytest.raises(ValueError, match=match):
        analysis.prediction_map(predictions, records)
