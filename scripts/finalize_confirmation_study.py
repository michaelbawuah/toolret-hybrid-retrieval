"""Fail-closed completion manifest for the frozen empirical study.

This combines immutable analysis, real conditional execution, repeated latency,
failure diagnostics, and a full independent audit. A failed or inconclusive
primary hypothesis can still complete the prospectively scoped experiment.
The original evaluator summary is never modified.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np


DOMAINS = ("apigen", "toolbench", "toolace")
LATENCY_POLICIES = ("hybrid", "always", "fixed_disagreement", "utility_75")
EXPECTED_CHECKS = (
    "corpus_identity", "eligible_gold_labels_and_corpus_membership", "independent_full_component_inventory",
    "router_schema_and_fixed_fit", "independent_router_coefficients", "public_protocol_tree_receipt",
    "independent_frozen_pointwise_predictions", "independent_all_policy_decision_sets",
    "independent_raw_per_query_metrics", "primary_noninferiority_conclusion",
    "independent_full_cohort_conditional_runtime_accounting",
    "latency_predefined_balanced_selection", "latency_exact_counterbalanced_schedule",
    "latency_raw_event_counts_timings_routes_and_rank_hashes", "latency_raw_events_hash",
    "evaluation_router_and_frozen_provenance", "public_committed_router_bytes_and_protocol_ancestry",
)


class StudyIncompleteError(ValueError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise StudyIncompleteError(message)


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def unique_index(rows: list[dict], key: str) -> dict[str, dict]:
    result = {row[key]: row for row in rows}
    require(len(result) == len(rows) and all(isinstance(value, str) and value for value in result), f"duplicate/invalid {key}")
    return result


def close(a: Any, b: Any) -> bool:
    return bool(np.allclose(a, b, atol=1e-10, rtol=1e-9))


def verify_hashes(payload: dict, mapping: dict[str, Path], label: str) -> None:
    for field, path in mapping.items():
        require(payload.get(field) == digest(path), f"{label} hash mismatch: {field}")


def verify_sources(payload: dict[str, str], root: Path, label: str) -> None:
    require(bool(payload), f"{label} source hashes missing")
    for source, expected in payload.items():
        path = (root / source).resolve()
        require(path.is_relative_to(root) and path.is_file() and digest(path) == expected, f"{label} source changed: {source}")


def verify_full_audit(audit: dict, evidence: dict[str, Path], root: Path) -> int:
    require(audit.get("schema_version") == "toolret-independent-confirmation-audit-v1", "independent audit schema mismatch")
    require(audit.get("independent_implementation") is True and audit.get("imports_study_metric_router_or_evaluator_code") is False, "audit independence declaration missing")
    checks = audit.get("checks", [])
    require(bool(checks) and all(check.get("status") == "pass" for check in checks), "independent audit has a failed or pending check")
    require(audit.get("all_executed_checks_pass") is True, "independent audit did not pass")
    require(Counter(check["status"] for check in checks) == audit.get("counts"), "independent audit counts disagree")
    check_names = {check["name"] for check in checks}
    require(len(check_names) == len(checks) and set(EXPECTED_CHECKS).issubset(check_names), "full evaluation/runtime/latency audit coverage missing")
    supplied = audit.get("input_evidence_sha256", {})
    require(isinstance(supplied, dict) and bool(supplied), "audit evidence fingerprints missing")
    resolved = {(root / name).resolve(): value for name, value in supplied.items()}
    for role, path in evidence.items():
        if role == "independent_audit":
            continue
        require(resolved.get(path.resolve()) == digest(path), f"independent audit is stale or missing evidence: {role}")
    for path, expected in resolved.items():
        require(path.is_relative_to(root) and path.is_file() and digest(path) == expected, f"audited input changed: {path}")
    require(audit.get("auditor_source_sha256") == digest(root / "scripts/audit_confirmation_study.py"), "independent auditor source changed")
    return len(checks)


def verify_runtime(runtime: dict, predictions: list[dict], expected_count: int = 1500) -> dict[str, Any]:
    predicted = unique_index(predictions, "query_id")
    records = unique_index(runtime.get("per_query", []), "query_id")
    require(len(records) == len(predicted) == runtime.get("queries") == expected_count and records.keys() == predicted.keys(), "actual runtime cohort mismatch")
    require(runtime.get("primary_policy") == "utility_75", "actual runtime primary policy mismatch")
    require(runtime.get("all_decisions_match_frozen_predictions") is True and runtime.get("all_rankings_match_evaluation") is True, "actual runtime did not match all decisions/rankings")
    calls = 0
    for qid, record in records.items():
        routed = predicted[qid]["rerank_decisions"]["utility_75"]
        require(type(routed) is bool and record.get("reranked") is routed, "actual runtime routing mismatch")
        require(record.get("scored_pairs") == (20 if routed else 0), "actual runtime per-query pair count mismatch")
        require(record.get("matches_frozen_decision") is True and record.get("matches_evaluated_ranking") is True, "actual runtime per-query rank/decision check failed")
        calls += routed
    require(runtime.get("actual_backend_calls") == calls and runtime.get("actual_backend_pairs") == 20 * calls
            and runtime.get("skipped_calls") == expected_count - calls, "actual runtime backend accounting mismatch")
    return {"queries": expected_count, "calls": calls, "pairs": 20 * calls, "skipped": expected_count - calls}


def verify_latency(
    summary: dict, selection: dict, events: list[dict], per_query: list[dict],
    cache: list[dict], predictions: list[dict], *, expected_per_source: int = 30,
    repetitions: int = 3,
) -> None:
    by_id = unique_index(cache, "query_id")
    frozen = unique_index(predictions, "query_id")
    selected = selection.get("query_ids", [])
    expected_count = len(DOMAINS) * expected_per_source
    require(len(selected) == len(set(selected)) == summary.get("queries") == expected_count, "latency selected query count mismatch")
    require(set(selected).issubset(by_id) and summary.get("repetitions") == repetitions, "latency cohort/repetitions mismatch")
    require(Counter(by_id[qid]["source_domain"] for qid in selected) == {source: expected_per_source for source in DOMAINS}, "latency source selection unbalanced")
    require(summary.get("source_domain_counts") == {source: expected_per_source for source in DOMAINS}, "latency declared source counts mismatch")
    require(set(summary.get("policies", {})) == set(LATENCY_POLICIES), "latency policy set mismatch")
    require(len(events) == expected_count * len(LATENCY_POLICIES) * repetitions, "latency event count mismatch")
    require(len(selection.get("schedule", [])) == len(events), "latency schedule count mismatch")
    require(summary.get("all_actual_rankings_match_cache") is True, "latency rankings did not all match cache")
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for event, scheduled in zip(events, selection["schedule"]):
        require(all(event.get(key) == value for key, value in scheduled.items()), "latency event/schedule mismatch")
        qid, policy = event["query_id"], event["policy"]
        require(qid in selected and policy in LATENCY_POLICIES, "latency event query/policy mismatch")
        route = policy == "always" or (policy not in ("always", "hybrid") and frozen[qid]["rerank_decisions"][policy])
        rank = by_id[qid]["rankings"]["reranked" if route else "hybrid"]
        require(event.get("reranked") is route and event.get("scored_pairs") == (20 if route else 0), "latency actual routing/calls mismatch")
        require(event.get("matches_cache") is True and event.get("ranking_sha256") == hashlib.sha256(json.dumps(rank).encode()).hexdigest(), "latency actual ranking fingerprint mismatch")
        require(event.get("source_domain") == by_id[qid]["source_domain"], "latency source mismatch")
        milliseconds = event["total_ms"]
        components = event["components_ms"]
        require(not isinstance(milliseconds, bool) and math.isfinite(milliseconds) and milliseconds > 0, "latency wall time invalid")
        require(set(components) == {"bm25", "dense", "rrf", "routing", "cross_encoder"}
                and all(math.isfinite(value) and value >= 0 for value in components.values())
                and math.isclose(sum(components.values()), milliseconds, rel_tol=1e-9, abs_tol=1e-8), "latency component boundary invalid")
        require((route and components["cross_encoder"] > 0) or (not route and components["cross_encoder"] == 0), "latency measured CE work differs from actual routing")
        grouped[(qid, policy)].append(event)
    query_means = unique_index(per_query, "query_id")
    require(query_means.keys() == set(selected), "latency query-mean cohort mismatch")
    for policy in LATENCY_POLICIES:
        means, policy_events = [], []
        for qid in sorted(selected):
            rows = grouped[(qid, policy)]
            require(len(rows) == repetitions and {row["repetition"] for row in rows} == set(range(repetitions)), "latency repeated query-policy observations missing/duplicated")
            average = mean(row["total_ms"] for row in rows)
            require(close(query_means[qid]["policies"][policy]["mean_total_ms"], average), "latency saved query mean mismatch")
            means.append(average)
            policy_events.extend(rows)
        value = summary["policies"][policy]
        require(value.get("requests") == expected_count * repetitions, "latency policy request count mismatch")
        require(value.get("actual_cross_encoder_calls") == sum(row["reranked"] for row in policy_events)
                and value.get("actual_cross_encoder_pairs") == sum(row["scored_pairs"] for row in policy_events), "latency policy actual-call accounting mismatch")
        require(close([mean(means), *np.percentile(means, [50, 95]).tolist()],
                      [value["mean_ms"], value["p50_query_mean_ms"], value["p95_query_mean_ms"]]), "latency saved aggregate mismatch")


def primary_conclusion(summary: dict, actual_calls: int) -> bool:
    value = summary["primary_noninferiority"]
    query = summary["paired_nDCG_comparisons"]["utility_75_minus_always"]["source_stratified_query_bootstrap"]
    component = summary["paired_nDCG_comparisons"]["utility_75_minus_always"]["source_stratified_component_bootstrap"]
    require(value.get("policy") == "utility_75" and value.get("reference") == "always" and value.get("metric") == "nDCG@10"
            and value.get("predeclared_engineering_margin") == .01, "primary rule differs from frozen design")
    for interval in (query, component):
        require(all(math.isfinite(interval[key]) for key in ("low", "high", "delta_mean")) and interval["low"] <= interval["high"], "primary confidence interval invalid")
        require(interval.get("resamples") == 10000 and interval.get("seed") == 20261003 and interval.get("confidence") == .95, "primary interval bootstrap settings mismatch")
    achieved = query["low"] > -.01 and component["low"] > -.01 and actual_calls < summary["num_queries"]
    require(value.get("quality_and_required_call_criterion_met") is achieved, "saved primary conclusion disagrees with frozen rule/actual calls")
    return achieved


def finalize(evidence: dict[str, Path], root: Path) -> dict:
    require(all(path.is_file() for path in evidence.values()), "required evidence file missing; research completion remains pending")
    protocol, router, manifest, summary, provenance, runtime, latency, selection, failure, audit = [load_json(evidence[role]) for role in (
        "protocol", "router", "cache_manifest", "evaluation_summary", "frozen_prediction_provenance", "runtime_verification",
        "latency_summary", "latency_selection", "failure_summary", "independent_audit")]
    require(protocol.get("scope") == "confirmation_study" and protocol.get("primary_policy") == "utility_75"
            and protocol.get("reference_policy") == "always" and protocol.get("noninferiority_margin") == .01, "protocol scope/primary rule mismatch")
    require(router.get("schema_version") == "toolret-frozen-utility-router-v1", "router schema mismatch")
    require(manifest.get("scope") == "confirmation_study" and manifest.get("split", {}).get("name") == "confirmation"
            and manifest.get("cache_provenance", {}).get("partition") == "confirmation", "cache confirmation role mismatch")
    verify_hashes(manifest, {"cache_sha256": evidence["cache"], "queries_sha256": evidence["queries"], "corpus_sha256": evidence["corpus"]}, "cache")
    require(protocol.get("corpus_documents") == 37292 and protocol.get("corpus_sha256") == digest(evidence["corpus"])
            and protocol["confirmation"].get("queries") == 1500
            and protocol["confirmation"].get("queries_sha256") == digest(evidence["queries"]), "frozen protocol corpus/cohort identity differs")
    require(manifest["cache_provenance"]["protocol"] == protocol, "cache protocol differs")
    verify_hashes(router["input_integrity"], {"protocol_sha256": evidence["protocol"], "corpus_sha256": evidence["corpus"]}, "router")
    verify_sources(router["source_sha256"], root, "router")
    require(summary.get("scope") == "confirmation_study" and summary.get("num_queries") == 1500 and summary.get("metrics_relevance") == "binary", "evaluation count/scope/relevance mismatch")
    require(summary.get("input_integrity") == provenance and provenance.get("created_before_quality_scoring") is True, "prediction provenance missing or altered")
    require(summary.get("router") == router and summary.get("manifest") == manifest, "evaluation embedded model/cache provenance differs")
    evaluation_mapping = {"router_sha256": evidence["router"], "protocol_sha256": evidence["protocol"], "predictions_sha256": evidence["frozen_predictions"],
                          "cache_sha256": evidence["cache"], "queries_sha256": evidence["queries"], "corpus_sha256": evidence["corpus"],
                          "data_manifest_sha256": evidence["data_manifest"], "cache_manifest_sha256": evidence["cache_manifest"]}
    verify_hashes(provenance, evaluation_mapping, "frozen evaluation")
    verify_sources(provenance["evaluation_source_sha256"], root, "evaluation")
    cache, queries, predictions = [read_jsonl(evidence[role]) for role in ("cache", "queries", "frozen_predictions")]
    cached = unique_index(cache, "query_id")
    require(len(cached) == len(queries) == len(predictions) == 1500 and set(cached) == {row["id"] for row in queries} == {row["query_id"] for row in predictions}, "full evidence cohort mismatch")
    require(Counter(row["source_domain"] for row in cache) == {source: 500 for source in DOMAINS}, "confirmation source counts mismatch")
    components = {query["component_id"] for query in queries}
    require(len(components) == protocol["confirmation"]["component_count"] == 1174, "confirmation component count mismatch")
    calls = verify_runtime(runtime, predictions)
    runtime_mapping = {**{key: evaluation_mapping[key] for key in ("router_sha256", "protocol_sha256", "predictions_sha256", "cache_sha256", "queries_sha256", "corpus_sha256")},
                       "evaluation_summary_sha256": evidence["evaluation_summary"]}
    require(set(runtime_mapping) == set(runtime["input_integrity"]), "runtime input evidence coverage mismatch")
    verify_hashes(runtime["input_integrity"], runtime_mapping, "actual runtime")
    verify_sources(runtime["source_sha256"], root, "actual runtime")
    require(summary["policies"]["utility_75"]["reranker_invocations"] == calls["calls"]
            and summary["policies"]["utility_75"]["reranker_candidate_pairs"] == calls["pairs"], "required and actual primary compute disagree")
    latency_mapping = {"cache": evidence["cache"], "manifest": evidence["cache_manifest"], "corpus": evidence["corpus"], "queries": evidence["queries"],
                       "protocol": evidence["protocol"], "router_model": evidence["router"]}
    verify_hashes(latency["inputs_sha256"], latency_mapping, "latency")
    require(latency.get("events_sha256") == digest(evidence["latency_events"]), "latency event fingerprint mismatch")
    verify_sources(latency["source_sha256"], root, "latency")
    events, latency_per_query = read_jsonl(evidence["latency_events"]), read_jsonl(evidence["latency_per_query"])
    verify_latency(latency, selection, events, latency_per_query, cache, predictions)
    require(failure.get("schema_version") == "toolret-confirmation-failure-analysis-v1"
            and failure.get("scope") == "exploratory_confirmation_failure_analysis_after_frozen_predictions", "confirmation failure diagnostic schema/scope mismatch")
    failure_mapping = {**evaluation_mapping, "cache_manifest_sha256": evidence["cache_manifest"],
                       "prediction_provenance_sha256": evidence["frozen_prediction_provenance"], "evaluation_summary_sha256": evidence["evaluation_summary"]}
    verify_hashes(failure["provenance"], failure_mapping, "failure diagnostics")
    require(failure["provenance"].get("created_after_frozen_predictions") is True
            and failure["provenance"].get("declared_predictions_created_before_quality_scoring") is True
            and failure["provenance"].get("labels_altered") is False
            and failure["provenance"].get("analysis_modifies_rankings_or_router") is False, "failure diagnostics changed labels/router or lack ordering evidence")
    failure_rows = unique_index(read_jsonl(evidence["failure_per_query"]), "query_id")
    require(failure_rows.keys() == cached.keys(), "failure diagnostic cohort mismatch")
    frozen_by_id = unique_index(predictions, "query_id")
    query_by_id = unique_index(queries, "id")
    require(all(record.get("component_id") == query_by_id[qid]["component_id"]
                and record.get("frozen_rerank_decisions") == frozen_by_id[qid]["rerank_decisions"]
                for qid, record in failure_rows.items()), "failure diagnostics altered frozen component/routing assignments")
    verify_hashes(failure["provenance"], {"script_sha256": root / "scripts/analyze_confirmation_failures.py",
                  "reused_analysis_script_sha256": root / "scripts/analyze_tool_retrieval_failures.py",
                  "text_serialization_sha256": root / "src/toolret_research/text.py"}, "failure diagnostic sources")
    protocol_receipt, router_receipt = [load_json(evidence[role]) for role in ("protocol_freeze_receipt", "router_freeze_receipt")]
    require(protocol_receipt.get("protocol_sha256") == digest(evidence["protocol"]) and protocol_receipt.get("published_before_new_rank_generation") is True, "prospective public protocol freeze missing")
    require(router_receipt.get("router_sha256") == digest(evidence["router"]) and router_receipt.get("published_before_confirmation_rank_generation") is True
            and router_receipt.get("public_protocol_commit") == protocol_receipt.get("public_protocol_commit"), "prospective public router freeze missing")
    require(router["input_integrity"].get("protocol_frozen_commit") == protocol_receipt.get("public_protocol_commit"), "router frozen protocol commit differs from published receipt")
    audit_checks = verify_full_audit(audit, evidence, root)
    achieved = primary_conclusion(summary, calls["calls"])
    quality = summary["paired_nDCG_comparisons"]["utility_75_minus_always"]
    latency_comparison = latency["paired_comparisons"]["utility_75_vs_always"]
    always_latency, utility_latency = latency["policies"]["always"], latency["policies"]["utility_75"]
    evidence_output = {role: {"path": path.resolve().relative_to(root).as_posix(), "sha256": digest(path)} for role, path in evidence.items()}
    return {
        "schema_version": "toolret-confirmation-study-completion-v1", "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "completed_scope": True, "scope": "Frozen component-disjoint empirical retrieval study with actual conditional runtime, repeated warm CPU latency, failure diagnostics, and full independent evidence audit.",
        "predeclared_primary_criterion_achieved": achieved, "quality_and_required_call_criterion_met": achieved,
        "actual_conditional_execution_verified": True, "runtime_verification_pending": False,
        "primary_interpretation": "The predeclared engineering noninferiority/compute criterion was achieved." if achieved else "The scoped study is complete; the predeclared engineering noninferiority/compute criterion was not established. No policy was retuned after confirmation.",
        "checks": {"prospective_protocol_and_router_freeze": True, "immutable_cross_file_input_hashes": True,
                   "full_confirmation_and_component_cohort": True, "all1500_actual_runtime_decisions_and_rankings": True,
                   "required_and_actual_primary_calls_match": True, "all1080_actual_latency_events_and_rankings": True,
                   "complete_confirmation_failure_diagnostics": True, "full_independent_audit_without_pending_or_failed_checks": True,
                   "primary_result_matches_predeclared_rule": True},
        "headline": {
            "confirmation_queries": 1500, "development_queries": 300, "calibration_queries": 300,
            "positive_tool_components": 1174, "corpus_tools": 37292,
            "utility75_nDCG@10": summary["policies"]["utility_75"]["metrics"]["nDCG@10"],
            "always_nDCG@10": summary["policies"]["always"]["metrics"]["nDCG@10"],
            "primary_nDCG_delta": quality["source_stratified_query_bootstrap"]["delta_mean"],
            "primary_nDCG_query_CI95": [quality["source_stratified_query_bootstrap"][key] for key in ("low", "high")],
            "primary_nDCG_component_CI95": [quality["source_stratified_component_bootstrap"][key] for key in ("low", "high")],
            "actual_primary_CE_calls": calls["calls"], "actual_primary_CE_pairs": calls["pairs"], "actual_skipped_CE_calls": calls["skipped"],
            "CE_call_reduction_fraction_vs_always": calls["skipped"] / 1500,
            "latency_queries": 90, "latency_repetitions": 3, "latency_policies": 4, "latency_events": 1080,
            "latency_scope": "warm sequential CPU query-to-ranking; includes routing; excludes loading/network/queueing/concurrency",
            "always_mean_latency_ms": always_latency["mean_ms"], "utility75_mean_latency_ms": utility_latency["mean_ms"],
            "always_p50_query_mean_ms": always_latency["p50_query_mean_ms"], "utility75_p50_query_mean_ms": utility_latency["p50_query_mean_ms"],
            "always_p95_query_mean_ms": always_latency["p95_query_mean_ms"], "utility75_p95_query_mean_ms": utility_latency["p95_query_mean_ms"],
            "mean_latency_reduction_fraction_vs_always": latency_comparison["mean_latency_reduction_fraction"],
            "mean_latency_reduction_CI95_fraction": latency_comparison["reduction_ci95_fraction"],
            "mean_latency_delta_ms": latency_comparison["mean_delta_ms"], "mean_latency_delta_CI95_ms": latency_comparison["delta_ci95_ms"],
            "independent_audit_passed_checks": audit_checks,
        },
        "evidence": evidence_output, "finalizer_source_sha256": digest(Path(__file__)),
        "claim_boundary": "Completion establishes the declared empirical scope; it does not establish publication acceptance, faculty supervision, a novel gating algorithm, unseen API-family transfer, or downstream agent task execution.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path("results/research_confirmation_20261003"))
    parser.add_argument("--data-dir", type=Path, default=Path("data/research_confirmation_20261003"))
    parser.add_argument("--protocol", type=Path, default=Path("configs/confirmation_protocol_20261003.json"))
    parser.add_argument("--corpus", type=Path, default=Path("data/research_20261003/corpus.jsonl"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    results, data = args.results_dir, args.data_dir
    evidence = {
        "protocol": args.protocol, "router": results / "router/model.json", "cache": results / "confirmation_cache/rankings.jsonl",
        "cache_manifest": results / "confirmation_cache/manifest.json", "queries": data / "confirmation_queries.jsonl", "corpus": args.corpus,
        "evaluation_summary": results / "confirmation_eval/summary.json", "frozen_predictions": results / "confirmation_eval/frozen_predictions.jsonl",
        "frozen_prediction_provenance": results / "confirmation_eval/frozen_prediction_provenance.json",
        "evaluation_per_query": results / "confirmation_eval/per_query.jsonl", "evaluation_decisions": results / "confirmation_eval/decisions.jsonl",
        "runtime_verification": results / "runtime_verification.json", "latency_summary": results / "latency/summary.json",
        "latency_events": results / "latency/events.jsonl", "latency_selection": results / "latency/selection.json", "latency_per_query": results / "latency/per_query.jsonl",
        "failure_summary": results / "confirmation_failure_analysis.json", "failure_per_query": results / "confirmation_failure_analysis_per_query.jsonl",
        "independent_audit": results / "independent_audit.json", "data_manifest": data / "manifest.json",
        "data_audit": results / "data_audit.json", "component_assignments": results / "component_assignments.json",
        "protocol_freeze_receipt": results / "protocol_freeze_receipt.json", "router_freeze_receipt": results / "router_freeze_receipt.json",
    }
    output = args.output or results / "study_completion.json"
    require(not output.exists(), "Refusing to replace an existing study completion manifest")
    root = Path(__file__).resolve().parents[1]
    result = finalize(evidence, root)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"completed_scope": result["completed_scope"], "predeclared_primary_criterion_achieved": result["predeclared_primary_criterion_achieved"],
                      "headline": result["headline"], "output": str(output.resolve())}, indent=2))


if __name__ == "__main__":
    main()
