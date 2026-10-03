"""Analyze held-out failures only after decisions and quality are frozen.

The sealed pilot analysis functions supply independent nDCG, token accounting,
and rank-transition calculations. Its report and manually annotated pilot
examples are deliberately not reused. This diagnostic cannot fit a router,
change predictions, or relabel the dataset.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
from collections import Counter
from pathlib import Path


SCRIPT_ROOT = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_ROOT.parent
SPEC = importlib.util.spec_from_file_location("sealed_pilot_failure_functions", SCRIPT_ROOT / "analyze_tool_retrieval_failures.py")
pilot = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(pilot)

SCHEMA = "toolret-confirmation-failure-analysis-v1"
SCOPE = "exploratory_confirmation_failure_analysis_after_frozen_predictions"
ROUTING_POLICIES = ("fixed_disagreement", "cheap_jaccard", "utility_25", "utility_50", "utility_75")


def verify_integrity(paths: dict[str, Path], manifest: dict, evaluation: dict,
                     frozen: dict) -> dict:
    """Require exact links to the scored immutable prediction artifact."""
    if evaluation.get("schema_version") != "toolret-confirmation-evaluation-v1" or evaluation.get("scope") != "confirmation_study":
        raise ValueError("Expected frozen confirmation evaluation schema/scope.")
    if manifest.get("scope") != "confirmation_study" or manifest.get("split", {}).get("name") != "confirmation":
        raise ValueError("Only a confirmation cache may be analyzed.")
    if frozen.get("created_before_quality_scoring") is not True:
        raise ValueError("Frozen predictions must declare creation before quality scoring.")
    if frozen != evaluation.get("input_integrity"):
        raise ValueError("Standalone prediction provenance differs from evaluation input_integrity.")
    if manifest != evaluation.get("manifest"):
        raise ValueError("Evaluation references a different confirmation cache manifest.")
    hashes = {key: pilot.digest(path) for key, path in paths.items()}
    for key in ("cache_sha256", "cache_manifest_sha256", "queries_sha256", "corpus_sha256", "predictions_sha256"):
        if hashes[key] != frozen.get(key):
            raise ValueError(f"Frozen prediction identity mismatch: {key}")
    for key in ("cache_sha256", "queries_sha256", "corpus_sha256"):
        if hashes[key] != manifest.get(key):
            raise ValueError(f"Cache manifest identity mismatch: {key}")
    for key in ("router_sha256", "protocol_sha256", "data_manifest_sha256", "protocol_frozen_commit"):
        if not isinstance(frozen.get(key), str) or not frozen[key]:
            raise ValueError(f"Missing frozen artifact identity: {key}")
    protocol = manifest.get("cache_provenance", {}).get("protocol", {})
    if protocol.get("rerank_k") != 20 or protocol.get("max_sequence_length") != 256:
        raise ValueError("This diagnostic requires frozen depth20 / sequence256.")
    expected_model = {"name": pilot.MODEL, "revision": pilot.REVISION, "fine_tuned_here": False}
    if manifest.get("models", {}).get("reranker") != expected_model:
        raise ValueError("Pinned production reranker differs from tokenizer audit.")
    serialized_hash = pilot.digest(REPO_ROOT / "src/toolret_research/text.py")
    if manifest["cache_provenance"]["source_sha256"].get("src/toolret_research/text.py") != serialized_hash:
        raise ValueError("Tool serialization differs from cache generation.")
    # Hash agreement verifies the evaluator source version described by the
    # receipt. The chronological claim itself remains an auditable declaration.
    source_hashes = frozen.get("evaluation_source_sha256", {})
    if not isinstance(source_hashes, dict) or not {"scripts/evaluate_confirmation_study.py", "src/toolret_research/confirmation.py", "src/toolret_research/utility_router.py"}.issubset(source_hashes):
        raise ValueError("Missing evaluator and router source provenance.")
    for relative, expected in source_hashes.items():
        path = (REPO_ROOT / relative).resolve()
        if not path.is_relative_to(REPO_ROOT) or pilot.digest(path) != expected:
            raise ValueError(f"Evaluation source differs from frozen provenance: {relative}")
    return {**hashes, **{key: frozen[key] for key in (
        "router_sha256", "protocol_sha256", "data_manifest_sha256", "protocol_frozen_commit")},
        "declared_predictions_created_before_quality_scoring": True,
        "created_after_frozen_predictions": True,
        "labels_altered": False, "analysis_modifies_rankings_or_router": False,
        "script_sha256": pilot.digest(Path(__file__)),
        "reused_analysis_script_sha256": pilot.digest(SCRIPT_ROOT / "analyze_tool_retrieval_failures.py"),
        "text_serialization_sha256": serialized_hash}


def prediction_map(predictions: list[dict], records: list[dict]) -> dict[str, dict]:
    by_id = pilot.index_unique(predictions, "query_id")
    if set(by_id) != {r["query_id"] for r in records} or len(records) != len(by_id):
        raise ValueError("Frozen predictions and diagnostic records differ in query IDs.")
    for record in records:
        prediction = by_id[record["query_id"]]
        decisions = prediction.get("rerank_decisions")
        if not isinstance(decisions, dict) or set(decisions) != set(ROUTING_POLICIES) or any(type(v) is not bool for v in decisions.values()):
            raise ValueError("Frozen policies must contain exactly five boolean decisions.")
        if prediction.get("source_domain") != record["source_domain"]:
            raise ValueError("Frozen prediction source differs from query/cache.")
        utility = prediction.get("predicted_reranking_utility")
        if type(utility) not in (float, int) or not math.isfinite(utility):
            raise ValueError("Frozen predicted utility must be a finite number.")
        if decisions["fixed_disagreement"] != record["gate_selected"]:
            raise ValueError("Frozen disagreement decision differs from first-stage rankings.")
    return by_id


def routing_summary(records: list[dict], predictions: dict[str, dict], policy: str) -> dict:
    n = len(records)
    selected = [r for r in records if predictions[r["query_id"]]["rerank_decisions"][policy]]
    bypassed = [r for r in records if not predictions[r["query_id"]]["rerank_decisions"][policy]]
    value = sum(r["reranked_ndcg10"] if predictions[r["query_id"]]["rerank_decisions"][policy]
                else r["hybrid_ndcg10"] for r in records) / n
    return {"queries": n, "reranker_invocations": len(selected), "bypassed_queries": len(bypassed),
            "ndcg10": value,
            "mean_delta_vs_hybrid_ndcg10": sum(r["delta_ndcg10"] for r in selected) / n,
            "mean_delta_vs_always_ndcg10": -sum(r["delta_ndcg10"] for r in bypassed) / n,
            "selected_outcomes_of_always_vs_hybrid": dict(Counter(r["outcome"] for r in selected)),
            "bypassed_outcomes_of_always_vs_hybrid": dict(Counter(r["outcome"] for r in bypassed)),
            "total_harm_avoided_by_bypass": sum(max(-r["delta_ndcg10"], 0.) for r in bypassed),
            "total_gain_forgone_by_bypass": sum(max(r["delta_ndcg10"], 0.) for r in bypassed)}


def check_evaluation(records: list[dict], predictions: dict[str, dict], evaluation: dict) -> dict:
    if len(records) != evaluation["num_queries"]:
        raise ValueError("Evaluation and diagnostic query count differ.")
    for policy, field in (("hybrid", "hybrid_ndcg10"), ("always", "reranked_ndcg10")):
        point = sum(r[field] for r in records) / len(records)
        if abs(point - evaluation["policies"][policy]["metrics"]["nDCG@10"]) > 1e-12:
            raise ValueError(f"Independent nDCG differs for {policy}.")
    out = {}
    for policy in ROUTING_POLICIES:
        point = routing_summary(records, predictions, policy)
        reference = evaluation["policies"][policy]
        if abs(point["ndcg10"] - reference["metrics"]["nDCG@10"]) > 1e-12 or point["reranker_invocations"] != reference["reranker_invocations"]:
            raise ValueError(f"Frozen policy metrics/calls mismatch: {policy}")
        by_source = {}
        for source in sorted({r["source_domain"] for r in records}):
            values = [r for r in records if r["source_domain"] == source]
            source_point = routing_summary(values, predictions, policy)
            source_reference = evaluation["source_domain_breakdown"][source]["policies"][policy]
            if abs(source_point["ndcg10"] - source_reference["metrics"]["nDCG@10"]) > 1e-12 or source_point["reranker_invocations"] != source_reference["reranker_invocations"]:
                raise ValueError(f"Frozen source policy mismatch: {source}/{policy}")
            by_source[source] = source_point
        out[policy] = {**point, "by_source_domain": by_source}
    return out


def select_cases(records: list[dict], queries: dict[str, dict], predictions: dict[str, dict], limit: int = 5) -> list[dict]:
    cases = []
    for source in sorted({r["source_domain"] for r in records}):
        rows = [r for r in records if r["source_domain"] == source]
        losses = sorted((r for r in rows if r["outcome"] == "loss"), key=lambda r: (r["delta_ndcg10"], r["query_id"]))[:limit]
        wins = sorted((r for r in rows if r["outcome"] == "win"), key=lambda r: (-r["delta_ndcg10"], r["query_id"]))[:limit]
        for row in losses + wins:
            text = queries[row["query_id"]]["query"]
            excerpt = text if len(text) <= 240 else text[:237].rsplit(" ", 1)[0] + "..."
            transitions = row["gold_transitions"]
            observations = []
            if transitions.get("left_top10", 0):
                observations.append("Labeled positive(s) leave top 10.")
            if transitions.get("entered_top10", 0):
                observations.append("Labeled positive(s) enter top 10.")
            if row["positive_count"] > row["prefix_positive_count"]:
                observations.append("Some positive labels are absent from the scored prefix.")
            if row["top1_unjudged_cross_source"]:
                observations.append("CE top 1 is unjudged and comes from another corpus source.")
            if row["top1_possible_name_counterparts"]:
                observations.append("CE top 1 has a name-counterpart hint; equivalence is unverified.")
            cases.append({**row, "query_excerpt": excerpt,
                          "frozen_rerank_decisions": predictions[row["query_id"]]["rerank_decisions"],
                          "descriptive_note": " ".join(observations) or "Ordering changes within the fixed scored prefix.",
                          "note_type": "automatic_rank_and_name_observation_not_causal_annotation"})
    return cases


def render_report(result: dict) -> str:
    count = result["summary"]["queries"]
    lines = ["# Confirmation tool-retrieval failure analysis", "",
             f"This exploratory analysis describes **{count} held-out confirmation queries after the frozen routing predictions and evaluation were saved**. It does not train a router, change routing decisions, or alter relevance labels. Its case examples and hypotheses are descriptive diagnostics, not additional primary tests.", "",
             "## Candidate coverage and ranking changes", "",
             "| Source | Queries | Hybrid nDCG@10 | Always-CE nDCG@10 | CE delta | Wins / losses / unchanged | Mean candidate-label recall@20 | No positive in prefix | Gold left / entered top 10 |",
             "|---|---:|---:|---:|---:|---|---:|---:|---|" ]
    for source, group in result["groups"]["by_source_domain"].items():
        outcomes = group["outcomes"]
        lines.append(f"| {source} | {group['queries']} | {group['hybrid_ndcg10']:.6f} | {group['reranked_ndcg10']:.6f} | {group['mean_delta_ndcg10']:+.6f} | {outcomes.get('win',0)} / {outcomes.get('loss',0)} / {outcomes.get('unchanged',0)} | {group['mean_candidate_label_recall20']:.4f} | {group['queries_without_positive_in_prefix20']} | {group['positive_labels_left_top10']} / {group['positive_labels_entered_top10']} |")
    lines += ["", "Candidate recall is averaged per query using original exact-ID labels. A top-20 permutation cannot recover a positive tool outside that prefix. Positive-label counts do not establish which tools are mandatory for executing the request.", "",
              "## Exact tokenizer replay and unjudged promotions", "",
              "| Source | Truncated prefix pairs / total | Truncated positive pairs / positive prefix pairs | Query-truncated pairs | Unjudged cross-source CE top 1 | Name-counterpart hint at CE top 1 |",
              "|---|---|---|---:|---:|---:|" ]
    for source, group in result["groups"]["by_source_domain"].items():
        lines.append(f"| {source} | {group['prefix_pairs_truncated']} / {group['prefix_pairs']} | {group['positive_prefix_pairs_truncated']} / {group['positive_labels_in_prefix20']} | {group['query_truncated_pairs']} | {group['top1_unjudged_cross_source_queries']} | {group['top1_possible_name_counterpart_queries']} |")
    lines += ["", "The pinned CE tokenizer is replayed with the original flattened tool serialization and longest-first truncation at 256 tokens. Token_type_ids and special-token masks give retained query/document counts. Name-counterpart hints require matching normalized names or a complete suffix match with the shorter name at least 12 characters; they are not executable-equivalence judgments. Unjudged tools are not automatically wrong, and the qrels remain unchanged.", "",
              "## Frozen router's selected and bypassed outcomes", "",
              "| Policy | Required CE calls | Selected CE wins / losses / neutral | Bypassed CE wins / losses / neutral | Mean policy minus always nDCG |",
              "|---|---:|---|---|---:|" ]
    for policy, point in result["policy_failure_summary"].items():
        selected, bypassed = point["selected_outcomes_of_always_vs_hybrid"], point["bypassed_outcomes_of_always_vs_hybrid"]
        lines.append(f"| {policy} | {point['reranker_invocations']} | {selected.get('win',0)} / {selected.get('loss',0)} / {selected.get('unchanged',0)} | {bypassed.get('win',0)} / {bypassed.get('loss',0)} / {bypassed.get('unchanged',0)} | {point['mean_delta_vs_always_ndcg10']:+.6f} |")
    lines += ["", "A bypassed CE win is gain forgone; a bypassed CE loss is harm avoided. These labels are assigned only after applying frozen predictions. They are not features available to a deployable router. Required call counts are checked against the evaluator; actual execution and repeated latency are separate artifacts.", "",
              "## Descriptive ToolBench truncation breakdown", "",
              "| Any positive pair truncated | Queries | CE wins / losses / unchanged | Mean CE minus hybrid nDCG |",
              "|---|---:|---|---:|" ]
    for name, group in result["groups"]["toolbench_by_any_positive_prefix_truncation"].items():
        outcomes = group["outcomes"]
        lines.append(f"| {name} | {group['queries']} | {outcomes.get('win',0)} / {outcomes.get('loss',0)} / {outcomes.get('unchanged',0)} | {group['mean_delta_ndcg10']:+.6f} |")
    lines += ["", "These post-outcome strata are confounded by query, schema, source, and annotation differences. They do not establish that truncation causes or prevents a reranking change. JSON also contains positive-count, query-length, gold-document-length, and candidate-coverage strata.", "",
              "## Deterministic confirmation cases", "",
              "For each source, the five largest losses and five largest wins are selected by signed nDCG delta and query-ID tie break. These are outcome-selected examples and cannot estimate population frequencies. No manual notes from development are reused.", "" ]
    for case in result["cases"]:
        decisions = case["frozen_rerank_decisions"]
        lines += [f"### {case['query_id']} ({case['delta_ndcg10']:+.6f} CE delta)", "",
                  case["descriptive_note"], "", f"Query excerpt: {case['query_excerpt']}", "",
                  f"Frozen utility75 decision: {'rerank' if decisions['utility_75'] else 'bypass'}.", "",
                  "| Labeled tool | Hybrid rank | CE rank | Document tokens | Positive pair truncated |",
                  "|---|---:|---:|---:|---|" ]
        for tool in case["gold_tools"]:
            pair = tool["pair_tokenization"]
            clipped = "not scored" if pair is None else str(pair["pair_truncated"]).lower()
            lines.append(f"| {tool['tool_id']} ({tool['name']}) | {tool['hybrid_rank'] or 'absent'} | {tool['reranked_rank'] or 'absent'} | {tool['document_tokens']} | {clipped} |")
        first = case["reranked_top1"]
        lines += ["", f"CE top 1: `{first['tool_id']}` — {first['name']} (hybrid rank {first['hybrid_rank']}; labeled positive: {first['is_positive_label']}).", ""]
    lines += ["## Limits and interpretation", "",
              "- This diagnostic does not alter the frozen primary hypothesis, noninferiority margin, router, or thresholds.",
              "- Gold-tool component separation is not verified API-family separation or proof of clean pretrained models.",
              "- The exact-ID benchmark does not evaluate execution success or adjudicate every plausible alternate tool.",
              "- A controlled ablation or independently judged equivalence set is needed to test causal explanations.",
              "- Hash checks link these outputs to the immutable predictions and scored cohort; chronology is verified against the evaluator's provenance declaration and source hashes, not an external timestamp authority.", "",
              "## Provenance", "", "```json", json.dumps(result["provenance"], indent=2), "```", "" ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("cache", "manifest", "queries", "corpus", "predictions", "evaluation-summary", "output-prefix"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--prediction-provenance", type=Path)
    args = parser.parse_args()
    prediction_provenance = args.prediction_provenance or args.predictions.parent / "frozen_prediction_provenance.json"
    manifest = json.loads(args.manifest.read_text())
    evaluation = json.loads(args.evaluation_summary.read_text())
    frozen = json.loads(prediction_provenance.read_text())
    paths = {"cache_sha256": args.cache, "cache_manifest_sha256": args.manifest,
             "queries_sha256": args.queries, "corpus_sha256": args.corpus,
             "predictions_sha256": args.predictions, "prediction_provenance_sha256": prediction_provenance,
             "evaluation_summary_sha256": args.evaluation_summary}
    provenance = verify_integrity(paths, manifest, evaluation, frozen)
    if evaluation["num_queries"] != 1500:
        raise ValueError("Frozen confirmation diagnostic expects exactly1500 queries.")
    cache = pilot.read_rows(args.cache)
    queries = pilot.index_unique(pilot.read_rows(args.queries), "id")
    corpus = pilot.index_unique(pilot.read_rows(args.corpus), "id")
    counts = Counter(row.get("domain") for row in queries.values())
    if counts != {"apigen": 500, "toolbench": 500, "toolace": 500}:
        raise ValueError("Frozen confirmation source counts differ from500/500/500.")
    if any(row.get("split") != "confirmation" or not row.get("component_id") for row in queries.values()):
        raise ValueError("Confirmation queries require frozen split/component identity.")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(pilot.MODEL, revision=pilot.REVISION, local_files_only=True)
    records, groups = pilot.analyze(cache, queries, corpus, tokenizer, 256)
    predictions = prediction_map(pilot.read_rows(args.predictions), records)
    policy_results = check_evaluation(records, predictions, evaluation)
    for record in records:
        record["component_id"] = queries[record["query_id"]]["component_id"]
        record["frozen_rerank_decisions"] = predictions[record["query_id"]]["rerank_decisions"]
    result = {"schema_version": SCHEMA, "scope": SCOPE, "summary": pilot.summarize(records), "groups": groups,
              "cases": select_cases(records, queries, predictions), "policy_failure_summary": policy_results,
              "tokenizer": {"model": pilot.MODEL, "revision": pilot.REVISION, "max_length": 256,
                            "truncation": "longest_first", "truncation_side": tokenizer.truncation_side},
              "provenance": provenance,
              "integrity_checks": {"input_hashes_match_frozen_evaluation": True,
                                   "frozen_prediction_provenance_matches_summary": True,
                                   "independent_ndcg_and_policy_invocation_counts_match": True,
                                   "exact_confirmation_cohort_and_source_counts_verified": True},
              "limitations": ["Exploratory post-freeze descriptive analysis; no causal identification.",
                               "Exact-ID judgments do not establish tool equivalence or execution success.",
                               "Deterministic largest-delta examples are not representative frequency estimates.",
                               "Required call counts do not substitute for actual runtime verification."]}
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    outputs = {"json": args.output_prefix.with_suffix(".json"), "markdown": args.output_prefix.with_suffix(".md"),
               "per_query": args.output_prefix.with_name(args.output_prefix.name + "_per_query.jsonl")}
    if any(path.exists() for path in outputs.values()):
        raise ValueError("Refusing to overwrite an existing confirmation diagnostic.")
    outputs["json"].write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    outputs["markdown"].write_text(render_report(result))
    outputs["per_query"].write_text("".join(json.dumps(row, sort_keys=True, allow_nan=False) + "\n" for row in records))
    print(json.dumps({"outputs": {key: str(path) for key, path in outputs.items()}, "queries": len(records),
                      "sources": groups["by_source_domain"], "integrity_checks": result["integrity_checks"]}, indent=2))


if __name__ == "__main__":
    main()
