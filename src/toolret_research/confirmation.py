"""Frozen confirmation-study routing and source/group-aware evaluation.

Routing accepts only first-stage rankings and query length. Qrels are consulted
only after routing decisions have been frozen. Gold components are positive-
tool connected components, not inferred API families.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
import math
import random
from statistics import mean
from typing import Any, Iterable
import unicodedata

import numpy as np

from .selective import METHODS, METRICS, binary_metrics, validate_rows
from .utility_router import FEATURE_NAMES, RidgeUtilityModel, features, predict, shouldroute

DOMAINS = ("apigen", "toolbench", "toolace")
BUDGETS = (25, 50, 75)
PRIMARY_POLICY = "utility_75"
BOOTSTRAP_SEED = 20261003
BOOTSTRAP_RESAMPLES = 10000
NONINFERIORITY_MARGIN = .01
RANDOM_SEEDS = tuple(range(20))
DEPLOYABLE = ("bm25", "dense", "hybrid", "always", "fixed_disagreement", "cheap_jaccard", "utility_25", "utility_50", "utility_75")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def validate_confirmation_rows(
    rows: Iterable[dict[str, Any]], corpus_ids: Iterable[str], manifest: dict[str, Any],
    expected_queries: Iterable[dict[str, Any]], *, expected_count: int = 1500,
    expected_per_domain: int = 500,
) -> list[dict[str, Any]]:
    """Reuse exact rank/RRF checks without changing the sealed pilot evaluator."""
    if manifest.get("schema_version") != "toolret-selective-v1" or manifest.get("scope") != "confirmation_study":
        raise ValueError("confirmation cache requires toolret-selective-v1 / confirmation_study")
    if manifest.get("split", {}).get("name") != "confirmation" or manifest.get("cache_provenance", {}).get("partition") != "confirmation":
        raise ValueError("confirmation cache manifest must declare confirmation split and partition")
    prepared = list(expected_queries)
    # The old validator's scope enum is deliberately sealed. Its synthetic
    # validation branch imposes all rank, hash-declared-count, and qrel checks
    # without adding a claim that this independent confirmation is a pilot.
    compatible = dict(manifest, scope="synthetic")
    output = validate_rows(rows, corpus_ids, compatible, expected_queries=prepared)
    by_id = {q.get("id", q.get("query_id")): q for q in prepared}
    if len(output) != expected_count:
        raise ValueError(f"confirmation count must be {expected_count}")
    source_counts = Counter(row.get("source_domain") for row in output)
    if source_counts != {source: expected_per_domain for source in DOMAINS}:
        raise ValueError("confirmation source counts differ from frozen protocol")
    component_sources: dict[str, str] = {}
    tool_components: dict[str, str] = {}
    for row in output:
        query = by_id[row["query_id"]]
        if query.get("split") != "confirmation" or row.get("split") != "confirmation":
            raise ValueError("confirmation queries/cache must carry the confirmation split")
        component = query.get("component_id")
        if not isinstance(component, str) or not component.strip():
            raise ValueError("prepared confirmation queries require component_id")
        if row.get("component_id", component) != component:
            raise ValueError("cache/prepared component IDs disagree")
        text = query.get("query")
        if not isinstance(text, str) or not text.strip():
            raise ValueError("prepared confirmation queries require query text")
        source = row["source_domain"]
        if component in component_sources and component_sources[component] != source:
            raise ValueError("positive-tool components cannot span sources")
        component_sources[component] = source
        for tool_id in row["relevant_ids"]:
            if tool_id in tool_components and tool_components[tool_id] != component:
                raise ValueError("shared positive tool must have the same component ID")
            tool_components[tool_id] = component
        row["component_id"] = component
        row["query"] = text
    return sorted(output, key=lambda row: row["query_id"])


def assert_split_disjointness(*query_partitions: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Reject training/calibration/confirmation ID, gold, or group leakage."""
    previous: dict[str, set[str]] = {"ids": set(), "components": set(), "positive_tools": set(), "normalized_texts": set()}
    details = []
    for index, partition in enumerate(query_partitions):
        rows = list(partition)
        current = {
            "ids": {row.get("id", row.get("query_id")) for row in rows},
            "components": {row["component_id"] for row in rows},
            "positive_tools": {tool for row in rows for tool in row["relevant_ids"]},
            "normalized_texts": {" ".join(unicodedata.normalize("NFKC", row["query"]).casefold().split()) for row in rows},
        }
        if any(not isinstance(value, str) or not value for values in current.values() for value in values):
            raise ValueError("partition IDs, components, and positive tools must be nonempty strings")
        if len(current["ids"]) != len(rows):
            raise ValueError("partition contains duplicate query IDs")
        if len(current["normalized_texts"]) != len(rows):
            raise ValueError("partition contains duplicate normalized query texts")
        for name, values in current.items():
            if previous[name] & values:
                raise ValueError(f"partition leakage in {name}")
            previous[name] |= values
        details.append({"partition": index, "queries": len(rows), **{name: len(values) for name, values in current.items()}})
    return {"disjoint_query_ids": True, "disjoint_positive_tools": True, "disjoint_positive_tool_components": True, "disjoint_normalized_query_texts": True, "partitions": details}


def freeze_predictions(rows: list[dict[str, Any]], router: dict[str, Any]) -> list[dict[str, Any]]:
    """Generate deployable decisions without reading qrels or CE rankings."""
    if router.get("schema_version") != "toolret-frozen-utility-router-v1":
        raise ValueError("router must use the frozen utility-router schema")
    model = RidgeUtilityModel.from_dict(router["model"])
    if model.alpha != 10 or model.training_rows != 300:
        raise ValueError("router must use frozen alpha10 and300 development observations")
    thresholds = router["thresholds"]
    if set(thresholds) != {str(budget / 100) for budget in BUDGETS}:
        raise ValueError("router must contain exactly the frozen25/50/75% thresholds")
    for budget in BUDGETS:
        entry = thresholds[str(budget / 100)]
        if (entry.get("nominal_budget_fraction") != budget / 100
                or entry.get("decision_rule") != "predicted_ndcg_delta > threshold"
                or entry.get("primary") is not (budget == 75)):
            raise ValueError("router threshold metadata differs from the frozen pointwise method")
    records = []
    for row in sorted(rows, key=lambda item: item["query_id"]):
        rankings = row["rankings"]
        # Explicitly pass only cheap first-stage rankings to the feature API.
        values = features(row["query"], {name: rankings[name] for name in ("bm25", "dense")})
        prediction = float(predict(model, values))
        bm25_10, dense_10 = set(rankings["bm25"][:10]), set(rankings["dense"][:10])
        jaccard = len(bm25_10 & dense_10) / len(bm25_10 | dense_10)
        decisions = {
            "fixed_disagreement": rankings["bm25"][0] != rankings["dense"][0],
            "cheap_jaccard": jaccard < .5,
            **{f"utility_{budget}": shouldroute(prediction, float(thresholds[str(budget / 100)]["threshold"])) for budget in BUDGETS},
        }
        records.append({"query_id": row["query_id"], "source_domain": row["source_domain"],
                        "predicted_reranking_utility": prediction,
                        "features": dict(zip(FEATURE_NAMES, map(float, values))),
                        "rerank_decisions": decisions})
    return records


def validate_predictions(rows: list[dict[str, Any]], predictions: list[dict[str, Any]]) -> dict[str, dict[str, bool]]:
    by_id: dict[str, dict[str, bool]] = {}
    expected_names = {"fixed_disagreement", "cheap_jaccard", *(f"utility_{budget}" for budget in BUDGETS)}
    for record in predictions:
        qid = record.get("query_id")
        decisions = record.get("rerank_decisions")
        if not isinstance(qid, str) or not qid or qid in by_id:
            raise ValueError("frozen predictions require unique IDs")
        if not isinstance(decisions, dict) or set(decisions) != expected_names or any(type(value) is not bool for value in decisions.values()):
            raise ValueError("frozen prediction decisions differ from declared policies")
        value = record.get("predicted_reranking_utility")
        if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
            raise ValueError("frozen predictions must be finite")
        by_id[qid] = decisions
    if set(by_id) != {row["query_id"] for row in rows}:
        raise ValueError("frozen prediction IDs differ from confirmation cache")
    return by_id


def stratified_bootstrap_deltas(
    rows: list[dict[str, Any]], differences: dict[str, list[float]], *,
    resamples: int = BOOTSTRAP_RESAMPLES, seed: int = BOOTSTRAP_SEED,
) -> dict[str, dict[str, Any]]:
    """Paired query and group percentile intervals with fixed 1/3 source weights.

    Component draws are ratio-of-sums within each source, so the target remains
    a query-macro mean rather than an equally weighted component mean.
    The same sampled indices are used for every paired comparison.
    """
    if isinstance(resamples, bool) or not isinstance(resamples, int) or resamples <= 0:
        raise ValueError("bootstrap resamples must be a positive integer")
    if not rows or not differences or any(len(values) != len(rows) for values in differences.values()):
        raise ValueError("bootstrap differences must align with nonempty rows")
    arrays = {name: np.asarray(values, dtype=np.float64) for name, values in differences.items()}
    if any(not np.isfinite(values).all() for values in arrays.values()):
        raise ValueError("bootstrap differences must be finite")
    component_sources: dict[str, str] = {}
    for row in rows:
        if row["source_domain"] not in DOMAINS:
            raise ValueError("unexpected bootstrap source")
        component = row["component_id"]
        if component in component_sources and component_sources[component] != row["source_domain"]:
            raise ValueError("bootstrap components cannot span sources")
        component_sources[component] = row["source_domain"]
    query_draws = {name: np.zeros(resamples) for name in arrays}
    component_draws = {name: np.zeros(resamples) for name in arrays}
    estimates = {name: 0. for name in arrays}
    query_rng, component_rng = np.random.default_rng(seed), np.random.default_rng(seed)
    source_counts, group_counts = {}, {}
    for source in DOMAINS:
        indices = np.asarray([i for i, row in enumerate(rows) if row["source_domain"] == source], dtype=np.int64)
        if not len(indices):
            raise ValueError("all frozen sources must have observations")
        grouped: dict[str, list[int]] = defaultdict(list)
        for index in indices:
            grouped[rows[int(index)]["component_id"]].append(int(index))
        groups = [grouped[key] for key in sorted(grouped)]
        sizes = np.asarray([len(group) for group in groups], dtype=np.int64)
        source_counts[source], group_counts[source] = len(indices), len(groups)
        # Batches bound memory and share samples across comparisons.
        for start in range(0, resamples, 250):
            stop = min(start + 250, resamples)
            qsample = query_rng.integers(0, len(indices), size=(stop - start, len(indices)))
            csample = component_rng.integers(0, len(groups), size=(stop - start, len(groups)))
            denominator = sizes[csample].sum(axis=1)
            for name, values in arrays.items():
                query_draws[name][start:stop] += values[indices][qsample].mean(axis=1) / len(DOMAINS)
                sums = np.asarray([values[group].sum() for group in groups])
                component_draws[name][start:stop] += sums[csample].sum(axis=1) / denominator / len(DOMAINS)
        for name, values in arrays.items():
            estimates[name] += float(values[indices].mean()) / len(DOMAINS)
    output = {}
    for name in arrays:
        def interval(values: np.ndarray) -> dict[str, Any]:
            low, high = np.quantile(values, [.025, .975], method="linear")
            return {"delta_mean": estimates[name], "low": float(low), "high": float(high),
                    "confidence": .95, "resamples": resamples, "seed": seed,
                    "source_weights": {source: 1 / len(DOMAINS) for source in DOMAINS}}
        output[name] = {
            "source_stratified_query_bootstrap": interval(query_draws[name]),
            "source_stratified_component_bootstrap": {
                **interval(component_draws[name]), "estimator": "per-source ratio of sampled component delta sums to sampled query counts; macro over sources",
                "components_by_source": group_counts, "queries_by_source": source_counts,
            },
        }
    return output


def evaluate_confirmation(
    rows: list[dict[str, Any]], predictions: list[dict[str, Any]], *,
    resamples: int = BOOTSTRAP_RESAMPLES, seed: int = BOOTSTRAP_SEED,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    """Score frozen decisions; this function never fits or calibrates a policy."""
    rows = sorted(rows, key=lambda row: row["query_id"])
    decision_map = validate_predictions(rows, predictions)
    if any(row["rerank_k"] != 20 for row in rows):
        raise ValueError("confirmation requires exactly 20 scored pairs per CE invocation")
    qids = [row["query_id"] for row in rows]
    selections = {name: {qid for qid in qids if decision_map[qid][name]} for name in decision_map[qids[0]]}
    global_random, source_random = {}, {}
    for target in ("fixed_disagreement", "utility_25", "utility_50", "utility_75"):
        count = len(selections[target])
        for random_seed in RANDOM_SEEDS:
            name = f"random_global_{target}_seed_{random_seed}"
            global_random[name] = set(random.Random(random_seed).sample(qids, count))
        if target in ("fixed_disagreement", "utility_75"):
            for random_seed in RANDOM_SEEDS:
                rng, chosen = random.Random(random_seed), set()
                for source in DOMAINS:
                    domain_qids = [row["query_id"] for row in rows if row["source_domain"] == source]
                    count_source = sum(qid in selections[target] for qid in domain_qids)
                    chosen.update(rng.sample(domain_qids, count_source))
                source_random[f"random_source_matched_{target}_seed_{random_seed}"] = chosen
    all_random = {**global_random, **source_random}
    selections.update(all_random)
    selections.update({"always": set(qids), "bm25": set(), "dense": set(), "hybrid": set()})
    vectors = {name: {metric: [] for metric in METRICS} for name in selections}
    per_query, decisions = [], []
    gains, hybrid_metrics, always_metrics = {}, {}, {}
    for row in rows:
        qid, rankings = row["query_id"], row["rankings"]
        hm = binary_metrics(rankings["hybrid"], row["relevant_ids"])
        am = binary_metrics(rankings["reranked"], row["relevant_ids"])
        hybrid_metrics[qid], always_metrics[qid] = hm, am
        gains[qid] = am["nDCG@10"] - hm["nDCG@10"]
        results = {}
        for name, chosen in selections.items():
            metrics = (binary_metrics(rankings[name], row["relevant_ids"]) if name in ("bm25", "dense")
                       else am if qid in chosen else hm)
            for metric, value in metrics.items():
                vectors[name][metric].append(value)
            if name in DEPLOYABLE:
                results[name] = metrics
        per_query.append({"query_id": qid, "source_domain": row["source_domain"], "component_id": row["component_id"],
                          "relevant_ids": row["relevant_ids"], "always_minus_hybrid_nDCG@10": gains[qid], "policies": results})
        decisions.append({"query_id": qid, "source_domain": row["source_domain"], "component_id": row["component_id"],
                          "rerank_policies": [name for name, chosen in selections.items() if qid in chosen]})
    policies = {}
    for name, chosen in selections.items():
        policies[name] = {"metrics": {metric: mean(values) for metric, values in vectors[name].items()},
                          "reranker_invocations": len(chosen), "reranker_invocation_fraction": len(chosen) / len(rows),
                          "reranker_candidate_pairs": 20 * len(chosen),
                          "reranker_invocation_reduction_fraction_vs_always": 1 - len(chosen) / len(rows)}
    random_summaries = {}
    for kind, collection in (("global", global_random), ("source_matched", source_random)):
        targets = ("fixed_disagreement", "utility_25", "utility_50", "utility_75") if kind == "global" else ("fixed_disagreement", "utility_75")
        for target in targets:
            names = [name for name in collection if f"_{target}_seed_" in name]
            random_summaries[f"{kind}_{target}"] = {
                "target_policy": target, "selection": kind, "seeds": list(RANDOM_SEEDS),
                "invocations_per_seed": len(selections[target]),
                "metrics": {metric: {"mean_across_seeds": mean(policies[name]["metrics"][metric] for name in names),
                                      "min_across_seeds": min(policies[name]["metrics"][metric] for name in names),
                                      "max_across_seeds": max(policies[name]["metrics"][metric] for name in names)} for metric in METRICS},
                "note": "Seed range is policy-randomization variation, not a confidence interval for new queries.",
            }
    comparisons = {}
    for after, before in (("utility_75", "always"), ("fixed_disagreement", "always"), ("hybrid", "always"), ("utility_75", "fixed_disagreement")):
        comparisons[f"{after}_minus_{before}"] = [a - b for a, b in zip(vectors[after]["nDCG@10"], vectors[before]["nDCG@10"])]
    for target in ("utility_75", "fixed_disagreement"):
        for kind, collection in (("global", global_random), ("source_matched", source_random)):
            names = [name for name in collection if f"_{target}_seed_" in name]
            random_means = [mean(vectors[name]["nDCG@10"][index] for name in names) for index in range(len(rows))]
            comparisons[f"{target}_minus_random_{kind}_mean"] = [a - b for a, b in zip(vectors[target]["nDCG@10"], random_means)]
    paired = stratified_bootstrap_deltas(rows, comparisons, resamples=resamples, seed=seed)
    primary = paired["utility_75_minus_always"]
    lower_query = primary["source_stratified_query_bootstrap"]["low"]
    lower_cluster = primary["source_stratified_component_bootstrap"]["low"]
    fewer_calls = policies["utility_75"]["reranker_invocations"] < len(rows)
    success = lower_query > -NONINFERIORITY_MARGIN and lower_cluster > -NONINFERIORITY_MARGIN and fewer_calls
    oracle = {}
    sorted_gain_ids = sorted(qids, key=lambda qid: (-gains[qid], qid))
    base = mean(hybrid_metrics[qid]["nDCG@10"] for qid in qids)
    for target in ("fixed_disagreement", "utility_25", "utility_50", "utility_75"):
        count = len(selections[target])
        oracle[target] = {"matched_invocations": count, "nDCG@10": base + sum(gains[qid] for qid in sorted_gain_ids[:count]) / len(rows),
                          "deployable": False, "uses_confirmation_labels": True,
                          "selection": "largest true always-minus-hybrid nDCG gain; exact realized invocation count"}
    domain_breakdown = {}
    for source in DOMAINS:
        indices = [i for i, row in enumerate(rows) if row["source_domain"] == source]
        domain_breakdown[source] = {"num_queries": len(indices), "positive_tool_components": len({rows[i]["component_id"] for i in indices}),
                                    "policies": {name: {"metrics": {metric: mean(vectors[name][metric][i] for i in indices) for metric in METRICS},
                                                        "reranker_invocations": sum(qids[i] in selections[name] for i in indices)} for name in DEPLOYABLE}}
    routing_effects = {}
    for target in ("fixed_disagreement", "cheap_jaccard", "utility_25", "utility_50", "utility_75"):
        skipped = [qid for qid in qids if qid not in selections[target]]
        routed = [qid for qid in qids if qid in selections[target]]
        routing_effects[target] = {
            "harmful_skips_count": sum(gains[qid] > 1e-12 for qid in skipped),
            "harmful_skips_total_nDCG_gain_forgone": sum(max(gains[qid], 0.) for qid in skipped),
            "beneficial_skips_count": sum(gains[qid] < -1e-12 for qid in skipped),
            "beneficial_skips_total_nDCG_harm_avoided": sum(max(-gains[qid], 0.) for qid in skipped),
            "neutral_skips_count": sum(abs(gains[qid]) <= 1e-12 for qid in skipped),
            "harmful_reranks_count": sum(gains[qid] < -1e-12 for qid in routed),
            "beneficial_reranks_count": sum(gains[qid] > 1e-12 for qid in routed),
        }
    summary = {
        "schema_version": "toolret-confirmation-evaluation-v1", "scope": "confirmation_study", "num_queries": len(rows),
        "metrics_relevance": "binary", "cutoff": 10, "source_weights": {source: 1 / 3 for source in DOMAINS},
        "policies": policies, "random_equal_budget_baselines": random_summaries, "paired_nDCG_comparisons": paired,
        "source_domain_breakdown": domain_breakdown, "routing_effects": routing_effects,
        "oracle_matched_budget_ceiling": oracle,
        "primary_noninferiority": {"policy": PRIMARY_POLICY, "reference": "always", "metric": "nDCG@10",
                                  "predeclared_engineering_margin": NONINFERIORITY_MARGIN,
                                  "criterion": "both two-sided95% bootstrap lower bounds > -0.01 AND fewer CE invocations",
                                  "query_bootstrap_lower": lower_query, "component_bootstrap_lower": lower_cluster,
                                  "fewer_required_reranker_invocations": fewer_calls,
                                  "quality_and_required_call_criterion_met": success,
                                  "runtime_verification_pending": True,
                                  "actual_conditional_execution_verified": False,
                                  "interpretation": "Meets the predeclared quality and required-call criteria on this held-out cohort; actual conditional execution must be verified separately." if success else "Does not establish the predeclared engineering noninferiority claim."},
        "limitations": [
            "Gold components connect queries sharing positive tool IDs; they are not verified API-provider families.",
            "Binary qrels and all-positive-label coverage do not demonstrate downstream agent execution success.",
            "Target budgets are calibrated pointwise thresholds; realized test invocation fractions can differ from 25/50/75%.",
            "Only utility75 noninferiority is primary; other policy/metric comparisons and domain breakdowns are exploratory.",
            "The 0.01 margin is a predeclared engineering tolerance, not an externally accepted universal quality threshold.",
            "Random seed ranges express routing randomness; they are not statistical confidence intervals.",
            "Oracle selection uses test labels and is diagnostic, never a deployed or competitive policy.",
            "Invocation and pair counts are compute proxies; actual repeated latency is reported separately.",
            "Public pretrained models may have seen related source data; dataset group separation does not prove pretraining cleanliness.",
        ],
    }
    return summary, per_query, decisions


def render_confirmation_report(summary: dict[str, Any]) -> str:
    lines = ["# Held-out selective tool-retrieval confirmation study", "",
             f"Queries: **{summary['num_queries']}**. Binary relevance; cutoff 10. Frozen pointwise routing; no confirmation policy selection.", "",
             "| Policy | nDCG@10 | Recall@10 | MRR@10 | All-positive-label coverage@10 | CE calls | CE pairs |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for name in DEPLOYABLE:
        value = summary["policies"][name]
        metric = value["metrics"]
        lines.append(f"| {name} | {metric['nDCG@10']:.6f} | {metric['Recall@10']:.6f} | {metric['MRR@10']:.6f} | {metric['All-positive-label coverage@10']:.6f} | {value['reranker_invocations']} | {value['reranker_candidate_pairs']} |")
    result = summary["primary_noninferiority"]
    lines.extend(["", "## Predeclared primary claim", "", result["interpretation"], "",
                  "Utility75 versus always-on reranking, engineering margin 0.01. Success requires both paired 95% interval lower bounds strictly above −0.01 and fewer CE invocations.", "",
                  "| Comparison | Mean nDCG difference | Source-stratified query 95% CI | Source-stratified gold-component 95% CI |",
                  "|---|---:|---:|---:|"])
    for name, comparison in summary["paired_nDCG_comparisons"].items():
        query, component = comparison["source_stratified_query_bootstrap"], comparison["source_stratified_component_bootstrap"]
        lines.append(f"| {name} | {query['delta_mean']:.6f} | [{query['low']:.6f}, {query['high']:.6f}] | [{component['low']:.6f}, {component['high']:.6f}] |")
    lines.extend(["", "Component bootstrap samples groups within each source and uses a ratio-of-sums estimator to retain the query-macro target. Both procedures keep source weights at 1/3 and reuse paired draws across comparisons.", "", "## Equal-budget random controls", "",
                  "| Control | Calls per seed | Mean nDCG | Seed range |", "|---|---:|---:|---:|"])
    for name, control in summary["random_equal_budget_baselines"].items():
        metric = control["metrics"]["nDCG@10"]
        lines.append(f"| {name} | {control['invocations_per_seed']} | {metric['mean_across_seeds']:.6f} | [{metric['min_across_seeds']:.6f}, {metric['max_across_seeds']:.6f}] |")
    lines.extend(["", "Seed ranges are random-policy variation, not confidence intervals.", "", "## Limitations", ""])
    lines.extend(f"- {limitation}" for limitation in summary["limitations"])
    return "\n".join(lines) + "\n"
