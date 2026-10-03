"""Auditable, zero-training selective reranking from compatible ranking caches.

This evaluates a fixed policy, not a learned gate or an end-to-end latency
benchmark. Binary qrels are deliberate; graded relevance is not inferred.
"""

from __future__ import annotations

import csv
from collections import Counter
import hashlib
import json
import math
import random
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

from .fusion import weighted_reciprocal_rank_fusion
from .metrics import ndcg_at_k, recall_at_k, reciprocal_rank

SCHEMA_VERSION = "toolret-selective-v1"
SCOPES = {"synthetic", "historical_reproduction", "preliminary_fresh_pilot"}
METHODS = ("bm25", "dense", "hybrid", "reranked")
METRICS = ("Recall@10", "MRR@10", "nDCG@10", "All-positive-label coverage@10")
GATE_RULE = "rerank iff rankings.bm25[0] != rankings.dense[0]"


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _id_list(value: Any, label: str, *, nonempty: bool = True) -> list[str]:
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item.strip() for item in value
    ):
        raise ValueError(f"{label} must be a list of nonempty string IDs")
    if nonempty and not value:
        raise ValueError(f"{label} cannot be empty")
    if len(value) != len(set(value)):
        raise ValueError(f"{label} contains duplicate IDs")
    return list(value)


def validate_manifest(manifest: Any) -> dict[str, Any]:
    if not isinstance(manifest, dict):
        raise ValueError("manifest must be a JSON object")
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"manifest.schema_version must be {SCHEMA_VERSION!r}")
    if manifest.get("scope") not in SCOPES:
        raise ValueError(f"manifest.scope must be one of {sorted(SCOPES)}")
    for field in ("dataset", "models", "cache_provenance", "split"):
        if not isinstance(manifest.get(field), dict) or not manifest[field]:
            raise ValueError(f"manifest.{field} must be a nonempty object")
    for field in ("name", "version", "source"):
        if not manifest["dataset"].get(field):
            raise ValueError(f"manifest.dataset.{field} is required")
    for field in ("dense", "reranker"):
        if not manifest["models"].get(field):
            raise ValueError(f"manifest.models.{field} is required")
    for field in ("name", "selection"):
        if not manifest["split"].get(field):
            raise ValueError(f"manifest.split.{field} is required")
    protocol = manifest["cache_provenance"].get("protocol", {})
    if not isinstance(protocol, dict):
        raise ValueError("manifest.cache_provenance.protocol must be an object")
    if "random_policy_seeds" in protocol:
        seeds = protocol["random_policy_seeds"]
        if (
            not isinstance(seeds, list)
            or len(seeds) != 20
            or any(isinstance(value, bool) or not isinstance(value, int) for value in seeds)
            or len(set(seeds)) != 20
        ):
            raise ValueError("protocol.random_policy_seeds must contain exactly 20 unique integer seeds")
    for field in ("timing_provenance", "api_family_provenance", "data_preparation_manifest"):
        if field in manifest and not isinstance(manifest[field], dict):
            raise ValueError(f"manifest.{field} must be an object")
    corpus_hash = manifest.get("corpus_sha256")
    if (
        not isinstance(corpus_hash, str)
        or len(corpus_hash) != 64
        or any(char not in "0123456789abcdef" for char in corpus_hash)
    ):
        raise ValueError("manifest.corpus_sha256 must be a lowercase SHA-256")
    if "rrf" in manifest:
        config = manifest["rrf"]
        if not isinstance(config, dict):
            raise ValueError("manifest.rrf must be an object")
        k = config.get("k")
        if isinstance(k, bool) or not isinstance(k, int) or k < 0:
            raise ValueError("manifest.rrf.k must be a nonnegative integer")
        weights = config.get("weights")
        if (
            not isinstance(weights, list)
            or len(weights) != 2
            or any(
                isinstance(weight, bool)
                or not isinstance(weight, (int, float))
                or not math.isfinite(weight)
                or weight < 0
                for weight in weights
            )
            or sum(weights) <= 0
        ):
            raise ValueError("manifest.rrf.weights must be two finite nonnegative weights")
    return manifest


def load_excluded_ids(path: str | Path) -> set[str]:
    """Read historical IDs from CSV, JSON list, JSONL, or one ID per line."""
    path = Path(path)
    raw = path.read_text(encoding="utf-8-sig")
    if not raw.strip():
        return set()
    if path.suffix.lower() == ".csv":
        reader = csv.DictReader(raw.splitlines())
        field = next(
            (name for name in ("query_id", "id", "qid") if name in (reader.fieldnames or [])),
            None,
        )
        if field is None:
            raise ValueError("historical CSV needs a query_id, id, or qid column")
        ids = [row[field] for row in reader]
    elif raw.lstrip().startswith("["):
        ids = json.loads(raw)
    elif raw.lstrip().startswith("{"):
        ids = []
        for line in raw.splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError("historical JSONL rows must be objects")
            value = next((row[key] for key in ("query_id", "id", "qid") if key in row), None)
            ids.append(value)
    else:
        ids = [line.strip() for line in raw.splitlines() if line.strip()]
    # Historical CSVs can repeat IDs across multiple failure records.
    if not isinstance(ids, list) or any(
        not isinstance(value, str) or not value.strip() for value in ids
    ):
        raise ValueError("historical query IDs must be nonempty strings")
    return set(ids)


def validate_rows(
    rows: Iterable[dict[str, Any]],
    corpus_ids: Iterable[str],
    manifest: dict[str, Any],
    *,
    excluded_ids: Iterable[str] = (),
    expected_queries: Iterable[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    validate_manifest(manifest)
    known_ids = set(corpus_ids)
    if not known_ids:
        raise ValueError("corpus cannot be empty")
    excluded = set(excluded_ids)
    seen: set[str] = set()
    output = []
    for position, original in enumerate(rows, 1):
        if not isinstance(original, dict):
            raise ValueError(f"cache row {position} must be an object")
        row = dict(original)
        qid = row.get("query_id")
        if not isinstance(qid, str) or not qid.strip():
            raise ValueError(f"cache row {position} needs a nonempty query_id")
        if qid in seen:
            raise ValueError(f"duplicate query_id: {qid}")
        seen.add(qid)
        if manifest["scope"] == "preliminary_fresh_pilot" and qid in excluded:
            raise ValueError(f"fresh pilot contains historically inspected query: {qid}")
        if "query" in row and (not isinstance(row["query"], str) or not row["query"].strip()):
            raise ValueError(f"{qid}: query must be a nonempty string")
        if "source_domain" in row and (
            not isinstance(row["source_domain"], str) or not row["source_domain"].strip()
        ):
            raise ValueError(f"{qid}: source_domain must be a nonempty string")
        gold = _id_list(row.get("relevant_ids"), f"{qid}.relevant_ids")
        unknown_gold = set(gold) - known_ids
        if unknown_gold:
            raise ValueError(f"{qid}: gold IDs missing from corpus: {sorted(unknown_gold)}")
        rankings = row.get("rankings")
        if not isinstance(rankings, dict):
            raise ValueError(f"{qid}.rankings must be an object")
        rankings = {
            method: _id_list(rankings.get(method), f"{qid}.rankings.{method}")
            for method in METHODS
        }
        for method, ranking in rankings.items():
            unknown = set(ranking) - known_ids
            if unknown:
                raise ValueError(f"{qid}.{method}: ranking IDs missing from corpus: {sorted(unknown)}")
        rerank_k = row.get("rerank_k")
        if (
            isinstance(rerank_k, bool)
            or not isinstance(rerank_k, int)
            or rerank_k <= 0
            or rerank_k > len(rankings["hybrid"])
        ):
            raise ValueError(f"{qid}: rerank_k must be positive and <= hybrid length")
        hybrid, reranked = rankings["hybrid"], rankings["reranked"]
        if (
            len(hybrid) != len(reranked)
            or set(hybrid[:rerank_k]) != set(reranked[:rerank_k])
            or hybrid[rerank_k:] != reranked[rerank_k:]
        ):
            raise ValueError(
                f"{qid}: reranked must permute only the hybrid rerank_k prefix and preserve its exact tail"
            )
        if "rrf" in manifest:
            config = manifest["rrf"]
            expected = weighted_reciprocal_rank_fusion(
                [rankings["bm25"], rankings["dense"]],
                weights=config["weights"],
                k=config["k"],
            )
            if hybrid != expected:
                raise ValueError(f"{qid}: hybrid ranking does not match declared weighted RRF")
        if "timings_ms" in row:
            timings = row["timings_ms"]
            if not isinstance(timings, dict) or set(timings) - set(METHODS):
                raise ValueError(f"{qid}: timings_ms contains unknown components")
            for method, value in timings.items():
                if (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(value)
                    or value < 0
                ):
                    raise ValueError(f"{qid}: timing {method} must be finite and nonnegative")
        if "api_family_ids" in row:
            _id_list(row["api_family_ids"], f"{qid}.api_family_ids")
        row["relevant_ids"], row["rankings"] = gold, rankings
        output.append(row)
    if not output:
        raise ValueError("ranking cache cannot be empty")
    declared_counts = {
        "cache_provenance.queries": manifest["cache_provenance"].get("queries"),
        "data_preparation_manifest.query_count": manifest.get("data_preparation_manifest", {}).get("query_count"),
        "query_count": manifest.get("query_count"),
    }
    for label, count in declared_counts.items():
        if count is None:
            continue
        if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
            raise ValueError(f"manifest.{label} must be a positive integer")
        if len(output) != count:
            raise ValueError(f"cache query count {len(output)} differs from manifest.{label}={count}")
    prepared = manifest.get("data_preparation_manifest", {})
    if "pilot_domains" in prepared and "queries_per_domain" in prepared:
        domains, per_domain = prepared["pilot_domains"], prepared["queries_per_domain"]
        if (
            not isinstance(domains, list) or not domains
            or any(not isinstance(domain, str) or not domain for domain in domains)
            or len(domains) != len(set(domains))
            or isinstance(per_domain, bool) or not isinstance(per_domain, int) or per_domain <= 0
        ):
            raise ValueError("prepared pilot domain counts are malformed")
        observed = Counter(row.get("source_domain") for row in output)
        if observed != {domain: per_domain for domain in domains}:
            raise ValueError("cache source_domain counts differ from declared prepared pilot")
    if expected_queries is not None:
        expected = {}
        for query in expected_queries:
            if not isinstance(query, dict):
                raise ValueError("prepared query rows must be objects")
            qid = query.get("id", query.get("query_id"))
            if not isinstance(qid, str) or not qid or qid in expected:
                raise ValueError("prepared queries need unique nonempty string IDs")
            expected[qid] = query
        if set(expected) != seen:
            raise ValueError("cache query IDs differ from the exact prepared query set")
        for row in output:
            query = expected[row["query_id"]]
            gold = _id_list(query.get("relevant_ids"), f"prepared {row['query_id']}.relevant_ids")
            if set(gold) != set(row["relevant_ids"]):
                raise ValueError(f"{row['query_id']}: cache gold IDs differ from prepared query")
            domain = query.get("source_domain", query.get("domain"))
            if domain is not None and row.get("source_domain") != domain:
                raise ValueError(f"{row['query_id']}: cache source_domain differs from prepared query")
            if "query" in row and row["query"] != query.get("query"):
                raise ValueError(f"{row['query_id']}: cache query text differs from prepared query")
    return output


def should_rerank(row: dict[str, Any]) -> bool:
    """Fixed preregistered gate: no scores, labels, training, or tuning."""
    return row["rankings"]["bm25"][0] != row["rankings"]["dense"][0]


def binary_metrics(ranking: list[str], gold_ids: list[str], k: int = 10) -> dict[str, float]:
    gold = set(gold_ids)
    prefix = ranking[:k]
    return {
        f"Recall@{k}": recall_at_k(ranking, gold, k),
        f"MRR@{k}": reciprocal_rank(prefix, gold),
        f"nDCG@{k}": ndcg_at_k(ranking, gold, k),
        f"All-positive-label coverage@{k}": float(gold.issubset(prefix)),
    }


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    return ordered[lower] + (position - lower) * (ordered[upper] - ordered[lower])


def evaluate_policies(
    rows: list[dict[str, Any]],
    manifest: dict[str, Any],
    *,
    seed: int = 20261003,
    bootstrap_resamples: int = 2000,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    from .statistics import exact_sign_test, paired_bootstrap

    if not rows:
        raise ValueError("ranking cache cannot be empty")
    rows = sorted(rows, key=lambda row: row["query_id"])
    selected = {row["query_id"] for row in rows if should_rerank(row)}
    qids = [row["query_id"] for row in rows]
    validate_manifest(manifest)
    declared_seeds = manifest["cache_provenance"].get("protocol", {}).get("random_policy_seeds")
    random_seeds = declared_seeds if declared_seeds is not None else list(range(seed, seed + 20))
    random_choices = {
        f"random_seed_{random_seed}": set(random.Random(random_seed).sample(qids, len(selected)))
        for random_seed in random_seeds
    }
    policy_names = ["bm25", "dense", "hybrid", "always", "gated", *random_choices]
    vectors = {policy: {metric: [] for metric in METRICS} for policy in policy_names}
    per_query, decisions = [], []
    for row in rows:
        qid, rankings = row["query_id"], row["rankings"]
        policy_rankings = {
            "bm25": rankings["bm25"],
            "dense": rankings["dense"],
            "hybrid": rankings["hybrid"],
            "always": rankings["reranked"],
            "gated": rankings["reranked"] if qid in selected else rankings["hybrid"],
            **{
                policy: rankings["reranked"] if qid in choices else rankings["hybrid"]
                for policy, choices in random_choices.items()
            },
        }
        results = {}
        for policy, ranking in policy_rankings.items():
            metrics = binary_metrics(ranking, row["relevant_ids"])
            results[policy] = {"metrics": metrics, "top_10": ranking[:10]}
            for metric, value in metrics.items():
                vectors[policy][metric].append(value)
        per_query.append({
            "query_id": qid,
            "query": row.get("query"),
            "relevant_ids": row["relevant_ids"],
            "rerank_k": row["rerank_k"],
            "api_family_ids": row.get("api_family_ids"),
            "source_domain": row.get("source_domain"),
            "policies": results,
        })
        decisions.append({
            "query_id": qid,
            "gate_rule": GATE_RULE,
            "bm25_top_1": rankings["bm25"][0],
            "dense_top_1": rankings["dense"][0],
            "gate_reranks": qid in selected,
            "random_policies_reranking": [policy for policy, choices in random_choices.items() if qid in choices],
        })
    policies = {}
    all_pairs = sum(row["rerank_k"] for row in rows)
    for policy in policy_names:
        choices = (
            set(qids) if policy == "always" else selected if policy == "gated"
            else random_choices.get(policy, set())
        )
        policies[policy] = {
            "metrics": {metric: mean(values) for metric, values in vectors[policy].items()},
            "reranker_invocations": len(choices),
            "reranker_invocation_fraction": len(choices) / len(rows),
            "reranker_candidate_pairs": sum(row["rerank_k"] for row in rows if row["query_id"] in choices),
        }
    comparisons = {}
    for before, after in (("hybrid", "always"), ("hybrid", "gated"), ("always", "gated")):
        comparisons[f"{after}_minus_{before}"] = {
            metric: {
                "paired_bootstrap": paired_bootstrap(
                    vectors[before][metric], vectors[after][metric],
                    seed=seed, resamples=bootstrap_resamples,
                ),
                "exact_sign_test": exact_sign_test(vectors[before][metric], vectors[after][metric]),
            }
            for metric in METRICS
        }
    random_summary = {
        "seeds": random_seeds,
        "seed_provenance": "manifest_protocol" if declared_seeds is not None else "fallback_seed_plus_offset",
        "invocations_per_seed": len(selected),
        "metrics": {
            metric: {
                "mean_across_seeds": mean(policies[policy]["metrics"][metric] for policy in random_choices),
                "min_across_seeds": min(policies[policy]["metrics"][metric] for policy in random_choices),
                "max_across_seeds": max(policies[policy]["metrics"][metric] for policy in random_choices),
            }
            for metric in METRICS
        },
        "note": "Equal invocation counts; variable rerank_k can produce unequal candidate-pair totals.",
    }
    source_domains = {}
    domains = sorted({row["source_domain"] for row in rows if "source_domain" in row})
    for domain in domains:
        indices = [index for index, row in enumerate(rows) if row.get("source_domain") == domain]
        source_domains[domain] = {
            "num_queries": len(indices),
            "gate_invocations": sum(rows[index]["query_id"] in selected for index in indices),
            "metrics": {
                policy: {
                    metric: mean(values[index] for index in indices)
                    for metric, values in vectors[policy].items()
                }
                for policy in ("bm25", "dense", "hybrid", "always", "gated")
            },
        }
    timings = {}
    for method in METHODS:
        values = [float(row["timings_ms"][method]) for row in rows if method in row.get("timings_ms", {})]
        if values:
            timings[method] = {
                "n": len(values), "mean_ms": mean(values),
                "p50_ms": _percentile(values, 0.5), "p95_ms": _percentile(values, 0.95),
            }
    estimated_sums = {}
    disjoint = manifest.get("timing_provenance", {}).get("disjoint_components") is True
    complete_timings = all(set(METHODS).issubset(row.get("timings_ms", {})) for row in rows)
    if disjoint and complete_timings:
        for policy in policy_names:
            values = []
            for row in rows:
                components = row["timings_ms"]
                if policy in ("bm25", "dense"):
                    value = components[policy]
                else:
                    value = components["bm25"] + components["dense"] + components["hybrid"]
                    use_ce = (
                        policy == "always"
                        or (policy == "gated" and row["query_id"] in selected)
                        or row["query_id"] in random_choices.get(policy, set())
                    )
                    if use_ce:
                        value += components["reranked"]
                values.append(float(value))
            estimated_sums[policy] = {
                "n": len(values), "mean_ms": mean(values),
                "p50_ms": _percentile(values, 0.5), "p95_ms": _percentile(values, 0.95),
            }
    limitations = [
        "Binary qrels; all-positive-label coverage means all positive relevance labels retrieved, not mandatory-tool coverage or proven task execution success.",
        "Confidence intervals resample queries; dependent queries or API families can make uncertainty look too small.",
        "No end-to-end adaptive latency is measured; invocation and candidate-pair reductions are compute proxies.",
        "Random-policy variation across seeds is not a confidence interval for performance on new queries.",
        "Reported sign tests are exploratory and are not corrected for multiple comparisons.",
        "Dataset-domain transfer does not establish generalization to unseen API families.",
        "Declared provenance and historical-ID exclusions are supplied evidence, not proof of absence of pretraining contamination.",
    ]
    if "rrf" not in manifest:
        limitations.append("Hybrid cache is supplied without RRF configuration; weighted fusion cannot be independently verified.")
    if not manifest.get("api_family_provenance"):
        limitations.append("API-family provenance is absent; no API-family generalization claim is supported.")
    if timings:
        limitations.append("Reported timings are separately supplied component measurements; they are not summed into adaptive latency.")
        if not manifest.get("timing_provenance"):
            limitations.append("Timing provenance is absent; component timing comparability is unverified.")
    else:
        limitations.append("No measured component timings were supplied; no latency result is available.")
    if estimated_sums:
        limitations.append("Offline replay estimates sum declared disjoint components sequentially; they exclude gate/routing overhead and are not measured service or adaptive-policy latency.")
    if manifest["scope"] == "synthetic":
        limitations.append("Synthetic inputs verify evaluator correctness only; they are not empirical research findings.")
    if manifest["scope"] == "historical_reproduction":
        limitations.append("Historically inspected queries support reproduction, not independent confirmation of the gate.")
    if len(rows) < 100:
        limitations.append(f"Small pilot: only {len(rows)} queries; results are exploratory.")
    return ({
        "schema_version": SCHEMA_VERSION,
        "scope": manifest["scope"],
        "num_queries": len(rows),
        "gate_rule": GATE_RULE,
        "gate_tuned": False,
        "manifest": manifest,
        "rrf_verification": "exact" if "rrf" in manifest else "not_verified",
        "metrics_relevance": "binary",
        "policies": policies,
        "random_equal_invocation_baseline": random_summary,
        "paired_comparisons": comparisons,
        "source_domain_breakdown": source_domains,
        "gated_candidate_pair_reduction_fraction_vs_always": 1 - policies["gated"]["reranker_candidate_pairs"] / all_pairs,
        "supplied_component_timings_ms": timings,
        "estimated_offline_replay_component_sums_ms": estimated_sums,
        "adaptive_end_to_end_latency_ms": None,
        "limitations": limitations,
    }, per_query, decisions)


def render_report(summary: dict[str, Any]) -> str:
    lines = [
        "# Fixed selective reranking evaluation", "",
        f"Scope: **{summary['scope']}**. Queries: **{summary['num_queries']}**.", "",
        f"Fixed gate: `{summary['gate_rule']}`. No training or threshold tuning.", "",
        "All quality metrics use binary qrels and a cutoff of 10.", "",
        "| Policy | Recall@10 | MRR@10 | nDCG@10 | All-positive-label coverage@10 | CE calls | CE pairs |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for policy in ("bm25", "dense", "hybrid", "always", "gated"):
        result = summary["policies"][policy]
        values = " | ".join(f"{result['metrics'][metric]:.6f}" for metric in METRICS)
        lines.append(f"| {policy} | {values} | {result['reranker_invocations']} | {result['reranker_candidate_pairs']} |")
    lines.extend(["", "Random reranking uses 20 deterministic seeds with the same invocation count as the gate.", ""])
    for metric, values in summary["random_equal_invocation_baseline"]["metrics"].items():
        lines.append(f"- {metric}: mean {values['mean_across_seeds']:.6f}; seed range {values['min_across_seeds']:.6f}–{values['max_across_seeds']:.6f}.")
    lines.extend(["", "| Paired comparison | nDCG@10 difference | Query-bootstrap 95% interval |", "|---|---:|---:|"])
    for name, values in summary["paired_comparisons"].items():
        estimate = values["nDCG@10"]["paired_bootstrap"]
        lines.append(f"| {name} | {estimate['delta_mean']:.6f} | [{estimate['low']:.6f}, {estimate['high']:.6f}] |")
    if summary["estimated_offline_replay_component_sums_ms"]:
        lines.extend(["", "## Estimated offline replay component sums", "",
                      "These are counterfactual sums of declared disjoint components, excluding gate/routing overhead. They are not measured adaptive or service latency.", "",
                      "| Policy | Estimated mean ms | Estimated p50 ms | Estimated p95 ms |", "|---|---:|---:|---:|"])
        for policy in ("bm25", "dense", "hybrid", "always", "gated"):
            timing = summary["estimated_offline_replay_component_sums_ms"][policy]
            lines.append(f"| {policy} | {timing['mean_ms']:.3f} | {timing['p50_ms']:.3f} | {timing['p95_ms']:.3f} |")
    lines.extend(["", "No end-to-end adaptive latency was measured. Separate supplied component timings, when present, are in summary.json.", "", "## Limitations", ""])
    lines.extend(f"- {limitation}" for limitation in summary["limitations"])
    return "\n".join(lines) + "\n"
