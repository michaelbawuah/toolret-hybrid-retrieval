"""Independently audit the confirmation study from source bytes and saved outputs.

This script imports no study preparation, routing, metric, or evaluator module.
It independently reconstructs the source universe, graph partition, ridge fit,
pointwise routing, metrics, random controls, and paired intervals. An early
``--data-only`` run never opens confirmation rankings or quality outputs.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import random
import subprocess
import unicodedata

import numpy as np

DOMAINS = ("apigen", "toolbench", "toolace")
SPLITS = ("development", "calibration", "confirmation", "quarantined_pilot_connected", "reserve")
FEATURE_NAMES = (
    "top1_disagreement", "top10_jaccard", "top20_jaccard",
    "top20_reciprocal_weighted_jaccard", "top20_rank_coherence",
    "query_token_count_log1p", "rrf_top2_normalized_margin",
)
METRICS = ("nDCG@10", "MRR@10", "Recall@10", "All-positive-label coverage@10")


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1048576), b""):
            result.update(chunk)
    return result.hexdigest()


def normalized(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def positives(row: dict) -> set[str]:
    labels = row["labels"]
    labels = json.loads(labels) if isinstance(labels, str) else labels
    return {label["id"] for label in labels if float(label["relevance"]) > 0}


def components(rows: list[dict]) -> dict[str, list[str]]:
    """Traverse the query/tool bipartite graph; avoid the author's union-find."""
    by_id = {row["id"]: row for row in rows}
    if len(by_id) != len(rows):
        raise ValueError("duplicate query ID")
    by_tool = defaultdict(set)
    for row in rows:
        if not row["relevant_ids"]:
            raise ValueError("query without positive tools")
        for tool in row["relevant_ids"]:
            by_tool[tool].add(row["id"])
    remaining, result = set(by_id), {}
    while remaining:
        first = min(remaining)
        visited, pending = set(), [first]
        while pending:
            qid = pending.pop()
            if qid in visited:
                continue
            visited.add(qid)
            for tool in by_id[qid]["relevant_ids"]:
                pending.extend(by_tool[tool] - visited)
        remaining -= visited
        result[min(visited)] = sorted(visited)
    return dict(sorted(result.items()))


def assigned_splits(rows: list[dict], development_ids: set[str], seed: str,
                    calibration_per_domain: int, confirmation_per_domain: int) -> tuple[dict, dict]:
    """Reconstruct all anchored components and whole-component quota admission."""
    by_id = {row["id"]: row for row in rows}
    if not development_ids <= by_id.keys():
        raise ValueError("development is not within the eligible universe")
    graph = components(rows)
    anchored = {component for component, qids in graph.items() if development_ids.intersection(qids)}
    free = sorted(set(graph) - anchored, key=lambda component: (
        hashlib.sha256((seed + "\0" + component).encode()).hexdigest(), component))
    assignment = {component: "quarantined_pilot_connected" for component in anchored}
    for split, quota in (("calibration", calibration_per_domain), ("confirmation", confirmation_per_domain)):
        remaining = dict.fromkeys(DOMAINS, quota)
        for component in free:
            if component in assignment:
                continue
            sizes = Counter(by_id[qid]["domain"] for qid in graph[component])
            if all(sizes[source] <= remaining[source] for source in sizes):
                assignment[component] = split
                for source in sizes:
                    remaining[source] -= sizes[source]
        if any(remaining.values()):
            raise ValueError(f"infeasible {split} quotas: {remaining}")
    output = {split: set() for split in SPLITS}
    for component, qids in graph.items():
        for qid in qids:
            output["development" if qid in development_ids else assignment.get(component, "reserve")].add(qid)
    return output, graph


def metric_values(ranking: list[str], gold: set[str]) -> dict[str, float]:
    if not gold or len(ranking) != len(set(ranking)):
        raise ValueError("metrics require positive gold and a unique ranking")
    relevant_positions = [i for i, tool in enumerate(ranking[:10], 1) if tool in gold]
    dcg = math.fsum(1 / math.log2(position + 1) for position in relevant_positions)
    ideal = math.fsum(1 / math.log2(position + 1) for position in range(1, min(10, len(gold)) + 1))
    return {"nDCG@10": dcg / ideal,
            "MRR@10": 1 / relevant_positions[0] if relevant_positions else 0.,
            "Recall@10": len(relevant_positions) / len(gold),
            "All-positive-label coverage@10": float(gold <= set(ranking[:10]))}


def independent_features(query: str, sparse: list[str], dense: list[str]) -> np.ndarray:
    """Evaluate the protocol's seven formulas, without the routing library."""
    if not query.strip() or not sparse or not dense or len(sparse) != len(set(sparse)) or len(dense) != len(set(dense)):
        raise ValueError("invalid feature input")
    s10, d10, s20, d20 = set(sparse[:10]), set(dense[:10]), set(sparse[:20]), set(dense[:20])
    s_rank = {tool: position for position, tool in enumerate(sparse[:20], 1)}
    d_rank = {tool: position for position, tool in enumerate(dense[:20], 1)}
    union = s20 | d20
    weighted_min = math.fsum(min(1 / s_rank[tool], 1 / d_rank[tool]) for tool in s20 & d20)
    weighted_max = math.fsum(max(1 / s_rank[tool] if tool in s_rank else 0.,
                                1 / d_rank[tool] if tool in d_rank else 0.) for tool in union)
    distance = math.fsum(abs(s_rank.get(tool, 21) - d_rank.get(tool, 21)) for tool in union)
    scores = defaultdict(float)
    for rank in (sparse, dense):
        for position, tool in enumerate(rank, 1):
            scores[tool] += 1 / (60 + position)
    leading = sorted(scores.values(), reverse=True)
    margin = (leading[0] - leading[1]) / leading[0] if len(leading) > 1 else 1.
    return np.asarray([float(sparse[0] != dense[0]), len(s10 & d10) / len(s10 | d10),
                       len(s20 & d20) / len(union), weighted_min / weighted_max,
                       1 - distance / len(union) / 20, math.log1p(len(query.split())), margin])


def independent_ridge(x: np.ndarray, y: np.ndarray, alpha: float = 10.) -> dict:
    means, scales = x.mean(axis=0), x.std(axis=0)
    scales = np.where(scales > 1e-12, scales, 1.)
    z = (x - means) / scales
    intercept = float(y.mean())
    # Independent augmented least-squares solution, rather than author's normal equations.
    coefficients = np.linalg.lstsq(np.vstack((z, math.sqrt(alpha) * np.eye(x.shape[1]))),
                                  np.concatenate((y - intercept, np.zeros(x.shape[1]))), rcond=None)[0]
    return {"means": means, "scales": scales, "coefficients": coefficients, "intercept": intercept}


def paired_intervals(rows: list[dict], differences: dict[str, np.ndarray], resamples: int, seed: int) -> dict:
    """Independent stratified paired query and ratio-of-component-sums draws.

    Draw order matches the frozen evaluator for exact reproducibility, while
    all component arithmetic is independently rebuilt from source IDs.
    """
    query_rng, cluster_rng = np.random.default_rng(seed), np.random.default_rng(seed)
    query_draws = {name: np.zeros(resamples) for name in differences}
    cluster_draws = {name: np.zeros(resamples) for name in differences}
    estimates = {name: 0. for name in differences}
    for source in DOMAINS:
        ids = np.asarray([i for i, row in enumerate(rows) if row["source_domain"] == source])
        grouped = defaultdict(list)
        for i in ids:
            grouped[rows[int(i)]["component_id"]].append(int(i))
        groups = [grouped[key] for key in sorted(grouped)]
        sizes = np.asarray([len(group) for group in groups])
        group_totals = {name: np.asarray([values[group].sum() for group in groups]) for name, values in differences.items()}
        for start in range(0, resamples, 250):
            stop = min(start + 250, resamples)
            q = query_rng.integers(len(ids), size=(stop - start, len(ids)))
            c = cluster_rng.integers(len(groups), size=(stop - start, len(groups)))
            n = sizes[c].sum(axis=1)
            for name, values in differences.items():
                query_draws[name][start:stop] += values[ids[q]].sum(axis=1) / len(ids) / 3
                cluster_draws[name][start:stop] += group_totals[name][c].sum(axis=1) / n / 3
        for name, values in differences.items():
            estimates[name] += float(values[ids].mean()) / 3
    return {name: {"delta_mean": estimates[name], "query_ci95": np.quantile(query_draws[name], [.025, .975]).tolist(),
                   "component_ci95": np.quantile(cluster_draws[name], [.025, .975]).tolist()} for name in differences}


def evidence_fingerprints(args) -> dict[str, str]:
    """Bind a passing audit to all supplied evidence and its source code bytes."""
    paths = {args.corpus, args.protocol, args.data_dir / "manifest.json", Path(__file__),
             Path("data/research_20261003/queries.jsonl"),
             Path("results/research_confirmation_20261003/data_manifest.json"),
             Path("results/research_confirmation_20261003/data_audit.json"),
             Path("results/research_confirmation_20261003/component_assignments.json"),
             Path("results/research_confirmation_20261003/selected_query_ids.json"),
             Path("scripts/prepare_confirmation_study.py"), Path("scripts/prepare_research_pilot.py")}
    paths.update(args.data_dir / f"{name}_queries.jsonl" for name in SPLITS)
    paths.update((args.data_dir / "raw").glob("*.parquet"))
    paths.add(args.data_dir / "raw/sources.json")
    for name in ("model", "calibration_cache", "calibration_manifest", "confirmation_cache",
                 "confirmation_manifest", "summary", "predictions", "runtime", "failure_summary"):
        path = getattr(args, name)
        if path:
            paths.add(path)
    if args.model:
        paths.update((Path("results/research_20261003/pilot_cache/rankings.jsonl"), Path("results/research_20261003/pilot_cache/manifest.json")))
        paths.update(args.model.parent.glob("*predictions.jsonl"))
        paths.update((args.model.parent.parent / name) for name in ("protocol_freeze_receipt.json", "router_freeze_receipt.json"))
    if args.summary:
        paths.update(args.summary.parent / name for name in ("per_query.jsonl", "decisions.jsonl", "frozen_prediction_provenance.json", "report.md"))
    if args.latency_dir:
        paths.update(args.latency_dir / name for name in ("summary.json", "events.jsonl", "selection.json", "per_query.jsonl", "report.md"))
    if args.failure_summary:
        paths.update((args.failure_summary.with_name(args.failure_summary.stem + "_per_query.jsonl"), args.failure_summary.with_suffix(".md"),
                      Path("scripts/analyze_tool_retrieval_failures.py"), Path("scripts/analyze_confirmation_failures.py"), Path("src/toolret_research/text.py")))
    # Every referenced executable/source hash is also independently fingerprinted
    # for finalization; changing any evidence requires a fresh complete audit.
    for path in list(paths):
        if path.exists() and path.suffix == ".json":
            obj = json.loads(path.read_text())
            if not isinstance(obj, dict):
                continue
            for owner in (obj, obj.get("cache_provenance", {}), obj.get("input_integrity", {})):
                for field in ("source_sha256", "evaluation_source_sha256"):
                    paths.update(Path(source) for source in owner.get(field, {}))
    repo = Path.cwd().resolve()
    result = {}
    for path in sorted(paths):
        if path.exists():
            resolved = path.resolve()
            key = resolved.relative_to(repo).as_posix() if resolved.is_relative_to(repo) else str(resolved)
            result[key] = digest(path)
    return dict(sorted(result.items()))


class Audit:
    def __init__(self):
        self.checks = []
        self.details = {}

    def check(self, name: str, condition: bool, detail=None):
        self.checks.append({"name": name, "status": "pass" if condition else "fail", "detail": detail})

    def close(self, left, right, atol=1e-11) -> bool:
        try:
            return bool(np.allclose(left, right, rtol=0, atol=atol))
        except (TypeError, ValueError):
            return False

    def pending(self, name: str):
        self.checks.append({"name": name, "status": "pending"})

    def data(self, root: Path, corpus_path: Path, protocol_path: Path):
        import pyarrow.parquet as parquet

        manifest = json.loads((root / "manifest.json").read_text())
        corpus = read_rows(corpus_path)
        corpus_ids = {row["id"] for row in corpus}
        self.check("corpus_identity", len(corpus) == len(corpus_ids) == 37292 and digest(corpus_path) == manifest["prepared_files"]["corpus.jsonl"]["sha256"])
        raw = {}
        for source in manifest["upstream_sources"]:
            domain = source["path"].split("/")[0]
            path = root / "raw" / f"{domain}.parquet"
            self.check(f"raw_source_{domain}", path.stat().st_size == source["bytes"] and digest(path) == source["sha256"]
                       and source["revision"] == manifest["source_revisions"][source["repository"]])
            raw[domain] = parquet.read_table(path).to_pylist()
        forbidden_texts = {normalized(row["query"]) for row in raw["apibank"]}
        eligible, seen, removed = [], set(), []
        for qid, domain, row in sorted((row["id"], domain, row) for domain in DOMAINS for row in raw[domain]):
            text = normalized(row["query"])
            if text in forbidden_texts or text in seen:
                removed.append(qid)
                continue
            seen.add(text)
            labels = json.loads(row["labels"]) if isinstance(row["labels"], str) else row["labels"]
            eligible.append({"id": qid, "query": row["query"], "domain": domain, "category": row["category"],
                             "instruction": row.get("instruction", ""), "relevant_ids": sorted(positives(row)), "labels": labels})
        self.check("deduplicated_raw_eligible_universe", len(eligible) == 3099 and removed == ["toolbench_query_738"]
                   and dict(Counter(row["domain"] for row in eligible)) == {"apigen": 1000, "toolbench": 1099, "toolace": 1000})
        self.check("eligible_gold_labels_and_corpus_membership", all(row["category"] == "web" and set(row["relevant_ids"]) <= corpus_ids and row["relevant_ids"] for row in eligible))
        prepared = {name: read_rows(root / f"{name}_queries.jsonl") for name in SPLITS}
        for name in SPLITS:
            path = root / f"{name}_queries.jsonl"
            entry = manifest["prepared_files"][path.name]
            self.check(f"prepared_file_identity_{name}", digest(path) == entry["sha256"] and path.stat().st_size == entry["bytes"] and len(prepared[name]) == entry["query_count"])
        old_path = Path("data/research_20261003/queries.jsonl")
        old = {row["id"]: row for row in read_rows(old_path)}
        annotated = {row["id"]: {k: v for k, v in row.items() if k not in ("component_id", "split")} for row in prepared["development"]}
        self.check("development_is_exact_observed_pilot", digest(old_path) == manifest["development_source_sha256"] and old == annotated and len(old) == 300)
        expected, graph = assigned_splits(eligible, set(old), manifest["seed"], manifest["calibration_per_domain"], manifest["confirmation_per_domain"])
        component_by_id = {qid: component for component, qids in graph.items() for qid in qids}
        by_id = {row["id"]: row for row in eligible}
        for name in SPLITS:
            self.check(f"whole_component_assignment_{name}", expected[name] == {row["id"] for row in prepared[name]}
                       and all(row["component_id"] == component_by_id[row["id"]] and row["split"] == name
                               and {k: v for k, v in row.items() if k not in ("component_id", "split")} == by_id[row["id"]] for row in prepared[name]))
        self.check("component_ids_are_global_lexical_minima", len(graph) == 2005 and all(key == min(qids) for key, qids in graph.items()))
        self.check("components_are_source_exclusive", all(len({by_id[qid]["domain"] for qid in qids}) == 1 for qids in graph.values()))
        for left, right in (("development", "calibration"), ("development", "confirmation"), ("calibration", "confirmation")):
            a, b = prepared[left], prepared[right]
            overlaps = {"ids": len({r["id"] for r in a} & {r["id"] for r in b}),
                        "normalized_texts": len({normalized(r["query"]) for r in a} & {normalized(r["query"]) for r in b}),
                        "positive_tools": len(set().union(*(set(r["relevant_ids"]) for r in a)) & set().union(*(set(r["relevant_ids"]) for r in b))),
                        "components": len({r["component_id"] for r in a} & {r["component_id"] for r in b})}
            self.check(f"split_disjointness_{left}_{right}", not any(overlaps.values()), overlaps)
        anchored = {component_by_id[qid] for qid in old}
        anchored_queries = set().union(*(set(graph[key]) for key in anchored))
        self.check("all_nontraining_pilot_connected_queries_quarantined", len(anchored_queries) == 929 and len(prepared["quarantined_pilot_connected"]) == 629
                   and anchored_queries - set(old) == expected["quarantined_pilot_connected"])
        inventory = json.loads(Path("results/research_confirmation_20261003/component_assignments.json").read_text())["components"]
        inventory_valid = inventory.keys() == graph.keys()
        for component, qids in graph.items():
            record = inventory[component]
            target = "quarantined_pilot_connected" if component in anchored else next(name for name in SPLITS if qids[0] in expected[name])
            inventory_valid &= record["assignment"] == target and record["query_ids"] == qids
            inventory_valid &= record["contains_observed_pilot"] == (component in anchored)
            inventory_valid &= record["query_count"] == len(qids) and record["domain_counts"] == dict(Counter(by_id[qid]["domain"] for qid in qids))
            inventory_valid &= record["positive_tool_count"] == len(set().union(*(set(by_id[qid]["relevant_ids"]) for qid in qids)))
            inventory_valid &= record["observed_pilot_query_ids"] == [qid for qid in qids if qid in old]
        self.check("independent_full_component_inventory", inventory_valid)
        self.check("preparation_source_hash", digest(Path("scripts/prepare_confirmation_study.py")) == manifest["preparation_source_sha256"])
        self.details["data"] = {"eligible_queries": len(eligible), "components": len(graph), "removed_query_ids": removed,
                                "counts": {name: len(prepared[name]) for name in SPLITS},
                                "components_per_split": {name: len({row["component_id"] for row in values}) for name, values in prepared.items()},
                                "domains_per_split": {name: dict(Counter(row["domain"] for row in values)) for name, values in prepared.items()}}
        if protocol_path.exists():
            self.protocol(protocol_path, prepared)
        else:
            self.pending("frozen_protocol")
        return prepared, corpus_ids

    def protocol(self, path: Path, prepared: dict):
        protocol = json.loads(path.read_text())
        for name in ("split_manifest", "data_audit", "component_assignments"):
            self.check(f"protocol_linked_{name}_hash", digest(Path(protocol[name])) == protocol[f"{name}_sha256"])
        self.check("protocol_primary_is_prespecified", protocol["primary_policy"] == "utility_75" and protocol["reference_policy"] == "always"
                   and protocol["noninferiority_margin"] == .01 and protocol["bootstrap_resamples"] == 10000
                   and protocol["bootstrap_seed"] == 20261003 and protocol["random_policy_seeds"] == list(range(20)))
        for name in ("development", "calibration", "confirmation"):
            entry = protocol[name]
            self.check(f"protocol_cohort_{name}", entry["queries"] == len(prepared[name]) and entry["component_count"] == len({r["component_id"] for r in prepared[name]})
                       and digest(Path(entry["queries_path"])) == entry["queries_sha256"])
        self.check("protocol_feature_order_and_fit", protocol["utility_router"]["feature_order"] == list(FEATURE_NAMES)
                   and protocol["utility_router"]["alpha"] == 10 and protocol["utility_router"]["budget_fractions"] == [.25, .5, .75])
        self.details["protocol_sha256"] = digest(path)

    def cache(self, path: Path, manifest_path: Path, queries: list[dict], corpus_ids: set[str], protocol_path: Path):
        rows = read_rows(path)
        manifest = json.loads(manifest_path.read_text())
        by_id = {row["id"]: row for row in queries}
        self.check(f"cache_hash_{path.parent.name}", digest(path) == manifest["cache_sha256"])
        self.check(f"cache_cohort_{path.parent.name}", len(rows) == len(by_id) and {r["query_id"] for r in rows} == by_id.keys())
        valid = True
        for row in rows:
            source = by_id[row["query_id"]]
            valid &= set(row["relevant_ids"]) == set(source["relevant_ids"]) and row["source_domain"] == source["domain"]
            valid &= row.get("component_id", source.get("component_id")) == source.get("component_id")
            rankings = row["rankings"]
            valid &= all(ranking and len(ranking) == len(set(ranking)) and set(ranking) <= corpus_ids for ranking in rankings.values())
            valid &= len(rankings["bm25"]) == len(rankings["dense"]) == 100
            scores = defaultdict(float)
            for method in ("bm25", "dense"):
                for position, tool in enumerate(rankings[method], 1):
                    scores[tool] += 1 / (60 + position)
            hybrid = sorted(scores, key=lambda tool: (-scores[tool], tool))
            valid &= hybrid == rankings["hybrid"] and row["rerank_k"] == 20
            valid &= len(rankings["reranked"]) == len(hybrid) and set(rankings["reranked"][:20]) == set(hybrid[:20]) and rankings["reranked"][20:] == hybrid[20:]
        self.check(f"cache_ranks_qrels_and_rrf_{path.parent.name}", valid)
        protocol = json.loads(protocol_path.read_text()) if protocol_path.exists() else None
        if protocol:
            self.check(f"cache_pinned_models_{path.parent.name}", all(manifest["models"][name]["revision"] == protocol[f"{name}_revision"]
                       and manifest["models"][name]["name"] == protocol[f"{name}_model"] for name in ("dense", "reranker")))
        if manifest.get("scope") == "confirmation_study":
            self.check(f"cache_protocol_{path.parent.name}", manifest["cache_provenance"]["protocol"] == protocol)
            for source, expected in manifest["cache_provenance"]["source_sha256"].items():
                self.check(f"cache_source_hash_{path.parent.name}_{source}", digest(Path(source)) == expected)
        if manifest.get("scope") in ("preliminary_fresh_pilot", "confirmation_study"):
            self.details.setdefault("cache_inputs", {})["training" if manifest.get("scope") == "preliminary_fresh_pilot" else manifest["split"]["name"]] = {
                "cache_sha256": digest(path), "manifest_sha256": digest(manifest_path), "queries_sha256": manifest["queries_sha256"]}
        return rows, manifest

    def router(self, model_path: Path, train: list[dict], calibration: list[dict], prepared: dict, protocol_path: Path):
        artifact = json.loads(model_path.read_text())
        model = artifact["model"]
        for cohort in ("training", "calibration"):
            declared = artifact["input_integrity"][cohort]
            for key, actual in self.details["cache_inputs"][cohort].items():
                self.check(f"router_input_hash_{cohort}_{key}", declared[key] == actual)
        qlookup = {name: {r["id"]: r for r in rows} for name, rows in prepared.items()}
        x = np.asarray([independent_features(qlookup["development"][r["query_id"]]["query"], r["rankings"]["bm25"], r["rankings"]["dense"]) for r in train])
        y = np.asarray([metric_values(r["rankings"]["reranked"], set(r["relevant_ids"]))["nDCG@10"] - metric_values(r["rankings"]["hybrid"], set(r["relevant_ids"]))["nDCG@10"] for r in train])
        fit = independent_ridge(x, y)
        self.check("router_schema_and_fixed_fit", artifact["schema_version"] == "toolret-frozen-utility-router-v1" and model["feature_names"] == list(FEATURE_NAMES)
                   and model["alpha"] == 10 and model["training_rows"] == 300 and artifact["calibration_labels_used_for_fit_or_thresholds"] is False and artifact["confirmation_inputs_accessed"] is False)
        for name in ("means", "scales", "coefficients", "intercept"):
            self.check(f"independent_router_{name}", self.close(fit[name], model[name]))
        cx = np.asarray([independent_features(qlookup["calibration"][r["query_id"]]["query"], r["rankings"]["bm25"], r["rankings"]["dense"]) for r in calibration])
        predictions = (cx - fit["means"]) / fit["scales"] @ fit["coefficients"] + fit["intercept"]
        for budget in (.25, .5, .75):
            threshold = float(np.quantile(predictions, 1 - budget, method="linear"))
            entry = artifact["thresholds"][str(budget)]
            self.check(f"label_free_threshold_{budget}", self.close(threshold, entry["threshold"])
                       and int((predictions > entry["threshold"]).sum()) == entry["calibration_invocations"])
        commit = artifact["input_integrity"]["protocol_frozen_commit"]
        frozen = subprocess.check_output(["git", "show", f"{commit}:{protocol_path.as_posix()}" ])
        self.check("router_protocol_frozen_commit", hashlib.sha256(frozen).hexdigest() == digest(protocol_path)
                   == artifact["input_integrity"]["protocol_sha256"])
        protocol = json.loads(protocol_path.read_text())
        for name in ("split_manifest", "data_audit", "component_assignments"):
            committed = subprocess.check_output(["git", "show", f"{commit}:{protocol[name]}"], stderr=subprocess.DEVNULL)
            self.check(f"public_protocol_committed_{name}", hashlib.sha256(committed).hexdigest() == protocol[f"{name}_sha256"])
        protocol_receipt_path = model_path.parent.parent / "protocol_freeze_receipt.json"
        if protocol_receipt_path.exists():
            protocol_receipt = json.loads(protocol_receipt_path.read_text())
            published_tree = subprocess.check_output(["git", "rev-parse", f"{commit}^{{tree}}"], text=True).strip()
            self.check("public_protocol_tree_receipt", published_tree == protocol_receipt["public_git_tree"]
                       and protocol_receipt["public_protocol_commit"] == commit and protocol_receipt["protocol_sha256"] == digest(protocol_path))
        for source, expected in artifact["source_sha256"].items():
            self.check(f"router_source_hash_{source}", digest(Path(source)) == expected)
        self.details["router"] = {"model_sha256": digest(model_path), "protocol_frozen_commit": commit,
                                  "coefficients": fit["coefficients"].tolist(), "thresholds": artifact["thresholds"]}
        receipt_path = model_path.parent.parent / "router_freeze_receipt.json"
        if receipt_path.exists():
            receipt = json.loads(receipt_path.read_text())
            self.check("router_freeze_receipt_hash", receipt["router_sha256"] == digest(model_path) and receipt["public_protocol_commit"] == commit)
            try:
                published = subprocess.check_output(["git", "show", f"{receipt['public_router_commit']}:{model_path.as_posix()}"], stderr=subprocess.DEVNULL)
                ancestry = subprocess.run(["git", "merge-base", "--is-ancestor", commit, receipt["public_router_commit"]], capture_output=True).returncode == 0
                self.check("public_committed_router_bytes_and_protocol_ancestry", hashlib.sha256(published).hexdigest() == digest(model_path) and ancestry)
                published_tree = subprocess.check_output(["git", "rev-parse", f"{receipt['public_router_commit']}^{{tree}}"], text=True).strip()
                self.check("public_router_tree_receipt", published_tree == receipt["public_git_tree"])
                self.details["router"]["public_router_commit"] = receipt["public_router_commit"]
            except subprocess.CalledProcessError:
                self.pending("public_committed_router_bytes_and_protocol_ancestry")
        return artifact

    def evaluation(self, rows: list[dict], prepared: list[dict], artifact: dict, summary_path: Path,
                   predictions_path: Path):
        summary = json.loads(summary_path.read_text())
        rows = sorted(rows, key=lambda row: row["query_id"])
        by_id = {r["id"]: r for r in prepared}
        saved = {r["query_id"]: r for r in read_rows(predictions_path)}
        self.check("frozen_predictions_exact_cohort", saved.keys() == by_id.keys())
        qids = [r["query_id"] for r in rows]
        selected = {name: set() for name in ("fixed_disagreement", "cheap_jaccard", "utility_25", "utility_50", "utility_75")}
        model = artifact["model"]
        predictions_valid = True
        for r in rows:
            qid, ranks = r["query_id"], r["rankings"]
            f = independent_features(by_id[qid]["query"], ranks["bm25"], ranks["dense"])
            p = float((f - np.asarray(model["means"])) / np.asarray(model["scales"]) @ np.asarray(model["coefficients"]) + model["intercept"])
            decisions = {"fixed_disagreement": bool(f[0]), "cheap_jaccard": bool(f[1] < .5),
                         **{f"utility_{budget}": p > artifact["thresholds"][str(budget / 100)]["threshold"] for budget in (25, 50, 75)}}
            predictions_valid &= self.close(p, saved[qid]["predicted_reranking_utility"]) and decisions == saved[qid]["rerank_decisions"]
            predictions_valid &= self.close(f, [saved[qid]["features"][name] for name in FEATURE_NAMES]) and saved[qid]["source_domain"] == r["source_domain"]
            for name, route in decisions.items():
                if route:
                    selected[name].add(qid)
        self.check("independent_frozen_pointwise_predictions", predictions_valid)
        for target in ("fixed_disagreement", "utility_25", "utility_50", "utility_75"):
            for seed in range(20):
                selected[f"random_global_{target}_seed_{seed}"] = set(random.Random(seed).sample(qids, len(selected[target])))
            if target in ("fixed_disagreement", "utility_75"):
                for seed in range(20):
                    rng, chosen = random.Random(seed), set()
                    for source in DOMAINS:
                        source_ids = [r["query_id"] for r in rows if r["source_domain"] == source]
                        chosen.update(rng.sample(source_ids, len(set(source_ids) & selected[target])))
                    selected[f"random_source_matched_{target}_seed_{seed}"] = chosen
        selected.update({"always": set(qids), "bm25": set(), "dense": set(), "hybrid": set()})
        decision_rows = read_rows(summary_path.parent / "decisions.jsonl")
        decision_by_id = {r["query_id"]: r for r in decision_rows}
        self.check("independent_all_policy_decision_sets", len(decision_rows) == len(qids) and decision_by_id.keys() == set(qids)
                   and all(set(decision_by_id[qid]["rerank_policies"]) == {name for name, chosen in selected.items() if qid in chosen} for qid in qids))
        vectors = {name: np.zeros((len(rows), len(METRICS))) for name in selected}
        for i, row in enumerate(rows):
            qid, gold = row["query_id"], set(row["relevant_ids"])
            base = {method: metric_values(row["rankings"][method], gold) for method in ("bm25", "dense", "hybrid", "reranked")}
            for name, routed in selected.items():
                method = name if name in ("bm25", "dense") else "reranked" if qid in routed else "hybrid"
                vectors[name][i] = [base[method][metric] for metric in METRICS]
        for name, vector in vectors.items():
            reported = summary["policies"][name]
            self.check(f"independent_policy_metrics_and_compute_{name}", self.close(vector.mean(axis=0), [reported["metrics"][metric] for metric in METRICS])
                       and reported["reranker_invocations"] == len(selected[name]) and reported["reranker_candidate_pairs"] == 20 * len(selected[name]))
        saved_per_query = {r["query_id"]: r for r in read_rows(summary_path.parent / "per_query.jsonl")}
        per_query_valid = saved_per_query.keys() == set(qids)
        for i, row in enumerate(rows):
            saved_record = saved_per_query[row["query_id"]]
            per_query_valid &= saved_record["source_domain"] == row["source_domain"] and saved_record["component_id"] == row["component_id"]
            per_query_valid &= set(saved_record["relevant_ids"]) == set(row["relevant_ids"])
            for name, metrics in saved_record["policies"].items():
                per_query_valid &= self.close(vectors[name][i], [metrics[metric] for metric in METRICS])
        self.check("independent_raw_per_query_metrics", per_query_valid)
        for control_name, control in summary["random_equal_budget_baselines"].items():
            prefix = "random_global" if control["selection"] == "global" else "random_source_matched"
            names = [f"{prefix}_{control['target_policy']}_seed_{seed}" for seed in range(20)]
            values = np.asarray([vectors[name].mean(axis=0) for name in names])
            valid = control["seeds"] == list(range(20)) and control["invocations_per_seed"] == len(selected[control["target_policy"]])
            for i, metric in enumerate(METRICS):
                reported = control["metrics"][metric]
                valid &= self.close([values[:, i].mean(), values[:, i].min(), values[:, i].max()],
                                    [reported["mean_across_seeds"], reported["min_across_seeds"], reported["max_across_seeds"]])
            self.check(f"independent_random_control_summary_{control_name}", valid)
        for source in DOMAINS:
            indices = [i for i, row in enumerate(rows) if row["source_domain"] == source]
            breakdown = summary["source_domain_breakdown"][source]
            valid = breakdown["num_queries"] == len(indices) and breakdown["positive_tool_components"] == len({rows[i]["component_id"] for i in indices})
            for name, reported in breakdown["policies"].items():
                valid &= self.close(vectors[name][indices].mean(axis=0), [reported["metrics"][metric] for metric in METRICS])
                valid &= reported["reranker_invocations"] == sum(qids[i] in selected[name] for i in indices)
            self.check(f"independent_source_breakdown_{source}", valid)
        gains = vectors["always"][:, 0] - vectors["hybrid"][:, 0]
        for name, effect in summary["routing_effects"].items():
            mask = np.asarray([qid in selected[name] for qid in qids])
            expected = {
                "harmful_skips_count": int(((gains > 1e-12) & ~mask).sum()),
                "harmful_skips_total_nDCG_gain_forgone": float(np.maximum(gains[~mask], 0).sum()),
                "beneficial_skips_count": int(((gains < -1e-12) & ~mask).sum()),
                "beneficial_skips_total_nDCG_harm_avoided": float(np.maximum(-gains[~mask], 0).sum()),
                "neutral_skips_count": int(((abs(gains) <= 1e-12) & ~mask).sum()),
                "harmful_reranks_count": int(((gains < -1e-12) & mask).sum()),
                "beneficial_reranks_count": int(((gains > 1e-12) & mask).sum()),
            }
            self.check(f"independent_routing_effects_{name}", all(self.close(value, effect[key]) for key, value in expected.items()))
        for name, oracle in summary["oracle_matched_budget_ceiling"].items():
            count = len(selected[name])
            best = float(vectors["hybrid"][:, 0].mean() + np.sort(gains)[::-1][:count].sum() / len(rows))
            self.check(f"independent_label_aware_oracle_{name}", oracle["deployable"] is False and oracle["uses_confirmation_labels"] is True
                       and oracle["matched_invocations"] == count and self.close(best, oracle["nDCG@10"]))
        self.details["evaluation"] = {"policies": {name: {"nDCG@10": float(vector[:, 0].mean()), "calls": len(selected[name])}
                                                  for name, vector in vectors.items() if not name.startswith("random_")}}
        differences = {}
        for name in summary["paired_nDCG_comparisons"]:
            after, before = name.split("_minus_", 1)
            if before in vectors:
                differences[name] = vectors[after][:, 0] - vectors[before][:, 0]
            else:
                # Paired comparisons to a query-wise mean over matched controls.
                aliases = {"random_global_mean": "random_global", "random_source_matched_mean": "random_source_matched",
                           "global_random_mean": "random_global", "source_matched_random_mean": "random_source_matched"}
                family = next((value for alias, value in aliases.items() if before.startswith(alias)), None)
                if family is None:
                    family = "random_source_matched" if "source" in before else "random_global" if "random" in before else None
                names = [key for key in vectors if family and key.startswith(f"{family}_{after}_seed_")]
                if not names:
                    raise ValueError(f"unknown paired comparison {name}")
                differences[name] = vectors[after][:, 0] - np.asarray([vectors[key][:, 0] for key in names]).mean(axis=0)
        paired = paired_intervals(rows, differences, 10000, 20261003)
        for name, values in paired.items():
            reported = summary["paired_nDCG_comparisons"][name]
            query, cluster = reported["source_stratified_query_bootstrap"], reported["source_stratified_component_bootstrap"]
            self.check(f"independent_paired_intervals_{name}", self.close(values["delta_mean"], query["delta_mean"])
                       and self.close(values["query_ci95"], [query["low"], query["high"]])
                       and self.close(values["component_ci95"], [cluster["low"], cluster["high"]]))
        primary = paired["utility_75_minus_always"]
        success = min(primary["query_ci95"][0], primary["component_ci95"][0]) > -.01 and len(selected["utility_75"]) < len(rows)
        self.check("primary_noninferiority_conclusion", success == summary["primary_noninferiority"]["quality_and_required_call_criterion_met"])
        self.details["evaluation"]["primary_paired_intervals"] = primary
        self.details["evaluation"]["quality_and_required_call_criterion_met"] = success
        return selected

    def runtime(self, path: Path, selected: dict, inputs: dict[str, Path]):
        record = json.loads(path.read_text())
        qids = {r["query_id"] for r in record["per_query"]}
        all_qids = selected["always"]
        primary = selected["utility_75"]
        valid = len(record["per_query"]) == len(qids) == len(all_qids) == record["queries"] == 1500 and qids == all_qids
        valid &= record["actual_backend_calls"] == len(primary) and record["actual_backend_pairs"] == len(primary) * 20 and record["skipped_calls"] == 1500 - len(primary)
        valid &= all(r["reranked"] == (r["query_id"] in primary) and r["scored_pairs"] == (20 if r["reranked"] else 0)
                     and r["matches_frozen_decision"] is True and r["matches_evaluated_ranking"] is True for r in record["per_query"])
        self.check("independent_full_cohort_conditional_runtime_accounting", valid)
        for key, actual_path in inputs.items():
            self.check(f"primary_runtime_input_{key}", digest(actual_path) == record["input_integrity"][key])
        for source, expected in record["source_sha256"].items():
            self.check(f"primary_runtime_source_{source}", digest(Path(source)) == expected)
        self.details["runtime"] = {"actual_calls": record["actual_backend_calls"], "actual_pairs": record["actual_backend_pairs"], "queries": record["queries"]}

    def latency(self, directory: Path, rows: list[dict], selected: dict, protocol_path: Path,
                input_paths: dict[str, Path], confirmation_manifest: Path):
        summary = json.loads((directory / "summary.json").read_text())
        selection = json.loads((directory / "selection.json").read_text())
        events = read_rows(directory / "events.jsonl")
        protocol = json.loads(protocol_path.read_text())
        cfg = protocol["latency_benchmark"]
        policies = cfg["policies"]
        by_id = {row["query_id"]: row for row in rows}
        expected_ids = []
        for source in sorted(DOMAINS):
            source_ids = [qid for qid, row in by_id.items() if row["source_domain"] == source]
            expected_ids.extend(sorted(source_ids, key=lambda qid: (hashlib.sha256((cfg["selection_seed"] + "\0" + qid).encode()).hexdigest(), qid))[:30])
        self.check("latency_predefined_balanced_selection", selection["query_ids"] == expected_ids and summary["queries"] == 90
                   and summary["source_domain_counts"] == dict.fromkeys(DOMAINS, 30))
        rng, schedule = random.Random(cfg["order_seed"]), []
        for repetition in range(3):
            ids, order = list(expected_ids), list(policies)
            rng.shuffle(ids)
            rng.shuffle(order)
            for position, qid in enumerate(ids):
                offset = position % 4
                for policy_position, policy in enumerate(order[offset:] + order[:offset]):
                    schedule.append({"query_id": qid, "repetition": repetition, "policy": policy, "policy_position": policy_position})
        self.check("latency_exact_counterbalanced_schedule", selection["schedule"] == schedule and len(events) == 1080
                   and all({key: event[key] for key in schedule[i]} == schedule[i] for i, event in enumerate(events)))
        valid, grouped = True, defaultdict(list)
        for event in events:
            qid, policy = event["query_id"], event["policy"]
            route = policy == "always" or policy not in ("always", "hybrid") and qid in selected[policy]
            expected_rank = by_id[qid]["rankings"]["reranked" if route else "hybrid"]
            valid &= event["reranked"] == route and event["scored_pairs"] == (20 if route else 0) and event["matches_cache"] is True
            valid &= event["ranking_sha256"] == hashlib.sha256(json.dumps(expected_rank).encode()).hexdigest()
            valid &= math.isfinite(event["total_ms"]) and event["total_ms"] > 0 and all(math.isfinite(v) and v >= 0 for v in event["components_ms"].values())
            valid &= set(event["components_ms"]) == {"bm25", "dense", "rrf", "routing", "cross_encoder"}
            valid &= event["components_ms"]["cross_encoder"] > 0 if route else event["components_ms"]["cross_encoder"] == 0
            valid &= math.isclose(event["total_ms"], sum(event["components_ms"].values()), rel_tol=1e-9, abs_tol=1e-8)
            grouped[(qid, policy)].append(event)
        self.check("latency_raw_event_counts_timings_routes_and_rank_hashes", valid)
        qids = sorted(expected_ids)
        values = {}
        per_query = read_rows(directory / "per_query.jsonl")
        per_query_map = {row["query_id"]: row for row in per_query}
        saved_means_valid = len(per_query) == len(per_query_map) == 90 and per_query_map.keys() == set(qids)
        for policy in policies:
            query_means = np.asarray([np.mean([event["total_ms"] for event in grouped[(qid, policy)]]) for qid in qids])
            values[policy] = query_means
            reported = summary["policies"][policy]
            policy_events = [event for event in events if event["policy"] == policy]
            measured = [float(query_means.mean()), *np.percentile(query_means, [50, 95]).tolist(),
                        *np.percentile([event["total_ms"] for event in policy_events], [50, 95]).tolist()]
            expected = [reported["mean_ms"], reported["p50_query_mean_ms"], reported["p95_query_mean_ms"], reported["p50_event_ms"], reported["p95_event_ms"]]
            self.check(f"latency_independent_aggregate_{policy}", self.close(measured, expected, 1e-9)
                       and reported["actual_cross_encoder_calls"] == sum(event["reranked"] for event in policy_events)
                       and reported["actual_cross_encoder_pairs"] == sum(event["scored_pairs"] for event in policy_events)
                       and reported["requests"] == len(policy_events) == 270
                       and all(len(grouped[(qid, policy)]) == 3 for qid in qids)
                       and all(self.close(np.mean([event["components_ms"][component] for event in policy_events]), value, 1e-9)
                               for component, value in reported["component_means_ms"].items()))
            for i, qid in enumerate(qids):
                saved = per_query_map[qid]
                saved_means_valid &= saved["source_domain"] == by_id[qid]["source_domain"]
                saved_means_valid &= self.close(saved["policies"][policy]["mean_total_ms"], query_means[i], 1e-9)
                saved_means_valid &= saved["policies"][policy]["reranked"] == grouped[(qid, policy)][0]["reranked"]
                saved_means_valid &= all(self.close(value, np.mean([event["components_ms"][component] for event in grouped[(qid, policy)]]), 1e-9)
                                         for component, value in saved["policies"][policy]["components_mean_ms"].items())
        self.check("latency_saved_per_query_repetition_means", saved_means_valid)
        rng = np.random.default_rng(cfg["bootstrap_seed"])
        sampled = np.concatenate([rng.choice([i for i, qid in enumerate(qids) if by_id[qid]["source_domain"] == source], size=(10000, 30)) for source in sorted(DOMAINS)], axis=1)
        reference = values["always"]
        sampled_reference = reference[sampled].mean(axis=1)
        for policy in policies:
            if policy == "always":
                continue
            delta = (values[policy] - reference)[sampled].mean(axis=1)
            reduction = 1 - values[policy][sampled].mean(axis=1) / sampled_reference
            reported = summary["paired_comparisons"][f"{policy}_vs_always"]
            self.check(f"latency_independent_paired_intervals_{policy}",
                       self.close(float((values[policy] - reference).mean()), reported["mean_delta_ms"], 1e-9)
                       and self.close(np.quantile(delta, [.025, .975]), reported["delta_ci95_ms"], 1e-9)
                       and self.close(float(1 - values[policy].mean() / reference.mean()), reported["mean_latency_reduction_fraction"])
                       and self.close(np.quantile(reduction, [.025, .975]), reported["reduction_ci95_fraction"]))
        self.check("latency_raw_events_hash", digest(directory / "events.jsonl") == summary["events_sha256"])
        for key, path in input_paths.items():
            self.check(f"latency_input_hash_{key}", digest(path) == summary["inputs_sha256"][key])
        configuration_keys = {"dense_model", "dense_revision", "reranker_model", "reranker_revision", "candidate_k", "rerank_k",
                              "rrf_k", "rrf_weights", "max_sequence_length", "threads", "batch_size"}
        self.check("latency_model_configuration_and_threads", summary["model_configuration"].keys() == configuration_keys
                   and all(value == protocol[key] for key, value in summary["model_configuration"].items())
                   and summary["environment"]["torch_threads"] == 4 and summary["environment"]["torch_interop_threads"] == 1
                   and summary["environment"]["device"] == "cpu" and summary["repetitions"] == 3)
        expected_shards = json.loads(confirmation_manifest.read_text())["embedding_cache"]["shard_sha256"]
        measured_shards = summary["embedding_provenance"]["shards"]
        self.check("latency_exact_corpus_embedding_shards", {r["file"]: r["sha256"] for r in measured_shards} == expected_shards
                   and sum(r["rows"] for r in measured_shards) == 37292
                   and summary["embedding_provenance"]["all_finite"] is True and summary["embedding_provenance"]["all_unit_norm_atol_1e-5"] is True)
        for source, expected in summary["source_sha256"].items():
            self.check(f"latency_source_{source}", digest(Path(source)) == expected)
        self.details["latency"] = {"queries": 90, "events": 1080, "policies": summary["policies"], "paired_comparisons": summary["paired_comparisons"]}

    def failure_analysis(self, path: Path, rows: list[dict], selected: dict, inputs: dict[str, Path]):
        analysis = json.loads(path.read_text())
        records = read_rows(path.with_name(path.stem + "_per_query.jsonl"))
        by_id = {row["query_id"]: row for row in rows}
        record_by_id = {row["query_id"]: row for row in records}
        self.check("failure_analysis_exact_confirmation_cohort", len(records) == len(record_by_id) == len(rows) == 1500
                   and record_by_id.keys() == by_id.keys() and "confirmation" in analysis["scope"])
        provenance = analysis["provenance"]
        for key, expected_path in inputs.items():
            self.check(f"failure_analysis_input_{key}", digest(expected_path) == provenance.get(key))
        self.check("failure_analysis_source_hashes", provenance["script_sha256"] == digest(Path("scripts/analyze_confirmation_failures.py"))
                   and provenance["reused_analysis_script_sha256"] == digest(Path("scripts/analyze_tool_retrieval_failures.py"))
                   and provenance["text_serialization_sha256"] == digest(Path("src/toolret_research/text.py")))
        self.check("failure_analysis_exploratory_scope", provenance["labels_altered"] is False and provenance["analysis_modifies_rankings_or_router"] is False)
        metric_valid, structure_valid = True, True
        for qid, source in by_id.items():
            record, ranks, gold = record_by_id[qid], source["rankings"], set(source["relevant_ids"])
            before, after = (metric_values(ranks[method], gold)["nDCG@10"] for method in ("hybrid", "reranked"))
            delta = after - before
            metric_valid &= self.close([before, after, delta], [record["hybrid_ndcg10"], record["reranked_ndcg10"], record["delta_ndcg10"]])
            metric_valid &= record["outcome"] == ("win" if delta > 1e-12 else "loss" if delta < -1e-12 else "unchanged")
            hits = len(gold & set(ranks["hybrid"][:20]))
            structure_valid &= record["source_domain"] == source["source_domain"] and record.get("component_id", source["component_id"]) == source["component_id"]
            structure_valid &= record["positive_count"] == len(gold) and record["prefix_positive_count"] == hits
            structure_valid &= self.close(record["candidate_label_recall20"], hits / len(gold)) and record["prefix_pairs"] == 20
            structure_valid &= {r["tool_id"] for r in record["gold_tools"]} == gold
            transitions = Counter()
            for entry in record["gold_tools"]:
                tool = entry["tool_id"]
                a = ranks["hybrid"].index(tool) + 1 if tool in ranks["hybrid"] else None
                b = ranks["reranked"].index(tool) + 1 if tool in ranks["reranked"] else None
                transition = ("absent_from_fused_candidates" if a is None and b is None else "left_top10" if a <= 10 < b else
                              "entered_top10" if b <= 10 < a else "promoted_within_top10" if b < a <= 10 else
                              "demoted_within_top10" if a < b <= 10 else "unchanged_top10" if a == b and a <= 10 else "outside_top10")
                transitions[transition] += 1
                structure_valid &= entry["hybrid_rank"] == a and entry["reranked_rank"] == b and entry["transition"] == transition
            structure_valid &= dict(transitions) == record["gold_transitions"]
            if "gate_selected" in record:
                structure_valid &= record["gate_selected"] == (qid in selected["fixed_disagreement"])
            for key in ("utility_75_selected", "primary_gate_selected"):
                if key in record:
                    structure_valid &= record[key] == (qid in selected["utility_75"])
            structure_valid &= record["frozen_rerank_decisions"] == {name: qid in selected[name] for name in ("fixed_disagreement", "cheap_jaccard", "utility_25", "utility_50", "utility_75")}
        self.check("failure_analysis_independent_binary_metrics", metric_valid)
        self.check("failure_analysis_independent_candidate_coverage_and_transitions", structure_valid)
        aggregate = analysis["summary"]
        summary_valid = aggregate["queries"] == 1500 and aggregate["outcomes"] == dict(Counter(r["outcome"] for r in records))
        for key, field in (("hybrid_ndcg10", "hybrid_ndcg10"), ("reranked_ndcg10", "reranked_ndcg10"), ("mean_delta_ndcg10", "delta_ndcg10"),
                           ("mean_candidate_label_recall20", "candidate_label_recall20")):
            summary_valid &= self.close(aggregate[key], np.mean([r[field] for r in records]))
        for key, expected in (("positive_labels", sum(r["positive_count"] for r in records)), ("positive_labels_in_prefix20", sum(r["prefix_positive_count"] for r in records)),
                              ("queries_without_positive_in_prefix20", sum(r["prefix_positive_count"] == 0 for r in records)),
                              ("queries_with_all_labels_in_prefix20", sum(r["prefix_positive_count"] == r["positive_count"] for r in records)),
                              ("positive_labels_left_top10", sum(r["gold_transitions"].get("left_top10", 0) for r in records)),
                              ("positive_labels_entered_top10", sum(r["gold_transitions"].get("entered_top10", 0) for r in records))):
            summary_valid &= aggregate[key] == expected
        self.check("failure_analysis_independent_summary", summary_valid)
        for policy, results in analysis["policy_failure_summary"].items():
            def check_policy_subset(subset, reported):
                chosen = [r for r in subset if r["query_id"] in selected[policy]]
                skipped = [r for r in subset if r["query_id"] not in selected[policy]]
                metric = np.mean([r["reranked_ndcg10"] if r["query_id"] in selected[policy] else r["hybrid_ndcg10"] for r in subset])
                return (reported["queries"] == len(subset) and reported["reranker_invocations"] == len(chosen)
                        and reported["bypassed_queries"] == len(skipped) and self.close(reported["ndcg10"], metric)
                        and reported["selected_outcomes_of_always_vs_hybrid"] == dict(Counter(r["outcome"] for r in chosen))
                        and reported["bypassed_outcomes_of_always_vs_hybrid"] == dict(Counter(r["outcome"] for r in skipped))
                        and self.close(reported["mean_delta_vs_hybrid_ndcg10"], sum(r["delta_ndcg10"] for r in chosen) / len(subset))
                        and self.close(reported["mean_delta_vs_always_ndcg10"], -sum(r["delta_ndcg10"] for r in skipped) / len(subset))
                        and self.close(reported["total_harm_avoided_by_bypass"], sum(max(-r["delta_ndcg10"], 0) for r in skipped))
                        and self.close(reported["total_gain_forgone_by_bypass"], sum(max(r["delta_ndcg10"], 0) for r in skipped)))
            valid = check_policy_subset(records, results)
            for source, reported in results["by_source_domain"].items():
                valid &= check_policy_subset([r for r in records if r["source_domain"] == source], reported)
            self.check(f"failure_analysis_independent_routed_policy_{policy}", valid)
        expected_cases = []
        for source in sorted(DOMAINS):
            source_records = [r for r in records if r["source_domain"] == source]
            expected_cases.extend(r["query_id"] for r in sorted((r for r in source_records if r["outcome"] == "loss"), key=lambda r: (r["delta_ndcg10"], r["query_id"]))[:5])
            expected_cases.extend(r["query_id"] for r in sorted((r for r in source_records if r["outcome"] == "win"), key=lambda r: (-r["delta_ndcg10"], r["query_id"]))[:5])
        self.check("failure_analysis_deterministic_case_selection", [r["query_id"] for r in analysis["cases"]] == expected_cases)
        self.details["failure_analysis"] = {"scope": analysis["scope"], "queries": len(records), "summary": aggregate,
                                            "audit_limitation": "Tokenization clipping statistics and qualitative causal interpretations are not independently recomputed by this audit."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/research_confirmation_20261003"))
    parser.add_argument("--corpus", type=Path, default=Path("data/research_20261003/corpus.jsonl"))
    parser.add_argument("--protocol", type=Path, default=Path("configs/confirmation_protocol_20261003.json"))
    parser.add_argument("--model", type=Path)
    parser.add_argument("--calibration-cache", type=Path)
    parser.add_argument("--calibration-manifest", type=Path)
    parser.add_argument("--confirmation-cache", type=Path)
    parser.add_argument("--confirmation-manifest", type=Path)
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--predictions", type=Path)
    parser.add_argument("--runtime", type=Path)
    parser.add_argument("--latency-dir", type=Path)
    parser.add_argument("--failure-summary", type=Path)
    parser.add_argument("--output", type=Path, default=Path("results/research_confirmation_20261003/independent_audit.json"))
    parser.add_argument("--data-only", action="store_true")
    parser.add_argument("--model-only", action="store_true")
    args = parser.parse_args()
    audit = Audit()
    prepared, corpus_ids = audit.data(args.data_dir, args.corpus, args.protocol)
    if not args.data_only:
        if not args.model or not args.model.exists():
            parser.error("Full audit requires the already-frozen router artifact; confirmation quality must not be opened before router freeze")
        if not all((args.calibration_cache, args.calibration_manifest)):
            parser.error("Model audit requires explicit calibration cache and manifest paths")
        train, _ = audit.cache(Path("results/research_20261003/pilot_cache/rankings.jsonl"),
                               Path("results/research_20261003/pilot_cache/manifest.json"), prepared["development"], corpus_ids, args.protocol)
        calibration, _ = audit.cache(args.calibration_cache, args.calibration_manifest, prepared["calibration"], corpus_ids, args.protocol)
        artifact = audit.router(args.model, train, calibration, prepared, args.protocol)
        if not args.model_only:
            if not all((args.confirmation_cache, args.confirmation_manifest, args.summary, args.predictions)):
                parser.error("Full audit requires explicit confirmation cache, manifest, summary and frozen predictions paths")
            confirmation, _ = audit.cache(args.confirmation_cache, args.confirmation_manifest, prepared["confirmation"], corpus_ids, args.protocol)
            evaluation = json.loads(args.summary.read_text())
            for key, path in (("router_sha256", args.model), ("protocol_sha256", args.protocol), ("predictions_sha256", args.predictions),
                              ("cache_sha256", args.confirmation_cache), ("queries_sha256", args.data_dir / "confirmation_queries.jsonl"),
                              ("corpus_sha256", args.corpus), ("data_manifest_sha256", args.data_dir / "manifest.json"), ("cache_manifest_sha256", args.confirmation_manifest)):
                audit.check(f"evaluation_input_{key}", digest(path) == evaluation["input_integrity"][key])
            for path, expected in evaluation["input_integrity"]["evaluation_source_sha256"].items():
                audit.check(f"evaluation_source_{path}", digest(Path(path)) == expected)
            audit.check("evaluation_router_and_frozen_provenance", evaluation["router"] == artifact
                        and json.loads((args.summary.parent / "frozen_prediction_provenance.json").read_text()) == evaluation["input_integrity"]
                        and evaluation["input_integrity"]["created_before_quality_scoring"] is True)
            selected = audit.evaluation(confirmation, prepared["confirmation"], artifact, args.summary, args.predictions)
            if args.runtime:
                audit.runtime(args.runtime, selected, {"router_sha256": args.model, "protocol_sha256": args.protocol,
                              "predictions_sha256": args.predictions, "cache_sha256": args.confirmation_cache,
                              "queries_sha256": args.data_dir / "confirmation_queries.jsonl", "corpus_sha256": args.corpus,
                              "evaluation_summary_sha256": args.summary})
            else:
                audit.pending("actual_full_cohort_runtime")
            if args.latency_dir:
                audit.latency(args.latency_dir, confirmation, selected, args.protocol,
                              {"cache": args.confirmation_cache, "manifest": args.confirmation_manifest, "corpus": args.corpus,
                               "queries": args.data_dir / "confirmation_queries.jsonl", "protocol": args.protocol, "router_model": args.model}, args.confirmation_manifest)
            else:
                audit.pending("actual_repeated_latency")
            if args.failure_summary:
                audit.failure_analysis(args.failure_summary, confirmation, selected,
                                       {"cache_sha256": args.confirmation_cache, "cache_manifest_sha256": args.confirmation_manifest,
                                        "queries_sha256": args.data_dir / "confirmation_queries.jsonl", "corpus_sha256": args.corpus,
                                        "predictions_sha256": args.predictions, "router_sha256": args.model,
                                        "prediction_provenance_sha256": args.summary.parent / "frozen_prediction_provenance.json", "evaluation_summary_sha256": args.summary,
                                        "protocol_sha256": args.protocol, "data_manifest_sha256": args.data_dir / "manifest.json"})
            else:
                audit.pending("confirmation_failure_analysis")
        else:
            audit.pending("confirmation_metrics_random_controls_and_intervals")
            audit.pending("actual_runtime_verification_and_repeated_latency")
    else:
        audit.pending("router_fit_and_thresholds")
        audit.pending("confirmation_metrics_random_controls_and_intervals")
        audit.pending("actual_runtime_verification_and_repeated_latency")
    result = {"schema_version": "toolret-independent-confirmation-audit-v1", "created_at_utc": datetime.now(timezone.utc).isoformat(),
              "audit_scope": "raw-data/protocol only" if args.data_only else "data/protocol/model" if args.model_only else "data/protocol/model/evaluation/runtime/latency as supplied",
              "independent_implementation": True, "imports_study_metric_router_or_evaluator_code": False,
              "checks": audit.checks, "counts": dict(Counter(check["status"] for check in audit.checks)),
              "all_executed_checks_pass": all(check["status"] != "fail" for check in audit.checks), "details": audit.details,
              "all_required_stages_complete": not any(check["status"] == "pending" for check in audit.checks),
              "input_evidence_sha256": evidence_fingerprints(args),
              "auditor_source_sha256": digest(Path(__file__))}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"output": str(args.output), "counts": result["counts"], "all_executed_checks_pass": result["all_executed_checks_pass"]}))
    if not result["all_executed_checks_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
