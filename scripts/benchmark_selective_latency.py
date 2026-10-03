"""Measure repeated, actual warm-service retrieval latency on confirmation queries.

Corpus/model loading and validation occur before measurement. Every measured
request recomputes BM25, a dense query embedding, exact full-corpus dense search,
RRF, and its policy decision, then actually invokes the cross-encoder if routed.
Cached rankings are used solely for validation after the timed request returns.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import random
import subprocess
import time
from typing import Any

from toolret_research.bm25 import BM25Index
from toolret_research.data import load_corpus, load_queries, read_jsonl
from toolret_research.fusion import weighted_reciprocal_rank_fusion
from toolret_research.text import flatten_tool

SELECTION_SEED = "latency-confirmation-20261003"
RETRIEVAL_KEYS = (
    "dense_model", "dense_revision", "reranker_model", "reranker_revision",
    "candidate_k", "rerank_k", "rrf_k", "rrf_weights", "max_sequence_length",
    "threads", "batch_size",
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def select_latency_rows(rows: list[dict], per_domain: int,
                        seed: str = SELECTION_SEED) -> list[dict]:
    """Select without consulting relevance labels, quality, or recorded timings."""
    if isinstance(per_domain, bool) or not isinstance(per_domain, int) or per_domain < 1:
        raise ValueError("per_domain must be a positive integer")
    domains: dict[str, list[dict]] = defaultdict(list)
    seen = set()
    for row in rows:
        qid, domain = row.get("query_id"), row.get("source_domain")
        if not isinstance(qid, str) or not qid or qid in seen:
            raise ValueError("Latency cache query IDs must be unique nonempty strings")
        if not isinstance(domain, str) or not domain:
            raise ValueError("Every latency query needs a source_domain")
        seen.add(qid)
        domains[domain].append(row)
    if not domains:
        raise ValueError("Latency cohort cannot be empty")
    output = []
    for domain in sorted(domains):
        if len(domains[domain]) < per_domain:
            raise ValueError(f"Insufficient queries in {domain} for balanced sample")
        ordered = sorted(domains[domain], key=lambda row: (
            hashlib.sha256((seed + "\0" + row["query_id"]).encode()).hexdigest(),
            row["query_id"],
        ))
        output.extend(ordered[:per_domain])
    return output


def counterbalanced_schedule(query_ids: list[str], policies: list[str],
                             repetitions: int, seed: int) -> list[dict]:
    """Randomize query order and cyclically balance policy positions per repeat."""
    if repetitions < 2 or not query_ids or not policies:
        raise ValueError("At least two repetitions, one query and one policy are required")
    if len(set(query_ids)) != len(query_ids) or len(set(policies)) != len(policies):
        raise ValueError("Queries and policies must be unique")
    rng = random.Random(seed)
    schedule = []
    for repetition in range(repetitions):
        shuffled_queries, base = list(query_ids), list(policies)
        rng.shuffle(shuffled_queries)
        rng.shuffle(base)
        for position, qid in enumerate(shuffled_queries):
            offset = position % len(base)
            order = base[offset:] + base[:offset]
            for policy_position, policy in enumerate(order):
                schedule.append({"query_id": qid, "repetition": repetition,
                                 "policy": policy, "policy_position": policy_position})
    return schedule


def validate_inputs(cache_path: Path, manifest: dict, corpus_path: Path,
                    queries_path: Path, protocol: dict, rows: list[dict],
                    corpus: dict, query_rows: list[dict]) -> None:
    for key, path in (("cache_sha256", cache_path), ("corpus_sha256", corpus_path),
                      ("queries_sha256", queries_path)):
        if manifest.get(key) != sha256_file(path):
            raise ValueError(f"{key} mismatch")
    declared = manifest.get("cache_provenance", {}).get("protocol", {})
    for key in RETRIEVAL_KEYS:
        if declared.get(key) != protocol.get(key) or key not in protocol:
            raise ValueError(f"Retrieval protocol mismatch: {key}")
    for label, name_key, revision_key in (("dense", "dense_model", "dense_revision"),
                                         ("reranker", "reranker_model", "reranker_revision")):
        model = manifest.get("models", {}).get(label, {})
        if model.get("name") != protocol[name_key] or model.get("revision") != protocol[revision_key]:
            raise ValueError(f"Manifest model identity mismatch: {label}")
    source_hashes = manifest.get("cache_provenance", {}).get("source_sha256", {})
    for path in ("src/toolret_research/text.py", "src/toolret_research/bm25.py", "src/toolret_research/fusion.py"):
        if source_hashes.get(path) != sha256_file(path):
            raise ValueError(f"Ranking source hash mismatch: {path}")
    queries = {str(row["id"]): row for row in query_rows}
    if len(queries) != len(query_rows) or set(queries) != {row["query_id"] for row in rows}:
        raise ValueError("Exact cache/query cohort mismatch")
    if len(rows) != len({row["query_id"] for row in rows}):
        raise ValueError("Duplicate cached query IDs")
    known = set(corpus)
    for row in rows:
        qid = row["query_id"]
        source = queries[qid]
        if not isinstance(source.get("query"), str) or not source["query"].strip():
            raise ValueError(f"Empty source query: {qid}")
        domain = source.get("source_domain", source.get("domain"))
        if domain != row["source_domain"]:
            raise ValueError(f"Source-domain mismatch: {qid}")
        gold = set(str(doc_id) for doc_id in source["relevant_ids"]) if "relevant_ids" in source else {
            str(label["id"]) for label in source.get("labels", []) if float(label.get("relevance", 0)) > 0
        }
        if not gold or not gold <= known or len(row.get("relevant_ids", [])) != len(gold) or set(row["relevant_ids"]) != gold:
            raise ValueError(f"Positive-label alignment mismatch: {qid}")
        ranks = row.get("rankings", {})
        for method in ("bm25", "dense", "hybrid", "reranked"):
            values = ranks.get(method)
            if not isinstance(values, list) or not values or len(set(values)) != len(values) or not set(values) <= known:
                raise ValueError(f"Malformed cached ranking: {qid}/{method}")
        expected = weighted_reciprocal_rank_fusion([ranks["bm25"], ranks["dense"]],
                                                   protocol["rrf_weights"], protocol["rrf_k"])
        if ranks["hybrid"] != expected:
            raise ValueError(f"Cached RRF mismatch: {qid}")
        depth = min(protocol["rerank_k"], len(expected))
        if row.get("rerank_k") != depth or len(ranks["reranked"]) != len(expected) or set(ranks["reranked"][:depth]) != set(expected[:depth]) or ranks["reranked"][depth:] != expected[depth:]:
            raise ValueError(f"Cached reranked-prefix mismatch: {qid}")


def validate_benchmark_configuration(protocol: dict, selected: list[dict], *,
                                     queries_path: Path, per_domain: int,
                                     repetitions: int, policies: list[str],
                                     order_seed: int, bootstrap_resamples: int) -> None:
    frozen = protocol.get("latency_benchmark")
    if not isinstance(frozen, dict) or protocol.get("scope") != "confirmation_study":
        raise ValueError("A frozen confirmation latency protocol is required")
    expected = {
        "queries": len(selected), "queries_per_source": per_domain,
        "selection_seed": SELECTION_SEED, "repetitions": repetitions,
        "policies": policies, "order_seed": order_seed,
        "bootstrap_seed": 20261003, "bootstrap_resamples": bootstrap_resamples,
    }
    for key, value in expected.items():
        if frozen.get(key) != value:
            raise ValueError(f"Frozen latency configuration mismatch: {key}")
    counts = Counter(row["source_domain"] for row in selected)
    if set(counts) != set(protocol["query_sources"]) or any(count != per_domain for count in counts.values()):
        raise ValueError("Latency selected source domains differ from the frozen study")
    if sha256_file(queries_path) != protocol.get("confirmation", {}).get("queries_sha256"):
        raise ValueError("Latency requests must use the frozen confirmation cohort")


def load_embeddings(directory: Path, corpus_path: Path, document_ids: list[str],
                    protocol: dict, dimension: int):
    """Validate the frozen embedding cache including every norm and shard hash."""
    import numpy as np
    identity = json.loads((directory / "identity.json").read_text())
    expected = {
        "corpus_sha256": sha256_file(corpus_path),
        "document_ids_sha256": hashlib.sha256(json.dumps(document_ids).encode()).hexdigest(),
        "model": protocol["dense_model"], "revision": protocol["dense_revision"],
        "max_sequence_length": protocol["max_sequence_length"],
        "serialization": "flatten_tool(id,text) at script/source SHA256",
        "text_source_sha256": sha256_file(Path("src/toolret_research/text.py")),
        "shard_size": 1024, "dtype": "float32", "normalized": True,
    }
    if identity != expected:
        raise ValueError("Embedding identity mismatch")
    shards, provenance = [], []
    for offset in range(0, len(document_ids), identity["shard_size"]):
        path = directory / f"shard_{offset:06d}.npy"
        values = np.load(path, allow_pickle=False)
        size = min(identity["shard_size"], len(document_ids) - offset)
        if values.shape != (size, dimension) or values.dtype != np.float32 or not np.isfinite(values).all():
            raise ValueError(f"Invalid embedding shard: {path.name}")
        norms = np.linalg.norm(values, axis=1)
        if not np.allclose(norms, 1.0, rtol=0, atol=1e-5):
            raise ValueError(f"Non-unit document embeddings: {path.name}")
        provenance.append({"file": path.name, "sha256": sha256_file(path),
                           "rows": size, "norm_min": float(norms.min()),
                           "norm_max": float(norms.max())})
        shards.append(values)
    return np.concatenate(shards), {"identity": identity, "shards": provenance,
                                   "all_finite": True, "all_unit_norm_atol_1e-5": True}


def summarize_events(events: list[dict], policies: list[str], repetitions: int,
                     *, bootstrap_resamples: int = 10000,
                     bootstrap_seed: int = 20261003) -> dict:
    """Aggregate repetitions within query before paired, stratified bootstraps."""
    import numpy as np
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for event in events:
        if not math.isfinite(event["total_ms"]) or event["total_ms"] <= 0:
            raise ValueError("Invalid measured wall latency")
        components = event["components_ms"]
        if any(not math.isfinite(value) or value < 0 for value in components.values()) or not math.isclose(sum(components.values()), event["total_ms"], rel_tol=1e-9, abs_tol=1e-8):
            raise ValueError("Latency components must be finite, disjoint and sum to wall time")
        if not isinstance(event["reranked"], bool) or isinstance(event["scored_pairs"], bool) or not isinstance(event["scored_pairs"], int) or event["scored_pairs"] < 0 or event["reranked"] != (event["scored_pairs"] > 0):
            raise ValueError("Inconsistent actual cross-encoder invocation accounting")
        grouped[(event["query_id"], event["policy"])].append(event)
    qids = sorted({event["query_id"] for event in events})
    if not qids or bootstrap_resamples < 1:
        raise ValueError("Events and positive bootstrap resamples are required")
    domains = {}
    per_query = []
    values: dict[str, list[float]] = {policy: [] for policy in policies}
    summaries = {}
    for qid in qids:
        record = {"query_id": qid, "policies": {}}
        for policy in policies:
            subset = grouped[(qid, policy)]
            if len(subset) != repetitions or {row["repetition"] for row in subset} != set(range(repetitions)):
                raise ValueError("Incomplete or duplicated query-policy repetitions")
            domain_values = {row["source_domain"] for row in subset}
            if len(domain_values) != 1:
                raise ValueError("A query cannot have conflicting source domains")
            domain = next(iter(domain_values))
            if qid in domains and domains[qid] != domain:
                raise ValueError("Source domains disagree across policies")
            domains[qid] = domain
            route_values = {row["reranked"] for row in subset}
            if len(route_values) != 1:
                raise ValueError("Routing changed across repetitions for the same query")
            average = float(np.mean([row["total_ms"] for row in subset]))
            values[policy].append(average)
            record["policies"][policy] = {"mean_total_ms": average,
                "reranked": next(iter(route_values)),
                "components_mean_ms": {component: float(np.mean([row["components_ms"][component] for row in subset]))
                                       for component in ("bm25", "dense", "rrf", "routing", "cross_encoder")}}
        record["source_domain"] = domains[qid]
        per_query.append(record)
    for policy in policies:
        selected_events = [event for event in events if event["policy"] == policy]
        array = np.asarray(values[policy], dtype=np.float64)
        event_array = np.asarray([event["total_ms"] for event in selected_events])
        summaries[policy] = {
            "mean_ms": float(array.mean()), "p50_query_mean_ms": float(np.percentile(array, 50)),
            "p95_query_mean_ms": float(np.percentile(array, 95)),
            "p50_event_ms": float(np.percentile(event_array, 50)),
            "p95_event_ms": float(np.percentile(event_array, 95)),
            "actual_cross_encoder_calls": sum(event["reranked"] for event in selected_events),
            "actual_cross_encoder_pairs": sum(event["scored_pairs"] for event in selected_events),
            "requests": len(selected_events),
            "component_means_ms": {component: float(np.mean([event["components_ms"][component] for event in selected_events]))
                                   for component in ("bm25", "dense", "rrf", "routing", "cross_encoder")},
        }
    rng = np.random.default_rng(bootstrap_seed)
    groups: dict[str, list[int]] = defaultdict(list)
    for index, qid in enumerate(qids):
        groups[domains[qid]].append(index)
    sample_indices = np.concatenate([rng.choice(indices, size=(bootstrap_resamples, len(indices)), replace=True)
                                     for _, indices in sorted(groups.items())], axis=1)
    comparisons = {}
    if "always" in policies:
        reference = np.asarray(values["always"])
        reference_means = reference[sample_indices].mean(axis=1)
        for policy in policies:
            if policy == "always":
                continue
            candidate = np.asarray(values[policy])
            deltas = (candidate - reference)[sample_indices].mean(axis=1)
            relative = 1.0 - candidate[sample_indices].mean(axis=1) / reference_means
            comparisons[f"{policy}_vs_always"] = {
                "mean_delta_ms": float((candidate - reference).mean()),
                "delta_ci95_ms": [float(value) for value in np.percentile(deltas, [2.5, 97.5])],
                "mean_latency_reduction_fraction": float(1.0 - candidate.mean() / reference.mean()),
                "reduction_ci95_fraction": [float(value) for value in np.percentile(relative, [2.5, 97.5])],
            }
    return {"queries": len(qids), "source_domain_counts": dict(Counter(domains.values())),
            "policies": summaries, "paired_comparisons": comparisons, "per_query": per_query,
            "bootstrap": {"resamples": bootstrap_resamples, "seed": bootstrap_seed,
                          "method": "paired percentile bootstrap, query means as clusters, stratified by source domain",
                          "independent_units": len(qids), "repetitions_per_unit": repetitions}}


class UtilityAdapter:
    """The utility-router module owns frozen features, scores and thresholds."""
    def __init__(self, path: Path):
        # This interface is deliberately explicit: no labels enter measured routing.
        from toolret_research.utility_router import RidgeUtilityModel, features, predict, shouldroute
        self.artifact = json.loads(path.read_text())
        self.model = RidgeUtilityModel.from_dict(self.artifact["model"])
        self.features, self.predict, self.shouldroute = features, predict, shouldroute

    def validate(self, protocol_path: Path, corpus_path: Path, protocol: dict,
                 budgets: list[float]) -> None:
        artifact = self.artifact
        if artifact.get("schema_version") != "toolret-frozen-utility-router-v1":
            raise ValueError("Router artifact schema mismatch")
        integrity = artifact.get("input_integrity", {})
        if integrity.get("protocol_sha256") != sha256_file(protocol_path) or integrity.get("corpus_sha256") != sha256_file(corpus_path):
            raise ValueError("Router protocol/corpus fingerprint mismatch")
        root = Path(__file__).resolve().parents[1]
        utility_relative = "src/toolret_research/utility_router.py"
        if artifact.get("source_sha256", {}).get(utility_relative) != sha256_file(root / utility_relative):
            raise ValueError("Router utility source fingerprint mismatch")
        for relative, expected in artifact.get("source_sha256", {}).items():
            source = (root / relative).resolve()
            if not source.is_relative_to(root) or sha256_file(source) != expected:
                raise ValueError("Router source fingerprint mismatch")
        relative_protocol = protocol_path.resolve().relative_to(root).as_posix()
        frozen_commit = integrity.get("protocol_frozen_commit")
        if not isinstance(frozen_commit, str) or len(frozen_commit) != 40 or any(char not in "0123456789abcdef" for char in frozen_commit):
            raise ValueError("Router needs a full frozen protocol commit SHA")
        try:
            frozen_content = subprocess.check_output(["git", "show", f"{frozen_commit}:{relative_protocol}"], cwd=root, stderr=subprocess.DEVNULL)
        except subprocess.CalledProcessError as exc:
            raise ValueError("Router frozen protocol commit is unavailable") from exc
        if hashlib.sha256(frozen_content).hexdigest() != sha256_file(protocol_path):
            raise ValueError("Router protocol differs from its frozen Git commit")
        config = protocol["utility_router"]
        if self.model.alpha != config["alpha"] or self.model.training_rows != config["training_query_count"]:
            raise ValueError("Router model differs from the frozen training design")
        if integrity.get("training", {}).get("queries") != config["training_query_count"] or integrity.get("calibration", {}).get("queries") != config["calibration_query_count"]:
            raise ValueError("Router training/calibration provenance count mismatch")
        if artifact.get("confirmation_inputs_accessed") is not False or artifact.get("calibration_labels_used_for_fit_or_thresholds") is not False:
            raise ValueError("Router provenance must declare no confirmation or calibration-label fitting")
        for budget in budgets:
            threshold = artifact.get("thresholds", {}).get(format(budget, "g"), {})
            value = threshold.get("threshold")
            if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) or threshold.get("nominal_budget_fraction") != budget or threshold.get("decision_rule") != "predicted_ndcg_delta > threshold":
                raise ValueError("Router threshold differs from the frozen pointwise policy")

    def decision(self, query: str, sparse: list[str], dense: list[str],
                 hybrid: list[str], budget: float) -> bool:
        score = self.predict(self.model, self.features(query, {"bm25": sparse, "dense": dense}))
        threshold = self.artifact["thresholds"][format(budget, "g")]["threshold"]
        return self.shouldroute(score, threshold)


def render_report(summary: dict) -> str:
    lines = ["# Actual repeated warm-service latency benchmark", "",
             f"{summary['queries']} hash-selected source-balanced confirmation queries; "
             f"{summary['repetitions']} repetitions; {summary['corpus_documents']:,} indexed tools; CPU {summary['environment']['torch_threads']} threads.", "",
             "Every request performs BM25, dense query encoding, exact full-corpus dense search/sort, RRF, routing and conditional cross-encoder scoring. "
             "Models, corpus embeddings and indexes remain loaded. Initialization, warmup, validation, network, queueing and concurrency are excluded.", "",
             "| Policy | Mean (ms) | p50 query mean (ms) | p95 query mean (ms) | Actual CE calls / requests |", "|---|---:|---:|---:|---:|"]
    for name, row in summary["policies"].items():
        lines.append(f"| {name} | {row['mean_ms']:.2f} | {row['p50_query_mean_ms']:.2f} | {row['p95_query_mean_ms']:.2f} | {row['actual_cross_encoder_calls']} / {row['requests']} |")
    lines.extend(["", "Percentiles above describe query means over repeated measurements. Event-level percentiles are separately available in summary.json. "
                  "Repeated observations are clustered by query; they are not treated as independent samples.", "",
                  "| Paired comparison | Mean difference (ms) | 95% CI (ms) | Mean latency reduction | 95% CI |", "|---|---:|---:|---:|---:|"])
    for name, row in summary["paired_comparisons"].items():
        low, high = row["delta_ci95_ms"]
        rel_low, rel_high = row["reduction_ci95_fraction"]
        lines.append(f"| {name} | {row['mean_delta_ms']:+.2f} | [{low:+.2f}, {high:+.2f}] | {row['mean_latency_reduction_fraction']:.1%} | [{rel_low:.1%}, {rel_high:.1%}] |")
    lines.extend(["", f"CI procedure: {summary['bootstrap']['resamples']:,} paired percentile bootstrap resamples, stratified by source domain, seed {summary['bootstrap']['seed']}.", "",
                  "All actual first-stage and final rankings matched the frozen confirmation cache. This benchmark establishes observed warm CPU retrieval behavior on this machine; it does not establish deployed, GPU, concurrent, or downstream task performance.", "",
                  "Selection, complete randomized request order, every measured event, per-query means, source/model/data hashes, package versions and embedding-shard fingerprints accompany this report.", ""])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("cache", "manifest", "corpus", "queries", "protocol", "embedding-dir", "output-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--router-model", type=Path)
    parser.add_argument("--budgets", type=float, nargs="+", default=[0.75])
    parser.add_argument("--per-domain", type=int, default=30)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--order-seed", type=int, default=20261003)
    parser.add_argument("--bootstrap-resamples", type=int, default=10000)
    args = parser.parse_args()
    if args.repetitions < 2 or any(not math.isfinite(budget) or not 0 < budget < 1 for budget in args.budgets) or len(set(args.budgets)) != len(args.budgets):
        raise ValueError("Require repeated measurements and unique budgets strictly between 0 and 1")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError("Refusing to overwrite an existing latency output directory")
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    import numpy as np
    import torch
    from sentence_transformers import CrossEncoder, SentenceTransformer

    protocol = json.loads(args.protocol.read_text())
    manifest = json.loads(args.manifest.read_text())
    rows = read_jsonl(args.cache)
    corpus = load_corpus(args.corpus)
    query_rows = read_jsonl(args.queries)
    validate_inputs(args.cache, manifest, args.corpus, args.queries, protocol,
                    rows, corpus, query_rows)
    selected = select_latency_rows(rows, args.per_domain)
    query_texts = {query.id: query.text for query in load_queries(args.queries)}
    policies = ["hybrid", "always", "fixed_disagreement"]
    utility = UtilityAdapter(args.router_model) if args.router_model else None
    budgets = {f"utility_{budget * 100:g}": budget for budget in args.budgets} if utility else {}
    policies.extend(budgets)
    validate_benchmark_configuration(protocol, selected, queries_path=args.queries,
                                     per_domain=args.per_domain, repetitions=args.repetitions,
                                     policies=policies, order_seed=args.order_seed,
                                     bootstrap_resamples=args.bootstrap_resamples)
    if utility:
        utility.validate(args.protocol, args.corpus, protocol, args.budgets)
    if len(corpus) != protocol["corpus_documents"] or len(rows) != protocol["confirmation"]["queries"]:
        raise ValueError("Latency input corpus/cohort count differs from the frozen study")
    schedule = counterbalanced_schedule([row["query_id"] for row in selected],
                                        policies, args.repetitions, args.order_seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output_dir / "selection.json", {
        "selection_seed": SELECTION_SEED, "selection": "SHA256(seed NUL query_id), source-balanced; no metric or timing inspection",
        "per_domain": args.per_domain, "query_ids": [row["query_id"] for row in selected],
        "source_domain_counts": dict(Counter(row["source_domain"] for row in selected)),
        "cache_sha256": manifest["cache_sha256"], "schedule": schedule,
    })
    torch.set_num_threads(protocol["threads"])
    torch.set_num_interop_threads(1)
    document_ids = list(corpus)
    documents = {doc_id: flatten_tool({"id": doc_id, "text": corpus[doc_id]["text"]}) for doc_id in document_ids}
    print(f"Loading pinned models and validating embeddings; {len(selected)} queries, {len(policies)} policies, {len(schedule)} actual requests", flush=True)
    dense_model = SentenceTransformer(protocol["dense_model"], revision=protocol["dense_revision"], device="cpu")
    dense_model.max_seq_length = protocol["max_sequence_length"]
    embeddings, embedding_provenance = load_embeddings(args.embedding_dir, args.corpus,
                                                       document_ids, protocol,
                                                       dense_model.get_sentence_embedding_dimension())
    document_embeddings = torch.from_numpy(embeddings).contiguous()
    document_id_array = np.asarray(document_ids)
    sparse_index = BM25Index.build(documents)
    cross_encoder = CrossEncoder(protocol["reranker_model"], revision=protocol["reranker_revision"],
                                 device="cpu", max_length=protocol["max_sequence_length"])

    def request(query: str, policy: str):
        total_start = time.perf_counter_ns()
        boundary = total_start
        sparse = [doc_id for doc_id, _ in sparse_index.search(query, k=protocol["candidate_k"])]
        now = time.perf_counter_ns()
        sparse_ms, boundary = (now - boundary) / 1e6, now
        query_embedding = dense_model.encode([query], normalize_embeddings=True, convert_to_tensor=True, show_progress_bar=False)
        similarities = torch.matmul(query_embedding, document_embeddings.T)[0].cpu().numpy()
        indices = np.lexsort((document_id_array, -similarities))[:protocol["candidate_k"]]
        dense = [document_ids[int(index)] for index in indices]
        now = time.perf_counter_ns()
        dense_ms, boundary = (now - boundary) / 1e6, now
        hybrid = weighted_reciprocal_rank_fusion([sparse, dense], protocol["rrf_weights"], protocol["rrf_k"])
        now = time.perf_counter_ns()
        fusion_ms, boundary = (now - boundary) / 1e6, now
        if policy == "always":
            rerank = True
        elif policy == "fixed_disagreement":
            rerank = sparse[0] != dense[0]
        elif policy == "hybrid":
            rerank = False
        else:
            rerank = utility.decision(query, sparse, dense, hybrid, budgets[policy])
        depth = min(protocol["rerank_k"], len(hybrid))
        now = time.perf_counter_ns()
        routing_ms, boundary = (now - boundary) / 1e6, now
        if rerank:
            scores = cross_encoder.predict([[query, documents[doc_id]] for doc_id in hybrid[:depth]],
                                           batch_size=protocol["batch_size"], show_progress_bar=False)
            if not np.isfinite(scores).all() or len(scores) != depth:
                raise ValueError("Invalid measured cross-encoder output")
            prefix = sorted(zip(hybrid[:depth], scores), key=lambda pair: (-float(pair[1]), pair[0]))
            final = [doc_id for doc_id, _ in prefix] + hybrid[depth:]
        else:
            final = hybrid
        finish = time.perf_counter_ns()
        tail_ms = (finish - boundary) / 1e6
        # Skipped requests do no CE work. Attribute their final branch and
        # timer overhead to routing while retaining the full measured time.
        if not rerank:
            routing_ms += tail_ms
        return final, sparse, dense, hybrid, bool(rerank), depth if rerank else 0, {
            "bm25": sparse_ms, "dense": dense_ms, "rrf": fusion_ms,
            "routing": routing_ms, "cross_encoder": tail_ms if rerank else 0.,
        }, (finish - total_start) / 1e6

    # Warm all code paths with a non-benchmark query; every warmup is excluded.
    with torch.inference_mode():
        for policy in policies:
            request("weather forecast", policy)
    selected_map = {row["query_id"]: row for row in selected}
    events = []
    event_path = args.output_dir / "events.jsonl"
    with event_path.with_suffix(".jsonl.tmp").open("w") as stream, torch.inference_mode():
        for number, scheduled in enumerate(schedule, 1):
            row = selected_map[scheduled["query_id"]]
            final, sparse, dense, hybrid, routed, pairs, components, total_ms = request(query_texts[row["query_id"]], scheduled["policy"])
            ranks = row["rankings"]
            if sparse != ranks["bm25"] or dense != ranks["dense"] or hybrid != ranks["hybrid"]:
                raise ValueError(f"Actual first-stage ranking differs from cache: {row['query_id']}")
            expected = ranks["reranked"] if routed else ranks["hybrid"]
            if final != expected:
                raise ValueError(f"Actual final ranking differs from cache: {row['query_id']}")
            event = {**scheduled, "source_domain": row["source_domain"],
                     "total_ms": total_ms, "components_ms": components,
                     "reranked": routed, "scored_pairs": pairs, "matches_cache": True,
                     "ranking_sha256": hashlib.sha256(json.dumps(final).encode()).hexdigest()}
            events.append(event)
            stream.write(json.dumps(event) + "\n")
            stream.flush()
            if number % 30 == 0 or number == len(schedule):
                print(f"Measured {number}/{len(schedule)} requests; last={scheduled['policy']} {total_ms:.1f}ms", flush=True)
    event_path.with_suffix(".jsonl.tmp").replace(event_path)
    summary = summarize_events(events, policies, args.repetitions,
                               bootstrap_resamples=args.bootstrap_resamples)
    per_query = summary.pop("per_query")
    with (args.output_dir / "per_query.jsonl").open("w") as stream:
        for row in per_query:
            stream.write(json.dumps(row) + "\n")
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except subprocess.CalledProcessError:
        commit = "unavailable"
    source_paths = [Path(__file__), Path("src/toolret_research/bm25.py"), Path("src/toolret_research/text.py"), Path("src/toolret_research/fusion.py")]
    if utility:
        source_paths.append(Path("src/toolret_research/utility_router.py"))
    cpu_model = "unavailable"
    cpu_info = Path("/proc/cpuinfo")
    if cpu_info.exists():
        cpu_model = next((line.split(":", 1)[1].strip() for line in cpu_info.read_text().splitlines()
                          if line.startswith("model name")), cpu_model)
    summary.update({
        "scope": "actual repeated warm-service sequential CPU end-to-end retrieval benchmark",
        "measurement_boundary": "query text to final ranking: BM25 + dense encode + exact full-corpus search/sort + RRF + routing + conditional CE/prefix sort",
        "excluded": ["model/corpus/index loading", "corpus embedding", "input and output validation", "model warmup", "network", "service queueing", "concurrency"],
        "repetitions": args.repetitions, "order_seed": args.order_seed,
        "policy_order": "query order shuffled per repetition; random base policy order with cyclic position balancing",
        "warmup": "one full request per policy on non-benchmark query 'weather forecast'",
        "all_actual_rankings_match_cache": True, "corpus_documents": len(corpus),
        "model_configuration": {key: protocol[key] for key in RETRIEVAL_KEYS},
        "environment": {"platform": platform.platform(), "processor": platform.processor(), "cpu_model": cpu_model,
                        "logical_cpus": os.cpu_count(), "torch_threads": torch.get_num_threads(),
                        "torch_interop_threads": torch.get_num_interop_threads(), "device": "cpu",
                        "python": platform.python_version()},
        "packages": {name: importlib.metadata.version(name) for name in ("torch", "sentence-transformers", "transformers", "numpy")},
        "source_sha256": {str(path): sha256_file(path) for path in source_paths}, "base_commit": commit,
        "inputs_sha256": {"cache": sha256_file(args.cache), "manifest": sha256_file(args.manifest),
                          "corpus": sha256_file(args.corpus), "queries": sha256_file(args.queries),
                          "protocol": sha256_file(args.protocol),
                          "router_model": sha256_file(args.router_model) if utility else None},
        "embedding_provenance": embedding_provenance,
        "events_sha256": sha256_file(event_path),
        "limitations": ["Single machine and CPU thread setting; results do not establish GPU/server behavior.",
                        "90-query default source-balanced latency subsample; quality results belong to the larger confirmation cohort.",
                        "Warm sequential requests exclude network, queueing and concurrent service load."],
    })
    atomic_json(args.output_dir / "summary.json", summary)
    (args.output_dir / "report.md").write_text(render_report(summary))
    print(json.dumps({"saved": str(args.output_dir), "queries": summary["queries"],
                      "policies": {name: {"mean_ms": value["mean_ms"], "calls": value["actual_cross_encoder_calls"]} for name, value in summary["policies"].items()}}, indent=2), flush=True)


if __name__ == "__main__":
    main()
