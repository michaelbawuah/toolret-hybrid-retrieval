"""Fit the frozen expected-benefit router and label-free calibration thresholds.

There is deliberately no confirmation/test input argument. The old observed
pilot supplies training labels; calibration supplies feature predictions only.
The requested protocol must already exist in an explicitly named Git commit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import unicodedata
from typing import Any

import numpy as np

from toolret_research.data import load_corpus, read_jsonl
from toolret_research.fusion import weighted_reciprocal_rank_fusion
from toolret_research.metrics import ndcg_at_k
from toolret_research.selective import file_sha256
from toolret_research.utility_router import (
    FEATURE_NAMES, FEATURE_VERSION, calibratebudgetthreshold, features,
    fitmodel, predict, shouldroute,
)


def _normalized_query(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def _gold(row: dict[str, Any]) -> set[str]:
    if "relevant_ids" in row:
        values = row["relevant_ids"]
    else:
        values = [label["id"] for label in row.get("labels", []) if float(label.get("relevance", 0)) > 0]
    if not isinstance(values, list) or not values or any(not isinstance(x, str) or not x for x in values):
        raise ValueError("prepared query must have nonempty string positive IDs")
    if len(values) != len(set(values)):
        raise ValueError("prepared relevant_ids cannot repeat IDs")
    return set(values)


def _load_cohort(cache_path: Path, queries_path: Path, manifest_path: Path,
                 corpus_hash: str, corpus_ids: set[str], expected_rows: int) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], dict[str, Any]]:
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("corpus_sha256") != corpus_hash:
        raise ValueError("corpus hash does not match cohort manifest")
    if manifest.get("cache_sha256") != file_sha256(cache_path):
        raise ValueError("ranking cache hash does not match cohort manifest")
    query_hash = manifest.get("queries_sha256") or manifest.get("data_preparation_manifest", {}).get("prepared_files", {}).get("queries.jsonl", {}).get("sha256")
    if query_hash != file_sha256(queries_path):
        raise ValueError("query hash does not match cohort manifest")
    queries: dict[str, dict[str, Any]] = {}
    for query in read_jsonl(queries_path):
        qid = query.get("id")
        if not isinstance(qid, str) or not qid or qid in queries:
            raise ValueError("prepared query IDs must be unique nonempty strings")
        if not isinstance(query.get("query"), str) or not query["query"].strip():
            raise ValueError("prepared query text must be nonempty")
        queries[qid] = query
    rows = read_jsonl(cache_path)
    if len(rows) != expected_rows or len(queries) != expected_rows:
        raise ValueError(f"cohort cache and queries must contain exactly {expected_rows} rows")
    seen = set()
    for row in rows:
        qid = row.get("query_id")
        if qid not in queries or qid in seen:
            raise ValueError("cache must have the exact unique prepared query IDs")
        seen.add(qid)
        gold = _gold(queries[qid])
        if not gold <= corpus_ids or set(row.get("relevant_ids", [])) != gold:
            raise ValueError("cache relevance alignment or corpus membership mismatch")
        if not isinstance(row.get("rankings"), dict):
            raise ValueError("cache needs rankings")
        # The feature API validates sparse/dense uniqueness; unknown IDs and
        # expensive-rank permutation are checked independently for provenance.
        features(queries[qid]["query"], row["rankings"])
        for method in ("bm25", "dense", "hybrid", "reranked"):
            ranking = row["rankings"].get(method)
            if not isinstance(ranking, list) or not ranking or len(set(ranking)) != len(ranking) or not set(ranking) <= corpus_ids:
                raise ValueError("cache rankings must be unique corpus IDs")
        if len(row["rankings"]["bm25"]) != 100 or len(row["rankings"]["dense"]) != 100:
            raise ValueError("frozen router expects top-100 first-stage candidate lists")
        depth = row.get("rerank_k")
        hybrid, reranked = row["rankings"]["hybrid"], row["rankings"]["reranked"]
        if isinstance(depth, bool) or not isinstance(depth, int) or depth != 20 or len(hybrid) < depth:
            raise ValueError("frozen router expects a top-20 reranker")
        if len(hybrid) != len(reranked) or set(hybrid[:depth]) != set(reranked[:depth]) or hybrid[depth:] != reranked[depth:]:
            raise ValueError("expensive ranking must preserve the hybrid tail")
        if hybrid != weighted_reciprocal_rank_fusion(
                [row["rankings"]["bm25"], row["rankings"]["dense"]], [1, 1], 60):
            raise ValueError("hybrid ranks differ from the frozen RRF definition")
    return rows, queries, manifest


def _frozen_protocol(path: Path, commit: str) -> tuple[dict[str, Any], str]:
    repo = Path(__file__).resolve().parents[1]
    try:
        relative = path.resolve().relative_to(repo).as_posix()
    except ValueError as exc:
        raise ValueError("protocol must be inside this Git repository") from exc
    try:
        frozen = subprocess.check_output(["git", "show", f"{commit}:{relative}"], cwd=repo)
        full_commit = subprocess.check_output(["git", "rev-parse", f"{commit}^{{commit}}"], cwd=repo, text=True).strip()
    except subprocess.CalledProcessError as exc:
        raise ValueError("protocol is not present in the declared frozen commit") from exc
    if hashlib.sha256(frozen).hexdigest() != file_sha256(path):
        raise ValueError("protocol differs from the explicitly frozen Git commit")
    return json.loads(frozen), full_commit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for cohort in ("train", "calibration"):
        parser.add_argument(f"--{cohort}-cache", type=Path, required=True)
        parser.add_argument(f"--{cohort}-queries", type=Path, required=True)
        parser.add_argument(f"--{cohort}-manifest", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--frozen-protocol-commit", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    protocol, protocol_commit = _frozen_protocol(args.protocol, args.frozen_protocol_commit)
    config = protocol.get("utility_router", {})
    if (config.get("model") != "ridge_expected_ndcg_delta" or config.get("alpha") != 10
            or config.get("feature_version") != FEATURE_VERSION
            or config.get("budget_fractions") != [.25, .5, .75]
            or config.get("primary_budget_fraction") != .75
            or config.get("threshold_calibration") != "prediction_quantile_strict_greater_no_labels"):
        parser.error("protocol does not match the frozen utility-router method")
    training_n = config.get("training_query_count")
    calibration_n = config.get("calibration_query_count")
    if training_n != 300 or isinstance(calibration_n, bool) or not isinstance(calibration_n, int) or calibration_n < 2:
        parser.error("protocol must declare training_query_count=300 and calibration_query_count>=2")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error("output directory must be empty or new; fitted artifacts cannot be overwritten")
    corpus = load_corpus(args.corpus)
    corpus_hash = file_sha256(args.corpus)
    train, train_queries, train_manifest = _load_cohort(
        args.train_cache, args.train_queries, args.train_manifest, corpus_hash, set(corpus), training_n)
    calibration, calibration_queries, calibration_manifest = _load_cohort(
        args.calibration_cache, args.calibration_queries, args.calibration_manifest, corpus_hash, set(corpus), calibration_n)
    if train_manifest.get("scope") != "preliminary_fresh_pilot" or calibration_manifest.get("scope") != "confirmation_study":
        parser.error("training must be the observed pilot and calibration must be the new confirmation-study scope")
    if (calibration_manifest.get("split", {}).get("name") != "calibration"
            or calibration_manifest.get("cache_provenance", {}).get("partition") != "calibration"
            or any(row.get("split") != "calibration" for row in calibration)
            or any(row.get("split") != "calibration" for row in calibration_queries.values())):
        parser.error("calibration manifest, cache rows, and prepared queries must explicitly declare the calibration partition")
    if set(train_queries) & set(calibration_queries):
        parser.error("training and calibration query IDs must be disjoint")
    train_texts = {_normalized_query(query["query"]) for query in train_queries.values()}
    calibration_texts = {_normalized_query(query["query"]) for query in calibration_queries.values()}
    if len(train_texts) != training_n or len(calibration_texts) != calibration_n or train_texts & calibration_texts:
        parser.error("training/calibration normalized query texts must be unique and disjoint")
    training_gold = set().union(*(_gold(row) for row in train_queries.values()))
    calibration_gold = set().union(*(_gold(row) for row in calibration_queries.values()))
    if training_gold & calibration_gold:
        parser.error("training and calibration positive-gold tool IDs must be disjoint")
    for manifest in (train_manifest, calibration_manifest):
        if manifest.get("rrf") != {"k": 60, "weights": [1, 1]}:
            parser.error("frozen features require equal-weight RRF with k=60")
        for name, field in (("dense", "dense"), ("reranker", "reranker")):
            entry = manifest.get("models", {}).get(name, {})
            if entry.get("name") != protocol.get(f"{field}_model") or entry.get("revision") != protocol.get(f"{field}_revision"):
                parser.error("cache model/revision differs from frozen confirmation protocol")
    x_train = np.asarray([features(train_queries[row["query_id"]]["query"], row["rankings"]) for row in train])
    # This is the only relevance-based target computation. Calibration qrels
    # validate dataset alignment above but never enter fitting or thresholds.
    y_train = np.asarray([
        ndcg_at_k(row["rankings"]["reranked"], row["relevant_ids"], 10)
        - ndcg_at_k(row["rankings"]["hybrid"], row["relevant_ids"], 10)
        for row in train
    ])
    model = fitmodel(x_train, y_train, alpha=10.)
    train_predictions = np.asarray(predict(model, x_train))
    x_calibration = np.asarray([features(calibration_queries[row["query_id"]]["query"], row["rankings"]) for row in calibration])
    calibration_predictions = np.asarray(predict(model, x_calibration))
    thresholds = {}
    for budget in config["budget_fractions"]:
        threshold = calibratebudgetthreshold(calibration_predictions, budget)
        calls = sum(shouldroute(float(value), threshold) for value in calibration_predictions)
        thresholds[str(budget)] = {
            "threshold": threshold, "nominal_budget_fraction": budget,
            "primary": budget == config["primary_budget_fraction"],
            "calibration_invocations": calls,
            "calibration_observed_call_fraction": calls / calibration_n,
            "calibration_threshold_ties_skipped": int(np.sum(calibration_predictions == threshold)),
            "decision_rule": "predicted_ndcg_delta > threshold",
        }
    sources = [Path(__file__), Path(__file__).resolve().parents[1] / "src/toolret_research/utility_router.py"]
    inputs = {
        "training": {"cache_sha256": file_sha256(args.train_cache), "queries_sha256": file_sha256(args.train_queries), "manifest_sha256": file_sha256(args.train_manifest), "queries": training_n},
        "calibration": {"cache_sha256": file_sha256(args.calibration_cache), "queries_sha256": file_sha256(args.calibration_queries), "manifest_sha256": file_sha256(args.calibration_manifest), "queries": calibration_n},
        "corpus_sha256": corpus_hash, "protocol_sha256": file_sha256(args.protocol),
        "protocol_frozen_commit": protocol_commit,
        "training_calibration_shared_query_ids": 0,
        "training_calibration_shared_normalized_query_texts": 0,
        "training_calibration_shared_positive_gold_tool_ids": 0,
    }
    artifact = {
        "schema_version": "toolret-frozen-utility-router-v1", "model": model.to_dict(),
        "thresholds": thresholds, "input_integrity": inputs,
        "source_sha256": {str(path.relative_to(Path(__file__).resolve().parents[1])): file_sha256(path) for path in sources},
        "numpy_version": np.__version__,
        "training_labels_used": True, "calibration_labels_used_for_fit_or_thresholds": False,
        "confirmation_inputs_accessed": False,
        "training_target_counts": {"positive": int(np.sum(y_train > 0)), "zero": int(np.sum(y_train == 0)), "negative": int(np.sum(y_train < 0))},
        "feature_inputs": "query whitespace-token count and first-stage BM25/dense positions only; tool ID values serve equality/position lookup only",
        "limitations": ["A model fit to the observed pilot is exploratory until the frozen confirmation evaluation.",
                        "Budget is calibrated pointwise; future invocation fraction is not forced to the nominal budget.",
                        "Threshold ties are skipped; no query/tool/source ID tie breaker is used."],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "model.json").write_text(json.dumps(artifact, indent=2, allow_nan=False) + "\n")
    for name, rows, matrix, predictions in (
        ("development_predictions.jsonl", train, x_train, train_predictions),
        ("calibration_predictions.jsonl", calibration, x_calibration, calibration_predictions),
    ):
        records = []
        for i, row in enumerate(rows):
            record = {"query_id": row["query_id"], "features": matrix[i].tolist(), "predicted_ndcg_delta": float(predictions[i])}
            if name.startswith("development"):
                record["observed_training_ndcg_delta"] = float(y_train[i])
            records.append(record)
        (args.output_dir / name).write_text("".join(json.dumps(record, allow_nan=False) + "\n" for record in records))
    print(json.dumps({"model": str((args.output_dir / "model.json").resolve()),
                      "training_queries": training_n, "calibration_queries": calibration_n,
                      "thresholds": thresholds, "calibration_labels_used": False}, indent=2))


if __name__ == "__main__":
    main()
