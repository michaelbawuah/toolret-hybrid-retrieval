"""Evaluate the frozen router on the untouched component-disjoint confirmation.

The model is fitted elsewhere, without confirmation inputs. This command checks
cohort/model/protocol provenance, writes immutable routing decisions before
quality scoring, and applies the predeclared primary noninferiority criterion.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

from toolret_research.confirmation import (
    BOOTSTRAP_RESAMPLES, BOOTSTRAP_SEED, RANDOM_SEEDS,
    assert_split_disjointness, canonical_sha256, evaluate_confirmation,
    freeze_predictions, render_confirmation_report, validate_confirmation_rows,
)
from toolret_research.data import load_corpus, read_jsonl
from toolret_research.selective import file_sha256


def verify_frozen_protocol(path: Path, commit: str) -> dict:
    root = Path(__file__).resolve().parents[1]
    relative = path.resolve().relative_to(root).as_posix()
    content = subprocess.check_output(["git", "show", f"{commit}:{relative}"], cwd=root)
    if hashlib.sha256(content).hexdigest() != file_sha256(path):
        raise ValueError("protocol differs from the frozen commit")
    protocol = json.loads(content)
    if (protocol.get("bootstrap_seed") != BOOTSTRAP_SEED
            or protocol.get("bootstrap_resamples") != BOOTSTRAP_RESAMPLES
            or protocol.get("random_policy_seeds") != list(RANDOM_SEEDS)):
        raise ValueError("bootstrap/random settings differ from frozen evaluator")
    if (protocol.get("primary_policy") != "utility_75" or protocol.get("reference_policy") != "always"
            or protocol.get("noninferiority_margin") != .01):
        raise ValueError("primary policy/reference/margin differs from frozen evaluator")
    router = protocol.get("utility_router", {})
    if (router.get("alpha") != 10 or router.get("budget_fractions") != [.25, .5, .75]
            or router.get("primary_budget_fraction") != .75):
        raise ValueError("utility method differs from frozen evaluator")
    return protocol


def _check_hash(path: Path, expected: str, label: str) -> str:
    actual = file_sha256(path)
    if actual != expected:
        raise ValueError(f"{label} SHA256 differs from declared provenance")
    return actual


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("cache", "manifest", "queries", "corpus", "data-manifest", "router", "protocol", "development-queries",
                 "training-cache", "training-queries", "training-manifest", "calibration-cache", "calibration-queries", "calibration-manifest"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error("output directory must be empty or new; frozen decisions/results cannot be overwritten")
    router = json.loads(args.router.read_text())
    if router.get("schema_version") != "toolret-frozen-utility-router-v1":
        parser.error("router artifact schema mismatch")
    integrity = router["input_integrity"]
    _check_hash(args.protocol, integrity["protocol_sha256"], "router protocol")
    protocol = verify_frozen_protocol(args.protocol, integrity["protocol_frozen_commit"])
    for cohort in ("training", "calibration"):
        for kind in ("cache", "queries", "manifest"):
            _check_hash(getattr(args, f"{cohort}_{kind}"), integrity[cohort][f"{kind}_sha256"], f"router {cohort} {kind}")
    root = Path(__file__).resolve().parents[1]
    for name in ("split_manifest", "data_audit", "component_assignments"):
        relative = protocol[name]
        _check_hash(root / relative, protocol[f"{name}_sha256"], f"frozen {name}")
    _check_hash(args.data_manifest, protocol["split_manifest_sha256"], "requested data manifest")
    for relative, expected in router["source_sha256"].items():
        source_path = (root / relative).resolve()
        if not source_path.is_relative_to(root):
            parser.error("router source path escapes repository")
        _check_hash(source_path, expected, f"router source {relative}")
    if router.get("confirmation_inputs_accessed") is not False or router.get("calibration_labels_used_for_fit_or_thresholds") is not False:
        parser.error("router provenance must declare no confirmation fitting or calibration-label fitting")
    manifest = json.loads(args.manifest.read_text())
    data_manifest = json.loads(args.data_manifest.read_text())
    _check_hash(args.corpus, integrity["corpus_sha256"], "router corpus")
    _check_hash(args.corpus, manifest["corpus_sha256"], "confirmation corpus")
    _check_hash(args.cache, manifest["cache_sha256"], "confirmation cache")
    _check_hash(args.queries, manifest["queries_sha256"], "confirmation queries")
    if manifest["split"].get("name") != "confirmation":
        parser.error("cache split must be confirmation")
    if canonical_sha256(manifest["cache_provenance"]["protocol"]) != canonical_sha256(protocol):
        parser.error("cache protocol content differs from frozen router protocol")
    if manifest["data_preparation_manifest"].get("parent_manifest_sha256") != file_sha256(args.data_manifest):
        parser.error("cache data-preparation parent SHA differs")
    for filename, path in (("development_queries.jsonl", args.development_queries), ("calibration_queries.jsonl", args.calibration_queries), ("confirmation_queries.jsonl", args.queries)):
        _check_hash(path, data_manifest["prepared_files"][filename]["sha256"], filename)
    development = read_jsonl(args.development_queries)
    calibration = read_jsonl(args.calibration_queries)
    confirmation = read_jsonl(args.queries)
    splits = assert_split_disjointness(development, calibration, confirmation)
    assignments = json.loads((root / protocol["component_assignments"]).read_text())["components"]
    component_query_ids = set()
    for component, info in assignments.items():
        if info["assignment"] == "confirmation":
            component_query_ids.update(info["query_ids"])
    if component_query_ids != {query["id"] for query in confirmation}:
        parser.error("confirmation does not contain the exact frozen whole-component assignment")
    for query in confirmation:
        info = assignments.get(query["component_id"])
        if info is None or info["assignment"] != "confirmation" or query["id"] not in info["query_ids"]:
            parser.error("confirmation component membership differs from frozen full-universe assignment")
    training = {row["id"]: row for row in read_jsonl(args.training_queries)}
    if set(training) != {row["id"] for row in development}:
        parser.error("component-tagged development IDs differ from router training IDs")
    for query in development:
        original = training[query["id"]]
        if original["query"] != query["query"] or set(original["relevant_ids"]) != set(query["relevant_ids"]):
            parser.error("component-tagged development queries differ from the original training cohort")
    if len(development) != 300 or len(calibration) != 300:
        parser.error("development/calibration counts differ from frozen300/300")
    corpus = load_corpus(args.corpus)
    rows = validate_confirmation_rows(read_jsonl(args.cache), corpus, manifest, confirmation)
    if manifest.get("rrf") != {"k": 60, "weights": [1, 1]}:
        parser.error("confirmation RRF differs from frozen equal-weight k60")
    if any(len(row["rankings"][method]) != 100 for row in rows for method in ("bm25", "dense")):
        parser.error("confirmation requires100 candidates per first-stage retriever")
    for name in ("dense", "reranker"):
        model = manifest["models"][name]
        if model.get("name") != protocol[f"{name}_model"] or model.get("revision") != protocol[f"{name}_revision"]:
            parser.error("confirmation model revision differs from frozen protocol")
    # Frozen decision artifact is written, then read back and hashed, BEFORE
    # evaluate_confirmation consults relevance outcomes or oracle controls.
    predictions = freeze_predictions(rows, router)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = args.output_dir / "frozen_predictions.jsonl"
    prediction_path.write_text("".join(json.dumps(record, allow_nan=False) + "\n" for record in predictions))
    frozen_integrity = {
        "created_before_quality_scoring": True, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "router_sha256": file_sha256(args.router), "protocol_sha256": file_sha256(args.protocol),
        "protocol_frozen_commit": integrity["protocol_frozen_commit"],
        "predictions_sha256": file_sha256(prediction_path), "cache_sha256": file_sha256(args.cache),
        "queries_sha256": file_sha256(args.queries), "corpus_sha256": file_sha256(args.corpus),
        "data_manifest_sha256": file_sha256(args.data_manifest), "cache_manifest_sha256": file_sha256(args.manifest),
        "evaluation_source_sha256": {
            relative: file_sha256(root / relative) for relative in (
                "scripts/evaluate_confirmation_study.py", "src/toolret_research/confirmation.py",
                "src/toolret_research/selective.py", "src/toolret_research/metrics.py",
                "src/toolret_research/utility_router.py", "src/toolret_research/fusion.py",
            )
        },
        "split_disjointness": splits,
    }
    (args.output_dir / "frozen_prediction_provenance.json").write_text(json.dumps(frozen_integrity, indent=2, allow_nan=False) + "\n")
    summary, per_query, decisions = evaluate_confirmation(rows, read_jsonl(prediction_path))
    summary["input_integrity"] = frozen_integrity
    summary["manifest"] = manifest
    summary["router"] = router
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    for filename, records in (("per_query.jsonl", per_query), ("decisions.jsonl", decisions)):
        (args.output_dir / filename).write_text("".join(json.dumps(record, allow_nan=False) + "\n" for record in records))
    (args.output_dir / "report.md").write_text(render_confirmation_report(summary))
    print(json.dumps({"queries": len(rows), "primary_noninferiority": summary["primary_noninferiority"],
                      "primary_metrics": summary["policies"]["utility_75"], "output_dir": str(args.output_dir.resolve())}, indent=2))


if __name__ == "__main__":
    main()
