"""Evaluate a fixed disagreement gate from provenance-tracked JSONL rankings.

Run with PYTHONPATH=src or an installed toolret-research package. Input rows:
{query_id, relevant_ids, rankings:{bm25,dense,hybrid,reranked}, rerank_k}
Query text and source_domain are optional; IDs and all positive qrels are required.
Optional timings_ms are separate measured components, never adaptive latency.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from toolret_research.data import load_corpus, read_jsonl
from toolret_research.selective import (
    evaluate_policies,
    file_sha256,
    load_excluded_ids,
    render_report,
    validate_manifest,
    validate_rows,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", required=True, help="Per-query ranking cache JSONL")
    parser.add_argument("--corpus", required=True, help="Corpus JSONL with unique id fields")
    parser.add_argument("--queries", help="Prepared query JSONL; checks exact ID/gold/domain alignment")
    parser.add_argument("--manifest", required=True, help="Dataset, model, split and cache provenance JSON")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--exclude-queries", help="Historically inspected IDs; required for a fresh pilot")
    parser.add_argument("--seed", type=int, default=20261003)
    parser.add_argument("--bootstrap-resamples", type=int, default=2000)
    args = parser.parse_args()
    if args.bootstrap_resamples <= 0:
        parser.error("--bootstrap-resamples must be positive")
    manifest = validate_manifest(json.loads(Path(args.manifest).read_text(encoding="utf-8")))
    if manifest["scope"] == "preliminary_fresh_pilot" and not args.exclude_queries:
        parser.error("--exclude-queries is required for preliminary_fresh_pilot")
    prepared_queries_hash = manifest.get("queries_sha256")
    nested_queries_hash = manifest.get("data_preparation_manifest", {}).get("prepared_files", {}).get("queries.jsonl", {}).get("sha256")
    if prepared_queries_hash and nested_queries_hash and prepared_queries_hash != nested_queries_hash:
        parser.error("prepared query SHA-256 declarations disagree")
    prepared_queries_hash = prepared_queries_hash or nested_queries_hash
    if manifest["scope"] == "preliminary_fresh_pilot" and prepared_queries_hash and not args.queries:
        parser.error("--queries is required for a fresh pilot with a declared query SHA-256")
    if args.queries and prepared_queries_hash and file_sha256(args.queries) != prepared_queries_hash:
        parser.error("prepared queries SHA-256 does not match manifest")
    if file_sha256(args.corpus) != manifest["corpus_sha256"]:
        parser.error("corpus SHA-256 does not match manifest")
    if "cache_sha256" in manifest and file_sha256(args.cache) != manifest["cache_sha256"]:
        parser.error("ranking cache SHA-256 does not match manifest")
    excluded = load_excluded_ids(args.exclude_queries) if args.exclude_queries else set()
    corpus = load_corpus(args.corpus)
    rows = validate_rows(
        read_jsonl(args.cache), corpus, manifest, excluded_ids=excluded,
        expected_queries=read_jsonl(args.queries) if args.queries else None,
    )
    summary, per_query, decisions = evaluate_policies(
        rows, manifest, seed=args.seed, bootstrap_resamples=args.bootstrap_resamples,
    )
    summary["input_integrity"] = {
        "corpus_sha256": file_sha256(args.corpus),
        "cache_sha256": file_sha256(args.cache),
        "manifest_sha256": file_sha256(args.manifest),
        "queries_sha256": file_sha256(args.queries) if args.queries else None,
        "exact_prepared_query_alignment_verified": bool(args.queries),
        "excluded_ids_file_sha256": file_sha256(args.exclude_queries) if args.exclude_queries else None,
        "excluded_id_count": len(excluded),
        "num_corpus_tools": len(corpus),
    }
    output = Path(args.output_dir)
    if output.exists() and any(output.iterdir()):
        parser.error("--output-dir must be empty or new to preserve prior experiment results")
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    for name, records in (("per_query.jsonl", per_query), ("decisions.jsonl", decisions)):
        (output / name).write_text(
            "".join(json.dumps(record, allow_nan=False) + "\n" for record in records), encoding="utf-8",
        )
    (output / "report.md").write_text(render_report(summary), encoding="utf-8")
    print(json.dumps({
        "scope": summary["scope"], "queries": len(rows),
        "gate_invocations": summary["policies"]["gated"]["reranker_invocations"],
        "output_dir": str(output.resolve()),
        "adaptive_latency_measured": False,
    }, indent=2))


if __name__ == "__main__":
    main()
