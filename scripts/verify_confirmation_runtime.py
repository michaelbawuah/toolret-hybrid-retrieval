"""Verify actual primary-policy CE calls over every frozen confirmation query.

First-stage rankings are cached. This execution verifies conditional model work
and exact rankings; end-to-end latency is a separate repeated benchmark.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path

from toolret_research.confirmation import validate_confirmation_rows
from toolret_research.data import load_corpus, read_jsonl
from toolret_research.selective import file_sha256
from toolret_research.text import flatten_tool
from toolret_research.utility_router import RidgeUtilityModel
from toolret_research.utility_runtime import route_utility_and_rerank


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("cache", "manifest", "corpus", "queries", "protocol", "router", "predictions", "evaluation-summary", "output"):
        p.add_argument(f"--{name}", type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise ValueError("Refusing to replace a prior runtime verification")
    manifest, protocol, artifact, evaluation = [json.loads(path.read_text()) for path in (a.manifest, a.protocol, a.router, a.evaluation_summary)]
    for key, path in (("cache_sha256", a.cache), ("corpus_sha256", a.corpus), ("queries_sha256", a.queries)):
        if manifest.get(key) != file_sha256(path):
            raise ValueError(f"Manifest input mismatch: {key}")
    integrity = evaluation["input_integrity"]
    for key, path in (("router_sha256", a.router), ("protocol_sha256", a.protocol), ("predictions_sha256", a.predictions), ("cache_sha256", a.cache)):
        if integrity.get(key) != file_sha256(path):
            raise ValueError(f"Evaluated input mismatch: {key}")
    if artifact.get("schema_version") != "toolret-frozen-utility-router-v1":
        raise ValueError("Expected frozen utility artifact")
    for key, path in (("protocol_sha256", a.protocol), ("corpus_sha256", a.corpus)):
        if artifact["input_integrity"].get(key) != file_sha256(path):
            raise ValueError(f"Router input mismatch: {key}")
    for relative, expected in artifact["source_sha256"].items():
        if file_sha256(relative) != expected:
            raise ValueError(f"Frozen router source changed: {relative}")
    if protocol.get("primary_policy") != "utility_75" or protocol["confirmation"]["queries"] != 1500:
        raise ValueError("Expected the frozen1500-query primary protocol")
    if protocol["confirmation"]["queries_sha256"] != file_sha256(a.queries):
        raise ValueError("Confirmation cohort is not the prospectively frozen file")
    declared = manifest["cache_provenance"]["protocol"]
    if declared != protocol or manifest["split"]["name"] != "confirmation" or manifest["cache_provenance"]["partition"] != "confirmation":
        raise ValueError("Wrong cache protocol or role")
    queries = list(read_jsonl(a.queries))
    corpus = load_corpus(a.corpus)
    rows = validate_confirmation_rows(read_jsonl(a.cache), corpus, manifest, queries)
    predictions = {r["query_id"]: r for r in read_jsonl(a.predictions)}
    if len(predictions) != 1500 or set(predictions) != {r["query_id"] for r in rows}:
        raise ValueError("Frozen prediction cohort mismatch")
    model = RidgeUtilityModel.from_dict(artifact["model"])
    threshold = float(artifact["thresholds"]["0.75"]["threshold"])
    documents = {doc: flatten_tool({"id": doc, "text": value["text"]}) for doc, value in corpus.items()}
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    import torch
    from sentence_transformers import CrossEncoder
    torch.set_num_threads(protocol["threads"])
    torch.set_num_interop_threads(protocol["interop_threads"])
    ce = CrossEncoder(protocol["reranker_model"], revision=protocol["reranker_revision"], device="cpu", max_length=protocol["max_sequence_length"])

    class CountingBackend:
        calls = 0
        pairs = 0

        def rerank(self, query, candidates):
            self.calls += 1
            self.pairs += len(candidates)
            scores = ce.predict([[query, text] for _, text in candidates], batch_size=protocol["batch_size"], show_progress_bar=False)
            return [(doc, float(score)) for (doc, _), score in zip(candidates, scores)]

    backend = CountingBackend()
    records = []
    with torch.inference_mode():
        for number, row in enumerate(rows, 1):
            ranks = row["rankings"]
            frozen = predictions[row["query_id"]]
            result = route_utility_and_rerank(row["query"], ranks["bm25"], ranks["dense"], ranks["hybrid"], documents, model, threshold, backend, depth=20)
            routed = frozen["rerank_decisions"]["utility_75"]
            if result.reranked != routed or abs(result.predicted_utility - frozen["predicted_reranking_utility"]) > 1e-12:
                raise ValueError(f"Frozen runtime decision mismatch: {row['query_id']}")
            expected = ranks["reranked"] if routed else ranks["hybrid"]
            if list(result.ranking) != expected:
                raise ValueError(f"Real CE ranking mismatch: {row['query_id']}")
            records.append({"query_id": row["query_id"], "reranked": result.reranked, "scored_pairs": result.scored_pairs, "matches_frozen_decision": True, "matches_evaluated_ranking": True})
            if number % 50 == 0:
                print(f"Primary runtime verified {number}/1500", flush=True)
    expected_calls = sum(record["rerank_decisions"]["utility_75"] for record in predictions.values())
    if backend.calls != expected_calls or backend.pairs != expected_calls * 20:
        raise ValueError("Actual CE invocation accounting mismatch")
    payload = {
        "scope": "actual primary conditional CE execution over cached first-stage rankings; separate from end-to-end latency",
        "primary_policy": "utility_75", "queries": 1500, "actual_backend_calls": backend.calls,
        "actual_backend_pairs": backend.pairs, "skipped_calls": 1500 - backend.calls,
        "all_decisions_match_frozen_predictions": True, "all_rankings_match_evaluation": True,
        "input_integrity": {key: file_sha256(path) for key, path in (("router_sha256", a.router), ("protocol_sha256", a.protocol), ("predictions_sha256", a.predictions), ("cache_sha256", a.cache), ("queries_sha256", a.queries), ("corpus_sha256", a.corpus), ("evaluation_summary_sha256", a.evaluation_summary))},
        "source_sha256": {path: file_sha256(path) for path in ("scripts/verify_confirmation_runtime.py", "src/toolret_research/utility_runtime.py", "src/toolret_research/utility_router.py")},
        "per_query": records,
    }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"actual_calls": backend.calls, "actual_pairs": backend.pairs, "skipped": 1500-backend.calls}), flush=True)


if __name__ == "__main__":
    main()
