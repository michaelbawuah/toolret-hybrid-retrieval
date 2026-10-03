"""Verify real selective model calls and rankings against the audited cache."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from toolret_research.data import load_corpus, load_queries, read_jsonl
from toolret_research.selective import file_sha256, should_rerank, validate_rows
from toolret_research.selective_runtime import route_and_rerank
from toolret_research.text import flatten_tool


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--queries", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    import torch
    from sentence_transformers import CrossEncoder
    manifest = json.loads(Path(args.manifest).read_text())
    if file_sha256(args.cache) != manifest["cache_sha256"] or file_sha256(args.corpus) != manifest["corpus_sha256"] or file_sha256(args.queries) != manifest["queries_sha256"]:
        raise ValueError("Runtime verification inputs must match the generated manifest")
    config = manifest["cache_provenance"]["protocol"]
    torch.set_num_threads(config["threads"])
    corpus = load_corpus(args.corpus)
    rows = validate_rows(read_jsonl(args.cache), corpus, manifest)
    queries = {query.id: query.text for query in load_queries(args.queries)}
    if set(queries) != {row["query_id"] for row in rows}:
        raise ValueError("Runtime cohort mismatch")
    documents = {doc_id: flatten_tool({"id": doc_id, "text": row["text"]}) for doc_id, row in corpus.items()}
    model = CrossEncoder(config["reranker_model"], revision=config["reranker_revision"], device="cpu", max_length=config["max_sequence_length"])

    class CountingBackend:
        def __init__(self):
            self.calls = 0
            self.pairs = 0

        def rerank(self, query, candidates):
            self.calls += 1
            self.pairs += len(candidates)
            values = model.predict([[query, text] for _, text in candidates], batch_size=config["batch_size"], show_progress_bar=False)
            return sorted([(doc_id, float(score)) for (doc_id, _), score in zip(candidates, values)], key=lambda pair: (-pair[1], pair[0]))

    backend = CountingBackend()
    records = []
    for number, row in enumerate(rows, 1):
        ranks = row["rankings"]
        result = route_and_rerank(queries[row["query_id"]], ranks["bm25"], ranks["dense"], ranks["hybrid"], documents, backend, depth=row["rerank_k"])
        expected = ranks["reranked"] if should_rerank(row) else ranks["hybrid"]
        if list(result.ranking) != expected:
            raise ValueError(f"Real runtime ranking mismatch: {row['query_id']}")
        records.append({"query_id": row["query_id"], "reranked": result.reranked, "scored_pairs": result.scored_pairs, "matches_cache": True})
        if number % 25 == 0 or number == len(rows):
            print(f"Runtime verified {number}/{len(rows)}", flush=True)
    expected_calls = sum(should_rerank(row) for row in rows)
    expected_pairs = sum(row["rerank_k"] for row in rows if should_rerank(row))
    if backend.calls != expected_calls or backend.pairs != expected_pairs:
        raise ValueError("Actual backend invocation accounting mismatch")
    output = Path(args.output)
    if output.exists():
        raise ValueError("Refusing to replace prior runtime verification")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({
        "scope": "actual selective reranker calls over cached first-stage rankings; not end-to-end latency measurement",
        "queries": len(rows), "backend_calls": backend.calls, "backend_pairs": backend.pairs,
        "skipped_calls": len(rows)-backend.calls, "all_rankings_match_cache": True,
        "cache_sha256": manifest["cache_sha256"], "script_sha256": file_sha256(__file__),
        "per_query": records,
    }, indent=2) + "\n")
    print(f"Verified actual calls={backend.calls}, pairs={backend.pairs}; saved {output}", flush=True)


if __name__ == "__main__":
    main()
