import argparse
import json
import time
from pathlib import Path

from .bm25 import BM25Index
from .data import load_corpus, load_queries
from .metrics import evaluate_rankings
from .text import flatten_tool

def run(corpus_path, query_path, ks):
    corpus = load_corpus(corpus_path)
    queries = load_queries(query_path)
    docs = {doc_id: flatten_tool(doc) for doc_id, doc in corpus.items()}

    t0 = time.perf_counter()
    index = BM25Index.build(docs)
    build_seconds = time.perf_counter() - t0

    rankings = {}
    latencies = []
    max_k = max(ks)

    for q in queries:
        t1 = time.perf_counter()
        hits = index.search(q.text, k=max_k)
        latencies.append((time.perf_counter() - t1) * 1000.0)
        rankings[q.id] = [doc_id for doc_id, _ in hits]

    qrels = {q.id: set(q.relevant_ids) for q in queries}
    metrics = evaluate_rankings(rankings, qrels, ks)

    ordered = sorted(latencies)
    p95_index = max(0, min(len(ordered)-1, int(0.95 * len(ordered))-1))
    return {
        "retriever": "bm25",
        "num_documents": len(corpus),
        "num_queries": len(queries),
        "build_seconds": build_seconds,
        "mean_query_latency_ms": sum(latencies) / max(len(latencies), 1),
        "p95_query_latency_ms": ordered[p95_index] if ordered else 0.0,
        "metrics": metrics,
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--queries", required=True)
    parser.add_argument("--k", nargs="+", type=int, default=[1, 5, 10])
    parser.add_argument("--output")
    args = parser.parse_args()

    result = run(args.corpus, args.queries, args.k)
    payload = json.dumps(result, indent=2)
    print(payload)

    if args.output:
        p = Path(args.output)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(payload + "\n", encoding="utf-8")

if __name__ == "__main__":
    main()
