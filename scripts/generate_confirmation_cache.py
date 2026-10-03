"""Generate pinned rankings for a frozen tool-disjoint confirmation partition.

Run from the repository root with PYTHONPATH=src. Corpus encoding is shard-
checkpointed so interruption does not discard already encoded documents.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import time
from pathlib import Path

from toolret_research.bm25 import BM25Index
from toolret_research.data import load_corpus, load_queries, read_jsonl
from toolret_research.fusion import weighted_reciprocal_rank_fusion
from toolret_research.text import flatten_tool


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def reranked_prefix(hybrid: list[str], scores: list[float], depth: int) -> list[str]:
    if len(scores) != depth or depth > len(hybrid):
        raise ValueError("Reranker must score exactly the requested prefix.")
    import math
    if not all(math.isfinite(float(value)) for value in scores):
        raise ValueError("Non-finite reranker scores.")
    prefix = sorted(zip(hybrid[:depth], scores), key=lambda pair: (-pair[1], pair[0]))
    return [doc_id for doc_id, _ in prefix] + hybrid[depth:]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("calibration", "confirmation"), required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--data-manifest", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--embedding-dir", type=Path, required=True)
    args = parser.parse_args()
    # Disable Xet's separate transfer layer; plain Hugging Face downloads suffice.
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    import numpy as np
    import torch
    from sentence_transformers import CrossEncoder, SentenceTransformer

    config = json.loads(args.protocol.read_text())
    full_data_manifest = json.loads(args.data_manifest.read_text())
    query_file_key = f"{args.split}_queries.jsonl"
    per_domain = full_data_manifest[f"{args.split}_per_domain"]
    data_manifest = {
        "source_revisions": full_data_manifest["source_revisions"],
        "corpus_count": full_data_manifest["corpus_count"],
        "query_count": full_data_manifest["prepared_files"][query_file_key]["query_count"],
        "pilot_domains": full_data_manifest["pilot_domains"],
        "queries_per_domain": per_domain,
        "prepared_files": {
            "corpus.jsonl": full_data_manifest["prepared_files"]["corpus.jsonl"],
            "queries.jsonl": full_data_manifest["prepared_files"][query_file_key],
        },
        "parent_manifest_sha256": sha256_file(args.data_manifest),
        "split": args.split,
        "selection": full_data_manifest["assignment_algorithm"],
        "scope": full_data_manifest["scope"],
    }
    if (args.output_dir / "rankings.jsonl").exists():
        raise ValueError("Refusing to replace an existing partition cache.")
    torch.set_num_threads(config["threads"])
    torch.set_num_interop_threads(1)
    corpus = load_corpus(args.corpus)
    queries = load_queries(args.queries)
    query_rows = {str(row["id"]): row for row in read_jsonl(args.queries)}
    if len(queries) != len(query_rows) or not queries:
        raise ValueError("Query IDs must be unique and the query set nonempty.")
    for query in queries:
        if not query.relevant_ids or not query.relevant_ids.issubset(corpus):
            raise ValueError(f"Missing or unknown positive labels for {query.id}")
    from collections import Counter
    observed_domains = Counter(row.get("domain") for row in query_rows.values())
    if observed_domains != {domain: per_domain for domain in data_manifest["pilot_domains"]}:
        raise ValueError("Partition source-domain counts differ from the frozen manifest.")
    if len(queries) != data_manifest["query_count"] or len(corpus) != data_manifest["corpus_count"]:
        raise ValueError("Partition/corpus count mismatch.")
    if any(not isinstance(row.get("component_id"), str) or row.get("split") != args.split for row in query_rows.values()):
        raise ValueError("Every query must declare its frozen component and partition.")
    document_ids = list(corpus)
    document_texts = {
        doc_id: flatten_tool({"id": doc_id, "text": corpus[doc_id]["text"]})
        for doc_id in document_ids
    }
    corpus_hash = sha256_file(args.corpus)
    if data_manifest["prepared_files"]["corpus.jsonl"]["sha256"] != corpus_hash:
        raise ValueError("Prepared corpus manifest hash mismatch.")
    if data_manifest["prepared_files"]["queries.jsonl"]["sha256"] != sha256_file(args.queries):
        raise ValueError("Prepared query manifest hash mismatch.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.embedding_dir.mkdir(parents=True, exist_ok=True)
    identity = {
        "corpus_sha256": corpus_hash,
        "document_ids_sha256": hashlib.sha256(json.dumps(document_ids).encode()).hexdigest(),
        "model": config["dense_model"], "revision": config["dense_revision"],
        "max_sequence_length": config["max_sequence_length"],
        "serialization": "flatten_tool(id,text) at script/source SHA256",
        "text_source_sha256": sha256_file(Path("src/toolret_research/text.py")),
        "shard_size": 1024,
        "dtype": "float32", "normalized": True,
    }
    identity_path = args.embedding_dir / "identity.json"
    if identity_path.exists() and json.loads(identity_path.read_text()) != identity:
        raise ValueError("Embedding cache identity mismatch; use a new embedding directory.")
    atomic_json(identity_path, identity)
    print(f"Loading pinned dense model; corpus={len(corpus)}, queries={len(queries)}", flush=True)
    dense = SentenceTransformer(config["dense_model"], revision=config["dense_revision"], device="cpu")
    dense.max_seq_length = config["max_sequence_length"]
    texts = list(document_texts.values())
    embedding_start = time.perf_counter()
    shards = []
    shard_hashes = {}
    for offset in range(0, len(texts), identity["shard_size"]):
        end = min(offset + identity["shard_size"], len(texts))
        shard_path = args.embedding_dir / f"shard_{offset:06d}.npy"
        if shard_path.exists():
            values = np.load(shard_path, allow_pickle=False)
            if values.shape != (end - offset, dense.get_sentence_embedding_dimension()):
                raise ValueError(f"Invalid cached embedding shape: {shard_path}")
        else:
            values = dense.encode(texts[offset:end], batch_size=config["batch_size"],
                                  normalize_embeddings=True, convert_to_numpy=True,
                                  show_progress_bar=False).astype(np.float32)
            with shard_path.with_suffix(".tmp").open("wb") as stream:
                np.save(stream, values, allow_pickle=False)
            shard_path.with_suffix(".tmp").replace(shard_path)
        if not np.isfinite(values).all():
            raise ValueError("Non-finite document embedding.")
        if values.dtype != np.float32 or not np.allclose(np.linalg.norm(values, axis=1), 1.0, atol=2e-5):
            raise ValueError("Cached embeddings must be unit-normalized float32.")
        shard_hashes[shard_path.name] = sha256_file(shard_path)
        shards.append(values)
        print(f"Encoded/reused {end}/{len(texts)} tools; elapsed={time.perf_counter()-embedding_start:.1f}s", flush=True)
    document_embeddings = torch.from_numpy(np.concatenate(shards)).contiguous()
    print("Building BM25 index", flush=True)
    index_start = time.perf_counter()
    bm25 = BM25Index.build(document_texts)
    index_seconds = time.perf_counter() - index_start
    reranker = CrossEncoder(config["reranker_model"], revision=config["reranker_revision"],
                            device="cpu", max_length=config["max_sequence_length"])
    # Warm models with unscored smoke input; pilot labels are never used here.
    dense.encode(["weather forecast"], normalize_embeddings=True, show_progress_bar=False)
    reranker.predict([["weather forecast", "retrieve a weather forecast"]], show_progress_bar=False)
    cache_path = args.output_dir / "rankings.jsonl"
    if cache_path.exists():
        raise ValueError("Rankings already exist; choose a fresh output directory to avoid accidental replacement.")
    temporary_cache = cache_path.with_suffix(".jsonl.tmp")
    with temporary_cache.open("w", encoding="utf-8") as stream, torch.inference_mode():
        for number, query in enumerate(queries, 1):
            start = time.perf_counter()
            sparse = [doc_id for doc_id, _ in bm25.search(query.text, k=config["candidate_k"])]
            sparse_ms = (time.perf_counter() - start) * 1000
            start = time.perf_counter()
            q_emb = dense.encode([query.text], normalize_embeddings=True,
                                 convert_to_tensor=True, show_progress_bar=False)
            similarity = torch.matmul(q_emb, document_embeddings.T)[0].cpu().numpy()
            # Stable score-descending / ID-ascending ties, consistent with BM25/RRF.
            indices = np.lexsort((np.asarray(document_ids), -similarity))[:config["candidate_k"]]
            dense_ids = [document_ids[int(index)] for index in indices]
            dense_ms = (time.perf_counter() - start) * 1000
            start = time.perf_counter()
            hybrid = weighted_reciprocal_rank_fusion([sparse, dense_ids], config["rrf_weights"], config["rrf_k"])
            fusion_ms = (time.perf_counter() - start) * 1000
            depth = min(config["rerank_k"], len(hybrid))
            start = time.perf_counter()
            scores = reranker.predict([[query.text, document_texts[doc_id]] for doc_id in hybrid[:depth]],
                                      batch_size=config["batch_size"], show_progress_bar=False)
            reranked = reranked_prefix(hybrid, [float(score) for score in scores], depth)
            ce_ms = (time.perf_counter() - start) * 1000
            row = {
                "query_id": query.id,
                "component_id": query_rows[query.id]["component_id"],
                "split": args.split,
                "source_domain": query_rows[query.id].get("source_domain", query_rows[query.id].get("domain", query.id.split("_query_")[0])),
                "relevant_ids": sorted(query.relevant_ids),
                "rankings": {"bm25": sparse, "dense": dense_ids, "hybrid": hybrid, "reranked": reranked},
                "rerank_k": depth,
                "timings_ms": {"bm25": sparse_ms, "dense": dense_ms, "hybrid": fusion_ms, "reranked": ce_ms},
            }
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            stream.flush()
            if number % 10 == 0 or number == len(queries):
                print(f"Ranked {number}/{len(queries)} queries", flush=True)
    temporary_cache.replace(cache_path)
    source_files = [args.protocol, args.data_manifest, Path(__file__), Path("src/toolret_research/bm25.py"),
                    Path("src/toolret_research/text.py"), Path("src/toolret_research/fusion.py")]
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except subprocess.CalledProcessError:
        commit = "unavailable"
    manifest = {
        "schema_version": "toolret-selective-v1", "scope": "confirmation_study",
        "dataset": {"name": "ToolRet positive-tool-component-disjoint confirmation study", "version": data_manifest["source_revisions"],
                    "source": "https://huggingface.co/datasets/mangopy/ToolRet-Queries"},
        "data_preparation_manifest": data_manifest,
        "full_data_preparation_manifest": full_data_manifest,
        "embedding_cache": {"identity": identity, "shard_sha256": shard_hashes, "reused": True},
        "corpus_sha256": corpus_hash, "queries_sha256": sha256_file(args.queries),
        "cache_sha256": sha256_file(cache_path),
        "models": {
            "dense": {"name": config["dense_model"], "revision": config["dense_revision"], "fine_tuned_here": False},
            "reranker": {"name": config["reranker_model"], "revision": config["reranker_revision"], "fine_tuned_here": False}},
        "split": {"name": args.split, "selection": full_data_manifest["assignment_algorithm"]},
        "rrf": {"k": config["rrf_k"], "weights": config["rrf_weights"]},
        "cache_provenance": {"script": "scripts/generate_confirmation_cache.py", "base_commit": commit,
                             "partition": args.split,
                             "source_sha256": {str(path): sha256_file(path) for path in source_files},
                             "protocol": config, "corpus_documents": len(corpus), "queries": len(queries)},
        "timing_provenance": {"hardware": platform.platform(), "processor": platform.processor(),
                              "disjoint_components": True,
                              "torch_threads": config["threads"], "device": "cpu", "warm_models": True,
                              "repetitions": 1, "order": "bm25,dense,rrf,cross_encoder per query",
                              "boundary": "component wall time; bm25 search, dense encode+exact search+sort, rrf fusion, CE predict+prefix sort",
                              "policy_latency": "counterfactual offline component sums only; routing overhead not included",
                              "excluded": "model download/load, corpus encoding, index build, service/network/concurrency",
                              "bm25_index_seconds": index_seconds},
        "packages": {name: importlib.metadata.version(name) for name in ["torch", "sentence-transformers", "transformers", "numpy", "huggingface-hub"]},
        "limitations": config["limitations"],
    }
    atomic_json(args.output_dir / "manifest.json", manifest)
    print(f"Saved {cache_path}; SHA256={manifest['cache_sha256']}", flush=True)


if __name__ == "__main__":
    main()
