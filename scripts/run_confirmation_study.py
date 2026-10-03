"""Reproduce the frozen confirmation study without replacing published evidence.

Default: verify/reconstruct exact data bytes, use published caches and router,
and rerun evaluation into a new directory. --recompute regenerates calibration
and confirmation inference artifacts and refits the router in that directory.
--measure-runtime adds actual conditional CE execution and repeated full warm
retrieval latency; absent corpus embeddings are regenerated independently.

Run with the research environment's Python. All child commands use that same
interpreter, set PYTHONPATH themselves, and run from the repository root.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
PUBLISHED = ROOT / "results/research_confirmation_20261003"
PILOT = ROOT / "results/research_20261003"
PROTOCOL = ROOT / "configs/confirmation_protocol_20261003.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_file(path: Path, expected: str, label: str) -> None:
    if not path.is_file() or sha256_file(path) != expected:
        raise ValueError(f"{label} is absent or differs from the frozen SHA256: {path}")


def verify_protocol_commit(commit: str, protocol_path: Path) -> None:
    relative = protocol_path.resolve().relative_to(ROOT).as_posix()
    try:
        frozen = subprocess.check_output(["git", "show", f"{commit}:{relative}"], cwd=ROOT, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError as error:
        raise ValueError(
            f"Frozen protocol Git object is unavailable. Use a full clone, or run: git fetch origin {commit}"
        ) from error
    if hashlib.sha256(frozen).hexdigest() != sha256_file(protocol_path):
        raise ValueError("Current protocol differs from its original frozen Git commit")


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(f"_reproduce_{name}", ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def prepared_paths(data_manifest: dict, pilot_manifest: dict, workspace: Path) -> dict[str, Path]:
    """Existing data must match; only absent data may be reconstructed elsewhere."""
    declarations = {
        "corpus": ("data/research_20261003/corpus.jsonl", data_manifest["prepared_files"]["corpus.jsonl"]),
        "training_queries": ("data/research_20261003/queries.jsonl", pilot_manifest["prepared_files"]["queries.jsonl"]),
    }
    for name in ("development", "calibration", "confirmation"):
        declarations[f"{name}_queries"] = (
            f"data/research_confirmation_20261003/{name}_queries.jsonl",
            data_manifest["prepared_files"][f"{name}_queries.jsonl"],
        )
    output = {}
    for name, (relative, declaration) in declarations.items():
        candidate = ROOT / relative
        if candidate.exists():
            verify_file(candidate, declaration["sha256"], name)
            output[name] = candidate
        else:
            output[name] = workspace / "data" / Path(relative).parent.name / Path(relative).name
    return output


def write_verified_jsonl(path: Path, rows: list[dict], expected: str, writer) -> None:
    """Never replace an existing file; reject reconstructed bytes that drift."""
    if path.exists():
        verify_file(path, expected, path.name)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".reconstruction.tmp")
    if temporary.exists():
        raise ValueError(f"Unexpected previous reconstruction temporary file: {temporary}")
    writer(temporary, rows)
    verify_file(temporary, expected, path.name)
    temporary.replace(path)


def reconstruct_missing_data(paths: dict[str, Path], data_manifest: dict,
                             pilot_manifest: dict, workspace: Path) -> dict:
    """Pinned source downloads reproduce corpus/query bytes, not timestamps."""
    missing = {name for name, path in paths.items() if not path.exists()}
    if not missing:
        return {"reconstructed_files": [], "upstream_downloads": []}
    import pyarrow.parquet as parquet

    prep = load_script("prepare_research_pilot")
    grouped = load_script("prepare_confirmation_study")
    downloads = []
    raw_dir = workspace / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    sources = {source["path"]: source for source in pilot_manifest["upstream_sources"]}
    for source in data_manifest["upstream_sources"]:
        old = sources.get(source["path"])
        if old and (old["revision"], old["sha256"]) != (source["revision"], source["sha256"]):
            raise ValueError("Frozen upstream source declarations disagree")
        sources[source["path"]] = source

    def fetch(relative: str):
        source = sources[relative]
        target = raw_dir / relative.replace("/", "_")
        declared_revision = data_manifest["source_revisions"][source["repository"]]
        if source["revision"] != declared_revision:
            raise ValueError(f"Upstream source revision differs from frozen parent: {relative}")
        if not target.exists():
            print(f"Downloading pinned source {relative}", flush=True)
            downloaded = prep.download_source(source["repository"], source["revision"], relative, target)
            downloads.append(downloaded)
        verify_file(target, source["sha256"], f"upstream {relative}")
        return parquet.read_table(target).to_pylist()

    if "corpus" in missing:
        corpus = []
        for row in fetch("web/tools-00000-of-00001.parquet"):
            corpus.append({"id": row["id"], "text": row["documentation"], "source_domain": prep.source_domain(row["id"])})
        write_verified_jsonl(paths["corpus"], corpus, data_manifest["prepared_files"]["corpus.jsonl"]["sha256"], prep.write_jsonl)
    else:
        corpus = [json.loads(line) for line in paths["corpus"].open()]
    corpus_ids = {row["id"] for row in corpus}
    if len(corpus_ids) != len(corpus) or len(corpus) != data_manifest["corpus_count"]:
        raise ValueError("Reconstructed corpus count or IDs differ from frozen parent")
    if missing - {"corpus"}:
        raw_queries = {
            domain: fetch(f"{domain}/queries-00000-of-00001.parquet")
            for domain in ("apibank", *data_manifest["pilot_domains"])
        }
        domains = {domain: raw_queries[domain] for domain in data_manifest["pilot_domains"]}
        selected, _ = prep.select_queries(domains, raw_queries["apibank"], pilot_manifest["queries_per_domain"], pilot_manifest["seed"])
        training = prep.prepare_queries(selected, corpus_ids)
        write_verified_jsonl(paths["training_queries"], training, pilot_manifest["prepared_files"]["queries.jsonl"]["sha256"], prep.write_jsonl)
        eligible, _ = grouped.eligible_queries(domains, raw_queries["apibank"])
        all_queries = prep.prepare_queries(eligible, corpus_ids)
        splits, _ = grouped.grouped_holdout_split(
            all_queries, training, data_manifest["calibration_per_domain"],
            data_manifest["confirmation_per_domain"], data_manifest["seed"],
        )
        for name in ("development", "calibration", "confirmation"):
            write_verified_jsonl(paths[f"{name}_queries"], splits[name], data_manifest["prepared_files"][f"{name}_queries.jsonl"]["sha256"], prep.write_jsonl)
        # Preserve complete cohort files together when reconstructing. The
        # quarantine/reserve rows remain excluded from fitting and evaluation.
        reconstructed_dir = workspace / "data" / "research_confirmation_20261003"
        for name, rows in splits.items():
            declaration = data_manifest["prepared_files"].get(f"{name}_queries.jsonl")
            if declaration is not None:
                write_verified_jsonl(reconstructed_dir / f"{name}_queries.jsonl", rows, declaration["sha256"], prep.write_jsonl)
        for name in ("development", "calibration", "confirmation"):
            paths[f"{name}_queries"] = reconstructed_dir / f"{name}_queries.jsonl"
    return {
        "reconstructed_files": sorted(missing), "upstream_downloads": downloads,
        "reconstructed_at_utc": datetime.now(timezone.utc).isoformat(),
        "provenance_note": "Original published study provenance remains canonical; this record describes later reconstruction of its verified source-data bytes.",
    }


def run_script(name: str, **arguments) -> None:
    command = [sys.executable, str(ROOT / "scripts" / f"{name}.py")]
    for key, value in arguments.items():
        command.extend(["--" + key.replace("_", "-"), str(value)])
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(ROOT / "src") + (os.pathsep + environment["PYTHONPATH"] if environment.get("PYTHONPATH") else "")
    environment.setdefault("HF_HUB_DISABLE_XET", "1")
    environment.setdefault("TOKENIZERS_PARALLELISM", "false")
    print(f"Running {name}", flush=True)
    subprocess.run(command, cwd=ROOT, env=environment, check=True)


def ensure_embeddings(directory: Path, corpus_path: Path, protocol: dict) -> None:
    """Create absent corpus embeddings only, leaving cached query ranks intact."""
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    import numpy as np
    import torch
    from sentence_transformers import SentenceTransformer

    benchmark = load_script("benchmark_selective_latency")
    from toolret_research.text import flatten_tool

    corpus = [json.loads(line) for line in corpus_path.open()]
    ids = [row["id"] for row in corpus]
    identity = {
        "corpus_sha256": sha256_file(corpus_path),
        "document_ids_sha256": hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
        "model": protocol["dense_model"], "revision": protocol["dense_revision"],
        "max_sequence_length": protocol["max_sequence_length"],
        "serialization": "flatten_tool(id,text) at script/source SHA256",
        "text_source_sha256": sha256_file(ROOT / "src/toolret_research/text.py"),
        "shard_size": 1024, "dtype": "float32", "normalized": True,
    }
    torch.set_num_threads(protocol["threads"])
    torch.set_num_interop_threads(protocol["interop_threads"])
    model = SentenceTransformer(protocol["dense_model"], revision=protocol["dense_revision"], device="cpu")
    model.max_seq_length = protocol["max_sequence_length"]
    dimension = model.get_sentence_embedding_dimension()
    identity_path = directory / "identity.json"
    if identity_path.exists():
        if json.loads(identity_path.read_text()) != identity:
            raise ValueError("Existing embedding identity differs; choose a new --embedding-dir")
    else:
        directory.mkdir(parents=True, exist_ok=True)
        benchmark.atomic_json(identity_path, identity)
    with torch.inference_mode():
        for offset in range(0, len(corpus), identity["shard_size"]):
            path = directory / f"shard_{offset:06d}.npy"
            if path.exists():
                continue
            texts = [flatten_tool({"id": row["id"], "text": row["text"]}) for row in corpus[offset:offset + identity["shard_size"]]]
            values = model.encode(texts, batch_size=protocol["batch_size"], normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False).astype(np.float32)
            temporary = path.with_suffix(".tmp")
            with temporary.open("wb") as stream:
                np.save(stream, values, allow_pickle=False)
            temporary.replace(path)
            print(f"Encoded corpus {min(offset + len(values), len(corpus))}/{len(corpus)}", flush=True)
    benchmark.load_embeddings(directory, corpus_path, ids, protocol, dimension)


def evaluate_arguments(paths: dict[str, Path], calibration_cache: Path,
                       confirmation_cache: Path, router: Path,
                       output: Path) -> dict:
    return {
        "cache": confirmation_cache / "rankings.jsonl", "manifest": confirmation_cache / "manifest.json",
        "queries": paths["confirmation_queries"], "corpus": paths["corpus"],
        "data_manifest": PUBLISHED / "data_manifest.json", "router": router,
        "protocol": PROTOCOL, "development_queries": paths["development_queries"],
        "training_cache": PILOT / "pilot_cache/rankings.jsonl", "training_queries": paths["training_queries"],
        "training_manifest": PILOT / "pilot_cache/manifest.json",
        "calibration_cache": calibration_cache / "rankings.jsonl", "calibration_queries": paths["calibration_queries"],
        "calibration_manifest": calibration_cache / "manifest.json", "output_dir": output,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data" / f"research_reproduction_{timestamp}")
    parser.add_argument("--recompute", action="store_true", help="Regenerate calibration/confirmation caches and refit the router; CPU inference takes substantial time")
    parser.add_argument("--measure-runtime", action="store_true", help="Execute full-cohort conditional CE verification plus the frozen repeated warm-latency benchmark")
    parser.add_argument("--embedding-dir", type=Path, help="Optional existing or new compatible embedding checkpoint directory")
    args = parser.parse_args()
    workspace = args.output_dir.resolve()
    if workspace.exists():
        parser.error("Choose a new --output-dir; reproduction outputs are never overwritten")
    if workspace == ROOT or workspace.is_relative_to(PUBLISHED) or workspace.is_relative_to(PILOT):
        parser.error("Output directory must not be a published evidence directory")
    workspace.mkdir(parents=True)
    # Immutable hashes are checked before any upstream download or inference.
    protocol = json.loads(PROTOCOL.read_text())
    verify_file(PUBLISHED / "data_manifest.json", protocol["split_manifest_sha256"], "canonical split manifest")
    data_manifest = json.loads((PUBLISHED / "data_manifest.json").read_text())
    pilot_manifest = json.loads((PILOT / "data_manifest.json").read_text())
    published_router = PUBLISHED / "router/model.json"
    receipt = json.loads((PUBLISHED / "router_freeze_receipt.json").read_text())
    verify_file(published_router, receipt["router_sha256"], "published router")
    router_artifact = json.loads(published_router.read_text())
    verify_protocol_commit(router_artifact["input_integrity"]["protocol_frozen_commit"], PROTOCOL)
    paths = prepared_paths(data_manifest, pilot_manifest, workspace)
    reconstruction = reconstruct_missing_data(paths, data_manifest, pilot_manifest, workspace)
    (workspace / "data_reconstruction_metadata.json").write_text(json.dumps(reconstruction, indent=2, sort_keys=True) + "\n")
    reconstructed_dir = workspace / "data" / "research_confirmation_20261003"
    if reconstructed_dir.exists():
        # This is a copy of the study's original provenance, not metadata newly
        # generated today. Reconstruction metadata is separate above. Evaluator
        # always receives the committed canonical parent manifest directly.
        (reconstructed_dir / "manifest.json").write_bytes((PUBLISHED / "data_manifest.json").read_bytes())
    calibration_cache, confirmation_cache, router = PUBLISHED / "calibration_cache", PUBLISHED / "confirmation_cache", published_router
    embedding_dir = args.embedding_dir.resolve() if args.embedding_dir else ROOT / "checkpoints/research_20261003"
    if not args.embedding_dir and not embedding_dir.exists():
        embedding_dir = workspace / "embeddings"
    if args.recompute:
        calibration_cache, confirmation_cache = workspace / "calibration_cache", workspace / "confirmation_cache"
        shared = {"corpus": paths["corpus"], "data_manifest": PUBLISHED / "data_manifest.json", "protocol": PROTOCOL, "embedding_dir": embedding_dir}
        run_script("generate_confirmation_cache", split="calibration", queries=paths["calibration_queries"], output_dir=calibration_cache, **shared)
        fitted_dir = workspace / "router"
        run_script("fit_utility_router", train_cache=PILOT / "pilot_cache/rankings.jsonl", train_queries=paths["training_queries"], train_manifest=PILOT / "pilot_cache/manifest.json", calibration_cache=calibration_cache / "rankings.jsonl", calibration_queries=paths["calibration_queries"], calibration_manifest=calibration_cache / "manifest.json", corpus=paths["corpus"], protocol=PROTOCOL, frozen_protocol_commit=router_artifact["input_integrity"]["protocol_frozen_commit"], output_dir=fitted_dir)
        router = fitted_dir / "model.json"
        run_script("generate_confirmation_cache", split="confirmation", queries=paths["confirmation_queries"], output_dir=confirmation_cache, **shared)
    evaluation = workspace / "evaluation"
    run_script("evaluate_confirmation_study", **evaluate_arguments(paths, calibration_cache, confirmation_cache, router, evaluation))
    if args.measure_runtime:
        run_script("verify_confirmation_runtime", cache=confirmation_cache / "rankings.jsonl", manifest=confirmation_cache / "manifest.json", corpus=paths["corpus"], queries=paths["confirmation_queries"], protocol=PROTOCOL, router=router, predictions=evaluation / "frozen_predictions.jsonl", evaluation_summary=evaluation / "summary.json", output=workspace / "runtime_verification.json")
        # Loading/encoding corpus documents is excluded from latency measurement.
        previous = Path.cwd()
        try:
            os.chdir(ROOT)
            sys.path.insert(0, str(ROOT / "src"))
            ensure_embeddings(embedding_dir, paths["corpus"], protocol)
        finally:
            os.chdir(previous)
        run_script("benchmark_selective_latency", cache=confirmation_cache / "rankings.jsonl", manifest=confirmation_cache / "manifest.json", corpus=paths["corpus"], queries=paths["confirmation_queries"], protocol=PROTOCOL, router_model=router, embedding_dir=embedding_dir, output_dir=workspace / "latency")
    provenance = {
        "scope": "Independent reproduction run; published artifacts were not replaced",
        "mode": "recomputed_calibration_confirmation_and_refitted_router" if args.recompute else "published_caches_and_frozen_router",
        "measured_runtime": args.measure_runtime,
        "canonical_data_manifest_sha256": sha256_file(PUBLISHED / "data_manifest.json"),
        "protocol_sha256": sha256_file(PROTOCOL), "router_sha256": sha256_file(router),
        "data_paths": {name: str(path) for name, path in paths.items()},
        "data_sha256": {name: sha256_file(path) for name, path in paths.items()},
        "reconstruction": reconstruction,
        "recomputed_artifact_interpretation": "Pinned source-data bytes must match exactly. Fresh inference timings, package versions, base Git commit and provenance paths may differ; new caches and weights are kept separate and no byte-identical inference or timestamp claim is made.",
        "runner_source_sha256": sha256_file(Path(__file__)),
    }
    (workspace / "reproduction_provenance.json").write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")
    print(f"Reproduction complete: {workspace}", flush=True)


if __name__ == "__main__":
    main()
