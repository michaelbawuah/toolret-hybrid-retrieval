"""Reproduction must reconstruct exact frozen bytes and preserve prior evidence."""

import importlib.util
import json
from pathlib import Path
import shutil
import subprocess

import pytest


def load(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).parents[1] / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


runner = load("run_confirmation_study")
prep = load("prepare_research_pilot")
grouped = load("prepare_confirmation_study")


def declaration(path):
    return {"sha256": runner.sha256_file(path), "bytes": path.stat().st_size}


def test_existing_data_is_reused_only_when_exact_and_modified_files_fail(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    corpus = tmp_path / "data/research_20261003/corpus.jsonl"
    corpus.parent.mkdir(parents=True)
    corpus.write_text("frozen corpus\n")
    frozen = declaration(corpus)
    data = {"prepared_files": {name: frozen for name in ("corpus.jsonl", "development_queries.jsonl", "calibration_queries.jsonl", "confirmation_queries.jsonl")}}
    pilot = {"prepared_files": {"queries.jsonl": frozen}}
    paths = runner.prepared_paths(data, pilot, tmp_path / "reproduction")
    assert paths["corpus"] == corpus
    assert paths["confirmation_queries"].is_relative_to(tmp_path / "reproduction")
    corpus.write_text("modified corpus\n")
    with pytest.raises(ValueError, match="differs from the frozen"):
        runner.prepared_paths(data, pilot, tmp_path / "reproduction")


def test_verified_writer_preserves_existing_bytes_and_refuses_drift(tmp_path):
    path = tmp_path / "output.jsonl"
    prep.write_jsonl(path, [{"id": "frozen"}])
    digest = runner.sha256_file(path)
    timestamp = path.stat().st_mtime_ns
    runner.write_verified_jsonl(path, [{"id": "different"}], digest, prep.write_jsonl)
    assert path.stat().st_mtime_ns == timestamp
    assert json.loads(path.read_text())["id"] == "frozen"
    with pytest.raises(ValueError, match="differs from the frozen"):
        runner.write_verified_jsonl(tmp_path / "drift.jsonl", [{"id": "wrong"}], digest, prep.write_jsonl)
    assert not (tmp_path / "drift.jsonl").exists()


def test_frozen_git_protocol_preflight_rejects_drift_and_gives_fetch_guidance(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    protocol = tmp_path / "protocol.json"
    protocol.write_bytes(b'{"frozen": true}\n')
    monkeypatch.setattr(runner.subprocess, "check_output", lambda *args, **kwargs: protocol.read_bytes())
    runner.verify_protocol_commit("a" * 40, protocol)
    monkeypatch.setattr(runner.subprocess, "check_output", lambda *args, **kwargs: b"different")
    with pytest.raises(ValueError, match="differs from its original frozen"):
        runner.verify_protocol_commit("a" * 40, protocol)

    def missing(*args, **kwargs):
        raise subprocess.CalledProcessError(128, args[0])

    monkeypatch.setattr(runner.subprocess, "check_output", missing)
    with pytest.raises(ValueError, match="git fetch origin " + "a" * 40):
        runner.verify_protocol_commit("a" * 40, protocol)


def test_complete_absent_data_reconstruction_from_pinned_parquets(tmp_path, monkeypatch):
    parquet = pytest.importorskip("pyarrow.parquet")
    arrow = pytest.importorskip("pyarrow")
    raw = tmp_path / "fixture_sources"
    raw.mkdir()
    domains, corpus_raw = {}, []
    for domain in ("apigen", "toolbench", "toolace"):
        domains[domain] = []
        for index in range(3):
            tool = f"{domain}_tool_{index}"
            domains[domain].append({"id": f"{domain}_query_{index}", "query": f"{domain} question {index}", "instruction": "", "category": "web", "labels": json.dumps([{"id": tool, "relevance": 1}])})
            corpus_raw.append({"id": tool, "documentation": json.dumps({"name": tool, "description": "fixture"})})
    excluded = [{"id": "apibank_query_1", "query": "Old excluded question", "category": "web", "instruction": "", "labels": "[]"}]
    records = {f"{domain}/queries-00000-of-00001.parquet": values for domain, values in {**domains, "apibank": excluded}.items()}
    records["web/tools-00000-of-00001.parquet"] = corpus_raw
    sources, by_relative = [], {}
    for relative, values in records.items():
        path = raw / relative.replace("/", "_")
        parquet.write_table(arrow.Table.from_pylist(values), path)
        repo = prep.TOOLS_REPO if relative.startswith("web/") else prep.QUERY_REPO
        revision = prep.TOOLS_REVISION if relative.startswith("web/") else prep.QUERY_REVISION
        sources.append({"path": relative, "repository": repo, "revision": revision, **declaration(path)})
        by_relative[relative] = path
    corpus = [{"id": item["id"], "text": item["documentation"], "source_domain": prep.source_domain(item["id"])} for item in corpus_raw]
    selected, _ = prep.select_queries(domains, excluded, 1, "pilot-seed")
    training = prep.prepare_queries(selected, {row["id"] for row in corpus})
    eligible, _ = grouped.eligible_queries(domains, excluded)
    all_queries = prep.prepare_queries(eligible, {row["id"] for row in corpus})
    splits, _ = grouped.grouped_holdout_split(all_queries, training, 1, 1, "confirmation-seed")
    expected_dir = tmp_path / "expected"
    expected_dir.mkdir()
    declarations = {}
    for filename, rows in [("corpus.jsonl", corpus), ("training_queries.jsonl", training), *[(f"{name}_queries.jsonl", splits[name]) for name in ("development", "calibration", "confirmation")]]:
        path = expected_dir / filename
        prep.write_jsonl(path, rows)
        declarations[filename] = declaration(path)
    data_manifest = {
        "prepared_files": {key: value for key, value in declarations.items() if key != "training_queries.jsonl"},
        "upstream_sources": [source for source in sources if source["repository"] == prep.QUERY_REPO],
        "source_revisions": {prep.QUERY_REPO: prep.QUERY_REVISION, prep.TOOLS_REPO: prep.TOOLS_REVISION},
        "pilot_domains": ["apigen", "toolbench", "toolace"], "corpus_count": 9,
        "calibration_per_domain": 1, "confirmation_per_domain": 1, "seed": "confirmation-seed",
    }
    pilot_manifest = {"prepared_files": {"queries.jsonl": declarations["training_queries.jsonl"]}, "upstream_sources": sources, "queries_per_domain": 1, "seed": "pilot-seed"}
    workspace = tmp_path / "run"
    paths = {"corpus": workspace / "corpus.jsonl", "training_queries": workspace / "training_queries.jsonl", **{f"{name}_queries": workspace / f"{name}_queries.jsonl" for name in ("development", "calibration", "confirmation")}}
    downloads = []

    def fake_download(repo, revision, relative, target):
        downloads.append((repo, revision, relative))
        shutil.copyfile(by_relative[relative], target)
        return {"repository": repo, "revision": revision, "path": relative, **declaration(target)}

    monkeypatch.setattr(prep, "download_source", fake_download)
    monkeypatch.setattr(runner, "load_script", lambda name: prep if name == "prepare_research_pilot" else grouped)
    outcome = runner.reconstruct_missing_data(paths, data_manifest, pilot_manifest, workspace)
    assert len(downloads) == 5
    assert set(outcome["reconstructed_files"]) == set(paths)
    for name, path in paths.items():
        assert path.read_bytes() == (expected_dir / ("corpus.jsonl" if name == "corpus" else name + ".jsonl")).read_bytes()
    # A second pass downloads nothing and does not rewrite reconstructed files.
    assert runner.reconstruct_missing_data(paths, data_manifest, pilot_manifest, workspace)["reconstructed_files"] == []
    assert len(downloads) == 5
