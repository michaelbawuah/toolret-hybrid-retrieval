"""Synthetic fixtures test code correctness; no empirical claims use these."""

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from toolret_research.fusion import weighted_reciprocal_rank_fusion
from toolret_research.selective import (
    binary_metrics,
    evaluate_policies,
    file_sha256,
    load_excluded_ids,
    should_rerank,
    validate_manifest,
    validate_rows,
)


@pytest.fixture
def manifest():
    return {
        "schema_version": "toolret-selective-v1", "scope": "synthetic",
        "dataset": {"name": "synthetic", "version": "1", "source": "tests only"},
        "models": {"dense": "synthetic fixture", "reranker": "synthetic fixture"},
        "cache_provenance": {"method": "hand-constructed fixtures"},
        "split": {"name": "synthetic", "selection": "code correctness only"},
        "corpus_sha256": "0" * 64,
        "rrf": {"k": 10, "weights": [1.0, 1.25]},
    }


def row(qid="q1", *, agree=False):
    bm25 = ["a", "b", "c"]
    dense = ["a", "c", "b"] if agree else ["b", "a", "c"]
    hybrid = weighted_reciprocal_rank_fusion([bm25, dense], [1, 1.25], 10)
    return {
        "query_id": qid, "query": "synthetic query", "relevant_ids": ["a"],
        "rankings": {"bm25": bm25, "dense": dense, "hybrid": hybrid,
                     "reranked": list(reversed(hybrid[:2])) + hybrid[2:]},
        "rerank_k": 2,
    }


def test_fixed_gate_and_binary_cutoff():
    assert should_rerank(row())
    assert not should_rerank(row(agree=True))
    values = binary_metrics(["a", "x", "b"], ["a", "b"], k=2)
    assert values["Recall@2"] == 0.5
    assert values["MRR@2"] == 1
    assert values["All-positive-label coverage@2"] == 0
    assert binary_metrics([str(i) for i in range(11)], ["10"])["MRR@10"] == 0


def test_valid_cache_and_rrf_verification(manifest):
    rows = validate_rows([row()], {"a", "b", "c"}, manifest)
    assert rows[0]["query_id"] == "q1"
    broken = row()
    broken["rankings"]["hybrid"] = list(reversed(broken["rankings"]["hybrid"]))
    broken["rankings"]["reranked"] = broken["rankings"]["hybrid"][:]
    with pytest.raises(ValueError, match="weighted RRF"):
        validate_rows([broken], {"a", "b", "c"}, manifest)


@pytest.mark.parametrize("mutation, match", [
    (lambda r: r["rankings"]["bm25"].append("a"), "duplicate"),
    (lambda r: r["rankings"]["dense"].append("unknown"), "missing from corpus"),
    (lambda r: r.update(relevant_ids=[]), "cannot be empty"),
    (lambda r: r.update(relevant_ids=["unknown"]), "gold IDs"),
    (lambda r: r.update(rerank_k=0), "rerank_k"),
    (lambda r: r.update(rerank_k=True), "rerank_k"),
    (lambda r: r["rankings"]["bm25"].clear(), "cannot be empty"),
    (lambda r: r["rankings"].update(reranked=["c", "a", "b"]), "permute only"),
    (lambda r: r.update(timings_ms={"dense": float("nan")}), "finite"),
    (lambda r: r.update(timings_ms={"adaptive": 1}), "unknown components"),
])
def test_invalid_cache_rejected(manifest, mutation, match):
    value = row()
    mutation(value)
    with pytest.raises(ValueError, match=match):
        validate_rows([value], {"a", "b", "c"}, manifest)


def test_duplicate_queries_and_historical_exclusion(manifest):
    with pytest.raises(ValueError, match="duplicate query_id"):
        validate_rows([row(), row()], {"a", "b", "c"}, manifest)
    manifest["scope"] = "preliminary_fresh_pilot"
    with pytest.raises(ValueError, match="historically inspected"):
        validate_rows([row()], {"a", "b", "c"}, manifest, excluded_ids={"q1"})


@pytest.mark.parametrize("field", ["cache", "prepared"])
def test_declared_query_count_cannot_drop_cache_rows(manifest, field):
    if field == "cache":
        manifest["cache_provenance"]["queries"] = 2
    else:
        manifest["data_preparation_manifest"] = {"query_count": 2}
    with pytest.raises(ValueError, match="query count"):
        validate_rows([row()], {"a", "b", "c"}, manifest)


def test_exact_prepared_query_alignment_rejects_substitution_and_gold_changes(manifest):
    expected = [{"id": "q1", "query": "synthetic query", "relevant_ids": ["a"]}]
    assert len(validate_rows([row()], {"a", "b", "c"}, manifest, expected_queries=expected)) == 1
    with pytest.raises(ValueError, match="exact prepared query set"):
        validate_rows([row("different")], {"a", "b", "c"}, manifest, expected_queries=expected)
    altered = row()
    altered["relevant_ids"] = ["b"]
    with pytest.raises(ValueError, match="gold IDs differ"):
        validate_rows([altered], {"a", "b", "c"}, manifest, expected_queries=expected)


def test_prepared_source_domain_counts_and_labels(manifest):
    manifest["data_preparation_manifest"] = {"pilot_domains": ["synthetic_domain"], "queries_per_domain": 1}
    value = row()
    with pytest.raises(ValueError, match="source_domain counts"):
        validate_rows([value], {"a", "b", "c"}, manifest)
    value["source_domain"] = "synthetic_domain"
    expected = [{"id": "q1", "query": "synthetic query", "relevant_ids": ["a"], "domain": "different"}]
    with pytest.raises(ValueError, match="source_domain differs"):
        validate_rows([value], {"a", "b", "c"}, manifest, expected_queries=expected)


def test_untouched_tail_is_preserved(manifest):
    value = row()
    value["rankings"]["hybrid"].append("d")
    value["rankings"]["reranked"].append("d")
    value["rankings"]["reranked"][-2:] = reversed(value["rankings"]["reranked"][-2:])
    with pytest.raises(ValueError, match="exact tail"):
        validate_rows([value], {"a", "b", "c", "d"}, manifest)


def test_policy_counts_reproducibility_and_no_false_latency(manifest):
    rows = validate_rows([row("q2", agree=True), row("q1")], {"a", "b", "c"}, manifest)
    first = evaluate_policies(rows, manifest, bootstrap_resamples=100)
    second = evaluate_policies(list(reversed(rows)), manifest, bootstrap_resamples=100)
    assert first == second
    summary, records, decisions = first
    assert summary["policies"]["gated"]["reranker_invocations"] == 1
    assert summary["policies"]["always"]["reranker_invocations"] == 2
    assert summary["gated_candidate_pair_reduction_fraction_vs_always"] == 0.5
    assert len(summary["random_equal_invocation_baseline"]["seeds"]) == 20
    for name, policy in summary["policies"].items():
        if name.startswith("random_seed_"):
            assert policy["reranker_invocations"] == 1
    assert summary["adaptive_end_to_end_latency_ms"] is None
    assert any("Synthetic" in limitation for limitation in summary["limitations"])
    assert len(records) == len(decisions) == 2


def test_declared_random_seeds_respected_with_separate_bootstrap_seed(manifest):
    manifest["cache_provenance"]["protocol"] = {"random_policy_seeds": list(range(20))}
    rows = validate_rows([row("q2", agree=True), row("q1")], {"a", "b", "c"}, manifest)
    first = evaluate_policies(rows, manifest, seed=1000, bootstrap_resamples=10)
    again = evaluate_policies(list(reversed(rows)), manifest, seed=1000, bootstrap_resamples=10)
    assert first == again
    summary, _, decisions = first
    baseline = summary["random_equal_invocation_baseline"]
    assert baseline["seeds"] == list(range(20))
    assert baseline["seed_provenance"] == "manifest_protocol"
    assert "random_seed_0" in summary["policies"]
    assert "random_seed_1000" not in summary["policies"]
    for random_seed in range(20):
        assert summary["policies"][f"random_seed_{random_seed}"]["reranker_invocations"] == 1
    other_bootstrap = evaluate_policies(rows, manifest, seed=9999, bootstrap_resamples=10)
    assert other_bootstrap[2] == decisions
    assert summary["paired_comparisons"]["gated_minus_hybrid"]["nDCG@10"]["paired_bootstrap"]["seed"] == 1000


@pytest.mark.parametrize("seeds", [list(range(19)), [0] * 20, [True] + list(range(1, 20)), [0.0] + list(range(1, 20))])
def test_invalid_declared_random_seeds_rejected(manifest, seeds):
    manifest["cache_provenance"]["protocol"] = {"random_policy_seeds": seeds}
    with pytest.raises(ValueError, match="20 unique integer seeds"):
        validate_manifest(manifest)


def test_component_timings_are_never_summed(manifest):
    value = row()
    value["timings_ms"] = {"bm25": 3, "dense": 2, "hybrid": 1, "reranked": 10}
    rows = validate_rows([value], {"a", "b", "c"}, manifest)
    summary, _, _ = evaluate_policies(rows, manifest, bootstrap_resamples=10)
    assert summary["supplied_component_timings_ms"]["reranked"]["mean_ms"] == 10
    assert summary["adaptive_end_to_end_latency_ms"] is None
    assert summary["estimated_offline_replay_component_sums_ms"] == {}


def test_estimates_require_disjoint_timings_and_exclude_gate_overhead(manifest):
    value = row(agree=True)
    value["timings_ms"] = {"bm25": 3, "dense": 2, "hybrid": 1, "reranked": 10}
    manifest["timing_provenance"] = {"disjoint_components": True, "boundary": "synthetic"}
    rows = validate_rows([value], {"a", "b", "c"}, manifest)
    summary, _, _ = evaluate_policies(rows, manifest, bootstrap_resamples=10)
    estimates = summary["estimated_offline_replay_component_sums_ms"]
    assert estimates["always"]["mean_ms"] == 16
    assert estimates["gated"]["mean_ms"] == 6
    assert summary["adaptive_end_to_end_latency_ms"] is None


def test_query_text_optional_and_source_domain_breakdown(manifest):
    value = row()
    del value["query"]
    value["source_domain"] = "synthetic_domain"
    rows = validate_rows([value], {"a", "b", "c"}, manifest)
    summary, records, _ = evaluate_policies(rows, manifest, bootstrap_resamples=10)
    assert records[0]["query"] is None
    assert summary["source_domain_breakdown"]["synthetic_domain"]["num_queries"] == 1


def test_manifest_rejects_invalid_rrf(manifest):
    for bad in ([-1, 1], [float("nan"), 1], [0, 0], [1]):
        changed = deepcopy(manifest)
        changed["rrf"]["weights"] = bad
        with pytest.raises(ValueError, match="weights"):
            validate_manifest(changed)


def test_historical_csv_and_json_ids(tmp_path):
    csv_path = tmp_path / "history.csv"
    csv_path.write_text("query_id,issue\nq1,error\nq1,another\nq2,error\n")
    assert load_excluded_ids(csv_path) == {"q1", "q2"}
    json_path = tmp_path / "history.json"
    json_path.write_text('["q1", "q2"]')
    assert load_excluded_ids(json_path) == {"q1", "q2"}


def test_cli_synthetic_artifacts_provenance_and_no_overwrite(tmp_path, manifest):
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text(''.join(json.dumps({"id": tool}) + "\n" for tool in ("a", "b", "c")))
    cache = tmp_path / "cache.jsonl"
    cache.write_text(json.dumps(row()) + "\n")
    manifest["corpus_sha256"] = file_sha256(corpus)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    output = tmp_path / "output"
    repo = Path(__file__).resolve().parents[1]
    env = dict(os.environ, PYTHONPATH=str(repo / "src"))
    command = [sys.executable, str(repo / "scripts/evaluate_selective_reranking.py"),
               "--cache", str(cache), "--corpus", str(corpus), "--manifest", str(manifest_path),
               "--output-dir", str(output), "--bootstrap-resamples", "10"]
    result = subprocess.run(command, capture_output=True, text=True, env=env)
    assert result.returncode == 0, result.stderr
    assert {path.name for path in output.iterdir()} == {"summary.json", "per_query.jsonl", "decisions.jsonl", "report.md"}
    summary = json.loads((output / "summary.json").read_text())
    assert summary["scope"] == "synthetic"
    assert summary["input_integrity"]["cache_sha256"] == file_sha256(cache)
    again = subprocess.run(command, capture_output=True, text=True, env=env)
    assert again.returncode != 0
    assert "empty or new" in again.stderr


def test_cli_fresh_pilot_requires_historical_ids(tmp_path, manifest):
    manifest["scope"] = "preliminary_fresh_pilot"
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    repo = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, str(repo / "scripts/evaluate_selective_reranking.py"),
         "--cache", "unused", "--corpus", "unused", "--manifest", str(manifest_path),
         "--output-dir", str(tmp_path / "output")],
        capture_output=True, text=True, env=dict(os.environ, PYTHONPATH=str(repo / "src")),
    )
    assert result.returncode != 0
    assert "--exclude-queries is required" in result.stderr
