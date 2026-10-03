"""Check scientific sampling, actual-cache alignment and latency clustering."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

SPEC = importlib.util.spec_from_file_location("latency_benchmark", Path(__file__).parents[1] / "scripts/benchmark_selective_latency.py")
latency = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(latency)


def test_hash_selection_balanced_invariant_to_results_and_input_order():
    rows = [{"query_id": f"{source}_{index}", "source_domain": source,
             "relevant_ids": ["a"], "timings_ms": {"reranked": index}}
            for source in ("a", "b", "c") for index in range(7)]
    selected = latency.select_latency_rows(rows, 3)
    altered = [{**row, "relevant_ids": ["bogus"], "timings_ms": {"reranked": -1}}
               for row in reversed(rows)]
    assert [row["query_id"] for row in selected] == [row["query_id"] for row in latency.select_latency_rows(altered, 3)]
    assert len(selected) == 9
    assert {source: sum(row["source_domain"] == source for row in selected) for source in ("a", "b", "c")} == {"a": 3, "b": 3, "c": 3}


@pytest.mark.parametrize("mutation", ["duplicate", "insufficient", "missing_domain"])
def test_selection_rejects_invalid_cohorts(mutation):
    rows = [{"query_id": "q", "source_domain": "a"}, {"query_id": "r", "source_domain": "a"}]
    count = 1
    if mutation == "duplicate":
        rows.append(rows[0])
    elif mutation == "insufficient":
        count = 3
    else:
        rows[0].pop("source_domain")
    with pytest.raises(ValueError):
        latency.select_latency_rows(rows, count)


def test_schedule_balances_positions_and_measures_every_pair_three_times():
    queries, policies = [f"q{index}" for index in range(90)], ["hybrid", "always", "fixed_disagreement", "utility_75"]
    schedule = latency.counterbalanced_schedule(queries, policies, 3, 20261003)
    assert schedule == latency.counterbalanced_schedule(queries, policies, 3, 20261003)
    assert schedule != latency.counterbalanced_schedule(queries, policies, 3, 20261004)
    assert len(schedule) == 1080
    for repetition in range(3):
        subset = [row for row in schedule if row["repetition"] == repetition]
        for policy in policies:
            counts = [sum(row["policy"] == policy and row["policy_position"] == position for row in subset) for position in range(4)]
            assert max(counts) - min(counts) <= 1
        assert {(row["query_id"], row["policy"]) for row in subset} == {(query, policy) for query in queries for policy in policies}


def measured_events():
    events = []
    for index in range(6):
        for repetition in range(3):
            for policy, offset in (("always", 0), ("utility_75", -10)):
                duration = 100 + index * 10 + repetition * 2 + offset
                events.append({"query_id": f"q{index}", "source_domain": "a" if index < 3 else "b",
                               "repetition": repetition, "policy": policy,
                               "total_ms": duration, "reranked": policy == "always" or index % 2 == 0,
                               "scored_pairs": 20 if policy == "always" or index % 2 == 0 else 0,
                               "components_ms": {"bm25": 5, "dense": 10, "rrf": 1, "routing": 0.1, "cross_encoder": duration - 16.1}})
    return events


def test_paired_bootstrap_clusters_query_repetitions_before_comparison():
    summary = latency.summarize_events(measured_events(), ["always", "utility_75"], 3,
                                       bootstrap_resamples=200, bootstrap_seed=1)
    assert summary["bootstrap"]["independent_units"] == 6
    assert summary["bootstrap"]["repetitions_per_unit"] == 3
    assert summary["source_domain_counts"] == {"a": 3, "b": 3}
    assert summary["paired_comparisons"]["utility_75_vs_always"]["delta_ci95_ms"] == [-10, -10]
    assert summary["policies"]["always"]["actual_cross_encoder_calls"] == 18
    assert summary["policies"]["utility_75"]["actual_cross_encoder_calls"] == 9
    assert summary["policies"]["utility_75"]["actual_cross_encoder_pairs"] == 180
    assert summary["policies"]["always"]["mean_ms"] == 127


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "changed_route", "changed_domain", "nonfinite"])
def test_latency_summary_refuses_incomplete_or_inconsistent_measurements(mutation):
    events = measured_events()
    if mutation == "missing":
        events.pop()
    elif mutation == "duplicate":
        events.append(events[-1])
    elif mutation == "changed_route":
        events[0]["reranked"] = False
    elif mutation == "changed_domain":
        events[0]["source_domain"] = "c"
    else:
        events[0]["total_ms"] = float("nan")
    with pytest.raises(ValueError):
        latency.summarize_events(events, ["always", "utility_75"], 3, bootstrap_resamples=20)


def test_embedding_cache_checks_all_rows_and_exact_fingerprint(tmp_path, monkeypatch):
    monkeypatch.chdir(Path(__file__).parents[1])
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text("corpus")
    ids = ["a", "b"]
    config = {"dense_model": "test", "dense_revision": "pin", "max_sequence_length": 256}
    identity = {
        "corpus_sha256": latency.sha256_file(corpus),
        "document_ids_sha256": latency.hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
        "model": "test", "revision": "pin", "max_sequence_length": 256,
        "serialization": "flatten_tool(id,text) at script/source SHA256",
        "text_source_sha256": latency.sha256_file("src/toolret_research/text.py"),
        "shard_size": 1024, "dtype": "float32", "normalized": True,
    }
    (tmp_path / "identity.json").write_text(json.dumps(identity))
    np.save(tmp_path / "shard_000000.npy", np.array([[1, 0], [0, 1]], dtype=np.float32))
    values, provenance = latency.load_embeddings(tmp_path, corpus, ids, config, 2)
    assert values.shape == (2, 2)
    assert provenance["all_unit_norm_atol_1e-5"]
    assert len(provenance["shards"][0]["sha256"]) == 64
    np.save(tmp_path / "shard_000000.npy", np.array([[1, 0], [0, 2]], dtype=np.float32))
    with pytest.raises(ValueError, match="Non-unit"):
        latency.load_embeddings(tmp_path, corpus, ids, config, 2)
    np.save(tmp_path / "shard_000000.npy", np.array([[1, 0], [0, 1]], dtype=np.float32))
    with pytest.raises(ValueError, match="identity"):
        latency.load_embeddings(tmp_path, corpus, list(reversed(ids)), config, 2)


def test_utility_adapter_only_passes_query_and_cheap_rankings(tmp_path):
    from toolret_research.utility_router import RidgeUtilityModel
    model = RidgeUtilityModel((0,) * 7, (1,) * 7, (1,) + (0,) * 6, 0, 10, 300)
    path = tmp_path / "router.json"
    path.write_text(json.dumps({"model": model.to_dict(), "thresholds": {"0.75": {"threshold": 0.5}}}))
    adapter = latency.UtilityAdapter(path)
    assert adapter.decision("some query", ["a", "b"], ["b", "a"], ["a", "b"], 0.75)
    assert not adapter.decision("some query", ["a", "b"], ["a", "b"], ["a", "b"], 0.75)


@pytest.mark.parametrize("mutation", [None, "hash", "protocol", "model", "source", "cohort", "gold", "domain", "rrf", "prefix"])
def test_inputs_reject_stale_or_misaligned_cache_and_provenance(tmp_path, monkeypatch, mutation):
    monkeypatch.chdir(Path(__file__).parents[1])
    config = json.loads(Path("configs/selective_pilot_20261003.json").read_text())
    corpus = {"a": {"id": "a", "text": "a"}, "b": {"id": "b", "text": "b"}, "c": {"id": "c", "text": "c"}}
    query_rows = [{"id": "q", "query": "some query", "domain": "apigen", "labels": [{"id": "b", "relevance": 1}, {"id": "c", "relevance": 0}]}]
    rows = [{"query_id": "q", "source_domain": "apigen", "relevant_ids": ["b"], "rerank_k": 2,
             "rankings": {"bm25": ["a", "b"], "dense": ["b", "a"], "hybrid": ["a", "b"], "reranked": ["b", "a"]}}]
    corpus_path, query_path, cache_path = [tmp_path / name for name in ("corpus.jsonl", "queries.jsonl", "rankings.jsonl")]
    corpus_path.write_text("\n".join(json.dumps(row) for row in corpus.values()))
    query_path.write_text(json.dumps(query_rows[0]))
    cache_path.write_text(json.dumps(rows[0]))
    manifest = {
        "corpus_sha256": latency.sha256_file(corpus_path), "queries_sha256": latency.sha256_file(query_path), "cache_sha256": latency.sha256_file(cache_path),
        "models": {"dense": {"name": config["dense_model"], "revision": config["dense_revision"]}, "reranker": {"name": config["reranker_model"], "revision": config["reranker_revision"]}},
        "cache_provenance": {"protocol": dict(config), "source_sha256": {path: latency.sha256_file(path) for path in ("src/toolret_research/text.py", "src/toolret_research/bm25.py", "src/toolret_research/fusion.py")}},
    }
    if mutation == "hash":
        cache_path.write_text("changed")
    elif mutation == "protocol":
        manifest["cache_provenance"]["protocol"]["candidate_k"] = 1
    elif mutation == "model":
        manifest["models"]["dense"]["revision"] = "wrong"
    elif mutation == "source":
        manifest["cache_provenance"]["source_sha256"]["src/toolret_research/text.py"] = "wrong"
    elif mutation == "cohort":
        query_rows[0]["id"] = "other"
    elif mutation == "gold":
        rows[0]["relevant_ids"] = ["c"]
    elif mutation == "domain":
        rows[0]["source_domain"] = "toolbench"
    elif mutation == "rrf":
        rows[0]["rankings"]["hybrid"] = ["b", "a"]
    elif mutation == "prefix":
        rows[0]["rankings"]["reranked"] = ["b", "c"]
    if mutation is None:
        latency.validate_inputs(cache_path, manifest, corpus_path, query_path, config, rows, corpus, query_rows)
    else:
        with pytest.raises(ValueError):
            latency.validate_inputs(cache_path, manifest, corpus_path, query_path, config, rows, corpus, query_rows)


@pytest.mark.parametrize("mutation", [None, "subset", "policy", "repetitions", "order", "bootstrap", "cohort"])
def test_benchmark_refuses_departures_from_frozen_protocol(tmp_path, mutation):
    query_path = tmp_path / "queries.jsonl"
    query_path.write_text("untouched confirmation")
    selected = [{"query_id": f"{source}_{index}", "source_domain": source} for source in ("a", "b", "c") for index in range(30)]
    policies = ["hybrid", "always", "fixed_disagreement", "utility_75"]
    protocol = {
        "scope": "confirmation_study", "query_sources": ["a", "b", "c"],
        "confirmation": {"queries_sha256": latency.sha256_file(query_path)},
        "latency_benchmark": {"queries": 90, "queries_per_source": 30,
                              "selection_seed": latency.SELECTION_SEED, "repetitions": 3,
                              "policies": policies, "order_seed": 20261003,
                              "bootstrap_seed": 20261003, "bootstrap_resamples": 10000},
    }
    arguments = dict(queries_path=query_path, per_domain=30, repetitions=3,
                     policies=list(policies), order_seed=20261003, bootstrap_resamples=10000)
    if mutation == "subset":
        selected.pop()
    elif mutation == "policy":
        arguments["policies"].append("utility_50")
    elif mutation == "repetitions":
        arguments["repetitions"] = 2
    elif mutation == "order":
        arguments["order_seed"] = 1
    elif mutation == "bootstrap":
        arguments["bootstrap_resamples"] = 100
    elif mutation == "cohort":
        query_path.write_text("calibration, not confirmation")
    if mutation is None:
        latency.validate_benchmark_configuration(protocol, selected, **arguments)
    else:
        with pytest.raises(ValueError):
            latency.validate_benchmark_configuration(protocol, selected, **arguments)


@pytest.mark.parametrize("mutation", [None, "protocol", "corpus", "source", "alpha", "count", "confirmation_fit", "calibration_fit", "threshold", "budget"])
def test_router_provenance_and_threshold_are_frozen(tmp_path, monkeypatch, mutation):
    from toolret_research.utility_router import RidgeUtilityModel
    root = Path(__file__).parents[1]
    protocol_path = root / "configs/confirmation_protocol_20261003.json"
    protocol = json.loads(protocol_path.read_text())
    corpus_path = tmp_path / "corpus.jsonl"
    corpus_path.write_text("corpus")
    model = RidgeUtilityModel((0,) * 7, (1,) * 7, (1,) + (0,) * 6, 0, 10, 300)
    artifact = {
        "schema_version": "toolret-frozen-utility-router-v1", "model": model.to_dict(),
        "source_sha256": {"src/toolret_research/utility_router.py": latency.sha256_file(root / "src/toolret_research/utility_router.py")},
        "input_integrity": {"protocol_sha256": latency.sha256_file(protocol_path), "corpus_sha256": latency.sha256_file(corpus_path),
                            "protocol_frozen_commit": "a" * 40, "training": {"queries": 300}, "calibration": {"queries": 300}},
        "confirmation_inputs_accessed": False, "calibration_labels_used_for_fit_or_thresholds": False,
        "thresholds": {"0.75": {"threshold": 0.5, "nominal_budget_fraction": .75, "decision_rule": "predicted_ndcg_delta > threshold"}},
    }
    if mutation == "protocol":
        artifact["input_integrity"]["protocol_sha256"] = "wrong"
    elif mutation == "corpus":
        artifact["input_integrity"]["corpus_sha256"] = "wrong"
    elif mutation == "source":
        artifact["source_sha256"]["src/toolret_research/utility_router.py"] = "wrong"
    elif mutation == "alpha":
        artifact["model"]["alpha"] = 1
    elif mutation == "count":
        artifact["input_integrity"]["calibration"]["queries"] = 1
    elif mutation == "confirmation_fit":
        artifact["confirmation_inputs_accessed"] = True
    elif mutation == "calibration_fit":
        artifact["calibration_labels_used_for_fit_or_thresholds"] = True
    elif mutation == "threshold":
        artifact["thresholds"]["0.75"]["threshold"] = float("nan")
    elif mutation == "budget":
        artifact["thresholds"]["0.75"]["nominal_budget_fraction"] = .5
    path = tmp_path / "model.json"
    path.write_text(json.dumps(artifact))
    monkeypatch.setattr(latency.subprocess, "check_output", lambda *args, **kwargs: protocol_path.read_bytes())
    adapter = latency.UtilityAdapter(path)
    if mutation is None:
        adapter.validate(protocol_path, corpus_path, protocol, [.75])
    else:
        with pytest.raises(ValueError):
            adapter.validate(protocol_path, corpus_path, protocol, [.75])
