"""Meaningful independent-auditor checks on hand-computable and tampered cases."""
from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location("confirmation_auditor", Path(__file__).parents[1] / "scripts/audit_confirmation_study.py")
audit_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit_module)


def row(qid, tools, domain="apigen"):
    return {"id": qid, "relevant_ids": tools, "domain": domain}


def test_bipartite_components_reconstruct_transitive_global_minimum():
    rows = [row("z", ["a"]), row("m", ["a", "b"]), row("d", ["b"]), row("q", ["c"])]
    assert audit_module.components(rows) == {"d": ["d", "m", "z"], "q": ["q"]}


def test_components_reject_duplicate_query_identity():
    with pytest.raises(ValueError, match="duplicate query"):
        audit_module.components([row("q", ["a"]), row("q", ["b"])])


def test_whole_component_anchor_quarantines_untrained_neighbors():
    rows = [row("pilot", ["a"]), row("neighbor", ["a", "b"]), row("transitive", ["b"])]
    rows.extend(row(f"{source}_{i}", [f"tool_{source}_{i}"], source) for source in audit_module.DOMAINS for i in range(3))
    splits, graph = audit_module.assigned_splits(rows, {"pilot"}, "fixed", 1, 1)
    assert splits["development"] == {"pilot"}
    assert splits["quarantined_pilot_connected"] == {"neighbor", "transitive"}
    assert len(splits["calibration"]) == len(splits["confirmation"]) == len(splits["reserve"]) == 3
    assert set.union(*splits.values()) == {r["id"] for r in rows}


def test_metrics_match_hand_calculation_for_multiple_positives():
    values = audit_module.metric_values(["a", "x", "b", "y"], {"a", "b", "c"})
    assert values["nDCG@10"] == pytest.approx(1.5 / (1 + 1 / math.log2(3) + .5))
    assert values["Recall@10"] == pytest.approx(2 / 3)
    assert values["MRR@10"] == 1
    assert values["All-positive-label coverage@10"] == 0


def test_metrics_cutoff_excludes_rank_eleven():
    values = audit_module.metric_values([str(i) for i in range(11)], {"10"})
    assert all(value == 0 for value in values.values())


def test_feature_values_invariant_under_arbitrary_consistent_tool_renaming():
    sparse = [str(i) for i in range(100)]
    dense = sparse[5:] + sparse[:5]
    a = audit_module.independent_features("one two three", sparse, dense)
    mapping = {tool: f"replacement_{1000 - int(tool)}" for tool in sparse}
    b = audit_module.independent_features("one two three", [mapping[t] for t in sparse], [mapping[t] for t in dense])
    np.testing.assert_allclose(a, b, rtol=0, atol=0)
    assert a[0] == 1 and a[5] == math.log(4)


def test_augmented_ridge_recovers_intercept_with_constant_features():
    x = np.full((4, 7), 9.)
    model = audit_module.independent_ridge(x, np.array([.1, .2, -.1, .4]))
    assert model["intercept"] == pytest.approx(.15)
    np.testing.assert_allclose(model["scales"], np.ones(7))
    np.testing.assert_allclose(model["coefficients"], np.zeros(7), atol=1e-14)


def test_paired_query_and_component_intervals_retain_constant_difference():
    rows = [{"source_domain": source, "component_id": f"{source}_{i // 2}"} for source in audit_module.DOMAINS for i in range(4)]
    result = audit_module.paired_intervals(rows, {"constant": np.full(12, -.02)}, 500, 11)["constant"]
    assert result["delta_mean"] == pytest.approx(-.02)
    np.testing.assert_allclose(result["query_ci95"], [-.02, -.02])
    np.testing.assert_allclose(result["component_ci95"], [-.02, -.02])


def test_cache_auditor_detects_tampered_rrf_and_tail(tmp_path):
    ids = [f"tool_{i:03d}" for i in range(100)]
    query = {"id": "q", "domain": "apigen", "component_id": "q", "relevant_ids": [ids[0]]}
    ranks = {"bm25": ids, "dense": ids, "hybrid": ids, "reranked": ids}
    cache_row = {"query_id": "q", "source_domain": "apigen", "component_id": "q", "relevant_ids": [ids[0]], "rerank_k": 20, "rankings": ranks}
    cache = tmp_path / "rankings.jsonl"
    manifest = tmp_path / "manifest.json"
    cache.write_text(json.dumps(cache_row) + "\n")
    manifest.write_text(json.dumps({"scope": "synthetic", "cache_sha256": audit_module.digest(cache)}))
    auditor = audit_module.Audit()
    auditor.cache(cache, manifest, [query], set(ids), tmp_path / "protocol.json")
    assert all(check["status"] == "pass" for check in auditor.checks)
    cache_row["rankings"]["hybrid"] = ids[:1] + ids[2:3] + ids[1:2] + ids[3:]
    cache.write_text(json.dumps(cache_row) + "\n")
    manifest.write_text(json.dumps({"scope": "synthetic", "cache_sha256": audit_module.digest(cache)}))
    auditor = audit_module.Audit()
    auditor.cache(cache, manifest, [query], set(ids), tmp_path / "protocol.json")
    assert any(check["status"] == "fail" and check["name"].startswith("cache_ranks") for check in auditor.checks)


def test_auditor_accepts_saved_evaluator_schema_and_rejects_metric_tampering(tmp_path):
    # Only this interoperability test imports the production evaluator. The
    # auditor itself calculates all scores, decisions and CIs independently.
    from toolret_research.confirmation import evaluate_confirmation, freeze_predictions
    from toolret_research.utility_router import RidgeUtilityModel

    ids = [f"tool_{i:03d}" for i in range(100)]
    model = RidgeUtilityModel(tuple(np.zeros(7)), tuple(np.ones(7)), (0., 0., 0., 0., 0., .1, 0.), 0., 10., 300)
    artifact = {"schema_version": "toolret-frozen-utility-router-v1", "model": model.to_dict(),
                "thresholds": {str(budget): {"threshold": threshold, "nominal_budget_fraction": budget,
                                              "decision_rule": "predicted_ndcg_delta > threshold", "primary": budget == .75}
                               for budget, threshold in ((.25, .17), (.5, .14), (.75, .10))}}
    rows, prepared = [], []
    for source in audit_module.DOMAINS:
        for i in range(3):
            qid = f"{source}_{i}"
            text = " ".join(["token"] * (i + 1))
            rows.append({"query_id": qid, "source_domain": source, "component_id": qid, "query": text,
                         "relevant_ids": [ids[0]], "rerank_k": 20,
                         "rankings": {"bm25": ids, "dense": ids, "hybrid": ids, "reranked": ids[:20][::-1] + ids[20:]}})
            prepared.append({"id": qid, "query": text, "domain": source, "component_id": qid, "relevant_ids": [ids[0]]})
    predictions = freeze_predictions(rows, artifact)
    summary, per_query, decisions = evaluate_confirmation(rows, predictions)
    summary_path, prediction_path = tmp_path / "summary.json", tmp_path / "frozen_predictions.jsonl"
    summary_path.write_text(json.dumps(summary))
    for name, records in (("per_query.jsonl", per_query), ("decisions.jsonl", decisions), ("frozen_predictions.jsonl", predictions)):
        (tmp_path / name).write_text("".join(json.dumps(record) + "\n" for record in records))
    auditor = audit_module.Audit()
    auditor.evaluation(rows, prepared, artifact, summary_path, prediction_path)
    assert all(check["status"] == "pass" for check in auditor.checks)
    summary["policies"]["utility_75"]["metrics"]["nDCG@10"] += .02
    summary_path.write_text(json.dumps(summary))
    auditor = audit_module.Audit()
    auditor.evaluation(rows, prepared, artifact, summary_path, prediction_path)
    assert any(check["status"] == "fail" and check["name"] == "independent_policy_metrics_and_compute_utility_75" for check in auditor.checks)
