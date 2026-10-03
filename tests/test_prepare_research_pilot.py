"""Selection/provenance regressions without network or parquet dependencies."""

import importlib.util
import json
from pathlib import Path

import pytest


_SPEC = importlib.util.spec_from_file_location(
    "prepare_research_pilot", Path(__file__).parents[1] / "scripts" / "prepare_research_pilot.py"
)
prep = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(prep)


def row(query_id, text, labels=None):
    return {
        "id": query_id, "query": text, "category": "web",
        "labels": json.dumps(labels or [{"id": "tool_1", "relevance": 1}]),
    }


def test_normalized_overlap_with_apibank_is_excluded():
    domains = {"x": [row("x1", " ＨＥＬＬＯ  World "), row("x2", "Unique")]}
    selected, audit = prep.select_queries(domains, [row("old", "hello world")], 1, "seed")
    assert [item["id"] for item in selected] == ["x2"]
    assert audit["domain_counts"]["x"]["removals"]["normalized_text_matches_apibank"] == 1


def test_cross_domain_duplicate_keeps_smallest_id_independent_of_input_order():
    first = {"b": [row("z", "Repeated"), row("b", "Second")], "a": [row("a", "repeated")]}
    second = {"a": first["a"], "b": list(reversed(first["b"]))}
    selected1, audit1 = prep.select_queries(first, [], 1, "seed")
    selected2, audit2 = prep.select_queries(second, [], 1, "seed")
    assert selected1 == selected2
    assert audit1 == audit2
    assert [item["id"] for item in selected1] == ["a", "b"]


def test_selection_is_label_independent_and_rejects_insufficient_domain():
    original = {"x": [row(f"q{i}", f"Query {i}") for i in range(10)]}
    changed = {"x": [{**item, "labels": "THIS IS NOT EVEN JSON"} for item in original["x"]]}
    left, _ = prep.select_queries(original, [], 3, "fixed")
    right, _ = prep.select_queries(changed, [], 3, "fixed")
    assert [item["id"] for item in left] == [item["id"] for item in right]
    with pytest.raises(ValueError, match="eligible queries"):
        prep.select_queries(original, [], 11, "fixed")


def test_missing_positive_gold_fails_instead_of_silently_dropping_query():
    selected = [{**row("x", "Query"), "domain": "source"}]
    with pytest.raises(ValueError, match="Missing relevant tool"):
        prep.prepare_queries(selected, {"another_tool"})


def test_graded_labels_are_preserved_and_binary_ids_are_deduplicated():
    labels = [{"id": "a", "relevance": 2}, {"id": "a", "relevance": 1}, {"id": "b", "relevance": 0}]
    selected = [{**row("x", "Query", labels), "domain": "source"}]
    output = prep.prepare_queries(selected, {"a", "b"})
    assert output[0]["relevant_ids"] == ["a"]
    assert output[0]["labels"] == labels


def test_apibank_and_duplicate_ids_are_rejected():
    with pytest.raises(ValueError, match="APIBank"):
        prep.select_queries({"apibank": [row("old", "Query")]}, [], 1, "seed")
    with pytest.raises(ValueError, match="Duplicate query ID"):
        prep.select_queries({"x": [row("same", "A"), row("same", "B")]}, [], 1, "seed")


def test_source_domain_is_prefix_provenance_not_guessed_family():
    assert prep.source_domain("toolbench_tool_12") == "toolbench"
    assert prep.source_domain("no_structured_source") == "unknown"
