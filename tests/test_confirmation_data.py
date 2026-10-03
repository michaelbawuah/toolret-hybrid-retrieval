"""Freeze untouched-query splits and audit positive-tool overlap without models."""

import importlib.util
from pathlib import Path

import pytest


_SPEC = importlib.util.spec_from_file_location(
    "confirmation_prep", Path(__file__).parents[1] / "scripts" / "prepare_confirmation_study.py"
)
prep = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(prep)


def row(query_id, domain="x", positive=None, text=None):
    return {"id": query_id, "domain": domain, "query": text or query_id, "relevant_ids": positive or [query_id + "_tool"]}


def test_split_is_order_invariant_and_hash_deterministic():
    rows = [row(f"q{i}") for i in range(12)]
    left = prep.query_holdout_split(rows, rows[:2], 3, 5, "fixed")
    right = prep.query_holdout_split(list(reversed(rows)), rows[:2], 3, 5, "fixed")
    assert left == right
    assert {name: len(values) for name, values in left.items()} == {"development": 2, "calibration": 3, "confirmation": 5, "reserve": 2}
    assert len({item["id"] for values in left.values() for item in values}) == 12


def test_primary_assignment_does_not_consult_labels():
    rows = [row(f"q{i}") for i in range(10)]
    changed = [{**item, "relevant_ids": ["same"]} for item in rows]
    left = prep.query_holdout_split(rows, rows[:1], 2, 6, "fixed")
    right = prep.query_holdout_split(changed, changed[:1], 2, 6, "fixed")
    assert {name: [item["id"] for item in values] for name, values in left.items()} == {
        name: [item["id"] for item in values] for name, values in right.items()
    }


def test_domains_are_balanced_and_insufficient_pool_fails():
    rows = [row(f"{domain}{i}", domain) for domain in ("a", "b") for i in range(6)]
    dev = [rows[0], rows[6]]
    split = prep.query_holdout_split(rows, dev, 2, 3, "fixed")
    assert sum(item["domain"] == "a" for item in split["confirmation"]) == 3
    assert sum(item["domain"] == "b" for item in split["confirmation"]) == 3
    with pytest.raises(ValueError, match="unseen queries"):
        prep.query_holdout_split(rows, dev, 3, 3, "fixed")


@pytest.mark.parametrize("field,value", [("query", "changed"), ("domain", "changed"), ("relevant_ids", ["changed"])])
def test_changed_development_identity_is_rejected(field, value):
    rows = [row(f"q{i}") for i in range(5)]
    with pytest.raises(ValueError, match="Development query changed"):
        prep.query_holdout_split(rows, [{**rows[0], field: value}], 1, 2, "fixed")


def test_normalized_development_text_overlap_is_rejected():
    rows = [row("q0", text=" HELLO "), row("q1", text="ｈｅｌｌｏ"), row("q2")]
    with pytest.raises(ValueError, match="Normalized development text overlap"):
        prep.query_holdout_split(rows, rows[:1], 0, 1, "fixed")


def test_connected_components_include_transitive_bridge():
    rows = [row("a", positive=["t1"]), row("b", positive=["t1", "t2"]), row("c", positive=["t2"]), row("d", positive=["t3"])]
    components = prep.positive_components(rows)
    assert sorted(sorted(item["id"] for item in values) for values in components.values()) == [["a", "b", "c"], ["d"]]
    assert components == prep.positive_components(list(reversed(rows)))


def test_overlap_audit_distinguishes_direct_from_component_disjoint():
    rows = [row("a", positive=["t1"]), row("bridge", positive=["t1", "t2"]), row("c", positive=["t2"]), row("d", positive=["t3"])]
    audit = prep.overlap_audit({"development": rows[:1], "calibration": [], "confirmation": rows[2:], "reserve": rows[1:2]}, rows)
    assert audit["confirmation_direct_positive_tool_disjoint_from_development_and_calibration"]["query_ids"] == ["c", "d"]
    assert audit["confirmation_component_disjoint_from_development_and_calibration"]["query_ids"] == ["d"]
    assert audit["api_family_holdout"] is False


def test_dedup_and_apibank_exclusion_before_assignment():
    def raw(qid, text):
        return {"id": qid, "query": text, "category": "web", "labels": []}
    domains = {"b": [raw("z", "HELLO"), raw("olddup", "old")], "a": [raw("a", " hello "), raw("unique", "Other")]}
    eligible, audit = prep.eligible_queries(domains, [raw("old", " ＯＬＤ ")])
    assert [item["id"] for item in eligible] == ["a", "unique"]
    assert len(audit["removed_queries"]) == 2


def test_duplicate_query_ids_and_unknown_development_are_rejected():
    with pytest.raises(ValueError, match="Duplicate eligible"):
        prep.query_holdout_split([row("a"), row("a")], [], 0, 1, "fixed")
    with pytest.raises(ValueError, match="unique subset"):
        prep.query_holdout_split([row("a")], [row("missing")], 0, 1, "fixed")


def test_grouped_split_keeps_gold_components_whole_and_quarantines_pilot_bridge():
    rows = [
        row("pilot", positive=["p1"]), row("linked", positive=["p1", "p2"]),
        row("transitive", positive=["p2"]),
        *[row(f"fresh{i}", positive=[f"fresh_tool_{i}"]) for i in range(7)],
    ]
    splits, assignments = prep.grouped_holdout_split(rows, rows[:1], 2, 3, "fixed")
    assert {item["id"] for item in splits["quarantined_pilot_connected"]} == {"linked", "transitive"}
    assert len(splits["calibration"]) == 2 and len(splits["confirmation"]) == 3
    sets = [{tool for item in splits[name] for tool in item["relevant_ids"]} for name in ("development", "calibration", "confirmation")]
    assert not (sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])
    assert assignments["component_counts_per_split"]["development"] == 1
    assert assignments["labels_used_for_component_partition"] is True
    assert splits == prep.grouped_holdout_split(list(reversed(rows)), rows[:1], 2, 3, "fixed")[0]


def test_grouped_split_never_truncates_oversized_components_to_hit_quota():
    rows = [row("pilot"), row("a", positive=["t1"]), row("b", positive=["t1"]), row("c", positive=["t2"]), row("d", positive=["t2"])]
    with pytest.raises(ValueError, match="Whole-component quota infeasible"):
        prep.grouped_holdout_split(rows, rows[:1], 1, 2, "fixed")


def test_grouped_split_honors_every_domain_for_cross_source_components():
    rows = [row("pilot_a", "a"), row("pilot_b", "b"), row("pair_a", "a", ["shared"]), row("pair_b", "b", ["shared"])]
    rows += [row(f"{domain}{i}", domain) for domain in ("a", "b") for i in range(4)]
    splits, assignments = prep.grouped_holdout_split(rows, rows[:2], 1, 2, "fixed")
    assert assignments["cross_source_component_count"] == 1
    for name, size in (("calibration", 1), ("confirmation", 2)):
        assert {domain: sum(item["domain"] == domain for item in splits[name]) for domain in ("a", "b")} == {"a": size, "b": size}
    locations = {item["id"]: name for name, values in splits.items() for item in values}
    assert locations["pair_a"] == locations["pair_b"]
