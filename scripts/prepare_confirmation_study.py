"""Prepare a deterministic untouched-query confirmation study, without rankings.

The already-observed 300-query pilot is development data. New query IDs and
normalized texts are held out from it. Positive-tool overlap is measured rather
than silently presented as an API-family holdout. Source revisions remain pinned.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path


_SPEC = importlib.util.spec_from_file_location(
    "_pilot_preparation", Path(__file__).with_name("prepare_research_pilot.py")
)
pilot = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(pilot)


def eligible_queries(domains: dict[str, list[dict]], exclusions: list[dict]) -> tuple[list[dict], dict]:
    """Apply the pilot's normalization and duplicate rule to every source row."""
    excluded_texts = {pilot.normalize_query(row["query"]) for row in exclusions}
    seen_ids, seen_texts = set(), {}
    candidates, removed = [], []
    for query_id, domain, row in sorted(
        (row["id"], domain, row) for domain, rows in domains.items() for row in rows
    ):
        if domain == "apibank":
            raise ValueError("APIBank cannot be a confirmation source")
        if not isinstance(query_id, str) or not query_id or query_id in seen_ids:
            raise ValueError(f"Invalid or duplicate query ID: {query_id}")
        seen_ids.add(query_id)
        if not isinstance(row["query"], str) or not pilot.normalize_query(row["query"]):
            raise ValueError(f"Empty query: {query_id}")
        normalized = pilot.normalize_query(row["query"])
        if normalized in excluded_texts:
            reason = "normalized_text_matches_apibank"
        elif normalized in seen_texts:
            reason = "duplicate_normalized_text_in_candidate_domains"
        else:
            seen_texts[normalized] = query_id
            candidates.append({**row, "domain": domain})
            continue
        removed.append({"id": query_id, "domain": domain, "reason": reason})
    return candidates, {
        "eligible_counts": dict(Counter(row["domain"] for row in candidates)),
        "source_counts": {domain: len(rows) for domain, rows in sorted(domains.items())},
        "removed_queries": removed,
        "query_normalization": "Unicode NFKC, casefold, collapse whitespace",
        "duplicate_policy": "keep lexicographically smallest ID across candidate sources",
    }


def positive_components(rows: list[dict]) -> dict[str, list[dict]]:
    """Group queries connected by shared positive tool IDs, including transitively."""
    parents = {row["id"]: row["id"] for row in rows}
    if len(parents) != len(rows):
        raise ValueError("Duplicate query IDs")

    def find(query_id: str) -> str:
        while query_id != parents[query_id]:
            parents[query_id] = parents[parents[query_id]]
            query_id = parents[query_id]
        return query_id

    by_tool = {}
    for row in rows:
        if not row["relevant_ids"]:
            raise ValueError(f"No positive tools: {row['id']}")
        for tool_id in row["relevant_ids"]:
            if tool_id in by_tool:
                left, right = find(row["id"]), find(by_tool[tool_id])
                if left != right:
                    parents[max(left, right)] = min(left, right)
            else:
                by_tool[tool_id] = row["id"]
    output = defaultdict(list)
    for row in sorted(rows, key=lambda item: item["id"]):
        output[find(row["id"])].append(row)
    return dict(sorted(output.items()))


def query_holdout_split(
    rows: list[dict], development: list[dict], calibration_per_domain: int,
    confirmation_per_domain: int, seed: str,
) -> dict[str, list[dict]]:
    """Assign by query ID hash only; labels do not affect cohort assignment."""
    if calibration_per_domain < 0 or confirmation_per_domain <= 0:
        raise ValueError("Invalid split sizes")
    by_id = {row["id"]: row for row in rows}
    if len(by_id) != len(rows):
        raise ValueError("Duplicate eligible query IDs")
    development_ids = {row["id"] for row in development}
    if len(development_ids) != len(development) or not development_ids <= by_id.keys():
        raise ValueError("Development cohort is not a unique subset of eligible queries")
    for row in development:
        eligible = by_id[row["id"]]
        if (eligible["query"], eligible["domain"], eligible["relevant_ids"]) != (
            row["query"], row["domain"], row["relevant_ids"]
        ):
            raise ValueError(f"Development query changed: {row['id']}")
    development_texts = {pilot.normalize_query(row["query"]) for row in development}
    candidates = defaultdict(list)
    for row in rows:
        if row["id"] not in development_ids:
            if pilot.normalize_query(row["query"]) in development_texts:
                raise ValueError(f"Normalized development text overlap: {row['id']}")
            candidates[row["domain"]].append(row)
    result = {"development": list(development), "calibration": [], "confirmation": [], "reserve": []}
    for domain in sorted(candidates):
        ordered = sorted(candidates[domain], key=lambda row: pilot.selection_key(row["id"], seed))
        needed = calibration_per_domain + confirmation_per_domain
        if len(ordered) < needed:
            raise ValueError(f"{domain}: {len(ordered)} unseen queries; need {needed}")
        result["calibration"].extend(ordered[:calibration_per_domain])
        result["confirmation"].extend(ordered[calibration_per_domain:needed])
        result["reserve"].extend(ordered[needed:])
    return result


def grouped_holdout_split(
    rows: list[dict], development: list[dict], calibration_per_domain: int,
    confirmation_per_domain: int, seed: str,
) -> tuple[dict[str, list[dict]], dict]:
    """Keep positive-tool components whole, anchoring observed pilot components.

    Greedy quotas are fixed before rankings exist. Each component is admitted
    only if all its source-domain query counts fit the remaining split quotas.
    """
    # Validate the observed cohort identities, not merely their query IDs.
    query_holdout_split(rows, development, 0, 1, seed)
    if calibration_per_domain <= 0 or confirmation_per_domain <= 0:
        raise ValueError("Grouped split sizes must be positive")
    components = positive_components(rows)
    development_ids = {row["id"] for row in development}
    anchored = {component for component, values in components.items() if any(row["id"] in development_ids for row in values)}
    free = sorted((component for component in components if component not in anchored), key=lambda component: pilot.selection_key(component, seed))
    assignment = {component: "quarantined_pilot_connected" for component in anchored}
    taken = set()
    split_components = {}
    domain_names = sorted({row["domain"] for row in rows})
    for name, quota in (("calibration", calibration_per_domain), ("confirmation", confirmation_per_domain)):
        remaining = {domain: quota for domain in domain_names}
        chosen = []
        for component in free:
            if component in taken:
                continue
            counts = Counter(row["domain"] for row in components[component])
            if all(count <= remaining[domain] for domain, count in counts.items()):
                chosen.append(component)
                taken.add(component)
                assignment[component] = name
                for domain, count in counts.items():
                    remaining[domain] -= count
        if any(remaining.values()):
            raise ValueError(f"Whole-component quota infeasible for {name}: {remaining}")
        split_components[name] = chosen
    for component in free:
        assignment.setdefault(component, "reserve")
    component_by_query = {row["id"]: component for component, values in components.items() for row in values}
    splits = {
        "development": [{**row, "component_id": component_by_query[row["id"]], "split": "development"} for row in development],
        "calibration": [], "confirmation": [], "quarantined_pilot_connected": [], "reserve": [],
    }
    for component, values in components.items():
        target = assignment[component]
        for row in values:
            if row["id"] in development_ids:
                continue
            splits[target].append({**row, "component_id": component, "split": target})
    for name in splits:
        splits[name].sort(key=lambda row: (row["domain"], pilot.selection_key(row["component_id"], seed), row["id"]))
    component_inventory = {
        component: {
            "assignment": assignment[component],
            "contains_observed_pilot": component in anchored,
            "query_count": len(values),
            "domain_counts": dict(Counter(row["domain"] for row in values)),
            "positive_tool_count": len({tool for row in values for tool in row["relevant_ids"]}),
            "query_ids": [row["id"] for row in values],
            "observed_pilot_query_ids": [row["id"] for row in values if row["id"] in development_ids],
        }
        for component, values in components.items()
    }
    return splits, {
        "component_id_definition": "lexicographically minimum query ID in component",
        "component_formation": "transitive sharing of positive gold tool IDs over full deduplicated eligible universe",
        "assignment_algorithm": "Anchor every component containing an observed pilot query; ascending SHA256(seed + NUL + component_id), ID tie-break; greedily admit whole components to calibration then confirmation only if source counts fit remaining fixed quotas; unused free components reserved",
        "seed": seed,
        "labels_used_for_component_partition": True,
        "retrieval_rankings_or_quality_not_used_for_partition": True,
        "component_counts_per_split": {
            name: len({row["component_id"] for row in values}) for name, values in splits.items()
        },
        "component_counts_per_split_and_domain": {
            name: {domain: len({row["component_id"] for row in values if row["domain"] == domain}) for domain in domain_names}
            for name, values in splits.items()
        },
        "cross_source_component_count": sum(len({row["domain"] for row in values}) > 1 for values in components.values()),
        "components": component_inventory,
    }


def overlap_audit(splits: dict[str, list[dict]], all_rows: list[dict]) -> dict:
    """Quantify exact positive-ID overlap and stricter connected-component overlap."""
    components = positive_components(all_rows)
    component_by_query = {row["id"]: component for component, rows in components.items() for row in rows}
    tools = {name: {tool for row in rows for tool in row["relevant_ids"]} for name, rows in splits.items()}
    ids = {name: {row["id"] for row in rows} for name, rows in splits.items()}
    texts = {name: {pilot.normalize_query(row["query"]) for row in rows} for name, rows in splits.items()}
    pairwise = {}
    names = sorted(splits)
    for left_index, left in enumerate(names):
        for right in names[left_index + 1:]:
            pairwise[f"{left}__{right}"] = {
                "query_id_overlap": len(ids[left] & ids[right]),
                "normalized_text_overlap": len(texts[left] & texts[right]),
                "positive_tool_id_overlap": len(tools[left] & tools[right]),
            }
    prior_tools = tools.get("development", set()) | tools.get("calibration", set())
    prior_components = {
        component_by_query[row["id"]]
        for name in ("development", "calibration") for row in splits.get(name, [])
    }
    direct_unseen = [row for row in splits["confirmation"] if not (set(row["relevant_ids"]) & prior_tools)]
    component_unseen = [row for row in splits["confirmation"] if component_by_query[row["id"]] not in prior_components]
    pilot_components = {component_by_query[row["id"]] for row in splits["development"]}
    anchored = [row for row in all_rows if component_by_query[row["id"]] in pilot_components]
    return {
        "component_definition": "connected through any shared positive tool ID; entire eligible universe",
        "api_family_holdout": False,
        "component_count": len(components),
        "largest_component_sizes": sorted((len(rows) for rows in components.values()), reverse=True)[:25],
        "pilot_anchored_component_count": len(pilot_components),
        "pilot_anchored_query_counts": dict(Counter(row["domain"] for row in anchored)),
        "pilot_component_disjoint_available_counts": dict(Counter(row["domain"] for row in all_rows if component_by_query[row["id"]] not in pilot_components)),
        "split_counts": {name: dict(Counter(row["domain"] for row in rows)) for name, rows in splits.items()},
        "positive_tool_counts": {name: len(values) for name, values in tools.items()},
        "pairwise_overlap": pairwise,
        "confirmation_direct_positive_tool_disjoint_from_development_and_calibration": {
            "count": len(direct_unseen), "domain_counts": dict(Counter(row["domain"] for row in direct_unseen)),
            "query_ids": sorted(row["id"] for row in direct_unseen),
        },
        "confirmation_component_disjoint_from_development_and_calibration": {
            "count": len(component_unseen), "domain_counts": dict(Counter(row["domain"] for row in component_unseen)),
            "query_ids": sorted(row["id"] for row in component_unseen),
        },
    }


def corpus_metadata_audit(corpus: list[dict]) -> dict:
    columns, object_count = Counter(), 0
    for row in corpus:
        documentation = json.loads(row["text"])
        if isinstance(documentation, dict):
            object_count += 1
            columns.update(documentation.keys())
    return {
        "corpus_count": len(corpus),
        "documentation_json_object_count": object_count,
        "documentation_field_counts": dict(sorted(columns.items())),
        "uniform_structured_api_family_mapping_available": False,
        "reason": "Names are present but category, category_name, toolkit and URL fields have heterogeneous partial coverage; source ID prefixes are dataset provenance, not API families.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("data/research_confirmation_20261003"))
    parser.add_argument("--results-dir", type=Path, default=Path("results/research_confirmation_20261003"))
    parser.add_argument("--corpus", type=Path, default=Path("data/research_20261003/corpus.jsonl"))
    parser.add_argument("--development", type=Path, default=Path("data/research_20261003/queries.jsonl"))
    parser.add_argument("--calibration-per-domain", type=int, default=100)
    parser.add_argument("--confirmation-per-domain", type=int, default=500)
    parser.add_argument("--seed", default="confirmation-20261003")
    args = parser.parse_args()
    # Raw downloads and a feasibility audit may already exist; final outputs must
    # never be overwritten, including after evaluation becomes possible.
    for filename in ("manifest.json", "calibration_queries.jsonl", "confirmation_queries.jsonl", "development_queries.jsonl"):
        if (args.output_dir / filename).exists():
            raise SystemExit(f"Refusing to overwrite frozen data: {args.output_dir / filename}")
    import pyarrow.parquet as parquet

    raw_dir = args.output_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw_sources = []
    previous_path = raw_dir / "sources.json"
    previous = json.loads(previous_path.read_text()) if previous_path.exists() else []
    by_path = {source["path"]: source for source in previous}

    def get_domain(domain):
        relative = f"{domain}/queries-00000-of-00001.parquet"
        path = raw_dir / f"{domain}.parquet"
        cached = by_path.get(relative)
        if cached and path.exists():
            if cached["revision"] != pilot.QUERY_REVISION or cached["sha256"] != pilot.sha256_file(path):
                raise ValueError(f"Cached source identity mismatch: {domain}")
            provenance = cached
        else:
            provenance = pilot.download_source(pilot.QUERY_REPO, pilot.QUERY_REVISION, relative, path)
        return domain, parquet.read_table(path).to_pylist(), provenance

    with ThreadPoolExecutor(max_workers=4) as executor:
        source_rows = list(executor.map(get_domain, ("apibank", *pilot.PILOT_DOMAINS)))
    domains = {}
    for domain, rows, provenance in source_rows:
        raw_sources.append(provenance)
        if domain == "apibank":
            excluded = rows
        else:
            domains[domain] = rows
    previous_path.write_text(json.dumps(raw_sources, indent=2, sort_keys=True) + "\n")
    corpus = [json.loads(line) for line in args.corpus.open()]
    development = [json.loads(line) for line in args.development.open()]
    eligible, dedup_audit = eligible_queries(domains, excluded)
    rows = pilot.prepare_queries(eligible, {row["id"] for row in corpus})
    splits, assignments = grouped_holdout_split(rows, development, args.calibration_per_domain, args.confirmation_per_domain, args.seed)
    audit = {
        "deduplication": dedup_audit,
        "overlap": overlap_audit(splits, rows),
        "corpus_metadata": corpus_metadata_audit(corpus),
        "labels_used_for_primary_split_assignment": True,
        "label_use_scope": "Positive tool IDs construct whole connected components; no retrieval outcomes or quality values used",
        "new_rankings_or_quality_results_observed_during_selection": False,
        "development_status": "Previously observed 300-query pilot; never reused as confirmation evidence",
        "grouped_partition": {key: value for key, value in assignments.items() if key != "components"},
    }
    args.results_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in splits.items():
        pilot.write_jsonl(args.output_dir / f"{name}_queries.jsonl", rows)
    prepared = {
        f"{name}_queries.jsonl": {"sha256": pilot.sha256_file(args.output_dir / f"{name}_queries.jsonl"), "bytes": (args.output_dir / f"{name}_queries.jsonl").stat().st_size, "query_count": len(rows)}
        for name, rows in splits.items()
    }
    prepared["corpus.jsonl"] = {"path": str(args.corpus), "sha256": pilot.sha256_file(args.corpus), "bytes": args.corpus.stat().st_size, "tool_count": len(corpus)}
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Untouched positive-tool-ID-component-disjoint query holdout across source domains; not an API-family holdout",
        "source_revisions": {pilot.QUERY_REPO: pilot.QUERY_REVISION, pilot.TOOLS_REPO: pilot.TOOLS_REVISION},
        "upstream_sources": raw_sources,
        "pilot_domains": list(pilot.PILOT_DOMAINS),
        "seed": args.seed,
        "assignment_algorithm": assignments["assignment_algorithm"],
        "calibration_per_domain": args.calibration_per_domain,
        "confirmation_per_domain": args.confirmation_per_domain,
        "corpus_count": len(corpus),
        "development_source_sha256": pilot.sha256_file(args.development),
        "relevance_policy": "binary relevance > 0; original graded labels retained",
        "retrieval_input": "query field only; instruction and labels excluded",
        "prepared_files": prepared,
        "preparation_source_sha256": pilot.sha256_file(Path(__file__)),
        "no_new_retrieval_rankings_or_quality_results_inspected": True,
    }
    for filename, content in (("manifest.json", manifest), ("audit.json", audit)):
        (args.output_dir / filename).write_text(json.dumps(content, indent=2, sort_keys=True) + "\n")
        (args.results_dir / f"data_{filename}").write_text(json.dumps(content, indent=2, sort_keys=True) + "\n")
    inventory = {
        name: [{"id": row["id"], "domain": row["domain"], "component_id": row["component_id"], "split": name} for row in rows]
        for name, rows in splits.items()
    }
    (args.results_dir / "selected_query_ids.json").write_text(json.dumps(inventory, indent=2, sort_keys=True) + "\n")
    (args.results_dir / "component_assignments.json").write_text(json.dumps(assignments, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"split_counts": audit["overlap"]["split_counts"], "direct_disjoint_confirmation": audit["overlap"]["confirmation_direct_positive_tool_disjoint_from_development_and_calibration"]["domain_counts"], "component_disjoint_confirmation": audit["overlap"]["confirmation_component_disjoint_from_development_and_calibration"]["domain_counts"]}, indent=2))


if __name__ == "__main__":
    main()
