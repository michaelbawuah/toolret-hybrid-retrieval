"""Prepare a pinned, label-independent ToolRet source-domain research pilot.

This creates new files only. It does not recreate, tune against, or overwrite the
original APIBank training/validation/frozen-test files. Requires ``pyarrow`` to
read the public upstream parquet files; the selection logic uses only stdlib.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
import tempfile
import unicodedata
from urllib.request import Request, urlopen


QUERY_REPO = "mangopy/ToolRet-Queries"
TOOLS_REPO = "mangopy/ToolRet-Tools"
QUERY_REVISION = "b8c76ad3349ff17497b6bdb28bb5b8f61a0f6445"
TOOLS_REVISION = "e06c38c75612b6536bd959e08cdd345894aba6a7"
PILOT_DOMAINS = ("apigen", "toolbench", "toolace")
DEFAULT_SEED = "research-20261003"


def normalize_query(text: str) -> str:
    """Normalize unicode, case, and whitespace, without interpreting intent."""
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def selection_key(query_id: str, seed: str) -> tuple[str, str]:
    digest = hashlib.sha256(f"{seed}\0{query_id}".encode()).hexdigest()
    return digest, query_id


def source_domain(tool_id: str) -> str:
    """ID-prefix provenance only: this is not an inferred API family."""
    if "_tool_" not in tool_id:
        return "unknown"
    return tool_id.rsplit("_tool_", 1)[0]


def select_queries(
    domains: dict[str, list[dict]],
    excluded_queries: list[dict],
    per_domain: int,
    seed: str,
) -> tuple[list[dict], dict]:
    """Select by ID hash after text deduplication, never consulting labels.

    Duplicate text across candidate sources retains the lexicographically first
    query ID. Domain names are preserved separately from the tool ID prefixes.
    """
    if per_domain <= 0:
        raise ValueError("per_domain must be positive")
    excluded_texts = {normalize_query(row["query"]) for row in excluded_queries}
    all_rows = []
    seen_ids = set()
    for domain in sorted(domains):
        if domain == "apibank":
            raise ValueError("APIBank cannot be included in the fresh pilot")
        for row in domains[domain]:
            query_id, text = row["id"], row["query"]
            if not isinstance(query_id, str) or not query_id:
                raise ValueError("Invalid query ID")
            if query_id in seen_ids:
                raise ValueError(f"Duplicate query ID: {query_id}")
            if not isinstance(text, str) or not normalize_query(text):
                raise ValueError(f"Empty query: {query_id}")
            seen_ids.add(query_id)
            all_rows.append((query_id, domain, row, normalize_query(text)))

    candidates = {domain: [] for domain in domains}
    removal_counts = {domain: Counter() for domain in domains}
    removals = []
    seen_texts = {}
    for query_id, domain, row, normalized in sorted(all_rows):
        if normalized in excluded_texts:
            reason = "normalized_text_matches_apibank"
        elif normalized in seen_texts:
            reason = "duplicate_normalized_text_in_candidate_domains"
        else:
            seen_texts[normalized] = query_id
            candidates[domain].append(row)
            continue
        removal_counts[domain][reason] += 1
        removals.append({"id": query_id, "domain": domain, "reason": reason})

    selected = []
    counts = {}
    for domain in sorted(domains):
        rows = sorted(candidates[domain], key=lambda row: selection_key(row["id"], seed))
        if len(rows) < per_domain:
            raise ValueError(
                f"{domain}: only {len(rows)} eligible queries; need {per_domain}"
            )
        picked = rows[:per_domain]
        selected.extend({**row, "domain": domain} for row in picked)
        counts[domain] = {
            "source_count": len(domains[domain]),
            "eligible_count": len(rows),
            "selected_count": len(picked),
            "removals": dict(removal_counts[domain]),
        }
    audit = {
        "selection_seed": seed,
        "selection_algorithm": "ascending SHA256(seed + NUL + query_id), ID tie-break",
        "query_normalization": "Unicode NFKC, casefold, collapse whitespace",
        "duplicate_policy": "keep lexicographically smallest ID across all candidate domains",
        "excluded_apibank_count": len(excluded_queries),
        "domain_counts": counts,
        "removed_queries": removals,
        "selected_query_ids": [row["id"] for row in selected],
        "labels_used_for_selection": False,
    }
    return selected, audit


def prepare_queries(selected: list[dict], corpus_ids: set[str]) -> list[dict]:
    """Validate all selected gold IDs, preserving labels with binary evaluation IDs."""
    output = []
    for row in selected:
        if row.get("category") != "web":
            raise ValueError(f"Query is not category web: {row['id']}")
        labels = json.loads(row["labels"]) if isinstance(row["labels"], str) else row["labels"]
        if not isinstance(labels, list):
            raise ValueError(f"Labels must be a list: {row['id']}")
        positives = []
        for label in labels:
            if not isinstance(label, dict) or "id" not in label or "relevance" not in label:
                raise ValueError(f"Malformed label: {row['id']}")
            relevance = float(label["relevance"])
            if relevance > 0:
                if label["id"] not in corpus_ids:
                    raise ValueError(f"Missing relevant tool {label['id']} for {row['id']}")
                positives.append(label["id"])
        if not positives:
            raise ValueError(f"No positive labels: {row['id']}")
        output.append({
            "id": row["id"],
            "query": row["query"],
            "domain": row["domain"],
            "category": row["category"],
            "instruction": row.get("instruction", ""),
            "relevant_ids": sorted(set(positives)),
            "labels": labels,
        })
    return output


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_source(repo: str, revision: str, relative_path: str, target: Path) -> dict:
    if not re.fullmatch(r"[a-f0-9]{40}", revision):
        raise ValueError("Revision must be a full 40-character commit hash")
    url = f"https://huggingface.co/datasets/{repo}/resolve/{revision}/{relative_path}"
    request = Request(url, headers={"User-Agent": "toolret-independent-research/0.1"})
    # Anonymous public downloads: no credentials or tokens are read.
    with urlopen(request, timeout=90) as response, target.open("wb") as handle:
        shutil.copyfileobj(response, handle)
    return {
        "repository": repo,
        "revision": revision,
        "path": relative_path,
        "url": url,
        "sha256": sha256_file(target),
        "bytes": target.stat().st_size,
    }


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("data/research_20261003"))
    parser.add_argument("--per-domain", type=int, default=100)
    parser.add_argument("--seed", default=DEFAULT_SEED)
    parser.add_argument("--query-revision", default=QUERY_REVISION)
    parser.add_argument("--tools-revision", default=TOOLS_REVISION)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise SystemExit(f"Refusing to overwrite existing directory: {args.output_dir}")
    try:
        import pyarrow.parquet as parquet
    except ImportError as error:
        raise SystemExit("Install pyarrow to read the upstream parquet files") from error

    sources = []
    with tempfile.TemporaryDirectory(prefix="toolret-pilot-") as temporary:
        raw_dir = Path(temporary)
        query_domains = {}
        for domain in ("apibank", *PILOT_DOMAINS):
            path = raw_dir / f"{domain}.parquet"
            print(f"Downloading pinned query domain {domain}...", flush=True)
            sources.append(download_source(
                QUERY_REPO, args.query_revision,
                f"{domain}/queries-00000-of-00001.parquet", path,
            ))
            rows = parquet.read_table(path).to_pylist()
            if domain == "apibank":
                excluded_queries = rows
            else:
                query_domains[domain] = rows

        selected, audit = select_queries(query_domains, excluded_queries, args.per_domain, args.seed)
        tools_path = raw_dir / "web.parquet"
        print("Downloading pinned full web tool corpus...", flush=True)
        sources.append(download_source(
            TOOLS_REPO, args.tools_revision, "web/tools-00000-of-00001.parquet", tools_path,
        ))
        raw_corpus = parquet.read_table(tools_path).to_pylist()
        corpus = []
        corpus_ids = set()
        for row in raw_corpus:
            if row["id"] in corpus_ids:
                raise ValueError(f"Duplicate corpus ID: {row['id']}")
            if not isinstance(row["documentation"], str) or not row["documentation"].strip():
                raise ValueError(f"Empty tool documentation: {row['id']}")
            corpus_ids.add(row["id"])
            corpus.append({
                "id": row["id"], "text": row["documentation"],
                "source_domain": source_domain(row["id"]),
            })
        queries = prepare_queries(selected, corpus_ids)
        audit["missing_relevant_tool_ids"] = []
        audit["all_selected_queries_have_positive_labels"] = True
        audit["query_categories"] = dict(Counter(row["category"] for row in queries))
        audit["corpus_source_prefix_counts"] = dict(Counter(row["source_domain"] for row in corpus))
        audit["api_family_metadata_available"] = False
        audit["pre_evaluation_scope_repair"] = {
            "original_candidate_domains": ["apigen", "toolbench", "metatool"],
            "replacement": "metatool -> toolace",
            "reason": (
                "metatool queries have source category customized, and selected "
                "metatool_which_query_13 referenced metatool_which_tool_172, which "
                "is absent from the web corpus; toolace has category web and "
                "sufficient source rows. Replacement made before any ranking "
                "or retrieval-quality evaluation."
            ),
            "ranking_or_quality_results_observed": False,
        }

        # Validate everything before creating the output directory. A failed
        # download or a missing gold ID cannot leave a seemingly usable pilot.
        args.output_dir.mkdir(parents=True)
        corpus_path = args.output_dir / "corpus.jsonl"
        queries_path = args.output_dir / "queries.jsonl"
        write_jsonl(corpus_path, corpus)
        write_jsonl(queries_path, queries)
        manifest = {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "scope": "zero-training source-domain pilot; not API-family-held-out evaluation",
            "source_revisions": {QUERY_REPO: args.query_revision, TOOLS_REPO: args.tools_revision},
            "upstream_sources": sources,
            "pilot_domains": list(PILOT_DOMAINS),
            "queries_per_domain": args.per_domain,
            "query_count": len(queries),
            "corpus_count": len(corpus),
            "seed": args.seed,
            "relevance_policy": "binary relevance > 0; original graded labels retained",
            "retrieval_input": "query field only; instruction and labels excluded",
            "frozen_apibank_data_modified": False,
            "prepared_files": {
                "corpus.jsonl": {"sha256": sha256_file(corpus_path), "bytes": corpus_path.stat().st_size},
                "queries.jsonl": {"sha256": sha256_file(queries_path), "bytes": queries_path.stat().st_size},
            },
        }
        for filename, data in (("manifest.json", manifest), ("audit.json", audit)):
            (args.output_dir / filename).write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"Prepared {len(queries)} queries and {len(corpus)} tools at {args.output_dir}")


if __name__ == "__main__":
    main()
