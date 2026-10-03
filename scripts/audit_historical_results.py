"""Audit saved historical ranks; this never runs or tunes a retrieval model."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path

from toolret_research.statistics import (
    HISTORICAL_SYSTEMS,
    exact_sign_test,
    paired_bootstrap,
    validate_rank_records,
)


LABELS = {
    "bm25": "BM25",
    "dense": "Hard-negative MiniLM",
    "hybrid": "Hybrid retrieval",
    "reranked": "Hybrid + reranker",
}


def readme_test_mrr(path: Path) -> dict[str, float]:
    """Read only README's frozen test table, not its validation table."""
    text = path.read_text(encoding="utf-8")
    match = re.search(r"## Frozen Test Results\n(.*?)(?=\n## |\Z)", text, re.S)
    if not match:
        return {}
    values = {}
    for line in match.group(1).splitlines():
        cells = [cell.strip().replace("**", "") for cell in line.strip().split("|")]
        if len(cells) > 3 and cells[1] in (*LABELS.values(), "Base MiniLM"):
            try:
                values[cells[1]] = float(cells[2])
            except ValueError:
                continue
    return values


def file_evidence(path: Path) -> dict[str, object]:
    return {
        "path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "bytes": path.stat().st_size,
    }


def audit(csv_path: Path, readme_path: Path, *, seed: int, resamples: int) -> dict[str, object]:
    with csv_path.open(encoding="utf-8", newline="") as handle:
        raw_rows = list(csv.DictReader(handle))
    rows = validate_rank_records(raw_rows)
    n = len(rows)
    system_values = {
        system: [row[f"{system}_rr"] for row in rows]
        for system in HISTORICAL_SYSTEMS
    }
    readme_mrr = readme_test_mrr(readme_path)
    aggregates = {}
    discrepancies = []
    for system, values in system_values.items():
        mean = math.fsum(values) / n
        bootstrap = paired_bootstrap([0.0] * n, values, seed=seed, resamples=resamples)
        reported = readme_mrr.get(LABELS[system])
        rounded_match = reported is not None and f"{mean:.3f}" == f"{reported:.3f}"
        aggregates[system] = {
            "label": LABELS[system],
            "mrr": mean,
            "bootstrap_ci": bootstrap,
            "readme_mrr": reported,
            "matches_readme_to_three_decimals": rounded_match,
        }
        if reported is not None and not rounded_match:
            discrepancies.append({"system": system, "csv_mrr": mean, "readme_mrr": reported})
    comparisons = {}
    for before, after in (
        ("bm25", "dense"),
        ("bm25", "hybrid"),
        ("dense", "hybrid"),
        ("hybrid", "reranked"),
        ("dense", "reranked"),
    ):
        comparisons[f"{after}_minus_{before}"] = {
            "before": before,
            "after": after,
            "bootstrap": paired_bootstrap(
                system_values[before], system_values[after], seed=seed, resamples=resamples
            ),
            "sign_test": exact_sign_test(system_values[before], system_values[after]),
        }
    limitations = [
        "This is a descriptive reanalysis of 16 saved historical queries, not a new held-out test or benchmark-wide reproduction.",
        "All saved query IDs use the apibank prefix; these rows do not demonstrate cross-domain or unseen-API-family generalization.",
        "The CSV stores only first-relevant ranks. It cannot independently recover Recall@K, nDCG@K, full candidate rankings, or downstream agent success.",
        "The CSV does not contain base MiniLM per-query ranks; README's base MiniLM MRR cannot be recomputed from it.",
        "The full processed corpus, original split files, trained checkpoints, training logs, and raw full-ranking evaluation JSON are not committed in this checkout. The claimed 37,292-tool run and validation-selected configuration are historical README/script claims, not newly verified artifacts.",
        "Blank positions mean missing from the saved candidate ranking; they do not prove that no relevant tool exists in the full catalog.",
        "The new CSV audit verifies internal consistency, not the origin or correctness of historical relevance labels, model outputs, train/test isolation, or latency measurements.",
        "Bootstrap intervals resample query rows, assume exchangeable query observations, and remain exploratory for this small and potentially correlated sample. They cannot correct selection bias or data leakage.",
        "Exact sign-test p-values concern win/loss balance and ignore magnitude. They are unadjusted across comparisons and do not establish mean-MRR significance.",
        "Do not train, choose a threshold, or change retrieval settings using these frozen historical test rows; future experiments need separate development and fresh evaluation data.",
    ]
    return {
        "schema_version": 1,
        "evidence_type": "exploratory_historical_rank_reanalysis",
        "not_a_new_model_evaluation": True,
        "sources": {"csv": file_evidence(csv_path), "readme": file_evidence(readme_path)},
        "row_audit": {
            "n_queries": n,
            "unique_query_ids": len({row["query_id"] for row in rows}),
            "duplicate_query_ids": [],
            "rank_rr_inconsistencies": [],
            "effect_label_inconsistencies": [],
            "query_id_prefix_counts": dict(Counter(row["query_id"].split("_query_")[0] for row in rows)),
            "hybrid_vs_dense_counts": dict(Counter(row["hybrid_vs_dense"] for row in rows)),
            "reranker_effect_counts": dict(Counter(row["reranker_effect"] for row in rows)),
        },
        "aggregates": aggregates,
        "paired_comparisons": comparisons,
        "readme_discrepancies": discrepancies,
        "unverifiable_base_minilm_readme_mrr": readme_mrr.get("Base MiniLM"),
        "limitations": limitations,
    }


def render_markdown(report: dict[str, object]) -> str:
    row_audit = report["row_audit"]
    lines = [
        "# Historical ToolRet rank audit — October 3, 2026",
        "",
        "**Exploratory reanalysis of saved historical evidence; no retrieval model was run, trained, or tuned.**",
        "",
        f"Validated {row_audit['n_queries']} rows with unique query IDs, positive-or-missing ranks, consistent reciprocal ranks, and consistent effect labels.",
        "",
        "## Recomputed mean reciprocal rank",
        "",
        "Intervals are 95% paired query bootstrap percentile intervals; absolute MRR uses a zero-valued reference.",
        "",
        "| System | CSV MRR | Bootstrap interval | README MRR | Match at 3 decimals |",
        "|---|---:|---:|---:|---|",
    ]
    for values in report["aggregates"].values():
        ci = values["bootstrap_ci"]
        readme_value = values["readme_mrr"]
        readme_cell = f"{readme_value:.3f}" if readme_value is not None else "missing"
        lines.append(
            f"| {values['label']} | {values['mrr']:.6f} | [{ci['low']:.6f}, {ci['high']:.6f}] | {readme_cell} | {'yes' if values['matches_readme_to_three_decimals'] else 'no'} |"
        )
    lines += [
        "",
        "The saved CSV contradicts README's BM25 and fine-tuned dense MRR values. It gives hybrid minus dense a negative mean difference. This audit does not infer which original evaluation produced README's alternative numbers.",
        "",
        "## Paired differences",
        "",
        "Delta is after minus before. Sign-test p-values test win/loss balance, exclude ties, and have no multiplicity adjustment.",
        "",
        "| Comparison | Mean ΔMRR | 95% bootstrap interval | Wins / losses / ties | Exact sign p |",
        "|---|---:|---:|---:|---:|",
    ]
    for comparison in report["paired_comparisons"].values():
        ci = comparison["bootstrap"]
        sign = comparison["sign_test"]
        lines.append(
            f"| {LABELS[comparison['after']]} − {LABELS[comparison['before']]} | {ci['delta_mean']:+.6f} | [{ci['low']:+.6f}, {ci['high']:+.6f}] | {sign['wins']} / {sign['losses']} / {sign['ties']} | {sign['two_sided_pvalue']:.6f} |"
        )
    lines += ["", "## Evidence limitations", ""]
    lines += [f"- {limitation}" for limitation in report["limitations"]]
    lines += ["", "## Source hashes", ""]
    for source in report["sources"].values():
        lines.append(f"- `{source['path']}` — SHA-256 `{source['sha256']}`")
    first_ci = next(iter(report["aggregates"].values()))["bootstrap_ci"]
    lines += ["", f"Bootstrap: {first_ci['resamples']:,} resamples, seed {first_ci['seed']}, linear-interpolated percentile endpoints. Sampling unit: query.", ""]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=Path("results/test_failure_analysis.csv"))
    parser.add_argument("--readme", type=Path, default=Path("README.md"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261003)
    parser.add_argument("--resamples", type=int, default=20000)
    args = parser.parse_args()
    report = audit(args.csv, args.readme, seed=args.seed, resamples=args.resamples)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "historical_audit.json"
    markdown_path = args.output_dir / "historical_audit.md"
    json_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    markdown_path.write_text(render_markdown(report), encoding="utf-8")
    print(f"Historical audit saved: {json_path}, {markdown_path}")


if __name__ == "__main__":
    main()
