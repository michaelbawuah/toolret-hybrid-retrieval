"""Export the measured selective-tool-retrieval pilot as a four-page PDF.

Read-only with respect to experiment inputs. Requires matplotlib and reportlab;
run using the Codex primary Python runtime or install these optional packages.
All reported estimates come from the supplied evaluator/audit JSON files.
"""

from __future__ import annotations

import argparse
from datetime import date
from html import escape
import hashlib
import json
from pathlib import Path


POLICIES = ("bm25", "dense", "hybrid", "always", "gated")
LABELS = {
    "bm25": "BM25", "dense": "Base MiniLM", "hybrid": "Hybrid RRF",
    "always": "Always rerank", "gated": "Disagreement gate",
}
METRICS = ("MRR@10", "nDCG@10", "Recall@10", "All-positive-label coverage@10")


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_inputs(summary_path: Path, manifest_path: Path, historical_path: Path):
    for path in (summary_path, manifest_path, historical_path):
        if not path.is_file():
            raise FileNotFoundError(f"Measured input is not ready: {path}")
    summary = json.loads(summary_path.read_text())
    manifest = json.loads(manifest_path.read_text())
    historical = json.loads(historical_path.read_text())
    if summary["scope"] != "preliminary_fresh_pilot":
        raise ValueError("Only the fresh preliminary pilot is supported")
    if summary["manifest"] != manifest:
        raise ValueError("Evaluator summary does not match the supplied cache manifest")
    if summary["adaptive_end_to_end_latency_ms"] is not None:
        raise ValueError("This exporter expects no measured adaptive service latency")
    if summary["num_queries"] != 300 or manifest["cache_provenance"]["corpus_documents"] != 37292:
        raise ValueError("This report is scoped to the frozen 300-query, 37,292-tool pilot")
    if sorted(summary["source_domain_breakdown"]) != ["apigen", "toolace", "toolbench"]:
        raise ValueError("Source domains differ from the fixed pilot")
    for domain in summary["source_domain_breakdown"].values():
        if domain["num_queries"] != 100:
            raise ValueError("Each fixed source domain must contain 100 queries")
    for policy in POLICIES:
        for metric in METRICS:
            value = summary["policies"][policy]["metrics"][metric]
            if not 0 <= value <= 1:
                raise ValueError("Invalid retrieval metric")
    return summary, manifest, historical


def make_chart(summary: dict, target: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    colors = ["#a7b4c5", "#758ba6", "#4c6384", "#324c70", "#168b83"]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})
    figure, axes = plt.subplots(1, 2, figsize=(7.15, 2.55), gridspec_kw={"width_ratios": [1.35, 1]})
    quality = [summary["policies"][policy]["metrics"]["nDCG@10"] for policy in POLICIES]
    short = ["BM25", "MiniLM", "Hybrid", "Always", "Gate"]
    axes[0].barh(short, quality, color=colors, height=0.62)
    axes[0].invert_yaxis()
    axes[0].set_xlim(0, min(1, max(quality) * 1.27))
    axes[0].set_xlabel("nDCG@10 (binary relevance)")
    axes[0].set_title("Retrieval quality", loc="left", weight="bold", pad=10)
    for index, value in enumerate(quality):
        axes[0].text(value + 0.008, index, f"{value:.3f}", va="center", fontsize=8)
    names = ("hybrid", "always", "gated")
    calls = [summary["policies"][policy]["reranker_invocations"] for policy in names]
    axes[1].barh(["Hybrid", "Always", "Gate"], calls, color=[colors[2], colors[3], colors[4]], height=0.55)
    axes[1].invert_yaxis()
    axes[1].set_xlim(0, summary["num_queries"] * 1.17)
    axes[1].set_xlabel("Cross-encoder query invocations")
    axes[1].set_title("Reranking compute proxy", loc="left", weight="bold", pad=10)
    axes[1].xaxis.set_major_locator(MaxNLocator(integer=True, nbins=4))
    for index, value in enumerate(calls):
        axes[1].text(value + 4, index, str(value), va="center", fontsize=8)
    for axis in axes:
        axis.spines[["top", "right", "left"]].set_visible(False)
        axis.spines["bottom"].set_color("#d5dce5")
        axis.tick_params(axis="y", length=0)
        axis.grid(axis="x", color="#eef1f5", linewidth=0.7)
        axis.set_axisbelow(True)
    figure.tight_layout(w_pad=2.0)
    figure.savefig(target, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def export_report(summary_path: Path, manifest_path: Path, historical_path: Path,
                  output_pdf: Path, protocol: str, repository_url: str,
                  runtime_path: Path | None = None, independent_path: Path | None = None) -> None:
    summary, manifest, historical = load_inputs(summary_path, manifest_path, historical_path)
    runtime = json.loads(runtime_path.read_text()) if runtime_path else None
    independent = json.loads(independent_path.read_text()) if independent_path else None
    if runtime:
        if runtime["cache_sha256"] != manifest["cache_sha256"] or runtime["all_rankings_match_cache"] is not True:
            raise ValueError("Runtime verification does not match the ranking cache")
        if runtime["backend_calls"] != summary["policies"]["gated"]["reranker_invocations"]:
            raise ValueError("Verified runtime calls differ from evaluator counts")
        if runtime["backend_pairs"] != summary["policies"]["gated"]["reranker_candidate_pairs"]:
            raise ValueError("Verified runtime pairs differ from evaluator counts")
    if independent:
        if independent["cache_sha256"] != manifest["cache_sha256"] or not all(independent["checks"].values()):
            raise ValueError("Independent cache audit has failed checks")
        if independent["summary_comparison"]["all_checked_values_match"] is not True:
            raise ValueError("Independent metric recomputation differs from the evaluator")
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_LEFT
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.platypus import (
        Image, KeepTogether, PageBreak, Paragraph, Preformatted,
        SimpleDocTemplate, Spacer, Table, TableStyle,
    )

    if not output_pdf.is_absolute():
        raise ValueError("--output-pdf must be an absolute path")
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    asset_dir = output_pdf.parent / "report_assets"
    asset_dir.mkdir(exist_ok=True)
    chart = asset_dir / "selective_retrieval_quality_compute.png"
    make_chart(summary, chart)
    font_dir = Path("/usr/share/fonts/truetype/dejavu")
    pdfmetrics.registerFont(TTFont("ResearchSans", str(font_dir / "DejaVuSans.ttf")))
    pdfmetrics.registerFont(TTFont("ResearchSansBold", str(font_dir / "DejaVuSans-Bold.ttf")))
    pdfmetrics.registerFont(TTFont("ResearchMono", str(font_dir / "DejaVuSansMono.ttf")))
    pdfmetrics.registerFontFamily("ResearchSans", normal="ResearchSans", bold="ResearchSansBold")
    navy = colors.HexColor("#20334f")
    teal = colors.HexColor("#168b83")
    muted = colors.HexColor("#5d6a7b")
    pale = colors.HexColor("#eef3f7")
    rule = colors.HexColor("#dce3ea")
    styles = {
        "title": ParagraphStyle("title", fontName="ResearchSansBold", fontSize=25, leading=29, textColor=navy, spaceAfter=13),
        "subtitle": ParagraphStyle("subtitle", fontName="ResearchSans", fontSize=10.5, leading=15, textColor=muted, spaceAfter=13),
        "h1": ParagraphStyle("h1", fontName="ResearchSansBold", fontSize=15, leading=19, textColor=navy, spaceAfter=12),
        "h2": ParagraphStyle("h2", fontName="ResearchSansBold", fontSize=10.5, leading=14, textColor=navy, spaceBefore=11, spaceAfter=5),
        "body": ParagraphStyle("body", fontName="ResearchSans", fontSize=9.3, leading=13.7, textColor=navy, spaceAfter=8),
        "small": ParagraphStyle("small", fontName="ResearchSans", fontSize=8.1, leading=11.5, textColor=muted, spaceAfter=6),
        "cell": ParagraphStyle("cell", fontName="ResearchSans", fontSize=8.2, leading=11.2, textColor=navy),
        "header": ParagraphStyle("header", fontName="ResearchSansBold", fontSize=7.7, leading=10.5, textColor=colors.white),
        "code": ParagraphStyle("code", fontName="ResearchMono", fontSize=7.4, leading=10.5, textColor=navy, spaceAfter=8),
        "kicker": ParagraphStyle("kicker", fontName="ResearchSansBold", fontSize=8, leading=11, textColor=teal, spaceAfter=12),
    }
    width = 7.05 * inch
    document = SimpleDocTemplate(str(output_pdf), pagesize=letter,
                                 leftMargin=0.725 * inch, rightMargin=0.725 * inch,
                                 topMargin=0.66 * inch, bottomMargin=0.65 * inch,
                                 title="Selective Tool Retrieval for AI Agents",
                                 author="Michael Baffour Awuah",
                                 subject="Preliminary independent research; fixed source-domain ToolRet pilot")
    story = []

    def para(text: str, kind="body"):
        return Paragraph(text, styles[kind])

    def add(text: str, kind="body"):
        story.append(para(text, kind))

    def table(headers, rows, widths, highlight=None, font="cell"):
        cells = [[para(escape(str(value)), "header") for value in headers]]
        cells += [[para(escape(str(value)), font) for value in row] for row in rows]
        item = Table(cells, colWidths=widths, repeatRows=1, hAlign="LEFT")
        commands = [
            ("BACKGROUND", (0, 0), (-1, 0), navy),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 7),
            ("RIGHTPADDING", (0, 0), (-1, -1), 7),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, pale]),
            ("LINEBELOW", (0, 0), (-1, 0), 0.4, navy),
            ("LINEBELOW", (0, -1), (-1, -1), 0.4, rule),
        ]
        if highlight is not None:
            commands.append(("BACKGROUND", (0, highlight), (-1, highlight), colors.HexColor("#e6f4f1")))
        item.setStyle(TableStyle(commands))
        story.append(item)
        story.append(Spacer(1, 6))

    def number(value):
        return f"{value:.4f}"

    def ci_text(value):
        return f"{value['delta_mean']:+.4f} [{value['low']:+.4f}, {value['high']:+.4f}]"

    config = manifest["cache_provenance"]["protocol"]
    models = manifest["models"]
    call_reduction = 1 - summary["policies"]["gated"]["reranker_invocations"] / summary["policies"]["always"]["reranker_invocations"]
    pair_reduction = summary["gated_candidate_pair_reduction_fraction_vs_always"]
    ndcg_diff = summary["paired_comparisons"]["gated_minus_always"]["nDCG@10"]["paired_bootstrap"]
    bootstrap = summary["paired_comparisons"]["gated_minus_always"]["MRR@10"]["paired_bootstrap"]

    # Page 1: research question and frozen design.
    add("PRELIMINARY INDEPENDENT RESEARCH REPORT", "kicker")
    add("Selective Tool Retrieval<br/>for AI Agents", "title")
    add("Michael Baffour Awuah<br/>October 3, 2026", "subtitle")
    table(["Fresh queries", "Candidate tools", "Source domains", "Training in this pilot"],
          [["300", "37,292", "3 x 100 queries", "None"]], [width / 4] * 4)
    add("Research question", "h2")
    add("Can agreement between a sparse retriever and a dense retriever identify queries that can skip cross-encoder reranking, while preserving retrieval quality across several ToolRet source domains?")
    add("Abstract", "h2")
    add(f"This study evaluates a fixed, zero-training disagreement gate on 300 fresh queries over the ToolRet web corpus. The gate reranks a query only when BM25 and base MiniLM retrieve different top-ranked tools. It uses {call_reduction:.1%} fewer reranker invocations than always-on reranking. The gated-minus-always nDCG@10 difference is {ndcg_diff['delta_mean']:+.4f}, with a paired query-bootstrap 95% interval [{ndcg_diff['low']:+.4f}, {ndcg_diff['high']:+.4f}]. These findings describe this pilot and its compute counts; no adaptive service latency or downstream agent success was measured.")
    add("Fixed experimental design", "h2")
    add("Select 100 queries each from APIGen, ToolBench, and ToolACE by ascending SHA256(seed + NUL + query ID). Normalize Unicode, case, and whitespace; remove duplicates and APIBank text overlaps before selection. Every positive label must refer to a tool in the corpus. Retrieval uses the query alone; label documents and instructions are excluded from model inputs.")
    table(["Component", "Configuration"], [
        ["Dense retriever", models["dense"]["name"]],
        ["Cross-encoder", models["reranker"]["name"]],
        ["Candidates / fusion", f"Top {config['candidate_k']} per retriever; weighted RRF k={config['rrf_k']}, weights={config['rrf_weights']}"],
        ["Gate / rerank depth", f"BM25 top-1 != MiniLM top-1; rerank top {config['rerank_k']}"],
    ], [1.3 * inch, width - 1.3 * inch])
    add(f"Public pretrained models are pinned to revisions; neural sequence length is limited to {config['max_sequence_length']} tokens. No weights, gate thresholds, or retrieval settings were trained or selected using this pilot. This is an independent extension, not a submitted paper or a faculty-supervised appointment.", "small")
    story.append(PageBreak())

    # Page 2: directly measured quality and paired uncertainty.
    add("Results: retrieval quality and compute", "h1")
    add("All quality metrics use binary relevance and cutoff 10. MRR@10 measures the reciprocal rank of the first positive result; Recall@10 averages the fraction of each query's positive labels retrieved; nDCG@10 rewards their ranking positions. All-label coverage requires every positive label to appear in the top 10.", "small")
    rows = []
    for policy in POLICIES:
        result = summary["policies"][policy]
        rows.append([LABELS[policy], *[number(result["metrics"][metric]) for metric in METRICS],
                     str(result["reranker_invocations"]), str(result["reranker_candidate_pairs"])])
    table(["Policy", "MRR@10", "nDCG@10", "Recall@10", "All-label coverage", "CE calls", "CE pairs"],
          rows, [1.6 * inch, .85 * inch, .85 * inch, .85 * inch, 1.15 * inch, .85 * inch, .9 * inch], highlight=5)
    story.append(Image(str(chart), width=width, height=2.53 * inch))
    add("The chart reports retrieval quality and cross-encoder invocation counts. The right panel is a compute proxy, not measured response latency.", "small")
    add("Paired uncertainty", "h2")
    paired_rows = []
    for name, label in (("always_minus_hybrid", "Always - hybrid"), ("gated_minus_hybrid", "Gate - hybrid"), ("gated_minus_always", "Gate - always")):
        values = summary["paired_comparisons"][name]
        paired_rows.append([label, ci_text(values["nDCG@10"]["paired_bootstrap"]), ci_text(values["MRR@10"]["paired_bootstrap"])])
    table(["Comparison", "nDCG difference [95% CI]", "MRR difference [95% CI]"], paired_rows,
          [1.3 * inch, 2.875 * inch, 2.875 * inch])
    add(f"Intervals use {bootstrap['resamples']:,} paired query-bootstrap resamples, seed {bootstrap['seed']}, and percentile bounds. They assume exchangeable query observations and do not account for dependency within API families. They are exploratory, not multiplicity-adjusted.", "small")
    if ndcg_diff["low"] <= 0 <= ndcg_diff["high"]:
        add("The primary gate-versus-always interval includes zero. This pilot establishes neither superiority nor formal noninferiority; the interval does not prove equivalent quality.", "small")
    random_baseline = summary["random_equal_invocation_baseline"]
    rnd = random_baseline["metrics"]["nDCG@10"]
    add(f"An equal-invocation random policy across 20 fixed seeds achieves mean nDCG@10 {rnd['mean_across_seeds']:.4f} (seed range {rnd['min_across_seeds']:.4f}-{rnd['max_across_seeds']:.4f}), using {random_baseline['invocations_per_seed']} reranker calls per seed. This seed range describes random-policy variation; it is not a confidence interval.", "small")
    story.append(PageBreak())

    # Page 3: domain breakdown, latency boundary, historical discrepancies.
    add("Domain behavior and evidence boundaries", "h1")
    add("Source-domain breakdown", "h2")
    domain_rows = []
    for name, values in sorted(summary["source_domain_breakdown"].items()):
        metrics = values["metrics"]
        domain_rows.append([name, values["num_queries"], values["gate_invocations"],
                            number(metrics["hybrid"]["nDCG@10"]), number(metrics["always"]["nDCG@10"]),
                            number(metrics["gated"]["nDCG@10"]), number(metrics["gated"]["MRR@10"])])
    table(["Source", "Queries", "Gate calls", "Hybrid nDCG", "Always nDCG", "Gate nDCG", "Gate MRR"],
          domain_rows, [.9 * inch, .75 * inch, .8 * inch, 1.15 * inch, 1.15 * inch, 1.15 * inch, 1.15 * inch])
    regressions = [
        (name, values["metrics"]["gated"]["nDCG@10"] - values["metrics"]["hybrid"]["nDCG@10"])
        for name, values in sorted(summary["source_domain_breakdown"].items())
        if values["metrics"]["gated"]["nDCG@10"] < values["metrics"]["hybrid"]["nDCG@10"]
    ]
    if regressions:
        add("Domain regressions matter: " + "; ".join(
            f"{escape(name)} gated-minus-hybrid nDCG@10 is {difference:+.4f}"
            for name, difference in regressions
        ) + ". Aggregate performance does not imply a benefit for every source.", "small")
    add("Source datasets are not API families: upstream metadata contains no structured API-family identity. Means are over queries; source groups have equal sample sizes.", "small")
    add("Compute savings and measured components", "h2")
    add(f"The gate saves {call_reduction:.1%} of reranker invocations and {pair_reduction:.1%} of scoring pairs versus always reranking. The paired-evaluation cache computes always-on rankings for every query.")
    if runtime:
        add(f"A separate selective-runtime check actually made {runtime['backend_calls']} CE calls over {runtime['backend_pairs']:,} pairs and skipped {runtime['skipped_calls']} calls. All {runtime['queries']} resulting rankings matched the cache. It reused cached first-stage rankings and does not measure full request latency.", "small")
    if independent:
        add(f"Independent recomputation passed {len(independent['checks'])} integrity checks and matched all recorded metrics within tolerance.", "small")
    timing_rows = []
    for method in ("bm25", "dense", "hybrid", "reranked"):
        values = summary["supplied_component_timings_ms"][method]
        label = {"bm25": "BM25 search", "dense": "Dense encode + exact search", "hybrid": "RRF fusion", "reranked": "CE scoring + prefix sorting"}[method]
        timing_rows.append([label, f"{values['mean_ms']:.2f}", f"{values['p50_ms']:.2f}", f"{values['p95_ms']:.2f}"])
    table(["Measured component", "Mean ms", "p50 ms", "p95 ms"], timing_rows,
          [3.15 * inch, 1.3 * inch, 1.3 * inch, 1.3 * inch])
    timing = manifest["timing_provenance"]
    add(f"CPU, {timing['torch_threads']} PyTorch threads; warmed models; one measurement per query. Timings exclude setup, encoding/index build, network, concurrency, and service overhead. Counterfactual offline sums omit routing overhead and do not measure adaptive request latency.", "small")
    add("Historical result audit", "h2")
    add("The saved 16-query APIBank CSV exposed old README discrepancies when MRR was recomputed. These are historical audit findings, not new model evaluations.")
    old_rows = []
    for policy in ("bm25", "dense", "hybrid", "reranked"):
        values = historical["aggregates"][policy]
        old_rows.append([values["label"], f"{values['readme_mrr']:.3f}", f"{values['mrr']:.6f}",
                         "Yes" if values["matches_readme_to_three_decimals"] else "No"])
    table(["Historical system", "Old README MRR", "Saved-rank MRR", "Matches rounded value"], old_rows,
          [2.2 * inch, 1.6 * inch, 1.55 * inch, 1.7 * inch])
    add("Historical MRR has no cutoff at 10 and refers to unavailable trained checkpoints, so it is not comparable to this pilot's base-model MRR@10. Saved first-relevant ranks cannot recover Recall/nDCG, full rankings, or original run provenance. Inspected test data were preserved and excluded from the fresh pilot.", "small")
    story.append(PageBreak())

    # Page 4: reproducibility and honest scope.
    add("Reproducibility, limitations, and context", "h1")
    add("What the artifact supports", "h2")
    add("The artifact supports a preliminary study with pinned data/models, deterministic selection, ranking caches, a fixed label-independent gate, paired uncertainty, source breakdowns, and a historical audit. It does not establish technique priority, benchmark-wide superiority, production speedup, or downstream task completion.", "small")
    add("Limitations and next experiment", "h2")
    add("The 300-query pilot does not cover the full benchmark. Query dependence and incomplete relevance annotations can affect estimates; public-model pretraining overlap is unknown. All-label coverage does not establish mandatory-tool coverage. No model was trained here, and source domains do not define API-family holdouts.", "small")
    add("Next, establish API-family metadata independently of labels, reserve separate development/evaluation groups, expand the cohort, and measure adaptive request latency repeatedly. Fit future learned gates or thresholds outside today's pilot.", "small")
    add("Reproduce from the repository root", "h2")
    commands = (
        "D=data/research_20261003; R=results/research_20261003\n"
        "python scripts/prepare_research_pilot.py\n"
        "PYTHONPATH=src python scripts/generate_research_cache.py \\\n"
        "  --corpus \"$D/corpus.jsonl\" --queries \"$D/queries.jsonl\" \\\n"
        "  --data-manifest \"$D/manifest.json\" \\\n"
        f"  --protocol {protocol} \\\n"
        "  --output-dir \"$R/pilot_cache\" --embedding-dir \"$D/embeddings\"\n"
        "PYTHONPATH=src python scripts/evaluate_selective_reranking.py \\\n"
        "  --cache \"$R/pilot_cache/rankings.jsonl\" --corpus \"$D/corpus.jsonl\" \\\n"
        "  --queries \"$D/queries.jsonl\" --manifest \"$R/pilot_cache/manifest.json\" \\\n"
        "  --exclude-queries results/test_failure_analysis.csv \\\n"
        f"  --bootstrap-resamples {bootstrap['resamples']} --seed {bootstrap['seed']} \\\n"
        "  --output-dir \"$R/pilot_eval\""
    )
    story.append(Preformatted(commands, styles["code"]))
    add("Use Python 3.12 with CPU PyTorch, sentence-transformers, and pyarrow; exact executed package versions are in the cache manifest. Reruns require new output directories. Evidence JSON files retain complete hashes and timing provenance.", "small")
    add("Pinned model revisions", "h2")
    for role in ("dense", "reranker"):
        add(f"{escape(models[role]['name'])}<br/><font face='ResearchMono' size='7'>{models[role]['revision']}</font>", "small")
    add("Primary sources and related work", "h2")
    add("[1] Shi et al. (2025). <i>Retrieval Models Aren't Tool-Savvy: Benchmarking Tool Retrieval for Large Language Models.</i> Findings of ACL. ToolRet supplies the benchmark; this pilot uses only its web corpus.<br/><link href='https://aclanthology.org/2025.findings-acl.1258/'>https://aclanthology.org/2025.findings-acl.1258/</link>", "small")
    add("[2] Zheng et al. (2024). <i>ToolRerank: Adaptive and Hierarchy-Aware Reranking for Tool Retrieval.</i> LREC-COLING. Adaptive tool reranking predates this study.<br/><link href='https://aclanthology.org/2024.lrec-main.1413/'>https://aclanthology.org/2024.lrec-main.1413/</link>", "small")
    add("[3] <i>Lookahead-R: Budget-Aware Tool Retrieval via Execution-Centric Planning.</i> arXiv:2609.35811 (2026), preprint. Related budget-aware tool retrieval work.<br/><link href='https://arxiv.org/abs/2609.35811'>https://arxiv.org/abs/2609.35811</link>", "small")
    add("Data: <link href='https://huggingface.co/datasets/mangopy/ToolRet-Queries'>mangopy/ToolRet-Queries</link> and <link href='https://huggingface.co/datasets/mangopy/ToolRet-Tools'>mangopy/ToolRet-Tools</link>. Both dataset revisions and downloaded-file hashes are in the preparation manifest.", "small")
    if repository_url:
        add(f"Code: <link href='{escape(repository_url)}'>{escape(repository_url)}</link>", "small")

    def page_decoration(canvas, doc):
        canvas.saveState()
        page_width, page_height = letter
        canvas.setStrokeColor(rule)
        canvas.setLineWidth(0.5)
        canvas.line(document.leftMargin, .46 * inch, page_width - document.rightMargin, .46 * inch)
        canvas.setFont("ResearchSans", 7.2)
        canvas.setFillColor(muted)
        canvas.drawString(document.leftMargin, .29 * inch, "Michael Baffour Awuah | Preliminary independent research | October 3, 2026")
        canvas.drawRightString(page_width - document.rightMargin, .29 * inch, str(doc.page))
        if doc.page > 1:
            canvas.setFont("ResearchSansBold", 7.5)
            canvas.drawString(document.leftMargin, page_height - .36 * inch, "SELECTIVE TOOL RETRIEVAL FOR AI AGENTS")
        canvas.restoreState()

    document.build(story, onFirstPage=page_decoration, onLaterPages=page_decoration)
    metadata = {
        "report": str(output_pdf),
        "generated_from": {
            str(path): file_hash(path)
            for path in (summary_path, manifest_path, historical_path, runtime_path, independent_path)
            if path is not None
        },
        "author": "Michael Baffour Awuah", "report_date": date(2026, 10, 3).isoformat(),
        "adaptive_service_latency_claimed": False,
    }
    (output_pdf.parent / "research_report_provenance.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--historical-audit", type=Path, required=True)
    parser.add_argument("--output-pdf", type=Path, required=True)
    parser.add_argument("--protocol", required=True, help="Repository-relative fixed protocol path")
    parser.add_argument("--repository-url", default="")
    parser.add_argument("--runtime-verification", type=Path)
    parser.add_argument("--independent-audit", type=Path)
    args = parser.parse_args()
    export_report(args.summary, args.manifest, args.historical_audit, args.output_pdf,
                  args.protocol, args.repository_url, args.runtime_verification, args.independent_audit)


if __name__ == "__main__":
    main()
