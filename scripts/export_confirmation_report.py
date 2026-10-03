"""Create a measured-input-only technical report for the frozen ToolRet study.

Requires matplotlib/reportlab in the authoring runtime. This script does not
modify experiments, metrics, decisions, or their provenance. It refuses a
pending audit or mismatched model/data fingerprints. Run from the repo root.
"""
from __future__ import annotations

import argparse
from html import escape
import hashlib
import json
import math
from pathlib import Path
from typing import Any


POLICIES = ("bm25", "dense", "hybrid", "always", "fixed_disagreement", "cheap_jaccard", "utility_25", "utility_50", "utility_75")
LABELS = {
    "bm25": "BM25", "dense": "Base MiniLM", "hybrid": "Hybrid RRF",
    "always": "Always rerank", "fixed_disagreement": "Disagreement gate",
    "cheap_jaccard": "Jaccard gate", "utility_25": "Utility 25% target",
    "utility_50": "Utility 50% target", "utility_75": "Utility 75% target",
}
DOMAINS = ("apigen", "toolbench", "toolace")
DOMAIN_LABELS = {"apigen": "APIGen", "toolbench": "ToolBench", "toolace": "ToolACE"}
REPOSITORY = "https://github.com/michaelbawuah/toolret-hybrid-retrieval"
REFERENCES = (
    ("Shi et al. (2025)", "Retrieval Models Aren't Tool-Savvy: Benchmarking Tool Retrieval for Large Language Models", "Findings of ACL 2025", "https://aclanthology.org/2025.findings-acl.1258/"),
    ("Zheng et al. (2024)", "ToolRerank: Adaptive and Hierarchy-Aware Reranking for Tool Retrieval", "LREC-COLING 2024", "https://aclanthology.org/2024.lrec-main.1413/"),
    ("Bacellar (2026)", "Per-Query Gating of LLM Rerankers for Multi-Hop Retrieval", "arXiv preprint 2609.22880", "https://arxiv.org/abs/2609.22880"),
    ("Wu, Guo and Li (2026)", "Lookahead-R: Budget-Aware Tool Retrieval via Execution-Centric Planning", "ICMR 2026; arXiv 2609.35811", "https://arxiv.org/abs/2609.35811"),
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Completed measured input is not ready: {path}")
    return json.loads(path.read_text())


def close(actual: float, expected: float) -> bool:
    return math.isclose(actual, expected, rel_tol=0, abs_tol=1e-12)


def load_inputs(study: Path, protocol_path: Path, failure_path: Path) -> dict[str, Any]:
    paths = {
        "evaluation": study / "confirmation_eval/summary.json",
        "runtime": study / "runtime_verification.json",
        "latency": study / "latency/summary.json",
        "audit": study / "independent_audit.json",
        "router": study / "router/model.json",
        "data": study / "data_manifest.json",
        "failure": failure_path,
        "freeze": study / "protocol_freeze_receipt.json",
        "completion": study / "study_completion.json",
        "reproduction": study / "reproduction_validation.json",
        "protocol": protocol_path,
    }
    values = {name: read(path) for name, path in paths.items()}
    evaluation, runtime, latency, audit = (values[name] for name in ("evaluation", "runtime", "latency", "audit"))
    if evaluation.get("scope") != "confirmation_study" or evaluation.get("num_queries") != 1500:
        raise ValueError("Report requires the exact 1,500-query confirmation evaluation")
    if not audit.get("all_executed_checks_pass") or audit.get("counts", {}).get("pending", 0) or audit.get("counts", {}).get("fail", 0):
        raise ValueError("The independent study audit is incomplete or failed")
    if any(check.get("status") != "pass" for check in audit.get("checks", [])):
        raise ValueError("Every reported independent audit check must pass")
    if not audit.get("checks"):
        raise ValueError("Independent audit checks are missing")
    if runtime.get("queries") != 1500 or not runtime.get("all_decisions_match_frozen_predictions") or not runtime.get("all_rankings_match_evaluation"):
        raise ValueError("Full-cohort primary actual runtime verification is incomplete")
    if latency.get("queries") != 90 or latency.get("repetitions") != 3 or not latency.get("all_actual_rankings_match_cache"):
        raise ValueError("Frozen repeated latency verification is incomplete")
    if set(latency["policies"]) != {"hybrid", "always", "fixed_disagreement", "utility_75"}:
        raise ValueError("Latency policy set differs from the frozen study")
    primary = evaluation["policies"]["utility_75"]
    if runtime["actual_backend_calls"] != primary["reranker_invocations"] or runtime["actual_backend_pairs"] != primary["reranker_candidate_pairs"]:
        raise ValueError("Actual primary CE work differs from evaluation accounting")
    integrity = evaluation["input_integrity"]
    hashes = {
        "router_sha256": digest(paths["router"]),
        "protocol_sha256": digest(paths["protocol"]),
        "evaluation_summary_sha256": digest(paths["evaluation"]),
    }
    for name in ("router_sha256", "protocol_sha256"):
        if integrity.get(name) != hashes[name] or runtime["input_integrity"].get(name) != hashes[name]:
            raise ValueError(f"Shared study fingerprint differs: {name}")
    if runtime["input_integrity"].get("evaluation_summary_sha256") != hashes["evaluation_summary_sha256"]:
        raise ValueError("Evaluation summary changed after actual runtime verification")
    for name in ("cache_sha256", "queries_sha256", "corpus_sha256", "predictions_sha256"):
        if integrity.get(name) != runtime["input_integrity"].get(name):
            raise ValueError(f"Runtime/evaluation input differs: {name}")
    mapping = {"router_model": "router_sha256", "protocol": "protocol_sha256", "cache": "cache_sha256", "queries": "queries_sha256", "corpus": "corpus_sha256"}
    for name, key in mapping.items():
        if latency["inputs_sha256"].get(name) != integrity.get(key):
            raise ValueError(f"Latency/evaluation input differs: {name}")
    for key in ("cache_sha256", "queries_sha256", "corpus_sha256"):
        if values["failure"]["provenance"].get(key) != integrity.get(key):
            raise ValueError(f"Failure/evaluation input differs: {key}")
    if values["failure"].get("scope") == "exploratory_original_pilot_failure_analysis":
        raise ValueError("Development failure analysis cannot stand in for confirmation diagnostics")
    if values["freeze"].get("protocol_sha256") != hashes["protocol_sha256"] or values["freeze"].get("published_before_new_rank_generation") is not True:
        raise ValueError("Prospective public protocol freeze is not verified")
    for policy in POLICIES:
        metric = evaluation["policies"][policy]["metrics"]
        if any(not math.isfinite(v) or not 0 <= v <= 1 for v in metric.values()):
            raise ValueError("Invalid measured retrieval metric")
    for source in DOMAINS:
        if evaluation["source_domain_breakdown"][source]["num_queries"] != 500:
            raise ValueError("Confirmation source quotas differ")
    if sum(v["positive_tool_components"] for v in evaluation["source_domain_breakdown"].values()) != 1174:
        raise ValueError("Confirmation component count differs")
    pair = evaluation["paired_nDCG_comparisons"]["utility_75_minus_always"]
    expected_delta = primary["metrics"]["nDCG@10"] - evaluation["policies"]["always"]["metrics"]["nDCG@10"]
    if not close(pair["source_stratified_query_bootstrap"]["delta_mean"], expected_delta):
        raise ValueError("Primary paired delta does not match aggregate metrics")
    success = all(pair[k]["low"] > -.01 for k in ("source_stratified_query_bootstrap", "source_stratified_component_bootstrap")) and runtime["actual_backend_calls"] < 1500
    if success != evaluation["primary_noninferiority"]["quality_and_required_call_criterion_met"]:
        raise ValueError("Final primary quality/call criterion disagrees with evaluator")
    completion = values["completion"]
    if (completion.get("completed_scope") is not True
            or completion.get("actual_conditional_execution_verified") is not True
            or completion.get("predeclared_primary_criterion_achieved") is not success
            or not completion.get("checks")
            or not all(value is True for value in completion["checks"].values())):
        raise ValueError("The companion final study-completion record is incomplete or inconsistent")
    roles = {"protocol": "protocol", "router": "router", "data": "data_manifest", "evaluation": "evaluation_summary", "runtime": "runtime_verification", "latency": "latency_summary", "failure": "failure_summary", "audit": "independent_audit", "freeze": "protocol_freeze_receipt"}
    for name, role in roles.items():
        item = completion.get("evidence", {}).get(role, {})
        if item.get("sha256") != digest(paths[name]):
            raise ValueError(f"Final study-completion fingerprint differs: {role}")
    reproduction = values["reproduction"]
    if (reproduction.get("default_runner_exit_code") != 0
            or reproduction.get("summary_numeric_results_and_all_other_fields_identical_except_evaluation_created_at_utc") is not True
            or reproduction["live_pinned_source_reconstruction"].get("all_five_public_parquets_downloaded_and_sha_verified") is not True
            or not all(item.get("identical_bytes") is True for item in reproduction.get("artifact_comparisons", {}).values())
            or len(reproduction.get("artifact_comparisons", {})) != 4):
        raise ValueError("Default cached replay / pinned source reconstruction is not verified")
    for name, item in reproduction["artifact_comparisons"].items():
        if item["sha256"] != digest(study / "confirmation_eval" / name):
            raise ValueError(f"Reproduction record does not match current evaluation: {name}")
    values.update({"paths": paths, "hashes": {name: digest(path) for name, path in paths.items()}, "primary_success": success})
    return values


def number(value: float, places: int = 4) -> str:
    return f"{value:.{places}f}"


def interval(value: dict, places: int = 4) -> str:
    def endpoint(v: float) -> str:
        precision = places
        while v != 0 and round(v, precision) == 0 and precision < 10:
            precision += 1
        return f"{v:+.{precision}f}"
    return f"[{endpoint(value['low'])}, {endpoint(value['high'])}]"


def delta_interval(value: dict) -> str:
    return f"{value['delta_mean']:+.4f} {interval(value)}"


def selected_cases(c: dict) -> list[dict]:
    """Take the first prespecified loss and win, not two loss-only anecdotes."""
    cases = [case for case in c["failure"].get("cases", []) if case.get("source_domain", "").casefold() == "toolbench"]
    output = []
    for outcome in ("loss", "win"):
        chosen = next((case for case in cases if case.get("outcome") == outcome), None)
        if chosen is not None:
            output.append(chosen)
    return output


def case_description(case: dict) -> str:
    note = case.get("manual_note", "")
    if note and not note.startswith("No manual semantic annotation"):
        return note
    if case["query_id"] == "toolbench_query_81":
        return ("Transactions and current-month quota are requested. CE promotes an unjudged ToolACE transaction tool; "
                "the labeled quota endpoint moves rank 2 to 15. Neither labeled pair is clipped. The primary router skips this rerank. "
                "Equivalence and execution are unverified.")
    if case["query_id"] == "toolbench_query_716":
        return ("Two labeled Nexweave template tools move from ranks 15 and 13 to 1 and 2. A third positive, the icon-search endpoint, "
                "is absent from fused candidates. Ordering improves, while a candidate miss remains outside the reranker's reach.")
    first = case["reranked_top1"]
    judged = "has a positive qrel" if first["is_positive_label"] else "has no positive qrel"
    return (f"The CE top result is {first.get('name') or first['tool_id']} and {judged}. "
            f"The scored prefix contains {case['prefix_positive_count']} of {case['positive_count']} positive tool labels. "
            f"Gold rank transitions are {case['gold_transitions']}. These observations describe ranking and labels, not verified execution correctness.")


def make_charts(c: dict, directory: Path) -> dict[str, Path]:
    """Draw the published measurements; never infer costs from call counts."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.ticker import PercentFormatter

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9.5,
        "axes.labelsize": 9.5, "axes.titlesize": 10,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": "#62676d", "axes.linewidth": .6,
        "xtick.color": "#454a50", "ytick.color": "#454a50",
        "text.color": "#20252b", "axes.labelcolor": "#20252b",
        "savefig.facecolor": "white", "pdf.fonttype": 42,
    })
    directory.mkdir(parents=True, exist_ok=True)
    blue, green, orange, grey, red = "#274b72", "#217367", "#ae6b29", "#929ba5", "#ad514b"
    output = {}

    def save(fig, key, stem):
        fig.savefig(directory / f"{stem}.png", dpi=300, bbox_inches="tight", pad_inches=.06)
        fig.savefig(directory / f"{stem}.pdf", bbox_inches="tight", pad_inches=.06)
        output[key] = directory / f"{stem}.png"
        plt.close(fig)

    e = c["evaluation"]
    fig, axes = plt.subplots(1, 2, figsize=(7.05, 2.45), gridspec_kw={"width_ratios": [1.22, 1]})
    offsets = {
        "hybrid": (8, -10), "always": (5, -12), "utility_25": (-8, 9),
        "utility_50": (-5, -13), "utility_75": (8, -2),
        "fixed_disagreement": (-6, 9), "cheap_jaccard": (-5, -11),
    }
    short = {"hybrid": "Hybrid", "always": "Always", "utility_25": "25%", "utility_50": "50%",
             "utility_75": "Learned 75%", "fixed_disagreement": "Disagreement", "cheap_jaccard": "Jaccard"}
    ax = axes[0]
    for name in ("hybrid", "always", "utility_25", "utility_50", "utility_75", "fixed_disagreement", "cheap_jaccard"):
        p = e["policies"][name]
        color = green if name == "utility_75" else orange if name == "fixed_disagreement" else blue if name in ("always", "hybrid") else grey
        marker = "D" if name == "fixed_disagreement" else "o"
        ax.scatter(p["reranker_invocation_fraction"], p["metrics"]["nDCG@10"], s=34, marker=marker, color=color, zorder=3)
        dx, dy = offsets[name]
        ax.annotate(short[name], (p["reranker_invocation_fraction"], p["metrics"]["nDCG@10"]),
                    textcoords="offset points", xytext=(dx, dy), ha="right" if dx < 0 else "left", fontsize=8.2, color=color)
    ax.set(xlim=(-.04, 1.15), ylim=(.516, .537), xlabel="Queries sent to cross-encoder", ylabel="nDCG@10")
    ax.xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    ax.set_title("(a) Quality and reranking budget", loc="left", pad=9)
    ax.grid(axis="y", color="#e6e8eb", linewidth=.6)
    ax.set_axisbelow(True)
    ax = axes[1]
    pair = e["paired_nDCG_comparisons"]["utility_75_minus_always"]
    for i, key in enumerate(("source_stratified_query_bootstrap", "source_stratified_component_bootstrap")):
        v = pair[key]
        ax.errorbar(v["delta_mean"], 1-i, xerr=[[v["delta_mean"]-v["low"]], [v["high"]-v["delta_mean"]]],
                    fmt="o", color=green, markersize=4.7, capsize=3, linewidth=1.3)
    ax.axvline(-.01, color=red, linestyle="--", linewidth=1)
    ax.axvline(0, color=grey, linewidth=.8)
    ax.set_yticks([1, 0], ["Query", "Component"])
    ax.set(xlim=(-.0125, .011), ylim=(-.65, 1.65), xlabel="Learned 75% minus always")
    ax.set_xticks([-.01, 0, .01], ["-0.01", "0", "+0.01"])
    ax.set_title("(b) Paired 95% intervals", loc="left", pad=9)
    ax.text(-.01, -.45, "loss tolerance", color=red, ha="center", fontsize=8)
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    fig.tight_layout(w_pad=1.7)
    save(fig, "quality", "confirmation_quality_compute")

    s = c["latency"]
    fig, axes = plt.subplots(1, 2, figsize=(7.05, 2.55), gridspec_kw={"width_ratios": [1, 1.05]})
    names = ("hybrid", "fixed_disagreement", "utility_75", "always")
    short_latency = ("Hybrid", "Disagreement", "Learned 75%", "Always")
    ax = axes[0]
    means = [s["policies"][name]["mean_ms"] for name in names]
    bars = ax.barh(range(4), means, color=[grey, orange, green, blue], height=.5)
    ax.set_yticks(range(4), short_latency)
    ax.invert_yaxis()
    ax.set(xlim=(0, 1150), xlabel="Mean retrieval time (ms)")
    for bar, value in zip(bars, means):
        ax.text(value+18, bar.get_y()+bar.get_height()/2, f"{value:.0f}", va="center", fontsize=9)
    ax.set_title("(a) Four measured policies", loc="left", pad=9)
    ax.grid(axis="x", color="#e6e8eb", linewidth=.6)
    ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax = axes[1]
    rows = c["latency_queries"]
    for routed, color, label in ((True, blue, "Reranked"), (False, green, "Skipped")):
        part = [r for r in rows if r["policies"]["utility_75"]["reranked"] is routed]
        ax.scatter([r["policies"]["always"]["mean_total_ms"] for r in part],
                   [r["policies"]["utility_75"]["mean_total_ms"] for r in part],
                   s=18, color=color, alpha=.75, linewidths=0, label=f"{label} ({len(part)})")
    maximum = max(r["policies"][n]["mean_total_ms"] for r in rows for n in ("always", "utility_75")) * 1.06
    ax.plot([0, maximum], [0, maximum], "--", color=grey, linewidth=.9)
    ax.set(xlim=(0, maximum), ylim=(0, maximum), xlabel="Always (ms)", ylabel="Learned 75% (ms)")
    ax.set_title("(b) Paired query times", loc="left", pad=9)
    ax.legend(loc="upper left", fontsize=8.1, frameon=False, handletextpad=.3)
    fig.tight_layout(w_pad=1.9)
    save(fig, "latency", "confirmation_latency")

    fig, axes = plt.subplots(1, 2, figsize=(7.05, 2.4), gridspec_kw={"width_ratios": [1, 1.1]})
    ax = axes[0]
    for i, source in enumerate(DOMAINS):
        d = e["source_domain_breakdown"][source]["policies"]
        change = d["always"]["metrics"]["nDCG@10"]-d["hybrid"]["metrics"]["nDCG@10"]
        ax.barh(i, change, height=.45, color=green if change >= 0 else red)
        ax.text(change + .002 if change >= 0 else change/2, i, f"{change:+.4f}",
                ha="left" if change >= 0 else "center", va="center", fontsize=9,
                color="#20252b" if change >= 0 else "white")
    ax.set_yticks(range(3), [DOMAIN_LABELS[n] for n in DOMAINS])
    ax.invert_yaxis()
    ax.axvline(0, color=grey, linewidth=.8)
    ax.set(xlim=(-.063, .052), xlabel="Always minus hybrid nDCG@10")
    ax.set_title("(a) Aggregate benefit varies by source", loc="left", pad=9)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax = axes[1]
    left = np.zeros(3)
    for outcome, label, color in (("win", "Improved", green), ("loss", "Worsened", red), ("unchanged", "Unchanged", "#cbd0d5")):
        vals = np.array([c["failure"]["groups"]["by_source_domain"][n]["outcomes"].get(outcome, 0) for n in DOMAINS])
        ax.barh(range(3), vals, left=left, height=.45, color=color, label=label)
        for i, value in enumerate(vals):
            ax.text(left[i]+value/2, i, str(value), ha="center", va="center", fontsize=8.6, color="white" if outcome != "unchanged" else "#30353b")
        left += vals
    ax.set_yticks(range(3), [DOMAIN_LABELS[n] for n in DOMAINS])
    ax.invert_yaxis()
    ax.set(xlim=(0, 500), xlabel="Queries (500 per source)")
    ax.set_title("(b) Per-query ranking outcomes", loc="left", pad=9)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.legend(loc="upper center", bbox_to_anchor=(.5, -.29), ncol=3, fontsize=7.9, frameon=False, columnspacing=.8, handlelength=1)
    fig.tight_layout(w_pad=2)
    save(fig, "sources", "confirmation_source_effects")
    return output


def report_blocks(c: dict, charts: dict[str, Path]) -> list[dict]:
    """One editorial source for PDF and Markdown; numbers come from sealed inputs."""
    e, s, f = (c[name] for name in ("evaluation", "latency", "failure"))
    p, a, fixed = (e["policies"][n] for n in ("utility_75", "always", "fixed_disagreement"))
    comp = e["paired_nDCG_comparisons"]
    pair = comp["utility_75_minus_always"]
    ciq, cic = (pair[n] for n in ("source_stratified_query_bootstrap", "source_stratified_component_bootstrap"))
    calls = c["runtime"]["actual_backend_calls"]
    time = s["paired_comparisons"]["utility_75_vs_always"]
    blocks = []
    def put(kind, **kw): blocks.append({"kind": kind, **kw})
    def para(text): put("paragraph", text=text)
    def heading(text, level=1): put("heading", text=text, level=level)
    def page(): put("page")
    def table(caption, headers, rows, widths, primary=None):
        put("table", caption=caption, headers=headers, rows=rows, widths=widths, primary=primary)
    def figure(key, caption, height): put("figure", key=key, caption=caption, height=height)

    put("title", text="When Is Tool Reranking Worth the Cost?", subtitle="Selective cross-encoder inference on a tool-disjoint ToolRet holdout")
    put("author", text="Michael Baffour Awuah", detail="Independent undergraduate research | October 3, 2026")
    heading("Abstract", 2)
    para(f"A cross-encoder can refine a retrieved tool list, but it adds inference time and can also demote a useful endpoint. "
         f"This study asks whether inexpensive signals from lexical and dense retrieval can identify queries worth reranking. "
         f"A ridge router is fitted on 300 development queries, calibrated on 300 separate queries, and tested on 1,500 queries "
         f"with no shared positively labeled tool IDs across the three sets. The searchable catalog contains 37,292 tools. "
         f"The preselected router makes {calls:,} cross-encoder calls, {1-calls/1500:.1%} fewer than always-on reranking, "
         f"with nDCG@10 of {p['metrics']['nDCG@10']:.4f} versus {a['metrics']['nDCG@10']:.4f}. Both paired 95% intervals satisfy "
         f"the predefined 0.01 loss tolerance. Mean warm CPU retrieval time falls by {time['mean_latency_reduction_fraction']:.1%} "
         f"on a separate 90-query, three-repetition benchmark. A simple top-result disagreement gate has higher observed "
         f"nDCG and fewer calls than the learned router. The results support selective inference in this configuration, "
         f"while leaving the value of learned allocation unresolved.")
    heading("1  Introduction")
    para("Consider a request for both transaction history and the current month's usage quota. A retriever must return "
         "endpoints for both operations. A model that moves a plausible transaction endpoint to the top can still make "
         "the list worse if it pushes the quota endpoint out of the first ten results. This occurs in the held-out example "
         "examined in Section 6. Tool retrieval therefore depends on coverage of a request as well as similarity to its wording.")
    para("The usual retrieve-then-rerank pipeline pays for the second model on every query. That is reasonable when reranking "
         "helps consistently. When its benefit varies, a cheaper first-stage signal may be enough to choose between the "
         "original list and the reranked list. The question here is whether such a choice can save actual inference work "
         "while keeping a declared level of ranking quality.")
    para("The study combines a frozen learned router, simple gates, and random allocations at matched call budgets. "
         "It separates ranking evaluation from conditional execution and measured elapsed time, then examines why the "
         "aggregate result differs across data sources. This is an empirical extension of existing gating methods to "
         "tool retrieval; the contribution is the controlled comparison and its reproducible evidence.")
    heading("2  Related work")
    para("ToolRet [1] provides heterogeneous tool-retrieval tasks and documents. ToolRerank [2] studies adaptive truncation "
         "and hierarchy-aware reranking, including the different needs of single-tool and multi-tool requests. "
         "Bacellar [3] investigates pre-reranker feature gating for multi-hop retrieval and reports harmful skips alongside "
         "aggregate performance. Lookahead-R [4] addresses budget-aware tool retrieval through execution-centric planning. "
         "The present study uses a smaller decision: rerank a fixed candidate prefix or return the fused first-stage list. "
         "The systems and evaluation scopes differ, so their reported scores are not used as directly comparable baselines.")

    page()
    heading("3  Experimental design")
    heading("3.1  Data and separation", 2)
    para("The corpus is the pinned ToolRet web-tool subset. The query universe contains 3,099 deduplicated APIGen, "
         "ToolBench, and ToolACE queries; APIBank queries are excluded. Queries that share a positive tool label are "
         "joined into connected components before partitioning. Keeping each component intact prevents the router "
         "from learning from a labeled endpoint that appears again in calibration or confirmation.")
    table("Table 1. Study cohorts. Each active cohort is balanced across the three sources.",
          ["Cohort", "Queries", "Components", "Use"],
          [["Development", "300", "252", "Fit and standardize the router"],
           ["Calibration", "300", "249", "Set score thresholds without labels"],
           ["Confirmation", "1,500", "1,174", "Evaluate the frozen policies"]], [.21,.12,.17,.50])
    para("The previously observed pilot supplies development data. Its components contain another 629 queries, which "
         "are quarantined. Whole remaining components are assigned by a deterministic hash order and quota-fitting rule; "
         "370 queries remain in reserve. The active sets share no query IDs, normalized query texts, positive tool IDs, "
         "or components. All tools remain searchable: the separation concerns supervised router exposure, not removal "
         "of held-out tools from the catalog. These components are not verified API families.")
    heading("3.2  Retrieval and the routing decision", 2)
    para("BM25 and normalized all-MiniLM-L6-v2 embeddings each retrieve 100 tools. Equal-weight reciprocal-rank fusion "
         "uses a constant of 60. The cross-encoder, ms-marco-MiniLM-L-6-v2, scores the first 20 fused candidates and "
         "reorders that prefix; the tail is retained. Both neural models are used off the shelf. Inference uses a maximum "
         "pair length of 256, batch size 64, four Torch threads, and one interop thread on CPU.")
    para("The router predicts the development-query change in nDCG@10 from reranking. Ridge regularization is fixed "
         "at 10; feature standardization uses development data only. Seven scalar features describe first-stage "
         "disagreement, overlap, rank coherence, query length, and the fused top-score margin (Appendix A). At inference, "
         "they require neither relevance labels nor cross-encoder scores. Tool IDs are used only for rank and equality comparisons.")
    put("equation", text="Predicted gain = ridge(first-stage features); rerank if predicted gain > frozen threshold.")
    para("Prediction quantiles on calibration set nominal call targets of 25%, 50%, and 75%. Ties skip. The 75% target "
         "is the sole primary policy; the other targets and simple gates are exploratory. Targets are calibration "
         "settings rather than exact quotas on new queries. The protocol was public before fresh rankings were "
         "generated, and the fitted router and thresholds were frozen before confirmation quality was evaluated.")
    heading("3.3  Metrics and the primary criterion", 2)
    para("Positive relevance grades are binarized. The primary metric is nDCG@10, with source means weighted equally. "
         "The router must use fewer cross-encoder calls and place both paired 95% interval lower bounds for its "
         "nDCG difference from always-on reranking above -0.01. The margin is a chosen engineering tolerance. "
         "The two bootstrap analyses resample queries and positive-tool components within source, respectively, "
         "using 10,000 draws. Component resampling retains a query-macro estimand through score-sum/query-count ratios.")

    page()
    heading("4  Ranking quality and inference work")
    table("Table 2. Confirmation results on 1,500 queries. Each cross-encoder call scores 20 pairs. "
          "Only learned 75% versus always is the primary comparison.",
          ["Policy", "nDCG@10", "MRR@10", "Recall@10", "Required CE calls", "Calls saved"],
          [[LABELS[n].replace("Utility", "Learned").replace(" target", ""),
            *[number(e["policies"][n]["metrics"][k]) for k in ("nDCG@10", "MRR@10", "Recall@10")],
            f"{e['policies'][n]['reranker_invocations']:,}", f"{1-e['policies'][n]['reranker_invocations']/1500:.1%}"] for n in POLICIES],
          [.29,.145,.14,.14,.12,.125], primary=9)
    figure("quality", "Figure 1. (a) Measured ranking quality against required call fraction. Learned-budget "
           "points other than the primary, and simple gates, are exploratory; the primary router is green. (b) Primary nDCG difference "
           "from always-on reranking. Both 95% intervals lie above the predefined loss tolerance and include zero.", 179)
    heading("4.1  The predefined tradeoff is achieved", 2)
    para(f"The primary difference is {ciq['delta_mean']:+.4f} nDCG@10: the query interval is {interval(ciq)} "
         f"and the component interval is {interval(cic)}. Both lower bounds exceed -0.01. Conditional execution "
         f"over every confirmation query verifies {calls:,} actual calls, {calls*20:,} scored pairs, and "
         f"{c['runtime']['skipped_calls']} skipped calls. Every routing decision and final ranking matches frozen evaluation. "
         "The result meets the stated quality-and-work criterion; the intervals do not establish a quality improvement.")
    heading("4.2  Learning is not the clear winner", 2)
    para(f"The disagreement gate reranks when BM25 and dense retrieval choose different top results. Its "
         f"nDCG@10 is {fixed['metrics']['nDCG@10']:.4f} with {fixed['reranker_invocations']:,} calls, giving a better "
         f"observed quality/cost point than the primary router. The learned-minus-disagreement component interval "
         f"is {interval(comp['utility_75_minus_fixed_disagreement']['source_stratified_component_bootstrap'])}. "
         "The primary policy remains fixed after evaluation.")

    page()
    heading("5  From avoided calls to elapsed time")
    para("Call counts do not measure user-visible delay. Skipping the cross-encoder still leaves BM25, dense "
         "encoding and search, fusion, and routing to run. A separate benchmark therefore recomputes the full "
         "query-to-ranking path for four policies, using 90 confirmation queries selected by hash, 30 per source. "
         "Each query is measured three times under each policy, for 1,080 requests in total.")
    figure("latency", "Figure 2. Warm sequential CPU retrieval on 90 queries, averaging three repetitions "
           "per query and policy. (a) Policy means. (b) Each point compares the same query under always and "
           "learned 75%. The dashed line marks equal time. Most savings occur on the skipped branch; reranked "
           "queries still incur cross-encoder inference.", 188)
    table("Table 3. Measured elapsed time. Percentiles describe query means over three repetitions, "
          "rather than individual request tail latency.",
          ["Policy", "Mean ms", "p50 ms", "p95 ms", "CE calls / requests"],
          [[LABELS[n].replace("Utility", "Learned").replace(" target", ""),
            f"{s['policies'][n]['mean_ms']:.1f}", f"{s['policies'][n]['p50_query_mean_ms']:.1f}",
            f"{s['policies'][n]['p95_query_mean_ms']:.1f}",
            f"{s['policies'][n]['actual_cross_encoder_calls']} / {s['policies'][n]['requests']}"]
           for n in ("hybrid", "always", "fixed_disagreement", "utility_75")], [.31,.14,.14,.14,.27], primary=4)
    heading("5.1  Measured speedup", 2)
    para(f"Mean time decreases from {s['policies']['always']['mean_ms']:.1f} ms to "
         f"{s['policies']['utility_75']['mean_ms']:.1f} ms, a {time['mean_latency_reduction_fraction']:.1%} reduction "
         f"(paired 95% interval [{time['reduction_ci95_fraction'][0]:.1%}, {time['reduction_ci95_fraction'][1]:.1%}]). "
         "This benchmark sends 219 of 270 primary-policy requests to the cross-encoder, saving 18.9% of calls. "
         "The 23.4% call saving in Section 4 describes the larger 1,500-query cohort. The different fractions "
         "reflect the benchmark subset and should not be treated as the same measurement.")
    heading("5.2  Measurement boundaries", 2)
    para("Models, the index, and corpus embeddings are already loaded. Timers include BM25, dense query encoding, "
         "exact corpus search and sorting, fusion, feature calculation, the routing decision, and conditional "
         "cross-encoder scoring and prefix sorting. Query and policy order are randomized and cyclically balanced. "
         "Every live ranking is checked against the frozen cache after timing.")
    para(f"The machine reports an AMD EPYC 9V74 CPU with nine logical CPUs available; Torch uses four threads "
         "and one interop thread. Loading, warmup, validation, networking, queueing, and concurrent serving are "
         "outside the measured interval. These results characterize warm sequential CPU retrieval on this machine. "
         "They leave production serving behavior and GPU performance open.")

    page()
    heading("6  Where reranking helps and hurts")
    para("The small aggregate gain from always-on reranking conceals a source-level reversal. It improves mean "
         "nDCG on APIGen and ToolACE while reducing it on ToolBench. The primary router sends 472, 303, and 374 "
         "queries from those sources to the cross-encoder, respectively. Source-matched controls matter because "
         "changing the allocation across these sources can change the overall score without better decisions within a source.")
    figure("sources", "Figure 3. Exploratory always-versus-hybrid diagnostics. Each source contributes 500 queries. "
           "The aggregate change (a) and per-query outcomes (b) reveal different reranking behavior across sources. "
           "Outcomes use binary-label nDCG@10, with a numerical equality tolerance of 1e-12.", 172)
    heading("6.1  Candidate coverage and request coverage", 2)
    para("On ToolBench, 184 queries lose nDCG under reranking and 116 improve. For 132 queries, none of the "
         "positive labels is in the scored prefix. A reranker cannot repair that candidate miss. For other queries, "
         "positively labeled candidates are present but their ordering deteriorates. These are different problems: improving "
         "candidate retrieval does not by itself solve the ordering of multiple requested operations.")
    table("Table 4. Two outcome-selected ToolBench cases. Names are shortened for readability; "
          "full IDs, query text, and tokenization are in the case artifacts. A dash indicates absence from fused candidates.",
          ["Query / labeled endpoint", "Hybrid rank", "CE rank", "Primary action"],
          [["81 / transaction history", "1", "4", "Skip reranking"],
           ["81 / monthly usage quota", "2", "15", "Skip reranking"],
           ["716 / all Nexweave templates", "15", "1", "Rerank"],
           ["716 / template details", "13", "2", "Rerank"],
           ["716 / icon search", "-", "-", "Rerank"]], [.48,.15,.13,.24])
    para("In query 81, the cross-encoder promotes a transaction endpoint from another corpus source that has no "
         "positive label for the query. Meanwhile, the labeled quota tool falls from rank 2 to rank 15, "
         "reducing nDCG by 0.7359. Neither positive pair is truncated. Skipping preserves both labeled endpoints "
         "at the top. Query 716 shows the converse: two template tools move into the first two positions, "
         "raising nDCG by 0.7654, but the missing icon-search endpoint remains unrecovered.")
    para("Of the 184 ToolBench losses, 152 have no truncated positive pair, so clipping a labeled tool "
         "cannot explain every loss. The cases illustrate ordering changes, rather than establish their cause "
         "or prevalence. They were selected by outcome size with query-ID tie breaks.")

    page()
    heading("7  Discussion")
    heading("7.1  The useful result is a deployment tradeoff", 2)
    para("For this frozen retrieval configuration, a gate can avoid real cross-encoder inference and meet a "
         "predefined ranking-loss tolerance. Conditional execution and elapsed-time measurement make the "
         "cost claim concrete. The result is narrower than saying reranking is unnecessary: some requests "
         "benefit substantially, and the router skips 60 queries whose nDCG would have improved. It also "
         "avoids 87 damaging reranks; the remaining 204 skipped queries are unchanged.")
    para("The simple disagreement rule is a consequential baseline, rather than a formality. It has the stronger "
         "observed quality/cost point and a lower measured mean time. The learned router's confidence intervals "
         "against disagreement and source-matched random allocation include zero. More model complexity "
         "is therefore not justified by this experiment alone. A new study should compare simple gating "
         "against learned allocation under another frozen cohort or retrieval configuration.")
    heading("7.2  Limits and next experiments", 2)
    para("The component split limits reuse of positive labels in supervised router fitting. It does not establish "
         "API-family separation or clean pretraining of the public neural models. Whole-component allocation "
         "and pilot quarantine also produce a particular benchmark cohort rather than a sample of live user traffic. "
         "Component bootstrap captures observed label sharing, while other semantic dependencies may remain.")
    para("Binary relevance judgments can omit interchangeable endpoints. Unjudged tools may be useful alternatives, "
         "but name similarity does not establish equivalence, and the original labels are retained. Recall and all-label coverage describe "
         "benchmark labels, and this study does not run the retrieved tools to completion. Controlled changes "
         "to document serialization and the sequence limit could help distinguish truncation from request-coverage "
         "effects. Independently judging candidate equivalence and testing downstream execution would address "
         "the larger question of whether a ranking improvement makes the agent more useful.")
    heading("8  Conclusion")
    para(f"The frozen learned router reduces actual cross-encoder calls by {1-calls/1500:.1%} across "
         "1,500 held-out queries and meets the declared 0.01 nDCG loss tolerance. A separate repeated CPU "
         f"benchmark measures {time['mean_latency_reduction_fraction']:.1%} lower mean retrieval time. "
         "The simpler disagreement gate has the better observed quality/cost point, and learned allocation "
         "superiority remains unresolved. The practical finding is that reranking should be evaluated as a "
         "query-dependent tradeoff, with simple controls and actual execution alongside aggregate ranking scores.")
    heading("References", 2)
    for i, (author, title, venue, url) in enumerate(REFERENCES, 1):
        put("reference", text=f"[{i}] {author}. {title}. {venue}.", url=url)

    page()
    heading("Appendix A  Router details and allocation controls")
    para("The following definitions are frozen in the protocol. Ranks are one-based; top-k lists contain unique "
         "tool IDs. Feature standardization is fitted on the 300 development rows. The ridge intercept is "
         "unpenalized, and the deployment decision uses a strict greater-than comparison with the threshold.")
    table("Table A1. Seven inference-time features. Two Jaccard cutoffs yield two separate features.",
          ["Feature", "Definition"],
          [["Top-1 disagreement", "1 if BM25 and dense top-1 IDs differ; otherwise 0."],
           ["Top-10 Jaccard", "Intersection size divided by union size over the two top-10 lists."],
           ["Top-20 Jaccard", "The same overlap measure at cutoff 20."],
           ["Weighted top-20 overlap", "Sum of minimum reciprocal-rank weights divided by sum of maximum weights over the union; missing-list weight is 0."],
           ["Top-20 rank coherence", "1 minus the mean absolute rank difference over the union, divided by 20. A missing rank is 21."],
           ["Log query length", "log1p of the whitespace-delimited query token count."],
           ["Normalized RRF margin", "(largest minus second-largest fused score) / largest score, using both top-100 lists. A singleton has margin 1."]], [.29,.71])
    table("Table A2. Calibration targets and realized confirmation work. Thresholds are set from "
          "label-free calibration predictions.",
          ["Call target", "Frozen threshold", "Test calls", "Realized call fraction"],
          [[f"{b}%", f"{c['router']['thresholds'][str(b/100)]['threshold']:.8f}", f"{e['policies'][f'utility_{b}']['reranker_invocations']:,}",
            f"{e['policies'][f'utility_{b}']['reranker_invocation_fraction']:.1%}"] for b in (25,50,75)], [.19,.28,.20,.33])
    para("The cheap Jaccard gate reranks when top-10 BM25/dense Jaccard overlap is below 0.5, "
         "a fixed rule without tuning. All secondary budget, gate, and "
         "allocation comparisons are exploratory. Global random controls sample the exact call count for "
         "each policy; source-matched controls additionally preserve calls per source. The range below "
         "describes variability across 20 seeds, rather than a confidence interval.")
    control_labels = {"global_fixed_disagreement":"Global / disagreement", "global_utility_25":"Global / learned 25%",
                      "global_utility_50":"Global / learned 50%", "global_utility_75":"Global / learned 75%",
                      "source_matched_fixed_disagreement":"Source-matched / disagreement", "source_matched_utility_75":"Source-matched / learned 75%"}
    table("Table A3. Random allocation controls, seeds 0-19.", ["Control", "Calls", "Mean nDCG", "Seed range"],
          [[control_labels[n], str(v["invocations_per_seed"]), f"{v['metrics']['nDCG@10']['mean_across_seeds']:.4f}",
            f"[{v['metrics']['nDCG@10']['min_across_seeds']:.4f}, {v['metrics']['nDCG@10']['max_across_seeds']:.4f}]"]
           for n,v in e["random_equal_budget_baselines"].items()], [.40,.12,.16,.32])
    para("A test-label-aware oracle at the primary call budget reaches nDCG@10 of "
         f"{e['oracle_matched_budget_ceiling']['utility_75']['nDCG@10']:.4f}. It selects queries using their true "
         "reranking gains and is only a diagnostic ceiling. It is unavailable to a deployed router.")

    page()
    heading("Appendix B  Additional results and reproduction")
    table("Table B1. Exploratory paired nDCG comparisons. Intervals use 10,000 paired "
          "source-stratified bootstrap draws, seed 20261003; no multiple-testing correction is applied.",
          ["Difference", "Mean", "Query 95% CI", "Component 95% CI"],
          [[label, f"{comp[key]['source_stratified_query_bootstrap']['delta_mean']:+.4f}",
            interval(comp[key]["source_stratified_query_bootstrap"]), interval(comp[key]["source_stratified_component_bootstrap"])]
           for key,label in (("fixed_disagreement_minus_always","Disagreement - always"),
                             ("hybrid_minus_always","Hybrid - always"),
                             ("utility_75_minus_fixed_disagreement","Learned 75% - disagreement"),
                             ("utility_75_minus_random_global_mean","Learned 75% - global random"),
                             ("utility_75_minus_random_source_matched_mean","Learned 75% - source random"),
                             ("fixed_disagreement_minus_random_global_mean","Disagreement - global random"),
                             ("fixed_disagreement_minus_random_source_matched_mean","Disagreement - source random"))], [.37,.10,.265,.265])
    table("Table B2. Source-level confirmation scores. Each source has 500 queries; these comparisons "
          "are exploratory.", ["Source", "Hybrid", "Always", "Disagreement", "Learned 75%"],
          [[DOMAIN_LABELS[n], *[number(e["source_domain_breakdown"][n]["policies"][v]["metrics"]["nDCG@10"])
                                  for v in ("hybrid","always","fixed_disagreement","utility_75")]] for n in DOMAINS], [.20,.18,.18,.22,.22])
    heading("B.1  Reproduction and verification", 2)
    para(f"An independent numerical implementation reconstructs components, router fitting and thresholds, "
         f"policy metrics, random allocations, bootstrap intervals, and runtime evidence. All {c['audit']['counts']['pass']} "
         "checks pass. The default reproduction runner was also executed: numerical results match, and frozen "
         "predictions, decisions, per-query metrics, and evaluation Markdown are byte-identical. Five live pinned "
         "parquet downloads reconstruct the canonical corpus and cohort fingerprints.")
    put("code", text=".venv/bin/python scripts/run_confirmation_study.py \\\n  --output-dir data/research_reproduction_run1")
    para("Install the recorded environment using docs/confirmation_reproduction.md. The default runner "
         "freshly evaluates published caches. --recompute regenerates rankings and the fit; --measure-runtime "
         "executes conditional inference and timing. These flags were not run in the default smoke test; "
         "the original study's inference and runtime are separate completed evidence.")
    heading("B.2  Evidence and pinned inputs", 2)
    para("The repository includes per-query metrics and decisions, raw latency events, failure cases, freeze "
         "receipts, source fingerprints, and the independent audit. The report sidecar binds its PDF, Markdown, "
         "and figures to the same measured inputs. Full hashes and model/data revisions are kept in the "
         "machine-readable artifacts rather than reproduced in the main paper. Exact model revisions and "
         "package versions are recorded in the protocol and research requirements.")
    put("code", text=f"Protocol freeze: {c['freeze']['public_protocol_commit']}\nRouter SHA-256: {c['hashes']['router']}")
    put("link", text="Code, measured evidence, and reproduction instructions", url=REPOSITORY)
    return blocks


def markdown(blocks: list[dict], charts: dict[str, Path], target: Path) -> None:
    import os
    lines = []
    for b in blocks:
        kind = b["kind"]
        if kind == "title": lines += ["# " + b["text"], "", b["subtitle"], ""]
        elif kind == "author": lines += [b["text"] + " | " + b["detail"], ""]
        elif kind == "heading": lines += [("## " if b["level"] == 1 else "### ") + b["text"], ""]
        elif kind in ("paragraph", "equation"): lines += [b["text"], ""]
        elif kind == "table":
            lines += ["**" + b["caption"] + "**", "", "| " + " | ".join(b["headers"]) + " |",
                      "| " + " | ".join("---" for _ in b["headers"]) + " |"]
            lines += ["| " + " | ".join(str(v) for v in row) + " |" for row in b["rows"]]
            lines.append("")
        elif kind == "figure":
            relative = os.path.relpath(charts[b["key"]].resolve(), target.parent.resolve())
            lines += [f"![{b['caption']}]({relative})", "", b["caption"], ""]
        elif kind == "code": lines += ["```text", b["text"], "```", ""]
        elif kind == "reference": lines += [b["text"] + " " + b["url"], ""]
        elif kind == "link": lines += [f"[{b['text']}]({b['url']})", ""]
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(line.rstrip() for line in lines).rstrip() + "\n")


def pdf(blocks: list[dict], charts: dict[str, Path], target: Path) -> None:
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak, KeepTogether
    import matplotlib
    fontroot = Path(matplotlib.get_data_path()) / "fonts/ttf"
    for name, filename in (("PaperSerif", "STIXGeneral.ttf"), ("PaperSerifBold", "STIXGeneralBol.ttf"),
                           ("PaperSerifItalic", "STIXGeneralItalic.ttf"), ("PaperSans", "DejaVuSans.ttf"),
                           ("PaperSansBold", "DejaVuSans-Bold.ttf"), ("PaperMono", "DejaVuSansMono.ttf")):
        pdfmetrics.registerFont(TTFont(name, str(fontroot / filename)))
    pdfmetrics.registerFontFamily("PaperSerif", normal="PaperSerif", bold="PaperSerifBold", italic="PaperSerifItalic", boldItalic="PaperSerifBold")
    pdfmetrics.registerFontFamily("PaperSans", normal="PaperSans", bold="PaperSansBold", italic="PaperSans", boldItalic="PaperSansBold")
    ink, muted, accent = (colors.HexColor(v) for v in ("#20252b", "#59616a", "#274b72"))
    width = 480
    styles = {
        "body": ParagraphStyle("Body", fontName="PaperSerif", fontSize=10.8, leading=14.1, textColor=ink,
                               alignment=TA_JUSTIFY, spaceAfter=8.5, splitLongWords=False),
        "h1": ParagraphStyle("H1", fontName="PaperSerifBold", fontSize=13.3, leading=17, textColor=ink, spaceBefore=9, spaceAfter=8, keepWithNext=True),
        "h2": ParagraphStyle("H2", fontName="PaperSerifBold", fontSize=10.8, leading=14.0, textColor=ink, spaceBefore=6, spaceAfter=5, keepWithNext=True),
        "title": ParagraphStyle("Title", fontName="PaperSerifBold", fontSize=21, leading=26, textColor=ink, alignment=TA_CENTER, spaceAfter=8),
        "subtitle": ParagraphStyle("Subtitle", fontName="PaperSerif", fontSize=10.5, leading=14.0, textColor=muted, alignment=TA_CENTER, spaceAfter=11),
        "author": ParagraphStyle("Author", fontName="PaperSerif", fontSize=10.5, leading=14.5, alignment=TA_CENTER, spaceAfter=4),
        "detail": ParagraphStyle("Detail", fontName="PaperSans", fontSize=8.5, leading=11.5, textColor=muted, alignment=TA_CENTER, spaceAfter=10),
        "caption": ParagraphStyle("Caption", fontName="PaperSans", fontSize=8.4, leading=11.3, textColor=muted, spaceAfter=6, keepWithNext=True),
        "figcaption": ParagraphStyle("FigureCaption", fontName="PaperSans", fontSize=8.4, leading=11.3, textColor=muted, spaceAfter=9),
        "cell": ParagraphStyle("Cell", fontName="PaperSans", fontSize=8.4, leading=11.2, textColor=ink),
        "head": ParagraphStyle("Head", fontName="PaperSansBold", fontSize=8.2, leading=10.8, textColor=ink),
        "reference": ParagraphStyle("Reference", fontName="PaperSerif", fontSize=8.5, leading=11.4, textColor=ink, spaceAfter=5),
        "code": ParagraphStyle("Code", fontName="PaperMono", fontSize=7.7, leading=11, textColor=ink, spaceBefore=2, spaceAfter=9),
        "equation": ParagraphStyle("Equation", fontName="PaperSerifItalic", fontSize=9.3, leading=13, textColor=ink, alignment=TA_CENTER, spaceBefore=3, spaceAfter=9),
    }
    story = []
    def p(text, key="body"): return Paragraph(escape(text), styles[key])
    for b in blocks:
        kind = b["kind"]
        if kind == "title": story.extend([p(b["text"], "title"), p(b["subtitle"], "subtitle")])
        elif kind == "author": story.extend([p(b["text"], "author"), p(b["detail"], "detail")])
        elif kind == "heading": story.append(p(b["text"], "h1" if b["level"] == 1 else "h2"))
        elif kind in ("paragraph", "equation"): story.append(p(b["text"], "body" if kind == "paragraph" else "equation"))
        elif kind == "page": story.append(PageBreak())
        elif kind == "table":
            cells = [[p(v, "head") for v in b["headers"]]] + [[p(str(v), "cell") for v in row] for row in b["rows"]]
            t = Table(cells, colWidths=[width*w for w in b["widths"]], hAlign="LEFT", repeatRows=1)
            commands = [("VALIGN",(0,0),(-1,-1),"TOP"), ("LEFTPADDING",(0,0),(-1,-1),5),
                        ("RIGHTPADDING",(0,0),(-1,-1),5), ("TOPPADDING",(0,0),(-1,-1),4),
                        ("BOTTOMPADDING",(0,0),(-1,-1),4), ("LINEABOVE",(0,0),(-1,0),.7,ink),
                        ("LINEBELOW",(0,0),(-1,0),.45,muted), ("LINEBELOW",(0,-1),(-1,-1),.7,ink)]
            if b["primary"] is not None:
                commands += [("BACKGROUND",(0,b["primary"]),(-1,b["primary"]),colors.HexColor("#edf3f5"))]
            t.setStyle(TableStyle(commands))
            story.append(KeepTogether([p(b["caption"], "caption"), t, Spacer(1,9)]))
        elif kind == "figure":
            from reportlab.lib.utils import ImageReader
            w,h = ImageReader(str(charts[b["key"]])).getSize()
            dw = min(width, b["height"]*w/h)
            story.append(KeepTogether([Image(str(charts[b["key"]]), width=dw, height=dw*h/w, hAlign="CENTER"),
                                      Spacer(1,5), p(b["caption"], "figcaption")]))
        elif kind == "code":
            # Wrap long hashes explicitly; table/path breaking is never left to glyph overflow.
            import textwrap
            lines = [part for line in b["text"].splitlines() for part in textwrap.wrap(line, width=91, subsequent_indent="  ", replace_whitespace=False)]
            story.append(Paragraph("<br/>".join(escape(v).replace(" ", "&#160;") for v in lines), styles["code"]))
        elif kind == "reference":
            story.append(Paragraph(escape(b["text"])+f" <link href='{b['url']}' color='#274b72'>{escape(b['url'])}</link>", styles["reference"]))
        elif kind == "link":
            story.append(Paragraph(f"<link href='{b['url']}' color='#274b72'>{escape(b['text'])}</link>", styles["reference"]))

    def furniture(canvas, doc):
        canvas.saveState()
        canvas.setFillColor(muted)
        canvas.setFont("PaperSans", 7.1)
        if doc.page > 1:
            canvas.drawString(66, 759, "WHEN IS TOOL RERANKING WORTH THE COST?")
            canvas.drawRightString(546, 759, "M. B. Awuah")
            canvas.setStrokeColor(colors.HexColor("#c5cbd1")); canvas.setLineWidth(.35)
            canvas.line(66, 751, 546, 751)
        canvas.drawString(66, 30, "ToolRet selective retrieval | October 2026")
        canvas.drawRightString(546, 30, str(doc.page))
        canvas.restoreState()
    target.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(str(target), pagesize=letter, leftMargin=60, rightMargin=60,
                            topMargin=54, bottomMargin=49, title="When Is Tool Reranking Worth the Cost?",
                            author="Michael Baffour Awuah", subject="Selective inference on a tool-disjoint ToolRet holdout")
    doc.build(story, onFirstPage=furniture, onLaterPages=furniture)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-dir", type=Path, default=Path("results/research_confirmation_20261003"))
    parser.add_argument("--protocol", type=Path, default=Path("configs/confirmation_protocol_20261003.json"))
    parser.add_argument("--failure-analysis", type=Path)
    parser.add_argument("--output-pdf", type=Path, required=True)
    parser.add_argument("--output-markdown", type=Path, default=Path("paper/tool_disjoint_reranking_study_20261003.md"))
    parser.add_argument("--figure-dir", type=Path, default=Path("paper/figures"))
    args = parser.parse_args()
    if not args.output_pdf.is_absolute(): parser.error("--output-pdf must be absolute")
    c = load_inputs(args.study_dir, args.protocol, args.failure_analysis or args.study_dir/"confirmation_failure_analysis.json")
    # Preserve the independent audit's entire evidence chain during editorial revisions.
    audited_files_verified = 0
    unavailable_source_inputs = []
    for path, expected in c["audit"]["input_evidence_sha256"].items():
        # Raw data and derived cohorts are reconstructible and deliberately
        # untracked. Rendering consumes published results, not those data bytes.
        if not Path(path).is_file() and path.startswith("data/"):
            unavailable_source_inputs.append(path)
            continue
        if digest(Path(path)) != expected:
            raise ValueError(f"Audited evidence changed: {path}")
        audited_files_verified += 1
    latency_path = args.study_dir/"latency/per_query.jsonl"
    c["paths"]["latency_per_query"] = latency_path
    c["hashes"]["latency_per_query"] = digest(latency_path)
    c["latency_queries"] = [json.loads(line) for line in latency_path.read_text().splitlines() if line.strip()]
    if len(c["latency_queries"]) != 90: raise ValueError("Expected 90 measured paired latency queries")
    charts = make_charts(c, args.figure_dir)
    blocks = report_blocks(c, charts)
    markdown(blocks, charts, args.output_markdown)
    pdf(blocks, charts, args.output_pdf)
    outputs = {"pdf":args.output_pdf, "markdown":args.output_markdown, **charts,
               **{name+"_vector": path.with_suffix(".pdf") for name,path in charts.items()}}
    def recorded_path(path: Path) -> str:
        try:
            return str(path.resolve().relative_to(Path.cwd().resolve()))
        except ValueError:
            return str(path.resolve())
    sidecar = {"scope":"completed_scoped_confirmation_study", "report_revision":"editorial-redesign-20261003",
               "input_paths":{name:recorded_path(path) for name,path in c["paths"].items()},
               "input_sha256":c["hashes"], "audited_evidence_files_verified":audited_files_verified,
               "unavailable_untracked_source_inputs_not_used_for_rendering":unavailable_source_inputs,
               "outputs":{name:{"path":recorded_path(path),"sha256":digest(path)} for name,path in outputs.items()},
               "report_exporter_sha256":digest(Path(__file__)),
               "primary_quality_and_verified_compute_criterion_met":c["primary_success"],
               "protocol_sha256":c["hashes"]["protocol"], "router_sha256":c["hashes"]["router"]}
    sidecar_path = args.output_pdf.with_suffix(".provenance.json")
    sidecar_path.write_text(json.dumps(sidecar,indent=2,sort_keys=True)+"\n")
    print(json.dumps({"outputs":{name:str(path.resolve()) for name,path in outputs.items()},
                      "provenance":str(sidecar_path),"primary_success":c["primary_success"]},indent=2))


if __name__ == "__main__":
    main()
