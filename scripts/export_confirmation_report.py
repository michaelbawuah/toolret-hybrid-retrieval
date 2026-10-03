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
    return f"[{value['low']:+.{places}f}, {value['high']:+.{places}f}]"


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


def primary_statement(c: dict) -> str:
    e = c["evaluation"]
    p, a = (e["policies"][name] for name in ("utility_75", "always"))
    calls = c["runtime"]["actual_backend_calls"]
    pair = e["paired_nDCG_comparisons"]["utility_75_minus_always"]
    q, g = (pair[name] for name in ("source_stratified_query_bootstrap", "source_stratified_component_bootstrap"))
    verdict = "meets" if c["primary_success"] else "does not establish"
    return (f"The primary utility router {verdict} the predeclared 0.01 absolute nDCG engineering noninferiority criterion. "
            f"Its nDCG@10 is {p['metrics']['nDCG@10']:.4f}, versus {a['metrics']['nDCG@10']:.4f} for always-on reranking "
            f"(difference {q['delta_mean']:+.4f}; query 95% CI {interval(q)}; component 95% CI {interval(g)}). "
            f"Real conditional execution verified {calls:,} CE calls and {calls * 20:,} scored pairs across all 1,500 queries, "
            f"a {1 - calls / 1500:.1%} call reduction.")


def baseline_statement(c: dict) -> str:
    e = c["evaluation"]
    primary, fixed = (e["policies"][name] for name in ("utility_75", "fixed_disagreement"))
    comp = e["paired_nDCG_comparisons"]
    keys = ("utility_75_minus_fixed_disagreement", "utility_75_minus_random_source_matched_mean")
    includes_zero = all(comp[key]["source_stratified_component_bootstrap"]["low"] <= 0 <= comp[key]["source_stratified_component_bootstrap"]["high"] for key in keys)
    wording = "The simpler disagreement gate is competitive" if fixed["metrics"]["nDCG@10"] <= primary["metrics"]["nDCG@10"] else "The simpler disagreement gate has a stronger observed quality point estimate"
    return (f"{wording}: nDCG@10 {fixed['metrics']['nDCG@10']:.4f} with {fixed['reranker_invocations']:,} required calls, "
            f"versus {primary['metrics']['nDCG@10']:.4f} with {primary['reranker_invocations']:,} for the primary router. "
            + ("Exploratory learned-minus-disagreement and learned-minus-source-matched-random component intervals both include zero; learned allocation superiority is not established."
               if includes_zero else "These exploratory comparisons do not replace the fixed primary hypothesis or establish multiple-testing-adjusted superiority."))


def toolbench_statement(c: dict) -> str:
    f = c["failure"]
    g = f["groups"]["by_source_domain"]["toolbench"]
    unclipped = f["groups"]["toolbench_by_any_positive_prefix_truncation"].get("no", {})
    losses = g["outcomes"].get("loss", 0)
    no_clipped_loss = unclipped.get("outcomes", {}).get("loss", 0)
    return (f"ToolBench always-minus-hybrid nDCG changes by {g['mean_delta_ndcg10']:+.4f}; {losses} of 500 queries worsen. "
            f"Of those losses, {no_clipped_loss} have no truncated labeled pair. Clipping a positive is therefore not necessary for every loss; "
            "this does not identify a cause or rule out clipped distractors. Exact-ID judgment risk and multi-request coverage remain distinct hypotheses.")


def timing_statement(c: dict) -> str:
    s = c["latency"]
    p, a = (s["policies"][name] for name in ("utility_75", "always"))
    v = s["paired_comparisons"]["utility_75_vs_always"]
    lo, hi = v["reduction_ci95_fraction"]
    return (f"On 90 source-balanced confirmation queries with three repetitions per policy, measured warm sequential CPU "
            f"mean retrieval time is {p['mean_ms']:.1f} ms for the primary router and {a['mean_ms']:.1f} ms for always-on reranking. "
            f"The observed mean-time reduction is {v['mean_latency_reduction_fraction']:.1%} "
            f"(paired query-bootstrap 95% CI [{lo:.1%}, {hi:.1%}]). Initialization, network, queueing, and concurrent serving are excluded.")


def make_charts(c: dict, directory: Path) -> dict[str, Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.titleweight": "bold"})
    directory.mkdir(parents=True, exist_ok=True)
    e = c["evaluation"]
    navy, teal, grey = "#20334f", "#13877d", "#8393a8"
    figure, axes = plt.subplots(1, 2, figsize=(7.2, 2.55), gridspec_kw={"width_ratios": [1.25, 1]})
    ax = axes[0]
    names = ("hybrid", "utility_25", "utility_50", "utility_75", "always")
    x = [e["policies"][name]["reranker_invocation_fraction"] for name in names]
    y = [e["policies"][name]["metrics"]["nDCG@10"] for name in names]
    ax.plot(x, y, color=teal, linewidth=1.4, alpha=.8)
    for i, name in enumerate(names):
        ax.scatter(x[i], y[i], color=navy if name == "always" else teal, s=40 if name == "utility_75" else 27, zorder=4)
        label = {"hybrid": "Hybrid", "always": "Always", "utility_25": "25% target", "utility_50": "50% target", "utility_75": "75% target"}[name]
        ax.annotate(label, (x[i], y[i]), textcoords="offset points", xytext=(-5 if name == "always" else 3, 8), fontsize=7.4, ha="right" if name == "always" else "left")
    for name, marker in (("fixed_disagreement", "s"), ("cheap_jaccard", "^")):
        v = e["policies"][name]
        ax.scatter(v["reranker_invocation_fraction"], v["metrics"]["nDCG@10"], s=32, marker=marker, color=grey, label=LABELS[name], zorder=3)
    ax.set_xlim(-.04, 1.1)
    span = max(y) - min(y)
    ax.set_ylim(max(0, min(y) - max(.009, span * .3)), min(1, max(y) + max(.016, span * .8)))
    ax.xaxis.set_major_formatter(PercentFormatter(1))
    ax.set_xlabel("Realized CE call fraction")
    ax.set_ylabel("Binary nDCG@10")
    ax.set_title("Held-out quality / compute", loc="left", fontsize=10)
    ax.legend(loc="lower right", fontsize=6.6, frameon=False)
    ax = axes[1]
    pair = e["paired_nDCG_comparisons"]["utility_75_minus_always"]
    for i, (key, label) in enumerate((("source_stratified_query_bootstrap", "Query"), ("source_stratified_component_bootstrap", "Gold component"))):
        v = pair[key]
        ax.errorbar(v["delta_mean"], i, xerr=[[v["delta_mean"] - v["low"]], [v["high"] - v["delta_mean"]]], fmt="o", color=teal, capsize=4)
    ax.axvline(-.01, color="#b35b4a", linestyle="--", linewidth=1.2)
    ax.axvline(0, color=grey, linewidth=.9)
    ax.set_yticks([0, 1], ["Query", "Gold component"])
    ax.set_ylim(-.65, 1.65)
    ax.set_xlabel("Primary minus always nDCG@10")
    ax.set_title("Primary paired 95% intervals", loc="left", fontsize=10)
    ax.text(.02, .98, "Dashed line: -0.01 tolerance", transform=ax.transAxes, fontsize=7, color="#9b5146", va="top")
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="x", color="#e9edf2", linewidth=.7)
        ax.set_axisbelow(True)
    figure.tight_layout(w_pad=2)
    quality = directory / "confirmation_quality_compute.png"
    figure.savefig(quality, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    s = c["latency"]
    names = ("hybrid", "always", "fixed_disagreement", "utility_75")
    means = [s["policies"][name]["mean_ms"] for name in names]
    tails = [s["policies"][name]["p95_query_mean_ms"] for name in names]
    figure, ax = plt.subplots(figsize=(7.1, 2.3))
    ax.barh(range(4), means, color=[grey, navy, "#587595", teal], height=.55, label="Mean query time")
    ax.scatter(tails, range(4), marker="D", s=30, facecolors="white", edgecolors=navy, zorder=5, label="p95 of query means")
    ax.set_yticks(range(4), [LABELS[name] for name in names])
    ax.invert_yaxis()
    ax.set_xlim(0, max(tails) * 1.18)
    for i, value in enumerate(means):
        ax.text(value - max(tails) * .02, i, f"{value:.0f} ms", fontsize=7.5, color="white", ha="right", va="center", fontweight="bold")
    ax.set_xlabel("Warm sequential CPU query-to-ranking time (ms)")
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color="#e9edf2", linewidth=.7)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", fontsize=7.2, frameon=False)
    figure.tight_layout()
    latency = directory / "confirmation_latency.png"
    figure.savefig(latency, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return {"quality": quality, "latency": latency}


def quality_rows(c: dict) -> list[list[str]]:
    result = []
    for name in POLICIES:
        v = c["evaluation"]["policies"][name]
        m = v["metrics"]
        result.append([LABELS[name], *[number(m[key]) for key in ("nDCG@10", "MRR@10", "Recall@10", "All-positive-label coverage@10")], f"{v['reranker_invocations']:,}", f"{v['reranker_candidate_pairs']:,}"])
    return result


def markdown(c: dict, charts: dict[str, Path], target: Path) -> None:
    e, s, f = (c[name] for name in ("evaluation", "latency", "failure"))
    p = e["paired_nDCG_comparisons"]["utility_75_minus_always"]
    lines = ["# When to rerank tool retrieval: a component-disjoint empirical study", "", "Michael Baffour Awuah | Independent undergraduate research | October 3, 2026", "", "## Abstract", "", primary_statement(c), "", baseline_statement(c), "", timing_statement(c), "", "The contribution is a reproducible transfer/evaluation of known pre-reranker gating to tool retrieval, with strict supervised-router holdout, deployment-path verification, source-level diagnostics, and honest uncertainty. Learned gating itself is not a new algorithm. This is a completed empirical study under a declared scope, not an accepted publication or state-of-the-art claim.", "", "## Prospective design", "", "The fixed web corpus contains 37,292 tools. Among 3,099 deduplicated eligible APIGen, ToolBench, and ToolACE queries, all 252 connected components touched by the previously observed pilot are anchored; they contain 929 queries. The original 300 pilot queries train the router; 629 additional pilot-connected queries are quarantined. Fresh calibration contains 300 queries (100/source, 249 components), and untouched confirmation contains 1,500 (500/source, 1,174 components). A further 370 free queries are reserved.", "", "Components are formed transitively from shared positive gold tool IDs over the full eligible universe, with minimum query ID as component ID. Labels are used for structural grouping, but retrieval results are not. Entire groups are assigned by a deterministic hash and quota-fitting rule. Development, calibration, and confirmation share no query IDs, normalized query texts, positive tool IDs, or components. These are tool-ID components, not verified API families; the relevant held-out tools remain available in the retrieval corpus.", "", "The protocol and linked data manifests were publicly frozen before fresh rankings were generated. The old pilot is development data and is never counted as confirmation evidence.", "", "### Fixed retrieval and router", "", "BM25 and normalized off-the-shelf MiniLM each retrieve 100 candidates. Equal-weight reciprocal-rank fusion uses k=60. A pinned MS-MARCO MiniLM cross-encoder scores only the first 20 fused candidates at maximum pair length 256; the tail is unchanged. Ranking ties use tool IDs deterministically. Models run on CPU with four Torch threads, one interop thread, and batch size 64.", "", "Ridge regression (fixed alpha=10) predicts development-query nDCG@10(always)-nDCG@10(hybrid). Seven features use rank equality/overlap/positions and whitespace query length only; source IDs, lexical tool-ID content, qrels, and CE outcomes are excluded from inference features. Feature means/stds are fit only on the 300 development rows. Calibration is used only for label-free prediction quantiles; strict prediction > threshold routes a query, and ties skip. Target fractions are 25%, 50%, and 75%; realized held-out call fractions may differ.", "", "| Feature | Frozen definition |", "|---|---|" ]
    for name in c["protocol"]["utility_router"]["feature_order"]:
        lines.append(f"| {name} | {c['protocol']['utility_router']['feature_definitions'][name]} |")
    lines += ["", "### Primary and secondary analyses", "", "The sole primary policy is the 75% calibration-target utility router versus always-on reranking. The predeclared engineering tolerance is 0.01 absolute binary nDCG@10: both paired source-stratified query and gold-component two-sided 95% interval lower bounds must exceed -0.01, with fewer CE calls. This tolerance is a design choice, not an externally accepted equivalence threshold or a power guarantee.", "", "Both intervals use 10,000 paired bootstrap draws, seed 20261003, and equal 1/3 source weights. The component estimator samples groups within each source and uses the ratio of sampled query-score sums to sampled query counts, retaining the query-macro target. Other budgets, metrics, baselines, domains, and diagnostics are exploratory.", "", "## Untouched confirmation results", "", "| Policy | nDCG@10 | MRR@10 | Recall@10 | All-label coverage@10 | Required CE calls | Required CE pairs |", "|---|---:|---:|---:|---:|---:|---:|"]
    lines += ["| " + " | ".join(row) + " |" for row in quality_rows(c)]
    lines += ["", "All-label coverage means all positive benchmark relevance labels retrieved, not a verified mandatory execution-tool set.", "", "![Confirmation quality and compute](figures/confirmation_quality_compute.png)", "", "### Primary result", "", primary_statement(c), "", "| Paired interval | Mean nDCG difference | 95% interval | Lower bound > -0.01? |", "|---|---:|---|---|"]
    for key, name in (("source_stratified_query_bootstrap", "Source-stratified query"), ("source_stratified_component_bootstrap", "Source-stratified component")):
        v = p[key]; lines.append(f"| {name} | {v['delta_mean']:+.6f} | [{v['low']:+.6f}, {v['high']:+.6f}] | {v['low'] > -.01} |")
    lines += ["", "### Equal-compute random controls", "", "Twenty seeds 0-19 select exact realized full-cohort call counts. Global controls exist for each utility budget and the disagreement gate; additional controls match per-source counts for the primary router and disagreement gate. The per-query mean random result is used for exploratory paired comparisons. Seed ranges describe random-allocation variation, not confidence intervals.", "", "| Control | Calls/seed | Mean nDCG | Full seed range |", "|---|---:|---:|---|"]
    for name, control in e["random_equal_budget_baselines"].items():
        m = control["metrics"]["nDCG@10"]
        lines.append(f"| {name} | {control['invocations_per_seed']} | {m['mean_across_seeds']:.6f} | [{m['min_across_seeds']:.6f}, {m['max_across_seeds']:.6f}] |")
    lines += ["", "### Exploratory paired allocation comparisons", "", "| Comparison | Mean nDCG difference | Query 95% CI | Component 95% CI |", "|---|---:|---|---|"]
    for name, comparison in e["paired_nDCG_comparisons"].items():
        if name == "utility_75_minus_always":
            continue
        q,g=(comparison[key] for key in ("source_stratified_query_bootstrap","source_stratified_component_bootstrap"))
        lines.append(f"| {name} | {q['delta_mean']:+.6f} | {interval(q, 6)} | {interval(g, 6)} |")
    lines += ["", "These comparisons are exploratory, without a new primary-policy selection or a multiple-testing superiority claim. A label-aware matched-budget oracle is retained as an unattainable diagnostic, never a deployed result.", "", "| Matched policy budget | Oracle calls | Oracle nDCG@10 (test-label-aware) |", "|---|---:|---:|"]
    for name,v in e["oracle_matched_budget_ceiling"].items():
        lines.append(f"| {name} | {v['matched_invocations']} | {v['nDCG@10']:.6f} |")
    lines += ["", "## Real conditional execution and measured runtime", "", "The primary router was executed across all 1,500 cached first-stage rankings. Every decision matched the frozen prediction, and every actual CE/hybrid output matched the evaluated ranking. These requests verify conditional CE work, not end-to-end first-stage time.", "", timing_statement(c), "", "![Actual repeated CPU retrieval timing](figures/confirmation_latency.png)", "", "| Policy | Mean ms | p50 query-mean ms | p95 query-mean ms | Actual CE calls / measured requests |", "|---|---:|---:|---:|---|"]
    for name in ("hybrid", "always", "fixed_disagreement", "utility_75"):
        v = s["policies"][name]
        lines.append(f"| {LABELS[name]} | {v['mean_ms']:.2f} | {v['p50_query_mean_ms']:.2f} | {v['p95_query_mean_ms']:.2f} | {v['actual_cross_encoder_calls']} / {v['requests']} |")
    lines += ["", "Latency queries are selected independently by hash, 30/source, with three repetitions of each of four policies. Query and policy order are randomized and cyclically balanced. Timers cover BM25, dense query encoding, exact dense search/sort, RRF, router feature/prediction overhead, and conditional CE/prefix sorting. Every actual ranking is checked after timing. Models/index/embeddings remain loaded. Initialization, validation, warmup, network, queueing, and concurrent load are excluded. Repetitions are averaged within query before uncertainty calculations; all-event percentiles are separately recorded.", "", "## Source-level and failure diagnostics", "", "| Source | Components | Hybrid nDCG | Always nDCG | Disagreement nDCG | Primary nDCG | Primary calls |", "|---|---:|---:|---:|---:|---:|---:|"]
    for name in DOMAINS:
        d = e["source_domain_breakdown"][name]
        vals = [number(d["policies"][p]["metrics"]["nDCG@10"]) for p in ("hybrid", "always", "fixed_disagreement", "utility_75")]
        lines.append(f"| {DOMAIN_LABELS[name]} | {d['positive_tool_components']} | " + " | ".join(vals) + f" | {d['policies']['utility_75']['reranker_invocations']} |")
    lines += ["", "### Candidate scope and tokenization", "", "| Source | Candidate-label recall@20 | No positive in prefix | CE wins / losses / same | Truncated positive pairs / positive prefix pairs | Unjudged cross-source top1 |", "|---|---:|---:|---|---|---:|"]
    for name in DOMAINS:
        d = f["groups"]["by_source_domain"][name]; o = d["outcomes"]
        lines.append(f"| {DOMAIN_LABELS[name]} | {d['mean_candidate_label_recall20']:.4f} | {d['queries_without_positive_in_prefix20']} | {o.get('win',0)} / {o.get('loss',0)} / {o.get('unchanged',0)} | {d['positive_prefix_pairs_truncated']} / {d['positive_labels_in_prefix20']} | {d['top1_unjudged_cross_source_queries']} |")
    lines += ["", toolbench_statement(c)]
    lines += ["", "A prefix permutation cannot recover relevant tools outside the scored 20. Candidate misses and within-prefix ordering changes have different causes/remedies. The exact pinned pair tokenizer is replayed with longest-first truncation; associations with losses are descriptive, not causal. An unjudged cross-source tool may be relevant: original exact-ID qrels remain unchanged. Name similarity is a textual hint, not verified tool equivalence or grounds for changing labels.", "", "### Primary skipped-query effects", ""]
    effects = e["routing_effects"]["utility_75"]
    lines += [f"The primary router skips {c['runtime']['skipped_calls']} queries: {effects['harmful_skips_count']} harmful, {effects['beneficial_skips_count']} beneficial, and {effects['neutral_skips_count']} unchanged under binary qrels (numerical tolerance 1e-12). Summed nDCG gain forgone is {effects['harmful_skips_total_nDCG_gain_forgone']:.4f}; summed harm avoided is {effects['beneficial_skips_total_nDCG_harm_avoided']:.4f}. These are ranking-label effects, not demonstrated tool-execution failures.", "", "### Deterministically selected examples", "", "Cases are deliberately selected by outcome size with query-ID tie breaks and cannot estimate prevalence.", ""]
    for case in selected_cases(c):
        lines += [f"- **{case['query_id']}**: always-minus-hybrid nDCG change {case['delta_ndcg10']:+.4f}. {case_description(case)} "]
    lines += ["", "## Integrity and reproducibility", "", f"Independent audit: **{c['audit']['counts']['pass']} checks passed, zero pending/failed**. The audit independently reconstructs data components, router fit/thresholds, metrics/random decisions/intervals, and runtime evidence without importing study metric/router/evaluator implementations.", "", f"Public protocol freeze: `{c['freeze']['public_protocol_commit']}`. Router thresholds and model were frozen before confirmation quality. Data/model/source fingerprints accompany all results.", "", "| Evidence | SHA-256 |", "|---|---|"]
    for name in ("protocol", "data", "router", "evaluation", "runtime", "latency", "failure", "audit", "completion", "reproduction"):
        lines.append(f"| {name} | `{c['hashes'][name]}` |")
    lines += ["", "### Reproduction entry points", "", "Install the recorded research environment and CPU Torch build, then run from the repository root:", "", "```bash", ".venv/bin/python scripts/run_confirmation_study.py \\", "  --output-dir data/research_reproduction_run1", "```", "", "The default reconstructs missing source data from pinned parquets, verifies frozen file fingerprints, and freshly evaluates published ranking caches and the frozen router. Add `--recompute` to regenerate calibration/ridge fitting/confirmation rankings; add `--measure-runtime` to execute full-cohort conditional CE calls and the 90-query repeated latency benchmark. Missing corpus embeddings are regenerated for real timing. Full prerequisites and argument examples are maintained in docs/confirmation_reproduction.md and the README. Generated stages refuse unintended overwrite. GPU results are not interchangeable with the recorded CPU experiment.", "", "The default runner was actually executed: all numerical summary fields matched except provenance creation time; frozen predictions, decisions, per-query metrics, and evaluation Markdown were byte-identical. Five live pinned parquet downloads reconstructed the complete corpus, original pilot, and all five grouped cohort files at their canonical fingerprints. The runner's neural-recomputation and runtime flags were not executed in that smoke test; original-study inference and actual runtime are separate measured evidence.", "", "| Stage | Script / evidence |", "|---|---|"]
    for a,b in (("Run ordered full evidence workflow","scripts/run_confirmation_study.py"),("Prepare exact grouped splits","scripts/prepare_confirmation_study.py"),("Generate pinned first-stage and CE rankings","scripts/generate_confirmation_cache.py"),("Fit development-only router","scripts/fit_utility_router.py"),("Apply frozen confirmation policies","scripts/evaluate_confirmation_study.py"),("Analyze ordering/tokenization/judgments","scripts/analyze_confirmation_failures.py"),("Verify all primary conditional CE calls","scripts/verify_confirmation_runtime.py"),("Measure repeated actual latency","scripts/benchmark_selective_latency.py"),("Independently audit / finalize evidence","scripts/audit_confirmation_study.py / scripts/finalize_confirmation_study.py"),("Regenerate this report","scripts/export_confirmation_report.py")):
        lines.append(f"| {a} | `{b}` |")
    lines += ["", "### Pinned models and runtime", "", f"- Dense: `{c['protocol']['dense_model']}` at `{c['protocol']['dense_revision']}`.", f"- CE: `{c['protocol']['reranker_model']}` at `{c['protocol']['reranker_revision']}`.", f"- Runtime: `{s['environment']}`.", f"- Packages: `{s['packages']}`.", "", "## Discussion and limitations", "", "The result evaluates a fixed routing decision under one tool-retrieval configuration, not a universal replacement for reranking. The primary model remains selected even if a secondary policy has a better observed point estimate. A failure to meet the tolerance is a completed negative/inconclusive empirical finding, not permission to retune on the confirmation set.", "", "Gold-tool grouping limits direct supervised-router label overlap, but does not identify API providers or prove model pretraining independence. Public models may have encountered related benchmarks. Deterministic whole-group quota assignment and pilot-component quarantine define a particular benchmark cohort; it is not sampled live-user traffic or a representative census of APIs. Binary qrels can omit interchangeable tools; all-label coverage does not establish mandatory tools, functional equivalence, or successful agent execution. Component resampling covers observed positive-tool sharing but cannot account for every semantic dependency. Source-specific results and outcome-selected cases are exploratory. The tolerance is a declared engineering tradeoff rather than a universally justified equivalence margin. CPU timing describes a single warm sequential machine and excludes deployment/network/concurrency costs. No faculty supervision, peer review, publication acceptance, new algorithm, or cross-paper state of the art is claimed.", "", "## Conclusion", "", primary_statement(c), "", "The full protocol-to-runtime evidence chain closes this scoped empirical study. Future independently frozen work can test additional retriever/reranker families, controlled truncation/serialization ablations, independently judged tool equivalence, and downstream execution success.", "", "## References", ""]
    for author,title,venue,url in REFERENCES:
        lines += [f"- {author}. [{title}]({url}). {venue}."]
    lines += ["", f"Code and measured artifacts: {REPOSITORY}", ""]
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(line.rstrip() for line in lines))


def pdf(c: dict, charts: dict[str, Path], target: Path) -> None:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
    fonts = Path("/usr/share/fonts/truetype/dejavu")
    for name, filename in (("StudySans", "DejaVuSans.ttf"), ("StudyBold", "DejaVuSans-Bold.ttf"), ("StudyMono", "DejaVuSansMono.ttf")):
        pdfmetrics.registerFont(TTFont(name, str(fonts / filename)))
    pdfmetrics.registerFontFamily("StudySans", normal="StudySans", bold="StudyBold")
    navy, teal, muted, pale = map(colors.HexColor, ("#20334f", "#13877d", "#617084", "#eef3f7"))
    styles = {
        "title": ParagraphStyle("title", fontName="StudyBold", fontSize=27, leading=32, textColor=navy, spaceAfter=12),
        "h1": ParagraphStyle("h1", fontName="StudyBold", fontSize=17, leading=22, textColor=navy, spaceAfter=12),
        "h2": ParagraphStyle("h2", fontName="StudyBold", fontSize=10.5, leading=14, textColor=navy, spaceBefore=11, spaceAfter=5),
        "body": ParagraphStyle("body", fontName="StudySans", fontSize=9.15, leading=13.4, textColor=navy, spaceAfter=8),
        "small": ParagraphStyle("small", fontName="StudySans", fontSize=8, leading=11.3, textColor=muted, spaceAfter=6),
        "cell": ParagraphStyle("cell", fontName="StudySans", fontSize=7.65, leading=10.5, textColor=navy),
        "header": ParagraphStyle("header", fontName="StudyBold", fontSize=7.5, leading=10.2, textColor=colors.white),
        "mono": ParagraphStyle("mono", fontName="StudyMono", fontSize=7.1, leading=10.2, textColor=navy, spaceAfter=6),
        "kicker": ParagraphStyle("kicker", fontName="StudyBold", fontSize=8, leading=11, textColor=teal, spaceAfter=13),
    }
    width = 7.05 * inch
    story = []
    def p(text: Any, kind="body"):
        return Paragraph(str(text), styles[kind])
    def add(text: Any, kind="body"):
        story.append(p(text, kind))
    def table(headers: list[str], rows: list[list[Any]], fractions: list[float], highlight: int | None = None):
        cells = [[p(escape(str(v)), "header") for v in headers]]
        cells += [[p(escape(str(v)), "cell") for v in row] for row in rows]
        t = Table(cells, colWidths=[width * f for f in fractions], repeatRows=1, hAlign="LEFT")
        commands = [("BACKGROUND",(0,0),(-1,0),navy),("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white,pale]),("VALIGN",(0,0),(-1,-1),"TOP"),("LEFTPADDING",(0,0),(-1,-1),6),("RIGHTPADDING",(0,0),(-1,-1),6),("TOPPADDING",(0,0),(-1,-1),3),("BOTTOMPADDING",(0,0),(-1,-1),3),("LINEBELOW",(0,-1),(-1,-1),.4,colors.HexColor("#d6dfe8"))]
        if highlight is not None:
            commands.append(("BACKGROUND",(0,highlight),(-1,highlight),colors.HexColor("#e5f3f0")))
        t.setStyle(TableStyle(commands)); story.append(t); story.append(Spacer(1,7))
    def new(title: str):
        story.append(PageBreak()); add(title,"h1")
    def image(path: Path, height: float):
        from reportlab.lib.utils import ImageReader
        w,h=ImageReader(str(path)).getSize()
        display_width=min(width,height*w/h)
        story.append(Image(str(path),width=display_width,height=display_width*h/w,hAlign="CENTER"));story.append(Spacer(1,6))
    e,s,f = (c[name] for name in ("evaluation","latency","failure"))
    pair = e["paired_nDCG_comparisons"]["utility_75_minus_always"]
    calls = c["runtime"]["actual_backend_calls"]
    add("INDEPENDENT UNDERGRADUATE RESEARCH - TECHNICAL REPORT","kicker")
    add("When to rerank<br/>tool retrieval","title")
    add("A component-disjoint empirical study<br/>Michael Baffour Awuah | October 3, 2026","small")
    table(["Held-out queries","Fixed tool corpus","Disjoint gold-tool groups","Full execution check"],[["1,500 / three sources","37,292 tools","1,174 components","1,500 primary decisions"]],[.25]*4)
    add("Abstract","h2"); add(primary_statement(c)); add(baseline_statement(c)); add(timing_statement(c))
    add("Question and declared contribution","h2")
    add("Can cheap first-stage ranking signals predict when tool reranking is worth its compute? This study transfers a known learned-gating design to tool retrieval and tests it using a strict supervised-router holdout, fixed budgets, matched random allocation, full conditional-execution checks, and repeated actual CPU timing.")
    add("Relation to existing work","h2")
    add("ToolRet supplies heterogeneous tasks/documents [1]; ToolRerank studies adaptive truncation and hierarchy [2]. Learned feature gating, held-out selection and matched controls already appear in Bacellar [3]; Lookahead-R studies budget-aware tool planning [4]. This study contributes transfer, execution evidence and diagnostics, rather than a new gating algorithm.")
    add("The complete result is reported under its frozen scope, including negative or inconclusive outcomes. No accepted publication, faculty supervision, or cross-paper state-of-the-art claim is implied.","small")
    new("1. Frozen design and supervised-router holdout")
    table(["Partition","Queries","Gold components","Permitted use"],[["Observed development","300 (100/source)","252","Sole supervised fit and standardization"],["Fresh calibration","300 (100/source)","249","Prediction quantiles only; no labels for fit/choice"],["Untouched confirmation","1,500 (500/source)","1,174","Frozen policies, metrics, runtime and diagnostics"]],[.22,.19,.16,.43])
    add("The 3,099 eligible queries form transitive shared-positive-tool groups. Pilot-touching groups anchor 929 queries: 300 train and 629 others are quarantined. Remaining whole groups are assigned by fixed hash/quota, reserving 370 queries.")
    add("The three partitions share no query IDs, normalized texts, positive tool IDs, or components. Labels define groups, not performance selection. These are not verified API families; held-out tools remain searchable.","small")
    add("Fixed first-stage features","h2")
    features = [("Top-1 disagreement","Indicator that BM25 and dense top results differ."),("Jaccard overlap, top 10 / top 20","Intersection / union of IDs at each cutoff."),("Reciprocal-weighted overlap, top 20","Sum of minimum reciprocal-rank weights / sum of maximum weights; absent weight zero."),("Rank coherence, top 20","1 - mean absolute rank difference / 20; missing position 21."),("Query length","log1p(whitespace token count)."),("Normalized RRF margin","Top-1 to top-2 score gap / top-1 score, from both top-100 lists.")]
    table(["Seven scalar features","Definition"],[list(row) for row in features],[.37,.63])
    add("Fitting and calibration","h2")
    add("Ridge(alpha 10) predicts development nDCG(always)-nDCG(hybrid). Means/stds use development only; the intercept is unpenalized. Calibration-score quantiles fix 25/50/75% target thresholds. Route iff prediction > threshold; ties skip. Confirmation call fractions can differ.")
    add("BM25/base MiniLM top 100 each; equal RRF k=60; CE top 20 only, unchanged tail. Pair length 256; CPU threads 4, interop 1, batch 64. Inference features exclude source/query IDs, lexical tool-ID content, qrels and CE outcomes.","small")
    new("2. Untouched confirmation: quality and compute")
    add("Binary relevance at cutoff 10; source means have equal 1/3 weights. Counts are policy-required CE work; primary counts were also verified by real conditional inference.","small")
    table(["Policy","nDCG@10","MRR@10","Recall@10","All-label coverage","CE calls","CE pairs"],quality_rows(c),[.28,.12,.12,.12,.14,.10,.12],highlight=9)
    image(charts["quality"],145)
    add("Sole primary hypothesis: utility 75% versus always","h2")
    table(["Paired bootstrap","Difference /95% interval","Lower bound > -0.01"],[["Source-stratified query",delta_interval(pair["source_stratified_query_bootstrap"]),str(pair["source_stratified_query_bootstrap"]["low"] > -.01)],["Source-stratified gold component",delta_interval(pair["source_stratified_component_bootstrap"]),str(pair["source_stratified_component_bootstrap"]["low"] > -.01)]],[.37,.43,.20])
    add(("Primary criterion met." if c["primary_success"] else "Primary noninferiority claim not established.") + " Both lower bounds must exceed -0.01 and CE calls must be fewer. The engineering tolerance is not universal equivalence or a power guarantee.")
    add("Paired percentile procedures: 10,000 draws, seed 20261003. Component draws preserve query means through per-source ratios. Smaller budgets, controls and sources are exploratory. All-label coverage concerns qrels, not mandatory execution tools.","small")
    new("3. Allocation controls and secondary tradeoffs")
    add(baseline_statement(c))
    add("Random controls distinguish reduced compute from informed allocation. Twenty seeds 0-19 match exact realized call counts. Global controls match total calls; source-matched controls also preserve call counts in each source. The primary policy is not chosen from these comparisons.")
    rows=[]
    for name,control in e["random_equal_budget_baselines"].items():
        m=control["metrics"]["nDCG@10"]
        name=name.replace("global_","Global / ").replace("source_matched_","Source matched / ").replace("utility_","utility ").replace("fixed_disagreement","fixed gate")
        rows.append([name,control["invocations_per_seed"],f"{m['mean_across_seeds']:.4f}",f"[{m['min_across_seeds']:.4f}, {m['max_across_seeds']:.4f}]"])
    table(["Random allocation control","Calls/seed","Mean nDCG","Full seed range"],rows,[.40,.15,.18,.27])
    add("Exploratory paired differences","h2")
    rows=[]
    for compared_policy in ("utility_75","fixed_disagreement"):
        for kind in ("global","source_matched"):
            comp=e["paired_nDCG_comparisons"][f"{compared_policy}_minus_random_{kind}_mean"]
            q,g=(comp[key] for key in ("source_stratified_query_bootstrap","source_stratified_component_bootstrap"))
            rows.append([("Primary" if compared_policy=="utility_75" else "Fixed gate")+" minus "+("global" if kind=="global" else "source-matched")+" random",f"{q['delta_mean']:+.4f}",interval(q, 5),interval(g, 5)])
    table(["Policy minus per-query 20-seed mean","Delta nDCG","Query 95% CI","Component 95% CI"],rows,[.36,.14,.25,.25])
    add("Paired intervals resample queries/components after averaging random-policy outcomes within query. Seed ranges describe allocation randomness, not uncertainty on new queries. These secondary comparisons do not replace the prespecified noninferiority hypothesis or establish multiple-testing-adjusted superiority.","small")
    add("Label-aware budget ceiling: diagnostic only","h2")
    table(["Realized budget from","Matched calls","Oracle nDCG@10"],[[LABELS[name],v["matched_invocations"],f"{v['nDCG@10']:.4f}"] for name,v in e["oracle_matched_budget_ceiling"].items()],[.45,.25,.30])
    add("The oracle allocates CE calls to the largest true confirmation-query gains. It uses test labels and reranking outcomes unavailable to the deployed router. It bounds attainable quality for this exact-call allocation problem; it is neither a usable policy nor a competitive result.","small")
    new("4. Actual conditional execution and CPU runtime")
    add(timing_statement(c))
    image(charts["latency"],135)
    table(["Policy","Mean ms","p50 query mean","p95 query mean","Actual CE calls / requests"],[[LABELS[name],f"{s['policies'][name]['mean_ms']:.1f}",f"{s['policies'][name]['p50_query_mean_ms']:.1f}",f"{s['policies'][name]['p95_query_mean_ms']:.1f}",f"{s['policies'][name]['actual_cross_encoder_calls']} / {s['policies'][name]['requests']}"] for name in ("hybrid","always","fixed_disagreement","utility_75")],[.29,.13,.16,.16,.26],highlight=4)
    add("Full-cohort primary verification","h2")
    add(f"Across 1,500 queries, actual execution made {calls:,} CE calls and {calls * 20:,} scores, skipping {1500-calls:,} calls. Every route and final ranking matched frozen/evaluated expectations. First-stage rankings are cached in this check; end-to-end timing is separate.")
    add("Measurement and uncertainty","h2")
    add("Hash-selected 30 queries/source, three repeats and four policies give 1,080 requests. Query order is randomized; policy positions are cyclically balanced. Timers include BM25, dense encoding/full search/sort, RRF, router features/prediction, and conditional CE/prefix sorting.")
    add("Loaded models/index/embeddings remain in memory. Loading, warmup, validation, network, queues and concurrency are excluded. Repeats are averaged within query before paired source-stratified bootstrap. p50/p95 describe query means; separate event-level percentiles remain in raw evidence.","small")
    add(f"Runtime: {escape(s['environment'].get('cpu_model','CPU'))}; Torch threads {s['environment']['torch_threads']}, interop {s['environment']['torch_interop_threads']}; Python {s['environment']['python']}. Single-machine warm sequential behavior does not establish production serving speed.","small")
    new("5. Source behavior and failure diagnostics")
    rows=[]
    for name in DOMAINS:
        d=e["source_domain_breakdown"][name]
        rows.append([DOMAIN_LABELS[name],d["positive_tool_components"],*[number(d["policies"][v]["metrics"]["nDCG@10"]) for v in ("hybrid","always","fixed_disagreement","utility_75")],d["policies"]["utility_75"]["reranker_invocations"]])
    table(["Source","Groups","Hybrid","Always","Fixed gate","Primary","Primary calls"],rows,[.18,.10,.14,.14,.16,.14,.14])
    add("All source-specific results are exploratory. Aggregate performance can conceal source regressions; no source is dropped and no budget is reselected using these outcomes.","small")
    rows=[]
    for name in DOMAINS:
        d=f["groups"]["by_source_domain"][name];o=d["outcomes"]
        rows.append([DOMAIN_LABELS[name],number(d["mean_candidate_label_recall20"]),d["queries_without_positive_in_prefix20"],f"{o.get('win',0)} / {o.get('loss',0)} / {o.get('unchanged',0)}",f"{d['positive_prefix_pairs_truncated']} / {d['positive_labels_in_prefix20']}",d["top1_unjudged_cross_source_queries"]])
    add("Candidate scope and judgment risk","h2")
    table(["Source","Label recall@20","No gold in top 20","CE wins/losses/same","Truncated gold pairs","Unjudged cross-source top 1"],rows,[.16,.15,.12,.20,.18,.19])
    add("The fixed CE prefix cannot recover tools it never scores. Exact pair-tokenizer replay measures query/document clipping, but truncation associations do not identify why a rank changed. An unjudged cross-source tool can still be useful; exact-ID qrels remain unchanged. Name counterparts are a textual warning, not independently established semantic equivalence.")
    add(toolbench_statement(c),"small")
    add("What the primary router skips","h2")
    effects=e["routing_effects"]["utility_75"]
    table(["Harmful skips","Beneficial skips","Unchanged skips","nDCG gain forgone / harm avoided"],[[effects["harmful_skips_count"],effects["beneficial_skips_count"],effects["neutral_skips_count"],f"{effects['harmful_skips_total_nDCG_gain_forgone']:.3f} / {effects['beneficial_skips_total_nDCG_harm_avoided']:.3f}"]],[.23,.23,.23,.31])
    add("Harmful means reranking would increase qrel nDCG but is skipped; beneficial means a reranking loss is avoided. Counts use numerical tolerance 1e-12. These are ranking-label effects, not demonstrated execution failures.","small")
    cases=selected_cases(c)
    if cases:
        add("Outcome-selected case review","h2")
        for case in cases:
            note=case_description(case)
            add(f"<b>{escape(case['query_id'])}</b> ({case['delta_ndcg10']:+.4f} CE-minus-hybrid nDCG): {escape(note)}","small")
        add("Cases are deterministically selected by outcome size and query-ID ties. They cannot estimate prevalence or justify a post-confirmation policy change.","small")
    new("6. Evidence chain and reproducibility")
    add(f"Independent audit: <b>{c['audit']['counts']['pass']} checks passed; zero failed or pending.</b> Data grouping, router fit/thresholds, metric/random/interval results, and runtime evidence are reconstructed independently of the study metric/router/evaluator implementations.")
    add(f"Public prospective protocol commit: <font name='StudyMono'>{c['freeze']['public_protocol_commit']}</font>. Protocol/data freeze preceded fresh rankings; the model and label-free thresholds were serialized before confirmation quality. Evaluation, full conditional verification and latency use the same router, protocol, corpus, query, and cache fingerprints.","small")
    table(["Evidence","SHA-256 prefix (full fingerprint chain in Markdown/sidecar)"],[[name,c["hashes"][name][:24]] for name in ("protocol","router","evaluation","runtime","latency")],[.25,.75])
    add("Reproduction stages","h2")
    add(".venv/bin/python scripts/run_confirmation_study.py<br/>  --output-dir data/research_reproduction_run1","mono")
    add("Default: reconstruct pinned source data, check hashes, and freshly evaluate published caches/router. Add --recompute to rerun calibration/fit/rank generation; add --measure-runtime for full conditional CE execution and repeated real timing. Missing embeddings are regenerated.","small")
    add("Verified default replay: numerical results match, and predictions/decisions/per-query/report bytes match exactly. Five live source downloads reconstruct all canonical corpus/cohort fingerprints. Neural-recomputation/runtime flags were not run in this smoke; original-study inference/runtime are separate evidence.","small")
    stages=[("Data / pinned rankings","prepare_confirmation_study.py / generate_confirmation_cache.py"),("Fit / frozen evaluation","fit_utility_router.py / evaluate_confirmation_study.py"),("Conditional execution / timing","verify_confirmation_runtime.py / benchmark_selective_latency.py"),("Failure diagnostics","analyze_confirmation_failures.py"),("Audit / finalization / report","audit_confirmation_study.py / finalize_confirmation_study.py / export_confirmation_report.py")]
    table(["Stage","Entry point under scripts/"],[list(row) for row in stages],[.45,.55])
    add("Run from the repository root with PYTHONPATH=src, the recorded research dependencies, CPU Torch wheel, and pinned Hugging Face revisions. The README supplies complete argument examples. Checkpointed embeddings avoid recomputing the corpus; generated evidence refuses unintended overwrite.","small")
    add("Pinned model checkpoints","h2")
    add(escape(c["protocol"]["dense_model"])+"<br/>"+c["protocol"]["dense_revision"],"mono")
    add(escape(c["protocol"]["reranker_model"])+"<br/>"+c["protocol"]["reranker_revision"],"mono")
    add(f"Recorded libraries: {escape(str(s['packages']))}","small")
    new("7. Interpretation, limitations and references")
    add("What this result supports","h2")
    add(primary_statement(c))
    add("The study evaluates a fixed router under one retrieval configuration. The primary policy remains fixed even if a secondary policy has a stronger observed point estimate. A failure to establish the tolerance is a completed negative/inconclusive result; it does not authorize tuning on the held-out cohort.")
    add("Limits on generalization","h2")
    limitations=["Tool-ID components constrain direct supervised label overlap but are not verified API families. All corpus tools remain searchable; this is a router-training holdout, not absent-tool retrieval.","Public pretrained models may have seen related benchmark content. Project-level group separation does not demonstrate pretraining independence.","Whole-group quota assignment and pilot-component quarantine define a particular benchmark cohort, not live-user traffic or a representative API census.","Binary qrels can omit interchangeable tools. All-label coverage, cross-source promotions, and case review do not prove mandatory-tool sets, equivalence, or downstream execution success.","The 0.01 tolerance is a declared engineering tradeoff. Both intervals remain visible. Component resampling covers observed positive-tool sharing, not all semantic dependence; secondary comparisons are exploratory.","Single-machine warm sequential CPU timing excludes loading, networking, queueing and concurrency. No GPU or production service conclusion follows."]
    for text in limitations:
        add(text,"small")
    add("Next independently frozen questions","h2")
    add("Additional retriever/reranker families, controlled truncation and document-serialization ablations, independently judged tool equivalence, and downstream execution would test how far these observations transfer. None is counted as completed in this report.","small")
    add("References","h2")
    for i,(author,title,venue,url) in enumerate(REFERENCES,1):
        add(f"[{i}] {escape(author)}. {escape(title)}. {escape(venue)}.<br/><link href='{url}' color='#13877d'>{url}</link>","small")
    add(f"<link href='{REPOSITORY}' color='#13877d'>{REPOSITORY}</link>","small")
    def footer(canvas,doc):
        canvas.saveState(); canvas.setStrokeColor(colors.HexColor("#d7e0e8")); canvas.setLineWidth(.4);canvas.line(.725*inch,.51*inch,7.775*inch,.51*inch)
        canvas.setFont("StudySans",7);canvas.setFillColor(muted);canvas.drawString(.725*inch,.34*inch,"Michael Baffour Awuah | ToolRet empirical study | October 3, 2026");canvas.drawRightString(7.775*inch,.34*inch,str(doc.page));canvas.restoreState()
    target.parent.mkdir(parents=True,exist_ok=True)
    doc=SimpleDocTemplate(str(target),pagesize=letter,leftMargin=.725*inch,rightMargin=.725*inch,topMargin=.64*inch,bottomMargin=.68*inch,title="When to rerank tool retrieval: a component-disjoint empirical study",author="Michael Baffour Awuah",subject="Completed scoped independent undergraduate empirical research")
    doc.build(story,onFirstPage=footer,onLaterPages=footer)


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-dir",type=Path,default=Path("results/research_confirmation_20261003"))
    parser.add_argument("--protocol",type=Path,default=Path("configs/confirmation_protocol_20261003.json"))
    parser.add_argument("--failure-analysis",type=Path)
    parser.add_argument("--output-pdf",type=Path,required=True)
    parser.add_argument("--output-markdown",type=Path,default=Path("paper/tool_disjoint_reranking_study_20261003.md"))
    parser.add_argument("--figure-dir",type=Path,default=Path("paper/figures"))
    args=parser.parse_args()
    if not args.output_pdf.is_absolute():
        parser.error("--output-pdf must be absolute")
    failure=args.failure_analysis or args.study_dir/"confirmation_failure_analysis.json"
    c=load_inputs(args.study_dir,args.protocol,failure)
    charts=make_charts(c,args.figure_dir)
    markdown(c,charts,args.output_markdown)
    pdf(c,charts,args.output_pdf)
    outputs={"pdf":args.output_pdf,"markdown":args.output_markdown,**charts}
    sidecar={"scope":"completed_scoped_confirmation_study","input_paths":{name:str(path.resolve()) for name,path in c["paths"].items()},"input_sha256":c["hashes"],"outputs":{name:{"path":str(path.resolve()),"sha256":digest(path)} for name,path in outputs.items()},"primary_quality_and_verified_compute_criterion_met":c["primary_success"],"protocol_sha256":c["hashes"]["protocol"],"router_sha256":c["hashes"]["router"]}
    sidecar_path=args.output_pdf.with_suffix(".provenance.json")
    sidecar_path.write_text(json.dumps(sidecar,indent=2,sort_keys=True)+"\n")
    print(json.dumps({"outputs":{name:str(path.resolve()) for name,path in outputs.items()},"provenance":str(sidecar_path),"primary_success":c["primary_success"]},indent=2))


if __name__=="__main__":
    main()
