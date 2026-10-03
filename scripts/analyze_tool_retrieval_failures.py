"""Describe retrieval/reranking failures without using findings to tune a router.

This exploratory audit retains the original binary qrels. Names that resemble
an unjudged tool are evidence of a judgment risk, never automatic new labels.
Only tokenizer inference is used: no model predictions or training occur.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

from toolret_research.text import flatten_tool


MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
REVISION = "233902d25c440f23af6f7d6e94d2946bac0bee0a"


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def index_unique(rows: list[dict], field: str) -> dict[str, dict]:
    out = {}
    for row in rows:
        key = str(row[field])
        if key in out:
            raise ValueError(f"Duplicate {field}: {key}")
        out[key] = row
    return out


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rank_of(ranking: list[str], tool_id: str) -> int | None:
    return ranking.index(tool_id) + 1 if tool_id in ranking else None


def ndcg(ranking: list[str], positives: set[str], cutoff: int = 10) -> float:
    if not positives:
        raise ValueError("Empty relevance labels are not supported.")
    ideal = sum(1 / math.log2(rank + 1) for rank in range(1, min(cutoff, len(positives)) + 1))
    return sum(1 / math.log2(rank + 1) for rank, tool in enumerate(ranking[:cutoff], 1)
               if tool in positives) / ideal


def rank_transition(before: int | None, after: int | None, cutoff: int = 10) -> str:
    if before is None and after is None:
        return "absent_from_fused_candidates"
    if before is None or after is None:
        raise ValueError("Reranking unexpectedly changed candidate membership.")
    if before <= cutoff < after:
        return "left_top10"
    if after <= cutoff < before:
        return "entered_top10"
    if before <= cutoff and after <= cutoff:
        return "promoted_within_top10" if after < before else "demoted_within_top10" if after > before else "unchanged_top10"
    return "outside_top10"


def tool_name(raw: dict) -> str:
    try:
        body = json.loads(raw["text"])
        return str(body.get("name", "")) if isinstance(body, dict) else ""
    except (TypeError, json.JSONDecodeError):
        return ""


def possible_name_counterpart(first: str, second: str) -> bool:
    """A reproducible name hint, not semantic equivalence or a relevance label."""
    def normal(value: str) -> str:
        value = re.sub(r"([a-z])([A-Z])", r"\1_\2", value)
        return "_".join(re.findall(r"[a-z0-9]+", value.lower()))
    left, right = normal(first), normal(second)
    shorter, longer = sorted([left, right], key=len)
    return len(shorter) >= 12 and (shorter == longer or longer.endswith("_" + shorter))


def exact_pair_tokens(tokenizer, query: str, documents: list[str], full_query_tokens: int,
                      full_document_tokens: list[int], max_length: int) -> list[dict]:
    """Count exact retained query/document tokens under the production tokenizer."""
    encoded = tokenizer([query] * len(documents), documents, truncation=True,
                        max_length=max_length, padding=False, return_special_tokens_mask=True)
    out = []
    special = tokenizer.num_special_tokens_to_add(pair=True)
    for i, (types, mask) in enumerate(zip(encoded["token_type_ids"], encoded["special_tokens_mask"])):
        retained_query = sum(kind == 0 and not is_special for kind, is_special in zip(types, mask))
        retained_document = sum(kind == 1 and not is_special for kind, is_special in zip(types, mask))
        full_document = full_document_tokens[i]
        full_pair = full_query_tokens + full_document + special
        out.append({"query_tokens": full_query_tokens, "document_tokens": full_document,
                    "full_pair_tokens": full_pair, "retained_query_tokens": retained_query,
                    "retained_document_tokens": retained_document,
                    "query_truncated": retained_query < full_query_tokens,
                    "document_truncated": retained_document < full_document,
                    "pair_truncated": len(encoded["input_ids"][i]) < full_pair,
                    "removed_document_tokens": full_document - retained_document})
    return out


def summarize(records: list[dict]) -> dict:
    if not records:
        return {"queries": 0}
    n = len(records)
    outcomes = Counter(record["outcome"] for record in records)
    return {"queries": n, "hybrid_ndcg10": sum(r["hybrid_ndcg10"] for r in records) / n,
            "reranked_ndcg10": sum(r["reranked_ndcg10"] for r in records) / n,
            "mean_delta_ndcg10": sum(r["delta_ndcg10"] for r in records) / n,
            "outcomes": dict(outcomes),
            "mean_candidate_label_recall20": sum(r["candidate_label_recall20"] for r in records) / n,
            "queries_without_positive_in_prefix20": sum(r["prefix_positive_count"] == 0 for r in records),
            "queries_with_all_labels_in_prefix20": sum(r["prefix_positive_count"] == r["positive_count"] for r in records),
            "positive_labels": sum(r["positive_count"] for r in records),
            "positive_labels_in_prefix20": sum(r["prefix_positive_count"] for r in records),
            "positive_labels_left_top10": sum(r["gold_transitions"].get("left_top10", 0) for r in records),
            "positive_labels_entered_top10": sum(r["gold_transitions"].get("entered_top10", 0) for r in records),
            "prefix_pairs": sum(r["prefix_pairs"] for r in records),
            "prefix_pairs_truncated": sum(r["prefix_pairs_truncated"] for r in records),
            "positive_prefix_pairs_truncated": sum(r["positive_prefix_pairs_truncated"] for r in records),
            "query_truncated_pairs": sum(r["query_truncated_pairs"] for r in records),
            "queries_with_query_truncation": sum(r["query_truncated_pairs"] > 0 for r in records),
            "top1_unjudged_cross_source_queries": sum(r["top1_unjudged_cross_source"] for r in records),
            "top1_possible_name_counterpart_queries": sum(bool(r["top1_possible_name_counterparts"]) for r in records),
            "mean_query_tokens": sum(r["query_tokens"] for r in records) / n,
            "mean_positive_document_tokens": sum(r["mean_positive_document_tokens"] for r in records) / n}


def group_summaries(records: list[dict], key) -> dict:
    buckets = defaultdict(list)
    for record in records:
        buckets[key(record)].append(record)
    return {str(name): summarize(values) for name, values in sorted(buckets.items())}


def analyze(cache: list[dict], queries: dict[str, dict], corpus: dict[str, dict],
            tokenizer, max_length: int) -> tuple[list[dict], dict]:
    if set(queries) != {str(r["query_id"]) for r in cache} or len(cache) != len(queries):
        raise ValueError("Cache and queries must have identical unique query IDs.")
    document_ids = sorted(set(tool for row in cache for tool in row["rankings"]["hybrid"][:row["rerank_k"]])
                          | {tool for row in cache for tool in row["relevant_ids"]})
    texts = {tool: flatten_tool({"id": tool, "text": corpus[tool]["text"]}) for tool in document_ids}
    doc_lengths = {tool: len(tokens) for tool, tokens in zip(document_ids, tokenizer(
        [texts[tool] for tool in document_ids], add_special_tokens=False, truncation=False)["input_ids"])}
    names = {tool: tool_name(corpus[tool]) for tool in document_ids}
    records = []
    for row in cache:
        qid = str(row["query_id"])
        query = queries[qid]
        positives = set(map(str, row["relevant_ids"]))
        if positives != set(map(str, query["relevant_ids"])):
            raise ValueError(f"Qrels mismatch for {qid}")
        if query.get("domain", query.get("source_domain")) != row["source_domain"]:
            raise ValueError(f"Source mismatch for {qid}")
        hybrid, reranked = row["rankings"]["hybrid"], row["rankings"]["reranked"]
        depth = row["rerank_k"]
        if len(set(hybrid)) != len(hybrid) or set(hybrid[:depth]) != set(reranked[:depth]) or hybrid[depth:] != reranked[depth:]:
            raise ValueError(f"Invalid prefix-only reranking for {qid}")
        prefix = hybrid[:depth]
        query_tokens = len(tokenizer(query["query"], add_special_tokens=False, truncation=False)["input_ids"])
        token_rows = exact_pair_tokens(tokenizer, query["query"], [texts[t] for t in prefix], query_tokens,
                                       [doc_lengths[t] for t in prefix], max_length)
        pair_tokens = dict(zip(prefix, token_rows))
        before, after = ndcg(hybrid, positives), ndcg(reranked, positives)
        delta = after - before
        first = reranked[0]
        first_source = corpus[first].get("source_domain", first.split("_tool_")[0])
        source = row["source_domain"]
        same_source = first_source.casefold() == source.casefold()
        counterparts = [tool for tool in sorted(positives)
                        if first not in positives and not same_source and possible_name_counterpart(names[first], names[tool])]
        gold = []
        for tool in sorted(positives):
            bh, ar = rank_of(hybrid, tool), rank_of(reranked, tool)
            gold.append({"tool_id": tool, "name": names[tool], "hybrid_rank": bh, "reranked_rank": ar,
                         "transition": rank_transition(bh, ar), "document_tokens": doc_lengths[tool],
                         "pair_tokenization": pair_tokens.get(tool)})
        hits = len(positives.intersection(prefix))
        records.append({"query_id": qid, "source_domain": source,
                        "hybrid_ndcg10": before, "reranked_ndcg10": after, "delta_ndcg10": delta,
                        "outcome": "loss" if delta < -1e-12 else "win" if delta > 1e-12 else "unchanged",
                        "positive_count": len(positives), "query_tokens": query_tokens,
                        "prefix_positive_count": hits, "candidate_label_recall20": hits / len(positives),
                        "prefix_pairs": depth,
                        "prefix_pairs_truncated": sum(r["pair_truncated"] for r in token_rows),
                        "positive_prefix_pairs_truncated": sum(pair_tokens[t]["pair_truncated"] for t in positives.intersection(prefix)),
                        "query_truncated_pairs": sum(r["query_truncated"] for r in token_rows),
                        "mean_positive_document_tokens": sum(doc_lengths[t] for t in positives) / len(positives),
                        "gold_transitions": dict(Counter(r["transition"] for r in gold)), "gold_tools": gold,
                        "reranked_top1": {"tool_id": first, "name": names[first], "source_domain": first_source,
                                          "is_positive_label": first in positives, "hybrid_rank": rank_of(hybrid, first),
                                          "pair_tokenization": pair_tokens[first]},
                        "top1_unjudged_cross_source": first not in positives and not same_source,
                        "top1_possible_name_counterparts": counterparts,
                        "gate_selected": row["rankings"]["bm25"][0] != row["rankings"]["dense"][0]})
    groups = {"by_source_domain": group_summaries(records, lambda r: r["source_domain"]),
              "by_positive_count": group_summaries(records, lambda r: "1" if r["positive_count"] == 1 else "2" if r["positive_count"] == 2 else "3+"),
              "by_query_tokens": group_summaries(records, lambda r: "0-31" if r["query_tokens"] < 32 else "32-63" if r["query_tokens"] < 64 else "64+"),
              "by_any_positive_prefix_truncation": group_summaries(records, lambda r: "yes" if r["positive_prefix_pairs_truncated"] else "no"),
              "by_mean_positive_document_tokens": group_summaries(records, lambda r: "0-127" if r["mean_positive_document_tokens"] < 128 else "128-255" if r["mean_positive_document_tokens"] < 256 else "256+"),
              "by_candidate_label_coverage20": group_summaries(records, lambda r: "none" if r["prefix_positive_count"] == 0 else "all" if r["prefix_positive_count"] == r["positive_count"] else "partial")}
    toolbench = [r for r in records if r["source_domain"].casefold() == "toolbench"]
    groups["toolbench_by_outcome"] = group_summaries(toolbench, lambda r: r["outcome"])
    groups["toolbench_by_any_positive_prefix_truncation"] = group_summaries(toolbench, lambda r: "yes" if r["positive_prefix_pairs_truncated"] else "no")
    groups["by_gate_selection"] = group_summaries(records, lambda r: "selected" if r["gate_selected"] else "bypassed")
    return records, groups


CASE_NOTES = {
    "toolbench_query_944": "The query names Ubidots and asks for two endpoints. CE promotes a general IoT sensor analyzer from another source; the labeled variable-details endpoint moves 1→15. This is a context-versus-endpoint-identity example, not proof of the model's reasoning.",
    "toolbench_query_588": "The query requests electricity emissions and fuel prices. CE puts an unjudged carbon-footprint calculator first and moves the labeled fuel-price endpoint 4→18. One request component is retained while the other loses top-10 coverage.",
    "toolbench_query_623": "CE places APIGen take_image_screenshot first; the labeled ToolBench Web_Capture_Take_Image_Screenshot moves 2→13. Their visible tool names and descriptions concern the same screenshot operation, but executable equivalence has not been checked. A second gold tool has an empty description and was already absent from fused candidates.",
    "toolbench_query_366": "CE puts an unjudged anime-ranking endpoint first. Both labeled anime-detail/filter endpoints are demoted; one crosses 6→12. The request mixes ranking, current-airing constraints, and detailed metadata. Truncation statistics are recorded separately and do not establish why the score changed.",
    "toolbench_query_497": "CE places an unjudged APIGen trivia endpoint first. The labeled trivia endpoint remains at rank 3, while the independent Steam special-offers endpoint moves 4→12. This is a multi-request coverage loss with a possible cross-source counterpart.",
    "toolbench_query_154": "Both labeled Trinidad-and-Tobago COVID endpoints improve (9→1 and 12→6). This win is useful alongside loss cases because it prevents treating long-document truncation or multi-request queries as universally harmful.",
    "toolbench_query_896": "The labeled email-format checker improves 9→1. The distinct email-existence checker remains at rank 88, outside the reranked prefix. Reranking can fix ordering but cannot rescue candidates it never scores.",
    "toolbench_query_113": "Four language-specific social-news tools improve into or within the top 10, despite an unjudged French-news tool being ranked first. A wrong top-1 label can coexist with a strong multi-label nDCG improvement.",
    "toolbench_query_119": "The two labeled SignNow role/field tools improve (5→2 and 15→9), while an unjudged APIGen get_role_ids tool is first. This again separates exact-ID labels from possible cross-source endpoint counterparts.",
    "toolbench_query_361": "Both labeled campaign-lead tools improve (2→1 and 8→5). This is a straightforward improvement in the ordering of existing prefix candidates, without adding a new candidate.",
}


def make_cases(records: list[dict], queries: dict[str, dict], limit: int = 5) -> list[dict]:
    rows = [r for r in records if r["source_domain"].casefold() == "toolbench"]
    losses = sorted((r for r in rows if r["outcome"] == "loss"), key=lambda r: (r["delta_ndcg10"], r["query_id"]))[:limit]
    wins = sorted((r for r in rows if r["outcome"] == "win"), key=lambda r: (-r["delta_ndcg10"], r["query_id"]))[:limit]
    return [{**r, "query_excerpt": queries[r["query_id"]]["query"][:240],
             "manual_note": CASE_NOTES.get(r["query_id"], "No manual semantic annotation; inspect the recorded names, ranks, and tokenization.")}
            for r in losses + wins]


def markdown_report(summary: dict, cases: list[dict]) -> str:
    lines = ["# Exploratory tool-retrieval failure analysis", "",
             "This analysis describes the already evaluated 300-query pilot. It does not tune the fixed router or alter binary relevance judgments. It is not confirmation evidence.", "",
             "## Candidate and ordering effects", "",
             "| Source | Queries | Hybrid nDCG@10 | CE nDCG@10 | Delta | Wins / losses / same | Candidate label recall@20 | No positive in prefix | Gold left / entered top 10 |",
             "|---|---:|---:|---:|---:|---|---:|---:|---|" ]
    for domain, group in summary["groups"]["by_source_domain"].items():
        outcomes = group["outcomes"]
        lines.append(f"| {domain} | {group['queries']} | {group['hybrid_ndcg10']:.6f} | {group['reranked_ndcg10']:.6f} | {group['mean_delta_ndcg10']:+.6f} | {outcomes.get('win',0)} / {outcomes.get('loss',0)} / {outcomes.get('unchanged',0)} | {group['mean_candidate_label_recall20']:.4f} | {group['queries_without_positive_in_prefix20']} | {group['positive_labels_left_top10']} / {group['positive_labels_entered_top10']} |")
    lines += ["", "A prefix permutation cannot recover positive tools outside its top-20 candidate set. Candidate misses and reranker ordering errors therefore need different remedies. Candidate label recall is averaged per query and refers to exact labeled tool IDs.", "",
              "## Token truncation and judgment risks", "",
              f"The exact pinned production tokenizer was replayed at max_length={summary['tokenizer']['max_length']}, with its longest-first pair truncation. Counts use token_type_ids and the special-token mask; query text and flattened tool text match ranking generation.", "",
              "| Source | Truncated prefix pairs / total | Truncated positive pairs / positive prefix pairs | Unjudged cross-source top 1 | Possible name-suffix counterpart at top 1 |",
              "|---|---|---|---:|---:|" ]
    for domain, group in summary["groups"]["by_source_domain"].items():
        lines.append(f"| {domain} | {group['prefix_pairs_truncated']} / {group['prefix_pairs']} | {group['positive_prefix_pairs_truncated']} / {group['positive_labels_in_prefix20']} | {group['top1_unjudged_cross_source_queries']} | {group['top1_possible_name_counterpart_queries']} |")
    lines += ["", "A name-suffix counterpart is only a deterministic textual hint: normalized names match or one ends in the other, with the shorter name at least 12 characters. It does not establish functional equivalence, relevance, or label error. The original qrels remain unchanged.", "",
              "### ToolBench descriptive breakdown", "",
              "| Outcome | Queries | Mean delta nDCG@10 | Mean query tokens | Mean gold document tokens | Truncated gold prefix pairs / total gold prefix pairs | Unjudged cross-source top 1 |",
              "|---|---:|---:|---:|---:|---|---:|" ]
    for outcome, group in summary["groups"]["toolbench_by_outcome"].items():
        lines.append(f"| {outcome} | {group['queries']} | {group['mean_delta_ndcg10']:+.6f} | {group['mean_query_tokens']:.1f} | {group['mean_positive_document_tokens']:.1f} | {group['positive_prefix_pairs_truncated']} / {group['positive_labels_in_prefix20']} | {group['top1_unjudged_cross_source_queries']} |")
    truncation_groups = summary["groups"]["toolbench_by_any_positive_prefix_truncation"]
    if "no" in truncation_groups and "yes" in truncation_groups:
        no, yes = truncation_groups["no"], truncation_groups["yes"]
        lines += ["", f"Of the {summary['groups']['toolbench_by_outcome'].get('loss',{}).get('queries',0)} ToolBench loss queries, {no['outcomes'].get('loss',0)} have no truncated positive pair. Among the {yes['queries']} queries with a truncated positive pair, {yes['outcomes'].get('win',0)} improve and {yes['outcomes'].get('loss',0)} worsen. This rules out the simple explanation that clipping a labeled positive is necessary for the observed regression; it does not rule out effects from clipped distractors or other interacting factors."]
    lines += ["", "These strata were examined after seeing pilot outcomes. They are descriptive associations, confounded by source, task, annotation, and tool schema; they do not identify causes. Full by-positive-count, query-length, document-length, and candidate-coverage breakdowns are in the JSON artifact.", "",
              "## Deterministic case review", "",
              "The five largest ToolBench losses and five largest wins are selected by signed nDCG delta, with query-ID tie breaks. Examples are deliberately outcome-selected and cannot estimate population frequencies.", "" ]
    for case in cases:
        lines += [f"### {case['query_id']} ({case['delta_ndcg10']:+.6f} nDCG@10)", "", case["manual_note"], "",
                  f"Query excerpt: {case['query_excerpt']}", "",
                  "| Labeled tool | Hybrid rank | CE rank | Document tokens | Positive pair truncated |",
                  "|---|---:|---:|---:|---|" ]
        for tool in case["gold_tools"]:
            tokens = tool["pair_tokenization"]
            truncated = "not scored" if tokens is None else str(tokens["pair_truncated"]).lower()
            lines.append(f"| {tool['tool_id']} ({tool['name']}) | {tool['hybrid_rank'] or 'absent'} | {tool['reranked_rank'] or 'absent'} | {tool['document_tokens']} | {truncated} |")
        first = case["reranked_top1"]
        lines += ["", f"CE top 1: `{first['tool_id']}` — {first['name']} (hybrid rank {first['hybrid_rank']}; labeled positive: {first['is_positive_label']}).", ""]
    lines += ["## Research implications", "",
              "1. Evaluate candidate coverage separately from ordering quality; a reranker cannot rescue absent candidates.",
              "2. Report source-level and multi-label results alongside the aggregate. An improvement on one source can conceal losses on another.",
              "3. Preserve exact-ID metrics, but independently adjudicate possible cross-source counterparts before claiming these are executable failures. Any alternate equivalence-aware evaluation needs frozen judging rules and should be reported separately.",
              "4. Test truncation and tool-document serialization as controlled ablations on future frozen data. Their associations here do not justify declaring either the cause of regression.",
              "5. A learned or heuristic router derived from these pilot observations must be evaluated on untouched confirmation queries. This analysis is not evidence that such a router already works.", "",
              "## Provenance", "", "```json", json.dumps(summary["provenance"], indent=2), "```", "" ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    parser.add_argument("--max-length", type=int, default=256)
    args = parser.parse_args()
    from transformers import AutoTokenizer
    manifest = json.loads(args.manifest.read_text())
    paths = {"cache_sha256": args.cache, "queries_sha256": args.queries, "corpus_sha256": args.corpus}
    for key, path in paths.items():
        if digest(path) != manifest[key]:
            raise ValueError(f"Manifest hash mismatch: {path}")
    if args.max_length != manifest["cache_provenance"]["protocol"]["max_sequence_length"]:
        raise ValueError("Tokenization length must match ranking generation.")
    if manifest["cache_provenance"]["protocol"]["rerank_k"] != 20:
        raise ValueError("This report's candidate-recall@20 fields require a rerank depth of 20.")
    if manifest["models"]["reranker"] != {"name": MODEL, "revision": REVISION, "fine_tuned_here": False}:
        raise ValueError("Pinned production reranker identity mismatch.")
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION, local_files_only=True)
    cache = read_rows(args.cache)
    queries, corpus = index_unique(read_rows(args.queries), "id"), index_unique(read_rows(args.corpus), "id")
    records, groups = analyze(cache, queries, corpus, tokenizer, args.max_length)
    cases = make_cases(records, queries)
    summary = {"scope": "exploratory_original_pilot_failure_analysis", "summary": summarize(records), "groups": groups,
               "cases": cases, "tokenizer": {"model": MODEL, "revision": REVISION, "max_length": args.max_length,
                                               "truncation": "longest_first", "truncation_side": tokenizer.truncation_side},
               "provenance": {**{key: digest(path) for key, path in paths.items()},
                              "manifest_sha256": digest(args.manifest), "script_sha256": digest(Path(__file__)),
                              "text_serialization_sha256": digest(Path("src/toolret_research/text.py")),
                              "labels_altered": False, "analysis_modifies_rankings_or_router": False},
               "limitations": ["Post-outcome exploratory analysis, no causal identification.",
                                "Exact-ID judgments do not establish tool equivalence or task execution.",
                                "Manual notes only for predetermined largest ToolBench pilot deltas.",
                                "Binary labels; positive count is not a proof of mandatory tool requirements."]}
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    output_json = args.output_prefix.with_suffix(".json")
    output_md = args.output_prefix.with_suffix(".md")
    output_rows = args.output_prefix.with_name(args.output_prefix.name + "_per_query.jsonl")
    for path in (output_json, output_md, output_rows):
        if path.exists():
            raise ValueError(f"Refusing to overwrite existing analysis: {path}")
    output_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    output_md.write_text(markdown_report(summary, cases))
    output_rows.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in records))
    print(json.dumps({"outputs": [str(output_json), str(output_md), str(output_rows)], "summary": summary["summary"],
                      "sources": groups["by_source_domain"]}, indent=2))


if __name__ == "__main__":
    main()
