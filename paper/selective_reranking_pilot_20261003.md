# Selective Reranking for Tool Retrieval: A Source-Domain Pilot

**Michael Baffour Awuah — Independent undergraduate research**

October 3, 2026 · Preliminary technical report; not peer reviewed

## Abstract

This pilot evaluates a fixed decision rule that invokes a cross-encoder only when BM25 and dense retrieval disagree on their top-ranked tool. Across 300 deterministically selected ToolRet queries and 37,292 web-category tools, the rule used 217 cross-encoder calls rather than 300, a 27.7% reduction. Binary nDCG@10 was 0.501864 versus 0.498600 for always reranking; the paired confidence interval includes zero difference. Performance varies substantially by query source. The contribution is an auditable experiment and verified implementation, with no claim of benchmark leadership, formal noninferiority, or generalization to unseen API families.

## Question and protocol

Does a cheap disagreement rule avoid some reranking work while retaining useful retrieval quality? The policy was fixed before retrieval-quality evaluation:

```python
rerank = bm25_ranked_ids[0] != dense_ranked_ids[0]
```

Relevance labels were used for data-integrity checks and evaluation, never for routing, threshold selection, or training. No model or gate was trained in this pilot. Historical fine-tuned checkpoints were unavailable, so this experiment uses frozen public models rather than recreating the earlier trained system.

BM25 and `sentence-transformers/all-MiniLM-L6-v2` each retrieve 100 candidates. Dense search uses normalized embeddings and exact full-corpus similarity search. Equal-weight reciprocal-rank fusion uses smoothing constant 60. `cross-encoder/ms-marco-MiniLM-L-6-v2` reorders only the fused top 20, preserving the exact remaining tail. Text length is capped at 256 model tokens; execution uses CPU and four Torch threads. Both model revisions are pinned in the protocol and cache manifest.

The comparisons are BM25, dense retrieval, hybrid fusion, always reranking, the disagreement gate, and 20 random reranking policies with seeds 0–19. Every random policy selects exactly 217 queries without replacement. Since reranking depth is always 20, each random policy also matches the gate's 4,340 query–tool scoring pairs.

## Data and integrity

The sample contains 100 queries each from APIGen, ToolACE, and ToolBench. Selection orders IDs by SHA-256 of `research-20261003`, a NUL separator, and query ID. Unicode normalization, case folding, and whitespace normalization remove exact normalized-text duplicates across candidate sources and matches to the excluded APIBank source. This does not establish semantic deduplication or pretraining independence.

Pinned upstream revisions are `b8c76ad3349ff17497b6bdb28bb5b8f61a0f6445` for ToolRet-Queries and `e06c38c75612b6536bd959e08cdd345894aba6a7` for ToolRet-Tools. The corpus is the 37,292-tool web category, not the entire heterogeneous ToolRet benchmark. An initial MetaTool source was replaced with ToolACE after category and gold-ID validation found tools absent from the web corpus; this amendment preceded ranking generation or quality measurement.

Integrity checks verify corpus and query hashes, all planned query IDs and positive labels, 100 queries per source, unique known ranking IDs, exact RRF reconstruction, and valid cross-encoder prefix permutations. Full hashes are recorded in the manifests; the prepared query SHA-256 is `ff727e5ce10933e0332acd1ad18f0556a8c03353325ce7074ccd6fb11857e9bd`, and corpus SHA-256 is `ac192e7a6d2b1930f3f61544748ce81741c372a104674a3520f8214b2ac3c09f`.

## Results

All metrics are query means at cutoff 10. Positive relevance grades are collapsed to binary labels. “All-positive-label coverage” means every labelled positive appears in the top ten; those labels are not independently established as mandatory execution tools.

| Policy | Recall@10 | MRR@10 | nDCG@10 | All-positive-label coverage@10 | CE calls |
|---|---:|---:|---:|---:|---:|
| BM25 | 0.561389 | 0.471509 | 0.455288 | 0.456667 | 0 |
| Dense | 0.533000 | 0.438721 | 0.420994 | 0.426667 | 0 |
| Hybrid | 0.632333 | 0.478474 | 0.482782 | 0.526667 | 0 |
| Always rerank | 0.619556 | 0.511315 | 0.498600 | 0.510000 | 300 |
| Disagreement gate | 0.617333 | 0.517995 | 0.501864 | 0.510000 | 217 |

The gate skips 83 calls and scores 4,340 rather than 6,000 pairs. Its nDCG@10 difference from always reranking is +0.003265, with a 95% paired query-bootstrap interval of [−0.005829, +0.012830]. The interval crosses zero. The corresponding exact sign test finds 12 wins, 10 losses, and 278 ties; its unadjusted two-sided p-value is 0.831812. These results do not establish superiority or formal noninferiority.

Bootstrap intervals use 20,000 resamples and seed 20261003. Query-level resampling assumes exchangeable observations; correlated requests or API families can understate uncertainty. Sign tests are exploratory, ignore effect magnitude, and receive no multiple-comparison correction.

Random reranking has mean nDCG@10 0.493190, with a seed range of 0.481646–0.501838. The gate exceeds that mean descriptively. The seed range is not an inferential confidence interval, and no superiority test against a random-policy population was performed.

| Query source | Queries | Gate CE calls | Hybrid nDCG@10 | Always nDCG@10 | Gate nDCG@10 |
|---|---:|---:|---:|---:|---:|
| APIGen | 100 | 69 | 0.611968 | 0.686829 | 0.681766 |
| ToolACE | 100 | 62 | 0.558829 | 0.555084 | 0.559673 |
| ToolBench | 100 | 86 | 0.277550 | 0.253885 | 0.264153 |

ToolBench exposes a failure boundary: both reranking policies score below hybrid fusion on nDCG@10. Aggregate nDCG also conceals lower recall and positive-label coverage than hybrid retrieval. This makes source-specific evaluation essential before adopting the policy.

## Implementation verification and timing limits

A separate execution over cached first-stage rankings actually invoked the pinned cross-encoder 217 times, scored 4,340 pairs, and skipped 83 calls. All 300 resulting rankings matched the cache-derived policy exactly. Independent recomputation also matched aggregate/domain metrics and all 21 recorded checks, including the primary-metric intervals and component timing sums.

Single-pass warm-model component measurements support counterfactual sequential sums, excluding gate/routing overhead: estimated mean 900.38 ms for always reranking versus 725.05 ms for the gate; estimated p95 1,189.77 versus 1,179.07 ms. These are not measured adaptive or deployed-service latency. This report claims fewer reranker calls and pairs, not a latency gain, total-compute reduction, or downstream agent-success improvement.

## Reproduction and next research step

The fixed protocol is `configs/selective_pilot_20261003.json`. Data preparation and cache generation use `scripts/prepare_research_pilot.py` and `scripts/generate_research_cache.py`. With prepared inputs and the supplied cache:

```bash
PYTHONPATH=src python scripts/evaluate_selective_reranking.py \
  --cache results/research_20261003/pilot_cache/rankings.jsonl \
  --corpus data/research_20261003/corpus.jsonl \
  --queries data/research_20261003/queries.jsonl \
  --manifest results/research_20261003/pilot_cache/manifest.json \
  --exclude-queries results/test_failure_analysis.csv \
  --output-dir results/research_20261003/reproduced_eval \
  --bootstrap-resamples 20000
```

The output directory must be new or empty. Results, query decisions, source hashes, runtime verification, and independent audit are retained under `results/research_20261003`.

The separate audit of 16 saved historical APIBank examples uses first-relevant ranks without a cutoff. Its historical uncut MRR is not directly comparable to this pilot's MRR@10, different queries, and off-the-shelf models; its evidence remains unchanged.

A subsequent study should establish reliable API-family metadata, allocate disjoint development/calibration/test families, freeze decisions before the new test, compare stronger retrievers, and measure adaptive execution with repeated timing. Current sources are dataset domains, not API-family holdouts. Public pretrained models may have seen benchmark content. This pilot therefore supports a reproducible preliminary finding, not a first-of-kind algorithm or benchmark-wide conclusion.

## Primary references

- [ToolRet: Retrieval Models Aren't Tool-Savvy, ACL Findings 2025](https://aclanthology.org/2025.findings-acl.1258/), with [official code](https://github.com/mangopy/tool-retrieval-benchmark).
- [ToolRerank: Adaptive and Hierarchy-Aware Reranking, 2024](https://arxiv.org/abs/2403.06551). Adaptive tool reranking predates this pilot.
- [Lookahead-R: Budget-Aware Tool Retrieval, 2026](https://arxiv.org/abs/2609.35811). Budget-aware retrieval is also prior work.
