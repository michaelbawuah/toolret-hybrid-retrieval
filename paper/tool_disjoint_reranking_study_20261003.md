# When to rerank tool retrieval: a component-disjoint empirical study

Michael Baffour Awuah | Independent undergraduate research | October 3, 2026

## Abstract

The primary utility router meets the predeclared 0.01 absolute nDCG engineering noninferiority criterion. Its nDCG@10 is 0.5273, versus 0.5241 for always-on reranking (difference +0.0032; query 95% CI [-0.0016, +0.0080]; component 95% CI [-0.0016, +0.0079]). Real conditional execution verified 1,149 CE calls and 22,980 scored pairs across all 1,500 queries, a 23.4% call reduction.

The simpler disagreement gate has a stronger observed quality point estimate: nDCG@10 0.5314 with 1,077 required calls, versus 0.5273 with 1,149 for the primary router. Exploratory learned-minus-disagreement and learned-minus-source-matched-random component intervals both include zero; learned allocation superiority is not established.

On 90 source-balanced confirmation queries with three repetitions per policy, measured warm sequential CPU mean retrieval time is 791.4 ms for the primary router and 920.7 ms for always-on reranking. The observed mean-time reduction is 14.0% (paired query-bootstrap 95% CI [8.2%, 20.0%]). Initialization, network, queueing, and concurrent serving are excluded.

The contribution is a reproducible transfer/evaluation of known pre-reranker gating to tool retrieval, with strict supervised-router holdout, deployment-path verification, source-level diagnostics, and honest uncertainty. Learned gating itself is not a new algorithm. This is a completed empirical study under a declared scope, not an accepted publication or state-of-the-art claim.

## Prospective design

The fixed web corpus contains 37,292 tools. Among 3,099 deduplicated eligible APIGen, ToolBench, and ToolACE queries, all 252 connected components touched by the previously observed pilot are anchored; they contain 929 queries. The original 300 pilot queries train the router; 629 additional pilot-connected queries are quarantined. Fresh calibration contains 300 queries (100/source, 249 components), and untouched confirmation contains 1,500 (500/source, 1,174 components). A further 370 free queries are reserved.

Components are formed transitively from shared positive gold tool IDs over the full eligible universe, with minimum query ID as component ID. Labels are used for structural grouping, but retrieval results are not. Entire groups are assigned by a deterministic hash and quota-fitting rule. Development, calibration, and confirmation share no query IDs, normalized query texts, positive tool IDs, or components. These are tool-ID components, not verified API families; the relevant held-out tools remain available in the retrieval corpus.

The protocol and linked data manifests were publicly frozen before fresh rankings were generated. The old pilot is development data and is never counted as confirmation evidence.

### Fixed retrieval and router

BM25 and normalized off-the-shelf MiniLM each retrieve 100 candidates. Equal-weight reciprocal-rank fusion uses k=60. A pinned MS-MARCO MiniLM cross-encoder scores only the first 20 fused candidates at maximum pair length 256; the tail is unchanged. Ranking ties use tool IDs deterministically. Models run on CPU with four Torch threads, one interop thread, and batch size 64.

Ridge regression (fixed alpha=10) predicts development-query nDCG@10(always)-nDCG@10(hybrid). Seven features use rank equality/overlap/positions and whitespace query length only; source IDs, lexical tool-ID content, qrels, and CE outcomes are excluded from inference features. Feature means/stds are fit only on the 300 development rows. Calibration is used only for label-free prediction quantiles; strict prediction > threshold routes a query, and ties skip. Target fractions are 25%, 50%, and 75%; realized held-out call fractions may differ.

| Feature | Frozen definition |
|---|---|
| top1_disagreement | Indicator that BM25 and dense top-1 tool IDs differ. |
| top10_jaccard | Intersection size / union size of the first 10 BM25 and dense tool IDs. |
| top20_jaccard | Intersection size / union size of the first 20 BM25 and dense tool IDs. |
| top20_reciprocal_weighted_jaccard | Over the top20 union, sum min(1/rank_B,1/rank_D) / sum max(1/rank_B,1/rank_D); missing-list weight is zero. |
| top20_rank_coherence | 1 minus mean absolute BM25/dense rank difference over the top20 union divided by 20; missing position is 21. |
| query_token_count_log1p | log1p(number of whitespace-delimited query tokens). |
| rrf_top2_normalized_margin | (largest RRF score - second-largest RRF score)/largest score, using both full top100 lists, equal weights and k60; singleton margin is 1. |

### Primary and secondary analyses

The sole primary policy is the 75% calibration-target utility router versus always-on reranking. The predeclared engineering tolerance is 0.01 absolute binary nDCG@10: both paired source-stratified query and gold-component two-sided 95% interval lower bounds must exceed -0.01, with fewer CE calls. This tolerance is a design choice, not an externally accepted equivalence threshold or a power guarantee.

Both intervals use 10,000 paired bootstrap draws, seed 20261003, and equal 1/3 source weights. The component estimator samples groups within each source and uses the ratio of sampled query-score sums to sampled query counts, retaining the query-macro target. Other budgets, metrics, baselines, domains, and diagnostics are exploratory.

## Untouched confirmation results

| Policy | nDCG@10 | MRR@10 | Recall@10 | All-label coverage@10 | Required CE calls | Required CE pairs |
|---|---:|---:|---:|---:|---:|---:|
| BM25 | 0.4689 | 0.4725 | 0.5840 | 0.4887 | 0 | 0 |
| Base MiniLM | 0.4602 | 0.4583 | 0.5911 | 0.5027 | 0 | 0 |
| Hybrid RRF | 0.5208 | 0.5196 | 0.6522 | 0.5620 | 0 | 0 |
| Always rerank | 0.5241 | 0.5301 | 0.6499 | 0.5480 | 1,500 | 30,000 |
| Disagreement gate | 0.5314 | 0.5388 | 0.6538 | 0.5553 | 1,077 | 21,540 |
| Jaccard gate | 0.5239 | 0.5276 | 0.6513 | 0.5520 | 1,391 | 27,820 |
| Utility 25% target | 0.5269 | 0.5272 | 0.6549 | 0.5633 | 409 | 8,180 |
| Utility 50% target | 0.5305 | 0.5339 | 0.6569 | 0.5620 | 829 | 16,580 |
| Utility 75% target | 0.5273 | 0.5327 | 0.6520 | 0.5527 | 1,149 | 22,980 |

All-label coverage means all positive benchmark relevance labels retrieved, not a verified mandatory execution-tool set.

![Confirmation quality and compute](figures/confirmation_quality_compute.png)

### Primary result

The primary utility router meets the predeclared 0.01 absolute nDCG engineering noninferiority criterion. Its nDCG@10 is 0.5273, versus 0.5241 for always-on reranking (difference +0.0032; query 95% CI [-0.0016, +0.0080]; component 95% CI [-0.0016, +0.0079]). Real conditional execution verified 1,149 CE calls and 22,980 scored pairs across all 1,500 queries, a 23.4% call reduction.

| Paired interval | Mean nDCG difference | 95% interval | Lower bound > -0.01? |
|---|---:|---|---|
| Source-stratified query | +0.003201 | [-0.001560, +0.007984] | True |
| Source-stratified component | +0.003201 | [-0.001558, +0.007931] | True |

### Equal-compute random controls

Twenty seeds 0-19 select exact realized full-cohort call counts. Global controls exist for each utility budget and the disagreement gate; additional controls match per-source counts for the primary router and disagreement gate. The per-query mean random result is used for exploratory paired comparisons. Seed ranges describe random-allocation variation, not confidence intervals.

| Control | Calls/seed | Mean nDCG | Full seed range |
|---|---:|---:|---|
| global_fixed_disagreement | 1077 | 0.522524 | [0.517310, 0.527781] |
| global_utility_25 | 409 | 0.521395 | [0.513734, 0.527221] |
| global_utility_50 | 829 | 0.521643 | [0.516458, 0.527598] |
| global_utility_75 | 1149 | 0.522497 | [0.518231, 0.528291] |
| source_matched_fixed_disagreement | 1077 | 0.520577 | [0.512594, 0.527697] |
| source_matched_utility_75 | 1149 | 0.524202 | [0.518584, 0.527429] |

### Exploratory paired allocation comparisons

| Comparison | Mean nDCG difference | Query 95% CI | Component 95% CI |
|---|---:|---|---|
| fixed_disagreement_minus_always | +0.007316 | [+0.002293, +0.012378] | [+0.002147, +0.012458] |
| hybrid_minus_always | -0.003235 | [-0.015542, +0.008995] | [-0.015779, +0.009131] |
| utility_75_minus_fixed_disagreement | -0.004114 | [-0.009234, +0.001147] | [-0.009354, +0.000967] |
| utility_75_minus_random_global_mean | +0.004761 | [+0.000032, +0.009600] | [+0.000197, +0.009326] |
| utility_75_minus_random_source_matched_mean | +0.003056 | [-0.001503, +0.007786] | [-0.001379, +0.007557] |
| fixed_disagreement_minus_random_global_mean | +0.008847 | [+0.004042, +0.013781] | [+0.003842, +0.013733] |
| fixed_disagreement_minus_random_source_matched_mean | +0.010795 | [+0.005759, +0.015963] | [+0.005520, +0.015959] |

These comparisons are exploratory, without a new primary-policy selection or a multiple-testing superiority claim. A label-aware matched-budget oracle is retained as an unattainable diagnostic, never a deployed result.

| Matched policy budget | Oracle calls | Oracle nDCG@10 (test-label-aware) |
|---|---:|---:|
| fixed_disagreement | 1077 | 0.592876 |
| utility_25 | 409 | 0.592876 |
| utility_50 | 829 | 0.592876 |
| utility_75 | 1149 | 0.591808 |

## Real conditional execution and measured runtime

The primary router was executed across all 1,500 cached first-stage rankings. Every decision matched the frozen prediction, and every actual CE/hybrid output matched the evaluated ranking. These requests verify conditional CE work, not end-to-end first-stage time.

On 90 source-balanced confirmation queries with three repetitions per policy, measured warm sequential CPU mean retrieval time is 791.4 ms for the primary router and 920.7 ms for always-on reranking. The observed mean-time reduction is 14.0% (paired query-bootstrap 95% CI [8.2%, 20.0%]). Initialization, network, queueing, and concurrent serving are excluded.

![Actual repeated CPU retrieval timing](figures/confirmation_latency.png)

| Policy | Mean ms | p50 query-mean ms | p95 query-mean ms | Actual CE calls / measured requests |
|---|---:|---:|---:|---|
| Hybrid RRF | 271.15 | 242.08 | 378.38 | 0 / 270 |
| Always rerank | 920.72 | 906.98 | 1172.28 | 270 / 270 |
| Disagreement gate | 772.52 | 847.68 | 1137.29 | 210 / 270 |
| Utility 75% target | 791.44 | 850.87 | 1083.67 | 219 / 270 |

Latency queries are selected independently by hash, 30/source, with three repetitions of each of four policies. Query and policy order are randomized and cyclically balanced. Timers cover BM25, dense query encoding, exact dense search/sort, RRF, router feature/prediction overhead, and conditional CE/prefix sorting. Every actual ranking is checked after timing. Models/index/embeddings remain loaded. Initialization, validation, warmup, network, queueing, and concurrent load are excluded. Repetitions are averaged within query before uncertainty calculations; all-event percentiles are separately recorded.

## Source-level and failure diagnostics

| Source | Components | Hybrid nDCG | Always nDCG | Disagreement nDCG | Primary nDCG | Primary calls |
|---|---:|---:|---:|---:|---:|---:|
| APIGen | 353 | 0.5851 | 0.6145 | 0.6225 | 0.6169 | 472 |
| ToolBench | 341 | 0.3588 | 0.3204 | 0.3333 | 0.3301 | 374 |
| ToolACE | 480 | 0.6186 | 0.6372 | 0.6383 | 0.6348 | 303 |

### Candidate scope and tokenization

| Source | Candidate-label recall@20 | No positive in prefix | CE wins / losses / same | Truncated positive pairs / positive prefix pairs | Unjudged cross-source top1 |
|---|---:|---:|---|---|---:|
| APIGen | 0.8010 | 65 | 130 / 106 / 264 | 17 / 481 | 187 |
| ToolBench | 0.5671 | 132 | 116 / 184 / 200 | 70 / 637 | 330 |
| ToolACE | 0.7882 | 85 | 118 / 107 / 275 | 87 / 489 | 123 |

ToolBench always-minus-hybrid nDCG changes by -0.0384; 184 of 500 queries worsen. Of those losses, 152 have no truncated labeled pair. Clipping a positive is therefore not necessary for every loss; this does not identify a cause or rule out clipped distractors. Exact-ID judgment risk and multi-request coverage remain distinct hypotheses.

A prefix permutation cannot recover relevant tools outside the scored 20. Candidate misses and within-prefix ordering changes have different causes/remedies. The exact pinned pair tokenizer is replayed with longest-first truncation; associations with losses are descriptive, not causal. An unjudged cross-source tool may be relevant: original exact-ID qrels remain unchanged. Name similarity is a textual hint, not verified tool equivalence or grounds for changing labels.

### Primary skipped-query effects

The primary router skips 351 queries: 60 harmful, 87 beneficial, and 204 unchanged under binary qrels (numerical tolerance 1e-12). Summed nDCG gain forgone is 15.7402; summed harm avoided is 20.5423. These are ranking-label effects, not demonstrated tool-execution failures.

### Deterministically selected examples

Cases are deliberately selected by outcome size with query-ID tie breaks and cannot estimate prevalence.

- **toolbench_query_81**: always-minus-hybrid nDCG change -0.7359. Transactions and current-month quota are requested. CE promotes an unjudged ToolACE transaction tool; the labeled quota endpoint moves rank 2 to 15. Neither labeled pair is clipped. The primary router skips this rerank. Equivalence and execution are unverified.
- **toolbench_query_716**: always-minus-hybrid nDCG change +0.7654. Two labeled Nexweave template tools move from ranks 15 and 13 to 1 and 2. A third positive, the icon-search endpoint, is absent from fused candidates. Ordering improves, while a candidate miss remains outside the reranker's reach.

## Integrity and reproducibility

Independent audit: **317 checks passed, zero pending/failed**. The audit independently reconstructs data components, router fit/thresholds, metrics/random decisions/intervals, and runtime evidence without importing study metric/router/evaluator implementations.

Public protocol freeze: `3e813e1af650104f33aaf77febce5a8c1a3f1213`. Router thresholds and model were frozen before confirmation quality. Data/model/source fingerprints accompany all results.

| Evidence | SHA-256 |
|---|---|
| protocol | `d33ceee4741679c5027b7b55cc96f5535f98aef0c2160292d8e0c948abd076bc` |
| data | `45d68454ecc950c56b2ecf9fd436cb4a6dafc780e4e64d3e154ef81fc304b2ea` |
| router | `f24613a552d36737f4a17797b94f0c9348a67072a97cfc3cb471ae4bd25c159c` |
| evaluation | `fda5bd3e67a6990c2e95076779c4f3dedf8faaa9ab7a7325eb6dfc5fee6f55d3` |
| runtime | `72b10b04b6527f023637f70ca5fe49bab0f345b7a125c565296bd82b7fc22316` |
| latency | `7ba78367007bc9ab3b5b7e72eb86abf2810b2e13e02fcf8093156b810682fb51` |
| failure | `c00cc3e8355f1f6149dfba2790360c4124e965a907cabd92c462f713f226fa72` |
| audit | `3ca05098a331e9f06c1b6443eed9746bde826a39147b794f5f9ce16eaf5520ef` |
| completion | `e6522e27612d32ddffd806a061c04f1bbad0efc5a6a239f1d68bc551aa25e300` |
| reproduction | `528ad7fdf2035a55ccd075a81e9c6d072f99fe44a3b2b58931e24ac82397dc70` |

### Reproduction entry points

Install the recorded research environment and CPU Torch build, then run from the repository root:

```bash
.venv/bin/python scripts/run_confirmation_study.py \
  --output-dir data/research_reproduction_run1
```

The default reconstructs missing source data from pinned parquets, verifies frozen file fingerprints, and freshly evaluates published ranking caches and the frozen router. Add `--recompute` to regenerate calibration/ridge fitting/confirmation rankings; add `--measure-runtime` to execute full-cohort conditional CE calls and the 90-query repeated latency benchmark. Missing corpus embeddings are regenerated for real timing. Full prerequisites and argument examples are maintained in docs/confirmation_reproduction.md and the README. Generated stages refuse unintended overwrite. GPU results are not interchangeable with the recorded CPU experiment.

The default runner was actually executed: all numerical summary fields matched except provenance creation time; frozen predictions, decisions, per-query metrics, and evaluation Markdown were byte-identical. Five live pinned parquet downloads reconstructed the complete corpus, original pilot, and all five grouped cohort files at their canonical fingerprints. The runner's neural-recomputation and runtime flags were not executed in that smoke test; original-study inference and actual runtime are separate measured evidence.

| Stage | Script / evidence |
|---|---|
| Run ordered full evidence workflow | `scripts/run_confirmation_study.py` |
| Prepare exact grouped splits | `scripts/prepare_confirmation_study.py` |
| Generate pinned first-stage and CE rankings | `scripts/generate_confirmation_cache.py` |
| Fit development-only router | `scripts/fit_utility_router.py` |
| Apply frozen confirmation policies | `scripts/evaluate_confirmation_study.py` |
| Analyze ordering/tokenization/judgments | `scripts/analyze_confirmation_failures.py` |
| Verify all primary conditional CE calls | `scripts/verify_confirmation_runtime.py` |
| Measure repeated actual latency | `scripts/benchmark_selective_latency.py` |
| Independently audit / finalize evidence | `scripts/audit_confirmation_study.py / scripts/finalize_confirmation_study.py` |
| Regenerate this report | `scripts/export_confirmation_report.py` |

### Pinned models and runtime

- Dense: `sentence-transformers/all-MiniLM-L6-v2` at `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`.
- CE: `cross-encoder/ms-marco-MiniLM-L-6-v2` at `233902d25c440f23af6f7d6e94d2946bac0bee0a`.
- Runtime: `{'cpu_model': 'AMD EPYC 9V74 80-Core Processor', 'device': 'cpu', 'logical_cpus': 9, 'platform': 'Linux-6.18.44-x86_64-with-glibc2.39', 'processor': 'x86_64', 'python': '3.12.14', 'torch_interop_threads': 1, 'torch_threads': 4}`.
- Packages: `{'numpy': '2.5.3', 'sentence-transformers': '6.1.0', 'torch': '2.14.1+cpu', 'transformers': '5.18.0'}`.

## Discussion and limitations

The result evaluates a fixed routing decision under one tool-retrieval configuration, not a universal replacement for reranking. The primary model remains selected even if a secondary policy has a better observed point estimate. A failure to meet the tolerance is a completed negative/inconclusive empirical finding, not permission to retune on the confirmation set.

Gold-tool grouping limits direct supervised-router label overlap, but does not identify API providers or prove model pretraining independence. Public models may have encountered related benchmarks. Deterministic whole-group quota assignment and pilot-component quarantine define a particular benchmark cohort; it is not sampled live-user traffic or a representative census of APIs. Binary qrels can omit interchangeable tools; all-label coverage does not establish mandatory tools, functional equivalence, or successful agent execution. Component resampling covers observed positive-tool sharing but cannot account for every semantic dependency. Source-specific results and outcome-selected cases are exploratory. The tolerance is a declared engineering tradeoff rather than a universally justified equivalence margin. CPU timing describes a single warm sequential machine and excludes deployment/network/concurrency costs. No faculty supervision, peer review, publication acceptance, new algorithm, or cross-paper state of the art is claimed.

## Conclusion

The primary utility router meets the predeclared 0.01 absolute nDCG engineering noninferiority criterion. Its nDCG@10 is 0.5273, versus 0.5241 for always-on reranking (difference +0.0032; query 95% CI [-0.0016, +0.0080]; component 95% CI [-0.0016, +0.0079]). Real conditional execution verified 1,149 CE calls and 22,980 scored pairs across all 1,500 queries, a 23.4% call reduction.

The full protocol-to-runtime evidence chain closes this scoped empirical study. Future independently frozen work can test additional retriever/reranker families, controlled truncation/serialization ablations, independently judged tool equivalence, and downstream execution success.

## References

- Shi et al. (2025). [Retrieval Models Aren't Tool-Savvy: Benchmarking Tool Retrieval for Large Language Models](https://aclanthology.org/2025.findings-acl.1258/). Findings of ACL 2025.
- Zheng et al. (2024). [ToolRerank: Adaptive and Hierarchy-Aware Reranking for Tool Retrieval](https://aclanthology.org/2024.lrec-main.1413/). LREC-COLING 2024.
- Bacellar (2026). [Per-Query Gating of LLM Rerankers for Multi-Hop Retrieval](https://arxiv.org/abs/2609.22880). arXiv preprint 2609.22880.
- Wu, Guo and Li (2026). [Lookahead-R: Budget-Aware Tool Retrieval via Execution-Centric Planning](https://arxiv.org/abs/2609.35811). ICMR 2026; arXiv 2609.35811.

Code and measured artifacts: https://github.com/michaelbawuah/toolret-hybrid-retrieval
