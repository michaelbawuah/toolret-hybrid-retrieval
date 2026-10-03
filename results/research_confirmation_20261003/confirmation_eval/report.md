# Held-out selective tool-retrieval confirmation study

Queries: **1500**. Binary relevance; cutoff 10. Frozen pointwise routing; no confirmation policy selection.

| Policy | nDCG@10 | Recall@10 | MRR@10 | All-positive-label coverage@10 | CE calls | CE pairs |
|---|---:|---:|---:|---:|---:|---:|
| bm25 | 0.468860 | 0.584044 | 0.472515 | 0.488667 | 0 | 0 |
| dense | 0.460233 | 0.591100 | 0.458337 | 0.502667 | 0 | 0 |
| hybrid | 0.520821 | 0.652178 | 0.519572 | 0.562000 | 0 | 0 |
| always | 0.524056 | 0.649944 | 0.530097 | 0.548000 | 1500 | 30000 |
| fixed_disagreement | 0.531372 | 0.653833 | 0.538817 | 0.555333 | 1077 | 21540 |
| cheap_jaccard | 0.523930 | 0.651300 | 0.527559 | 0.552000 | 1391 | 27820 |
| utility_25 | 0.526901 | 0.654878 | 0.527166 | 0.563333 | 409 | 8180 |
| utility_50 | 0.530525 | 0.656922 | 0.533859 | 0.562000 | 829 | 16580 |
| utility_75 | 0.527257 | 0.651956 | 0.532669 | 0.552667 | 1149 | 22980 |

## Predeclared primary claim

Meets the predeclared quality and required-call criteria on this held-out cohort; actual conditional execution must be verified separately.

Utility75 versus always-on reranking, engineering margin 0.01. Success requires both paired 95% interval lower bounds strictly above −0.01 and fewer CE invocations.

| Comparison | Mean nDCG difference | Source-stratified query 95% CI | Source-stratified gold-component 95% CI |
|---|---:|---:|---:|
| utility_75_minus_always | 0.003201 | [-0.001560, 0.007984] | [-0.001558, 0.007931] |
| fixed_disagreement_minus_always | 0.007316 | [0.002293, 0.012378] | [0.002147, 0.012458] |
| hybrid_minus_always | -0.003235 | [-0.015542, 0.008995] | [-0.015779, 0.009131] |
| utility_75_minus_fixed_disagreement | -0.004114 | [-0.009234, 0.001147] | [-0.009354, 0.000967] |
| utility_75_minus_random_global_mean | 0.004761 | [0.000032, 0.009600] | [0.000197, 0.009326] |
| utility_75_minus_random_source_matched_mean | 0.003056 | [-0.001503, 0.007786] | [-0.001379, 0.007557] |
| fixed_disagreement_minus_random_global_mean | 0.008847 | [0.004042, 0.013781] | [0.003842, 0.013733] |
| fixed_disagreement_minus_random_source_matched_mean | 0.010795 | [0.005759, 0.015963] | [0.005520, 0.015959] |

Component bootstrap samples groups within each source and uses a ratio-of-sums estimator to retain the query-macro target. Both procedures keep source weights at 1/3 and reuse paired draws across comparisons.

## Equal-budget random controls

| Control | Calls per seed | Mean nDCG | Seed range |
|---|---:|---:|---:|
| global_fixed_disagreement | 1077 | 0.522524 | [0.517310, 0.527781] |
| global_utility_25 | 409 | 0.521395 | [0.513734, 0.527221] |
| global_utility_50 | 829 | 0.521643 | [0.516458, 0.527598] |
| global_utility_75 | 1149 | 0.522497 | [0.518231, 0.528291] |
| source_matched_fixed_disagreement | 1077 | 0.520577 | [0.512594, 0.527697] |
| source_matched_utility_75 | 1149 | 0.524202 | [0.518584, 0.527429] |

Seed ranges are random-policy variation, not confidence intervals.

## Limitations

- Gold components connect queries sharing positive tool IDs; they are not verified API-provider families.
- Binary qrels and all-positive-label coverage do not demonstrate downstream agent execution success.
- Target budgets are calibrated pointwise thresholds; realized test invocation fractions can differ from 25/50/75%.
- Only utility75 noninferiority is primary; other policy/metric comparisons and domain breakdowns are exploratory.
- The 0.01 margin is a predeclared engineering tolerance, not an externally accepted universal quality threshold.
- Random seed ranges express routing randomness; they are not statistical confidence intervals.
- Oracle selection uses test labels and is diagnostic, never a deployed or competitive policy.
- Invocation and pair counts are compute proxies; actual repeated latency is reported separately.
- Public pretrained models may have seen related source data; dataset group separation does not prove pretraining cleanliness.
