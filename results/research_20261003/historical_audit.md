# Historical ToolRet rank audit — October 3, 2026

**Exploratory reanalysis of saved historical evidence; no retrieval model was run, trained, or tuned.**

Validated 16 rows with unique query IDs, positive-or-missing ranks, consistent reciprocal ranks, and consistent effect labels.

## Recomputed mean reciprocal rank

Intervals are 95% paired query bootstrap percentile intervals; absolute MRR uses a zero-valued reference.

| System | CSV MRR | Bootstrap interval | README MRR | Match at 3 decimals |
|---|---:|---:|---:|---|
| BM25 | 0.146065 | [0.029444, 0.295979] | 0.132 | no |
| Hard-negative MiniLM | 0.521205 | [0.307104, 0.736545] | 0.499 | no |
| Hybrid retrieval | 0.514667 | [0.315515, 0.716158] | 0.515 | yes |
| Hybrid + reranker | 0.473000 | [0.288257, 0.665923] | 0.473 | yes |

The saved CSV contradicts README's BM25 and fine-tuned dense MRR values. It gives hybrid minus dense a negative mean difference. This audit does not infer which original evaluation produced README's alternative numbers.

## Paired differences

Delta is after minus before. Sign-test p-values test win/loss balance, exclude ties, and have no multiplicity adjustment.

| Comparison | Mean ΔMRR | 95% bootstrap interval | Wins / losses / ties | Exact sign p |
|---|---:|---:|---:|---:|
| Hard-negative MiniLM − BM25 | +0.375141 | [+0.159851, +0.594728] | 14 / 1 / 1 | 0.000977 |
| Hybrid retrieval − BM25 | +0.368602 | [+0.208880, +0.541255] | 15 / 0 / 1 | 0.000061 |
| Hybrid retrieval − Hard-negative MiniLM | -0.006539 | [-0.155722, +0.154254] | 4 / 6 / 6 | 0.753906 |
| Hybrid + reranker − Hybrid retrieval | -0.041667 | [-0.187500, +0.093750] | 2 / 3 / 11 | 1.000000 |
| Hybrid + reranker − Hard-negative MiniLM | -0.048205 | [-0.259194, +0.175154] | 5 / 9 / 2 | 0.423950 |

## Evidence limitations

- This is a descriptive reanalysis of 16 saved historical queries, not a new held-out test or benchmark-wide reproduction.
- All saved query IDs use the apibank prefix; these rows do not demonstrate cross-domain or unseen-API-family generalization.
- The CSV stores only first-relevant ranks. It cannot independently recover Recall@K, nDCG@K, full candidate rankings, or downstream agent success.
- The CSV does not contain base MiniLM per-query ranks; README's base MiniLM MRR cannot be recomputed from it.
- The full processed corpus, original split files, trained checkpoints, training logs, and raw full-ranking evaluation JSON are not committed in this checkout. The claimed 37,292-tool run and validation-selected configuration are historical README/script claims, not newly verified artifacts.
- Blank positions mean missing from the saved candidate ranking; they do not prove that no relevant tool exists in the full catalog.
- The new CSV audit verifies internal consistency, not the origin or correctness of historical relevance labels, model outputs, train/test isolation, or latency measurements.
- Bootstrap intervals resample query rows, assume exchangeable query observations, and remain exploratory for this small and potentially correlated sample. They cannot correct selection bias or data leakage.
- Exact sign-test p-values concern win/loss balance and ignore magnitude. They are unadjusted across comparisons and do not establish mean-MRR significance.
- Do not train, choose a threshold, or change retrieval settings using these frozen historical test rows; future experiments need separate development and fresh evaluation data.

## Source hashes

- `results/test_failure_analysis.csv` — SHA-256 `4b81c678a902d6bbca29493ac8923d4027579c5ad2df7eb13bd3e0c30050f5f4`
- `results/research_20261003/historical_README.md` — SHA-256 `e76dcd7d327f966d024b450e13d98862b0c4c91e86c41f513d47501932b90eac`

Bootstrap: 20,000 resamples, seed 20261003, linear-interpolated percentile endpoints. Sampling unit: query.
