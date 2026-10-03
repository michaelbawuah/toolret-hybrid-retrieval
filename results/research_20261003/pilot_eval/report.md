# Fixed selective reranking evaluation

Scope: **preliminary_fresh_pilot**. Queries: **300**.

Fixed gate: `rerank iff rankings.bm25[0] != rankings.dense[0]`. No training or threshold tuning.

All quality metrics use binary qrels and a cutoff of 10.

| Policy | Recall@10 | MRR@10 | nDCG@10 | All-positive-label coverage@10 | CE calls | CE pairs |
|---|---:|---:|---:|---:|---:|---:|
| bm25 | 0.561389 | 0.471509 | 0.455288 | 0.456667 | 0 | 0 |
| dense | 0.533000 | 0.438721 | 0.420994 | 0.426667 | 0 | 0 |
| hybrid | 0.632333 | 0.478474 | 0.482782 | 0.526667 | 0 | 0 |
| always | 0.619556 | 0.511315 | 0.498600 | 0.510000 | 300 | 6000 |
| gated | 0.617333 | 0.517995 | 0.501864 | 0.510000 | 217 | 4340 |

Random reranking uses 20 deterministic seeds with the same invocation count as the gate.

- Recall@10: mean 0.621444; seed range 0.612889–0.631222.
- MRR@10: mean 0.501100; seed range 0.485337–0.513728.
- nDCG@10: mean 0.493190; seed range 0.481646–0.501838.
- All-positive-label coverage@10: mean 0.512833; seed range 0.500000–0.523333.

| Paired comparison | nDCG@10 difference | Query-bootstrap 95% interval |
|---|---:|---:|
| always_minus_hybrid | 0.015818 | [-0.009010, 0.040982] |
| gated_minus_hybrid | 0.019082 | [-0.004081, 0.042260] |
| gated_minus_always | 0.003265 | [-0.005829, 0.012830] |

## Estimated offline replay component sums

These are counterfactual sums of declared disjoint components, excluding gate/routing overhead. They are not measured adaptive or service latency.

| Policy | Estimated mean ms | Estimated p50 ms | Estimated p95 ms |
|---|---:|---:|---:|
| bm25 | 233.474 | 221.088 | 376.699 |
| dense | 31.154 | 28.921 | 45.983 |
| hybrid | 264.837 | 250.609 | 420.333 |
| always | 900.383 | 890.202 | 1189.774 |
| gated | 725.048 | 822.711 | 1179.068 |

No end-to-end adaptive latency was measured. Separate supplied component timings, when present, are in summary.json.

## Limitations

- Binary qrels; all-positive-label coverage means all positive relevance labels retrieved, not mandatory-tool coverage or proven task execution success.
- Confidence intervals resample queries; dependent queries or API families can make uncertainty look too small.
- No end-to-end adaptive latency is measured; invocation and candidate-pair reductions are compute proxies.
- Random-policy variation across seeds is not a confidence interval for performance on new queries.
- Reported sign tests are exploratory and are not corrected for multiple comparisons.
- Dataset-domain transfer does not establish generalization to unseen API families.
- Declared provenance and historical-ID exclusions are supplied evidence, not proof of absence of pretraining contamination.
- API-family provenance is absent; no API-family generalization claim is supported.
- Reported timings are separately supplied component measurements; they are not summed into adaptive latency.
- Offline replay estimates sum declared disjoint components sequentially; they exclude gate/routing overhead and are not measured service or adaptive-policy latency.
