# Actual repeated warm-service latency benchmark

90 hash-selected source-balanced confirmation queries; 3 repetitions; 37,292 indexed tools; CPU 4 threads.

Every request performs BM25, dense query encoding, exact full-corpus dense search/sort, RRF, routing and conditional cross-encoder scoring. Models, corpus embeddings and indexes remain loaded. Initialization, warmup, validation, network, queueing and concurrency are excluded.

| Policy | Mean (ms) | p50 query mean (ms) | p95 query mean (ms) | Actual CE calls / requests |
|---|---:|---:|---:|---:|
| hybrid | 271.15 | 242.08 | 378.38 | 0 / 270 |
| always | 920.72 | 906.98 | 1172.28 | 270 / 270 |
| fixed_disagreement | 772.52 | 847.68 | 1137.29 | 210 / 270 |
| utility_75 | 791.44 | 850.87 | 1083.67 | 219 / 270 |

Percentiles above describe query means over repeated measurements. Event-level percentiles are separately available in summary.json. Repeated observations are clustered by query; they are not treated as independent samples.

| Paired comparison | Mean difference (ms) | 95% CI (ms) | Mean latency reduction | 95% CI |
|---|---:|---:|---:|---:|
| hybrid_vs_always | -649.57 | [-671.47, -626.08] | 70.6% | [67.0%, 73.2%] |
| fixed_disagreement_vs_always | -148.21 | [-207.79, -89.64] | 16.1% | [9.8%, 22.5%] |
| utility_75_vs_always | -129.29 | [-189.39, -73.89] | 14.0% | [8.2%, 20.0%] |

CI procedure: 10,000 paired percentile bootstrap resamples, stratified by source domain, seed 20261003.

All actual first-stage and final rankings matched the frozen confirmation cache. This benchmark establishes observed warm CPU retrieval behavior on this machine; it does not establish deployed, GPU, concurrent, or downstream task performance.

Selection, complete randomized request order, every measured event, per-query means, source/model/data hashes, package versions and embedding-shard fingerprints accompany this report.
