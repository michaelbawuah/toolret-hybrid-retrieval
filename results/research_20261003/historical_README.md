# Efficient Tool Retrieval for LLM Agents

Independent reproduction and extension of **ToolRet (ACL 2025)** exploring whether a lightweight multi-stage retrieval pipeline can improve tool retrieval for LLM agents.

The system combines **BM25 sparse retrieval, dense retrieval, hard-negative fine-tuning, weighted Reciprocal Rank Fusion (RRF), and compact cross-encoder reranking**. Experiments measure both retrieval quality and computational cost on a ToolRet subset containing **37,292 candidate tools**.

> **Status:** End-to-end experimental pipeline complete, including frozen held-out test evaluation.

---

## Research Question

Can a lightweight multi-stage retrieval pipeline combining sparse retrieval, dense retrieval, hard-negative training, and compact reranking substantially improve tool retrieval quality while keeping inference costs practical?

This repository investigates that question through controlled retrieval experiments rather than treating the original ToolRet implementation as a black box.

---

## Motivation

LLM agents increasingly rely on external tools and APIs. Before an agent can call the correct tool, however, it must first retrieve that tool from a potentially large catalog.

Tool retrieval is therefore an information-retrieval problem:

```text
Natural-language query
        |
        v
Large tool catalog
        |
        v
Retrieve relevant tools
        |
        v
LLM / agent tool selection
```

Poor retrieval can prevent an otherwise capable language model from ever seeing the correct tool.

This project studies whether relatively small retrieval models can be made substantially stronger through better training and multi-stage retrieval.

---

## System Architecture

The final experimental pipeline is:

```text
                         User Query
                             |
                 +-----------+-----------+
                 |                       |
                 v                       v
          BM25 Retrieval         Fine-tuned MiniLM
          Sparse Ranking          Dense Retrieval
                 |                       |
                 +-----------+-----------+
                             |
                             v
                    Weighted RRF Fusion
                  BM25 = 1.0
                  Dense = 1.25
                  RRF k = 10
                             |
                             v
                    Top-500 Candidates
                             |
                             v
                 Cross-Encoder Reranker
                     Rerank Top 3
                             |
                             v
                    Final Ranked Tools
                             |
                             v
                MRR / Recall@K / nDCG@K
```

The dense retriever is fine-tuned using mined hard negatives. Sparse and dense rankings are then combined with weighted Reciprocal Rank Fusion. A compact cross-encoder reranks only the highest-ranked candidates.

---

## Experimental Setup

The experiments use a processed subset of the ToolRet benchmark.

| Component | Configuration |
|---|---|
| Candidate tools | 37,292 |
| Validation queries | 15 |
| Frozen test queries | 16 |
| Base dense model | `sentence-transformers/all-MiniLM-L6-v2` |
| Fine-tuned retriever | MiniLM + hard-negative training |
| Sparse retriever | BM25 |
| Fusion | Weighted Reciprocal Rank Fusion |
| BM25 weight | 1.0 |
| Dense weight | 1.25 |
| RRF k | 10 |
| Candidate pool | 500 |
| Reranker | Compact cross-encoder |
| Final rerank depth | 3 |
| Evaluation | MRR, Recall@K, nDCG@K |

Hyperparameters were selected using the validation split before the final test evaluation.

The **16-query test split was treated as frozen** during final evaluation and was not used for subsequent hyperparameter tuning.

---

# Results

## Validation Ablation

The validation experiments show how retrieval quality changed as components were introduced.

| System | MRR | nDCG@10 |
|---|---:|---:|
| BM25 | 0.337 | 0.328 |
| Base MiniLM | 0.274 | 0.272 |
| Hard-negative MiniLM | 0.522 | 0.544 |
| Hybrid retrieval | 0.667 | 0.648 |
| Hybrid + reranker | **0.711** | **0.677** |

![Validation ablation](assets/validation_ablation.png)

### Validation Findings

The largest dense-retrieval improvement came from **hard-negative fine-tuning**.

MRR increased from approximately:

```text
0.274 -> 0.522
```

after fine-tuning MiniLM on mined difficult negatives.

Combining sparse and dense retrieval produced another substantial improvement:

```text
0.522 -> 0.667 MRR
```

The compact cross-encoder then improved validation MRR further:

```text
0.667 -> 0.711
```

The validation experiments therefore support a multi-stage design in which sparse lexical matching and learned semantic retrieval contribute complementary signals.

---

## Frozen Test Results

After selecting the pipeline using validation experiments, the configuration was evaluated on the held-out test split.

| System | MRR | Recall@1 | Recall@3 | Recall@5 | Recall@10 | nDCG@10 |
|---|---:|---:|---:|---:|---:|---:|
| BM25 | 0.132 | 0.063 | 0.188 | 0.188 | 0.208 | 0.150 |
| Base MiniLM | 0.135 | 0.083 | 0.083 | 0.083 | 0.146 | 0.114 |
| Hard-negative MiniLM | 0.499 | 0.313 | 0.427 | 0.448 | 0.552 | 0.469 |
| Hybrid retrieval | **0.515** | **0.313** | **0.458** | **0.490** | **0.604** | **0.490** |
| Hybrid + reranker | 0.473 | 0.271 | 0.458 | 0.490 | 0.604 | 0.468 |

![Frozen test performance](assets/test_results.png)

---

## Main Findings

### 1. Hard-negative training was the largest single improvement

The base MiniLM retriever achieved only **0.135 test MRR**.

After hard-negative fine-tuning, test MRR increased to approximately **0.499**.

This suggests that training on difficult, retrieval-specific negatives was considerably more important than simply using an off-the-shelf embedding model.

### 2. Sparse and dense retrieval were complementary

The weighted hybrid system increased test MRR from approximately:

```text
0.499 -> 0.515
```

and Recall@10 from:

```text
0.552 -> 0.604
```

relative to the fine-tuned dense retriever alone.

This supports the hypothesis that lexical BM25 evidence and dense semantic similarity capture different useful signals.

### 3. Reranking improved validation performance but did not generalize to test MRR

The cross-encoder increased validation MRR:

```text
0.667 -> 0.711
```

but decreased frozen-test MRR:

```text
0.515 -> 0.473
```

while Recall@10 remained approximately **0.604**.

This is an important negative result: adding a learned reranker did not automatically improve held-out retrieval performance.

One plausible explanation is sensitivity to the small training/validation sample, but the present experiment is not large enough to establish the cause.

### 4. Validation and test performance differed substantially

The final reranked system achieved approximately **0.711 validation MRR** but **0.473 test MRR**.

Because the current validation and test subsets contain only 15 and 16 queries respectively, these estimates have high variance.

The project therefore reports both validation and held-out results rather than presenting the best validation score as final performance.

---

## Latency

The frozen reranked pipeline recorded approximately:

| Stage | Mean query latency |
|---|---:|
| BM25 retrieval | 34.6 ms |
| Dense retrieval | 212.2 ms |
| RRF fusion | 0.18 ms |
| Cross-encoder reranking | 14.6 ms |

The dense retrieval stage dominates query-time cost in the current implementation.

The reranker operates only on the top three fused candidates, limiting its additional latency.

These measurements reflect the current experimental implementation and local hardware rather than an optimized production retrieval service.

---

## Why This Repository Is Independent

The official ToolRet implementation is treated as a benchmark reference.

This repository independently implements the core experimental components so that individual design decisions can be inspected and modified:

- ToolRet data processing
- BM25 retrieval
- dense embedding retrieval
- retrieval metrics
- reciprocal-rank fusion
- weighted RRF
- hard-negative mining
- dense retriever fine-tuning
- cross-encoder training
- reranking
- validation ablations
- latency measurement
- frozen test evaluation
- result visualization

The objective is not simply to reproduce a reported number, but to understand **which retrieval components help, when they help, and what they cost**.

---

## Evaluation Metrics

### Mean Reciprocal Rank (MRR)

MRR measures how highly the first relevant tool appears in the ranking.

Higher is better.

### Recall@K

Recall@K measures how much of the relevant tool set appears within the first `K` retrieved results.

The experiments report:

```text
Recall@1
Recall@3
Recall@5
Recall@10
```

### nDCG@K

Normalized Discounted Cumulative Gain rewards systems that place relevant tools nearer the top of the ranking while accounting for ranking position.

---

## Repository Structure

```text
toolret-research/
├── README.md
├── pyproject.toml
├── assets/
│   ├── test_results.png
│   └── validation_ablation.png
├── checkpoints/
├── data/
│   ├── sample/
│   └── toolret/
│       ├── corpus.jsonl
│       ├── queries.jsonl
│       └── splits/
├── models/
├── paper/
│   └── notes.md
├── results/
├── scripts/
│   ├── create_query_splits.py
│   ├── evaluate_dense_models.py
│   ├── evaluate_hybrid_finetuned.py
│   ├── evaluate_hybrid_reranker.py
│   ├── mine_hard_negatives.py
│   ├── plot_results.py
│   ├── prepare_toolret.py
│   ├── run_bm25.py
│   ├── run_candidate_ablation.py
│   ├── run_dense.py
│   ├── run_hybrid.py
│   ├── run_rrf_ablation.py
│   ├── train_dense_hard_negatives.py
│   └── train_toolret_reranker.py
├── src/toolret_research/
│   ├── bm25.py
│   ├── data.py
│   ├── dense.py
│   ├── fusion.py
│   ├── hard_negatives.py
│   ├── hybrid.py
│   ├── metrics.py
│   ├── reranker.py
│   └── text.py
└── tests/
    └── run_tests.py
```

---

## Reproducing the Core Experiments

Create and activate a Python environment, install the project dependencies, and run commands from the repository root.

For scripts executed directly from `scripts/`, expose the source package with:

```bash
PYTHONPATH=src
```

For example, the BM25 evaluation interface can be inspected with:

```bash
PYTHONPATH=src python scripts/run_bm25.py --help
```

The dense-model comparison can be inspected with:

```bash
PYTHONPATH=src python scripts/evaluate_dense_models.py --help
```

The hybrid retrieval experiment can be inspected with:

```bash
PYTHONPATH=src python scripts/evaluate_hybrid_finetuned.py --help
```

Figures are generated with:

```bash
PYTHONPATH=src python scripts/plot_results.py
```

---

## Research Integrity

Results in this repository are separated conceptually into:

1. **Paper-reported results** — values reported by the ToolRet authors.
2. **Reproduced results** — results generated by independent baseline experiments in this repository.
3. **Extension results** — results produced by the hard-negative, hybrid retrieval, and reranking experiments developed here.

No metric should be presented as a paper result unless it was actually reported by the original authors.

No metric should be presented as a project result unless it was generated by a reproducible experiment in this repository.

The frozen test set is not used for post-hoc hyperparameter selection.

---

## Limitations
## Limitations and Failure Analysis

Although the experiments show substantial improvements over the original sparse and dense baselines, several limitations are important when interpreting the results.

### Small Evaluation Sets

The validation and frozen test subsets contain only **15 and 16 queries**, respectively. Because these evaluation sets are small, individual queries can have a large effect on aggregate metrics such as MRR and nDCG.

The reported results should therefore be interpreted as evidence about the behavior of the proposed retrieval pipeline rather than definitive benchmark-wide performance estimates. A larger evaluation would provide more stable estimates and stronger statistical confidence.

### Hybrid Retrieval Is Not Universally Better

The weighted BM25 + fine-tuned dense retrieval system achieved the strongest frozen-test performance before reranking, increasing MRR from **0.499 to 0.515** and Recall@10 from **0.552 to 0.604** relative to the hard-negative dense retriever.

However, per-query analysis shows that fusion does not improve every query. Relative to the fine-tuned dense retriever, hybrid fusion improved the rank of the first relevant tool on **4 test queries** while hurting it on **6**.

This apparent discrepancy is possible because MRR is sensitive not only to how many queries improve, but also to the magnitude and location of those rank changes. A small number of improvements near the top of the ranking can outweigh several smaller regressions.

### Reranker Generalization

The cross-encoder produced the strongest validation result, improving hybrid validation MRR from **0.667 to 0.711**. This improvement did not generalize to the frozen test set, where MRR decreased from **0.515 to 0.473**.

A descriptive analysis of the 16 frozen test queries showed:

| Reranker effect | Number of queries |
|---|---:|
| Improved relevant-tool rank | 2 |
| Hurt relevant-tool rank | 3 |
| Unchanged | 11 |

The two successful cases promoted a relevant tool from **rank 2 to rank 1**.

In contrast, the reranker made several costly top-rank mistakes. For one reservation query, a relevant tool moved from **rank 1 to rank 3**. For two additional queries, the first relevant result moved from **rank 1 to rank 2**.

These errors were sufficient to outweigh the two successful promotions and explain the reduction in test MRR.

### Candidate Recall vs. Ranking Quality

The reranker operates only on the **top 3 candidates** returned by the hybrid retriever. It reorders those candidates and then preserves the remainder of the fused ranking.

Consequently, test Recall@10 remained unchanged at **0.604** before and after reranking even though MRR decreased.

This distinction is important: the reranker's test regression is primarily a **ranking-quality failure**, not a candidate-recall failure. The relevant tools were already being retrieved; the cross-encoder sometimes placed them in worse positions.

### Experimental Retrieval Efficiency

The current dense retrieval implementation performs exact similarity search across the corpus of **37,292 tools**. This design is useful for controlled experimentation and keeps the implementation transparent, but it is not intended to represent a production-scale retrieval architecture.

A deployed system would likely use an approximate nearest-neighbor index such as HNSW or another vector-search backend to reduce retrieval latency as the corpus grows.

### Limited Reranker Supervision

The reranker was trained using a relatively small hard-negative training set. The difference between its validation and frozen-test behavior suggests that the cross-encoder may be more sensitive to the limited supervision than the first-stage dense retriever.

In contrast, hard-negative fine-tuning of the dense retriever produced the most consistent improvement in the project, substantially outperforming the base MiniLM model on both validation and test data.

This makes hard-negative dense training the strongest robust result of the current experiments, while the reranker should be considered a promising but less stable component.

### Test-Set Integrity

All model choices and retrieval hyperparameters were selected using the validation split before final test evaluation.

The test set was then frozen. The per-query failure analysis reported above was performed **only after the final test results were obtained** and is strictly descriptive.

No model, fusion weight, candidate depth, RRF constant, reranking depth, or other configuration was changed in response to test-set performance.

This separation is important for preventing test-set leakage and preserving the validity of the held-out evaluation.

### Future Work

Several extensions could strengthen the conclusions of this project:

- evaluate on substantially larger query sets and additional ToolRet domains;
- repeat experiments across multiple random seeds or cross-validation folds;
- investigate stronger hard-negative mining strategies;
- evaluate approximate nearest-neighbor indexing for production-scale retrieval;
- train the reranker with substantially more diverse supervision;
- analyze query categories to determine when sparse, dense, or hybrid retrieval is most effective;
- evaluate statistical uncertainty and significance of differences between retrieval stages.

These extensions are intentionally left as future work rather than being optimized against the current frozen test set.

---

## Next Steps

Planned extensions include:

- larger-scale evaluation across additional ToolRet domains
- query-category breakdowns
- confidence intervals and repeated evaluation splits
- approximate nearest-neighbor indexing
- memory and index-size measurements
- additional hard-negative mining strategies
- more diverse reranker training data
- full technical report

---

## Acknowledgment

This project is inspired by **ToolRet: Retrieval Models Aren't Tool-Savvy: Benchmarking Tool Retrieval for Large Language Models (ACL 2025)**.

The original work provides the benchmark and motivation; the retrieval pipeline, experiments, extensions, analysis, and visualizations in this repository are developed as an independent research reproduction and extension.