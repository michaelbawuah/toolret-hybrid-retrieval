# ToolRet: technical interview walkthrough

## What the study establishes

This is an empirical study of when an expensive reranker is worth running for tool retrieval. It applies known retrieval and regression methods to a reproducible tool benchmark; it does not claim a new gating algorithm, a publication, or faculty supervision.

Across **1,500 held-out queries and 37,292 tools**, the frozen learned `utility_75` router required **1,149 cross-encoder calls**, versus 1,500 for always-on reranking: **23.4% fewer calls and scored pairs**. Binary nDCG@10 was **0.527257 versus 0.524056**. These are retrieval and compute-count results; latency needs separate measurement.

## How retrieval works

**BM25** rewards matching query words, adjusting for how common a word is and document length. It is useful for endpoint names and exact terms. **Dense MiniLM retrieval** compares normalized query/tool embeddings, helping retrieve descriptions with related wording. Each returns 100 candidates.

**Reciprocal rank fusion (RRF)** combines positions rather than incompatible raw scores. A tool receives `1/(60 + rank)` from each list where it appears, with equal weights. **The cross-encoder (CE)** then reads the query and each of the top 20 fused tools together and scores relevance. It changes the prefix order while preserving candidate membership and the remaining tail. Its maximum pair length is 256 tokens.

**Routing** decides whether to invoke CE or return the fused ranking directly. Skipping CE cannot fix missing candidates; invoking it can improve or harm ordering.

## The learned router

Seven cheap features describe only the first-stage rankings and query length:

| Feature | Meaning |
|---|---|
| Top-1 disagreement | Whether BM25 and dense choose different first tools |
| Top-10 Jaccard | Shared top-10 tools divided by their union |
| Top-20 Jaccard | The same overlap measure at 20 |
| Reciprocal-weighted overlap | Overlap weighted toward earlier ranks |
| Rank coherence | Agreement in top-20 positions, treating absence as rank 21 |
| Log query length | `log(1 + whitespace-token count)` |
| Normalized RRF margin | Separation between the first two fused scores |

Ridge regression predicts **CE nDCG@10 minus hybrid nDCG@10** from 300 development queries. Feature standardization uses development data only; the frozen regularization parameter is 10. Ridge penalizes large coefficients to limit overfitting on this small training set.

Another 300 calibration queries supply prediction quantiles for nominal 25%, 50%, and 75% routing budgets. Their relevance labels and CE outcomes do not set thresholds. At inference, rerank only when predicted benefit exceeds the frozen threshold. A nominal 75% budget is not an exact test quota: the observed routing rate was 76.6%.

Features do not read CE scores, relevance labels, source names, or endpoint identities. Tool IDs are used only to compare overlap and positions; consistent renaming leaves features unchanged.

## Why the holdout and statistical claim matter

Queries connected through shared positive tool IDs form components. Development, calibration, and confirmation have disjoint query IDs, normalized texts, positive tools, and entire components. Additional queries connected to the observed pilot are quarantined. The 1,500-query confirmation has 500 queries per source; routing decisions are saved before quality scoring.

The primary question was **noninferiority**, with a predeclared engineering margin of 0.01 nDCG: both query and component bootstrap lower bounds must exceed −0.01, with fewer required CE calls. The learned-minus-always intervals were **[−0.001560, 0.007984]** and **[−0.001558, 0.007931]**, satisfying that quality criterion.

This does **not** establish superiority: both intervals include zero. Component separation also does not prove unseen API-family generalization or clean pretrained-model data.

## What surprised us

The simple rule “rerank when first choices disagree” achieved **0.531372 nDCG with 1,077 calls**, a stronger observed tradeoff than the learned primary router. The learned-versus-simple comparison includes zero; do not claim the learned method wins. This is a useful negative finding about added complexity.

The learned router bypassed **60 CE improvements, 87 CE regressions, and 204 neutral cases**. Skipping sacrifices some useful reranks while avoiding others that hurt; the net gain versus always was 0.003201 nDCG.

ToolBench exposes a real weakness: CE lowered nDCG by **0.038396**. Candidate-label recall@20 was **0.567067**; **132/500 queries** had no positive in the scored prefix. CE moved **94 positive labels out of top 10 and 47 in**. Case reviews show secondary request components being demoted and unjudged counterparts being promoted. Unjudged alternatives are not automatically wrong or executable equivalents. **152/184 loss queries had no truncated positive pair**, so clipping positives cannot explain every regression.

## Discussing engineering evidence

Read the [frozen evaluation](../results/research_confirmation_20261003/confirmation_eval/report.md), [failure analysis](../results/research_confirmation_20261003/confirmation_failure_analysis.md), and [reproduction commands](confirmation_reproduction.md). For timing questions, follow the measured-runtime/report links in the [evidence index](../README.md); never convert fewer calls into an assumed latency speedup. Use these artifacts to explain implementation choices, observed tradeoffs, and the study's limits.
