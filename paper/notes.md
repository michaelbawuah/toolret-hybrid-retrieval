# ToolRet paper notes

Paper: *Retrieval Models Aren't Tool-Savvy: Benchmarking Tool Retrieval for Large Language Models* (ACL 2025)

## Core problem

A tool-using LLM cannot place every possible tool/API definition in its context.
A retrieval system must identify a small candidate set of useful tools.

## Paper's core questions

1. How well do existing IR models perform on tool retrieval?
2. How much does retrieval quality affect downstream tool-use success?

## Benchmark scale reported by the paper

- about 7.6k retrieval tasks
- about 43k tools
- more than 200k training instances

## Concepts we must master

- inverted index
- TF / IDF / BM25
- sparse vs dense retrieval
- bi-encoder vs cross-encoder
- hard negatives
- Recall@K / MRR / nDCG@K
- latency-quality trade-offs

## Our first hypothesis

Sparse and dense retrieval will fail on different query/tool pairs. A cheap fusion
method may recover relevant tools that either retriever misses alone.

This is a hypothesis, not a conclusion.
