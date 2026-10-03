# Reproduce the confirmation study

Use Python 3.12 and a full checkout of this repository. The default command
re-evaluates the published frozen router and ranking caches; it does not rerun
neural inference or replace any published evidence.

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install 'torch==2.14.1+cpu' --index-url https://download.pytorch.org/whl/cpu
.venv/bin/python -m pip install -r requirements-research-20261003.txt
.venv/bin/python scripts/run_confirmation_study.py --output-dir data/research_reproduction_run1
```

The runner sets `PYTHONPATH`, uses the same Python interpreter for every child
command, and runs those commands from the repository root. The first run needs
network access to download missing public source-data parquets. Pinned source
revision and parquet hashes are checked before use. Existing ignored data files
are reused only when their corpus/query SHA256 matches the frozen manifests.
Modified existing files cause failure rather than silent replacement.

The fresh output contains `evaluation/summary.json`, complete policy decisions,
per-query metrics, a Markdown report, and `reproduction_provenance.json`. A new
output directory is required for every run. If `--output-dir` is omitted, a
unique UTC-stamped directory under `data/` is chosen.

## Regenerate inference and refit the router

```bash
.venv/bin/python scripts/run_confirmation_study.py --recompute --output-dir data/research_reproduction_recomputed1
```

This regenerates calibration rankings, fits the router using the published
300-query development cache and calibration features, then regenerates
confirmation rankings and evaluates the new router. Confirmation inputs do not
enter fitting. Corpus, model revisions, split membership, protocol, and query
bytes remain fixed. Fresh calibration/confirmation caches and router artifacts
are written only beneath the new output directory.

The published development ranking cache remains the training input: this option
is not a complete regeneration of the historical pilot. Fresh inference timing,
package versions, base Git commit, metadata paths and provenance may differ.
The runner verifies source-data byte identity but does not promise byte-identical
inference artifacts, weights across environments, or timestamps. CPU neural
inference takes substantially longer than the default cached evaluation.

## Measure actual runtime

```bash
.venv/bin/python scripts/run_confirmation_study.py --measure-runtime --output-dir data/research_reproduction_measured1
```

This adds actual primary conditional cross-encoder execution across all 1,500
confirmation queries, checking every policy decision and final ranking against
the evaluated evidence. It then runs the frozen 90-query latency subset with
three repetitions and four policies, recomputing BM25, dense query encoding,
exact corpus search, fusion, routing and conditional reranking for every request.
Models, corpus embeddings and indexes stay loaded. Initialization, network,
queueing and concurrent-service behavior are outside the timing boundary.

Compatible checkpoints at `checkpoints/research_20261003` are reused. If absent,
the runner regenerates corpus embeddings independently without regenerating
query ranking caches. Use `--embedding-dir PATH` to select another compatible
existing or new checkpoint directory. Embedding identity, shapes, unit norms,
finite values and shard hashes are validated before measurement.

`--recompute` and `--measure-runtime` may be combined. Run latency measurement
without another heavy workload on the same machine. Differences from published
timings are expected across hardware and library environments.

## Original versus reconstruction provenance

The evaluator always receives the committed
`results/research_confirmation_20261003/data_manifest.json` as its canonical
frozen parent. Absent query files are regenerated from pinned source rows using
the original deterministic positive-tool component assignment, then checked
against every declared query-file SHA256. If grouping is reconstructed, all five
cohort files are kept together, including excluded quarantine and reserve rows.

The reconstructed cohort directory receives a copy of the original published
`manifest.json`. That file describes the original study, not the later download.
`data_reconstruction_metadata.json` separately records the reconstruction time,
verified downloads and reconstructed files. No published manifest, ranking
cache, router weight or result is overwritten.

## Git ancestry

The evaluator checks that the protocol's bytes existed in the public frozen
commit `3e813e1af650104f33aaf77febce5a8c1a3f1213`. A full clone with the study
branch's history includes this object. A shallow checkout may require:

```bash
git fetch origin 3e813e1af650104f33aaf77febce5a8c1a3f1213 9a34862b31881545a266145b2096c0673b449182
```

The runner checks the protocol object before downloads or inference and reports
an explicit fetch command if it is absent. Source fingerprints intentionally
reject edits to the frozen router's fitting and feature implementation. Use the
study's published checkout for exact frozen-policy reproduction; developing a
changed method requires a separately scoped experiment.

## Regenerate the technical report

The published report is rendered from the completed measured artifacts and
refuses incomplete audits or mismatched evidence hashes. To regenerate its
eight-page PDF, Markdown, and three measured figures from the published completed study:

```bash
.venv/bin/python -m pip install -e '.[report]'
.venv/bin/python scripts/export_confirmation_report.py --output-pdf "$PWD/data/research_report_reproduced/report.pdf" --output-markdown data/research_report_reproduced/report.md --figure-dir data/research_report_reproduced/figures
```

This authoring step does not rerun inference, alter results, or establish a
new experiment. The report's fingerprint sidecar records its input and output
hashes; document-rendering details can vary across environments.
The figures are exported as 300-dpi PNGs for Markdown and vector PDFs for reuse.
The PDF and Markdown share one editorial source in the exporter. The exporter
checks the audit's published evidence files and any locally present source data;
absent untracked raw data is not needed to render the report.
