# Report redesign notes

The first edition presented the experiment as a verification summary. Its abstract began with an acceptance criterion, several paragraphs repeated the same numerical result, and a full page emphasized audit counts and fingerprints. Large sans-serif headings, shaded summary tables, and fixed page breaks reinforced that presentation. The revision explains the retrieval problem and the observed behavior before discussing the evidence checks.

## Public examples examined

- [Stanford CS224N outstanding reports, Winter 2024](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1244/project.html) identifies the following two student reports as outstanding custom projects. This establishes their selection; it does not imply any Stanford affiliation for this project.
- [Count Your Words Before They Hatch](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1244/final-projects/KatherineLi.pdf), by Katherine Li, introduces the problem through a concrete failed word-count request on page 1. Its analysis connects performance by sentence length to the training distribution and discusses specific errors. Borrow the sequence from task example to measured finding to interpretation, rather than its wording or claims.
- [Agent Retrieval on Textual and Relational Knowledge Bases](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1244/final-projects/ShiyuZhao.pdf), by Shiyu Zhao, pairs a numbered results table with interpretation. Figure 5 on page 8 places the query, labels, retrieved candidates, and comparison together. Borrow the explanatory role of a case study and self-contained captions.
- [BEIR](https://arxiv.org/pdf/2104.08663), by Thakur and colleagues, connects heterogeneous retrieval results to computational cost and then examines annotation bias. Its latency table names the corpus and measurement conditions. Borrow that separation of effectiveness, efficiency, and relevance-judgment limitations.
- [Stanford’s report instructions](https://web.stanford.edu/class/cs224n/project/Project_Report_Instructions.pdf) and [Jennifer Widom’s writing guide](https://cs.stanford.edu/people/widom/paper-writing.html) emphasize concise problem–approach–finding abstracts, introductions that explain difficulty, and conclusions that do not repeat the abstract verbatim.

## Design for this study

The revised paper has six main pages: motivation and related work; experimental design and routing; ranking quality; measured latency; source behavior and case analysis; discussion, conclusion, and references. Two appendix pages contain complete feature definitions, additional controls, fingerprints, and reproduction details. Detailed evidence remains linked from the repository.

Use restrained serif typography, a modest title, single-column text, numbered tables and figures, meaningful captions, and thin table rules. Keep policy colors consistent across three measured figures: quality/compute with primary intervals; paired query latency and policy means; source changes and win/loss counts. Average timing repetitions within query before plotting paired observations.

The main argument must retain the competitive disagreement baseline, inconclusive learned-versus-source-matched-random comparison, ToolBench regression, and distinction between full-cohort calls and timing-subset requests. Explain these findings in plain scientific prose. Preserve every measured input, primary policy, uncertainty interval, and reproduction command; redesign the presentation rather than the experiment.
