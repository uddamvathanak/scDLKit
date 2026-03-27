# Bioinformatics Original Paper target

## Decision

- target journal: [`Bioinformatics`](https://academic.oup.com/bioinformatics)
- article type: `Original Paper`
- checked date: `2026-03-26`
- repo visibility: planning only
- paper scope: full toolkit, not annotation-only

## Why this journal

`Bioinformatics` is the best primary target for the paper shape scDLKit is
trying to become. The journal explicitly accepts original research on methods,
software, and computational advances in molecular biology, expects real
biological data, and expects comparison against relevant existing methods. That
matches a toolkit plus benchmark paper much better than a protocol journal or a
short software note.

Why this over nearby options:

- versus [`Nature Protocols`](https://www.nature.com/nprot/aims): `Nature
  Protocols` publishes secondary, step-by-step protocol articles based on
  published primary research and is not a primary methods journal. That makes
  it a possible follow-on protocol paper later, not the main first target for
  scDLKit's current software and benchmark story. The distinction is explicit
  in Nature's own guidance and in its explanation of the difference from
  `Nature Methods`
  ([Aims & Scope](https://www.nature.com/nprot/aims),
  [Relationship to Other Nature Journals](https://www.nature.com/nprot/for-authors/relationship)).
- versus [`Bioinformatics Advances`](https://academic.oup.com/bioinformaticsadvances/pages/author-guidelines):
  that journal is a strong fallback and is clearly in scope for the work, but
  the working goal should be the stricter, more established `Bioinformatics`
  original-paper bar first.
- versus `Bioinformatics` `Application Note`:
  the application-note format is too small for the intended full-toolkit paper.
  The story needs room for benchmark design, task regimes, efficiency-aware
  evaluation, and cross-task evidence rather than just a concise software
  announcement.

## Official requirements

The maintainer-facing constraints to optimize around are:

- `Original Papers` are limited to **7 pages**, approximately **5,000 words
  excluding figures**
- the required manuscript sequence is:
  - Title page
  - Structured Abstract
  - Introduction
  - System and methods
  - Algorithm
  - Implementation
  - Discussion
  - References
- the structured abstract for original papers uses these headings:
  - `Motivation`
  - `Results`
  - `Availability and Implementation`
  - `Contact`
  - `Supplementary Information`
- the abstract should stay succinct; the journal recommends about **150 words**
- titles should be short, specific, and informative; avoid generic words such
  as `tool`, `package`, or `software` unless they are genuinely necessary
- if software is a central contribution, it should be freely available to
  non-commercial users and remain available for at least two years after
  publication
- every submission requires a cover letter explaining why the paper belongs in
  `Bioinformatics`
- the journal warns that it pre-screens submissions and can reject papers
  without review if they are not significant enough for its readership
- the journal expects methods, systems, and data to be public, and it places
  strong emphasis on realistic biological applicability and real-data
  evaluation

Primary source:

- [`Bioinformatics` author guidelines](https://academic.oup.com/bioinformatics/pages/author-guidelines)

## Subjective pass criteria

What is likely to get desk-rejected:

- a manuscript that reads like an internal roadmap or package announcement
  instead of a finished scientific contribution
- a full-toolkit claim backed only by annotation results
- incremental wrapper engineering without a clear computational biology advance
- weak or missing comparison against strong baselines and contemporary methods
- evidence based mostly on toy data, smoke tests, or engineering convenience
- page-budget sprawl that makes the central claim hard to see
- unclear availability of code, data, or benchmark artifacts

What is more likely to feel in scope:

- a clear computational bottleneck in single-cell foundation-model adaptation
  and evaluation
- a unified adaptation and benchmark design that is broader than a single
  dataset or tutorial
- rigorous comparison on real biological data with explicit held-out regimes
- efficiency-aware reporting, not just best-score reporting
- an evidence package that is reproducible, public, and easy to audit
- a discussion that is honest about current biological and modeling limits

## Style and narration study

Exemplar classes to learn from:

- framework and benchmark paper:
  [`CellBench`](https://academic.oup.com/bioinformatics/article/36/7/2288/5645177)
- method paper with biological validation:
  [`DiSC`](https://academic.oup.com/bioinformatics/article/doi/10.1093/bioinformatics/btaf327/8153702)
- software and resource paper:
  [`OpenBioLink`](https://academic.oup.com/bioinformatics/article/36/13/4097/5825726)

Recurring narrative moves across those papers:

- open with a field bottleneck, not with package history
- narrow quickly to one unresolved gap that the paper will actually close
- state exactly what is introduced before drifting into implementation detail
- quantify the main comparative claim early
- treat availability as a short factual statement, not as the main story
- put future work in the discussion, not in the headline claim
- use the discussion to frame limitations, realistic applicability, and next
  expansions
- avoid sounding like a changelog, benchmark logbook, or milestone recap

What to copy from the exemplar styles:

- from the framework and benchmark class: reproducibility, benchmark
  extensibility, and consistent I/O abstractions are presented as scientific
  enablers, not just engineering preferences
- from the method-validation class: method novelty is tied to biological
  realism, real datasets, and runtime or scalability claims that are directly
  measurable
- from the software and resource class: architecture is described cleanly, but
  only after the paper explains why the benchmark or resource matters to the
  field

## Implications for scDLKit

Current repo truth:

- scDLKit is currently closer to a software and methods paper than to a protocol
  paper
- M1 annotation is a strong first results pillar with a real benchmark bundle,
  PEFT comparisons, held-out regimes, efficiency metrics, and publication-grade
  figures
- the repo already contains the core ingredients for the annotation part of a
  `Bioinformatics` paper: real biological data, classical baselines, held-out
  evaluation, and public artifacts

What the full-toolkit `Bioinformatics` paper still requires:

- all four downstream task pillars, not only annotation
- common benchmark-object language across tasks
- efficiency-aware reporting across tasks, not only within annotation
- broader model and adaptation scope consistent with the abstract's unified-
  interface claim
- a frozen manuscript evidence package rather than scattered per-milestone
  artifacts

What must not be claimed before M2-M5 land:

- that the full toolkit has been validated across all intended downstream tasks
- that the unified adaptation interface already spans the broader model set
- that self-supervised adaptation is part of the shipped evidence story
- that the final `Bioinformatics` paper is submission-ready

Practical conclusion:

- M1 annotation results are the first validated results section, not the whole
  paper
- the paper should eventually be written as:
  - problem in single-cell foundation-model adaptation
  - toolkit design
  - benchmark protocol
  - multi-task results
  - efficiency and generalization results
  - limitations and scope

## Writing constraints for this repo

Title style:

- prefer a short, specific title centered on the scientific problem and the
  benchmark or adaptation contribution
- avoid names like `scDLKit: a software package for ...` unless there is no
  cleaner option

Abstract style:

- `Motivation` should state the field bottleneck in fragmented single-cell
  foundation-model adaptation and weakly standardized evaluation
- `Results` should contain the concrete evidence claim, not generic ambition
- `Availability and Implementation` should name the repo and package channels
  concisely
- the abstract should read like a claim and evidence summary, not a feature list

Introduction expectations:

- open with the scientific and benchmarking problem in the field
- explain why current practice is insufficient
- define what scDLKit changes and what the paper contributes
- avoid opening with project history, milestone order, or documentation status

Results section shape:

- organize results around claims, not repo milestones
- move from benchmark design and task setup into quantitative results
- keep annotation as the first validated pillar, then add the later task pillars
  as they land
- quantify generalization and efficiency claims instead of describing them
  loosely

Discussion scope:

- be explicit about scGPT-only evidence where that remains true
- discuss dataset-quality limits, vocabulary overlap, and full fine-tuning
  collapse where relevant
- treat future multi-model expansion and self-supervised work as forward-looking
  scope, not current evidence

### Target manuscript structure

1. Title page
   - short, specific title
   - avoid `tool`, `software`, or `package` unless unavoidable
2. Structured Abstract
   - `Motivation`
   - `Results`
   - `Availability and Implementation`
   - `Contact`
   - `Supplementary Information`
3. Introduction
   - fragmented single-cell FM adaptation and weakly standardized evaluation
   - why current practice is insufficient
   - what scDLKit changes
4. System and methods
   - task definitions
   - regime definitions
   - shared adaptation interface
   - efficiency metrics
5. Algorithm
   - API and model-adaptation abstraction
   - PEFT strategies
   - regime objects and benchmark orchestration
6. Implementation
   - package structure
   - benchmark runner
   - docs, tutorials, accessibility, and availability
7. Discussion
   - current scope limits
   - biological realism
   - scGPT-only phase and future multi-model work
   - vocabulary overlap and dataset-quality caveats
8. References

The repo currently has the strongest evidence for sections 3-6 around
annotation. The full-paper version of that structure is not defensible end to
end until later milestones land their own evidence.

## Submission package checklist

- manuscript draft in `Bioinformatics` original-paper section order
- figure inventory that fits the 7-page budget
- table inventory that fits the 7-page budget
- supplementary-material inventory used to keep the main paper focused
- claims-versus-evidence map tied to milestone artifacts
- code, data, and benchmark-artifact availability statement
- concise `Availability and Implementation` text for the abstract
- cover-letter argument for why the paper belongs in `Bioinformatics`

## Open questions to resolve later

- which representation-transfer datasets and baselines become the anchor M2
  evidence for the paper
- which perturbation and spatial tasks are strong enough to survive the paper's
  page budget
- whether the final M5 model breadth is enough to support the unified-interface
  claim without narrowing the title
- whether the full toolkit still fits the 7-page original-paper format cleanly
  or needs scope tightening before submission
