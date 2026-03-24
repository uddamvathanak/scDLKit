# Roadmap

## Paper vision

scDLKit is moving toward a publication-first identity rather than a feature
inventory. The paper target is:

**scDLKit is a minimal-code, AnnData-native framework for parameter-efficient adaptation and reproducible benchmarking of single-cell and spatial foundation models.**

That target has four model pillars:

- `scGPT`
- `scFoundation`
- `CellFM`
- `Nicheformer`

It also has four research-task pillars:

- cell type annotation
- integration / representation transfer
- perturbation-response prediction
- spatial domain / niche classification

And it has a common adaptation comparison set:

- frozen embeddings plus linear probe
- full fine-tuning
- LoRA
- adapters
- prefix tuning
- IA3-style scaling

The paper-level benchmark target should eventually evaluate those methods across:

- full-label regimes
- low-label regimes
- cross-study regimes

## Current implementation truth

The current repo is intentionally narrower than the paper target.

### Implemented

- stable baseline workflows through `TaskRunner`
- lower-level training and extension through `Trainer` plus adapters
- reproducible evaluation, reports, and docs-contract validation
- experimental scGPT frozen embedding support
- experimental scGPT annotation adaptation with a wrapper-first path
- generic annotation PEFT configs:
  - `LoRAConfig`
  - `AdapterConfig`
  - `PrefixTuningConfig`
  - `IA3Config`
- scGPT annotation strategies:
  - frozen probe
  - head-only tuning
  - full fine-tuning
  - LoRA
  - adapters
  - prefix tuning
  - IA3
- a dedicated annotation benchmark runner covering full-label, low-label, and
  cross-study regimes
- beyond-PBMC annotation evidence on cached human-pancreas subsets

### Pilot

- experimental foundation-model support is currently `scGPT` only
- the strongest current research-task story is annotation
- the current beyond-PBMC evidence story is annotation-focused rather than
  fully task-balanced
- the annotation pillar still needs its first frozen benchmark artifact bundle
  before it should be promoted publicly from `Pilot` to `Implemented`

### Planned

- `scFoundation`
- `CellFM`
- `Nicheformer`
- integration benchmark pipeline
- perturbation benchmark pipeline
- spatial benchmark pipeline

The roadmap should never imply that paper-target scope is already available in
the current release line.

## Current objective

The current active milestone is still the annotation pillar, but the work has
shifted from interface design to evidence freeze.

Why annotation is next:

- it is the strongest implemented research-facing capability already in the repo
- it already has the generic PEFT layer, benchmark script, and main published
  tutorial in place
- it sets the benchmark, artifact, and PEFT comparison conventions that later
  task pillars should reuse

Done for the annotation pillar means:

- annotation has a task spec and a benchmark matrix with frozen, head,
  full-finetune, and PEFT comparisons on the current scGPT path
- the benchmark workflow produces reviewable artifact bundles for full-label,
  low-label, cross-study, and Pareto reporting
- the main annotation tutorial is the static executed human-pancreas notebook
  with visible last-run metadata
- figure-ready outputs are defined, generated, and tracked
- the milestone checklist can be closed without guessing

(annotation-pillar)=
## Milestone 1: Annotation pillar

Status: `Active`

Primary objective:
- make annotation the first paper-ready task pillar using the current scGPT
  adaptation path as the starting point

Required outcomes:
- annotation task spec
- dataset shortlist and registry requirements
- frozen / full-FT / LoRA / adapters / prefix / IA3 comparison matrix
- low-label and cross-study regime implementation
- one main research-facing annotation tutorial
- figure-ready artifact inventory
- benchmark workflow plus artifact freeze

(spatial-pillar)=
## Milestone 2: Spatial pillar

Status: `Planned`

Primary objective:
- make spatial a real pillar of the paper rather than a future note

Required outcomes:
- Nicheformer integration plan
- spatial domain or niche classification task spec
- spatial metric pipeline
- first spatial tutorial
- first spatial qualitative figure plan

(integration-pillar)=
## Milestone 3: Integration pillar

Status: `Planned`

Primary objective:
- define representation-transfer benchmarking with task-specific metrics and
  datasets rather than treating integration as generic embedding inspection

Required outcomes:
- integration task spec
- metric pipeline for kBET, iLISI / cLISI, ASW, and clustering metrics where
  appropriate
- dataset registry entries
- one main integration tutorial

(perturbation-pillar)=
## Milestone 4: Perturbation pillar

Status: `Planned`

Primary objective:
- define perturbation-response prediction as a first-class benchmark task

Required outcomes:
- perturbation task spec
- dataset shortlist
- metric pipeline for correlation, error, and DE recovery
- one main perturbation tutorial

## Cross-model expansion

After the task pillars are defined, model breadth should expand toward the
paper target in a controlled way:

- bring `scFoundation`, `CellFM`, and `Nicheformer` into the common wrapper
  and benchmark story
- keep model parity explicit instead of implying equal maturity
- track wrapper, inference, PEFT, tests, tutorials, and benchmarks separately

## Maintenance rules

- public docs must distinguish `Implemented`, `Pilot`, and `Planned`
- no model, PEFT method, or task should be described as supported until code,
  tutorial, tests, and benchmark artifacts exist
- the high-level roadmap stays public and concise
- execution detail belongs in repo-tracked checklist files under `planning/`
