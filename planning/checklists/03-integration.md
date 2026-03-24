# Milestone 2: representation transfer pillar

Status: planned

## Objective

Define representation transfer as a first-class paper pillar with task-specific
metrics, standardized datasets, formal evaluation regimes, and efficiency
recording.

## Why this matters for the paper

The abstract explicitly names "representation transfer" as the second downstream
task alongside annotation, perturbation, and spatial. It cannot remain an
implied embedding visualization step. The abstract also requires that all four
tasks share the same evaluation regime structure (full-label, low-label,
cross-study) and the same efficiency metric recording — so M2 is where the
patterns established in M1 are first applied to a second task.

Note: this milestone was previously labelled "integration pillar." The abstract
calls this task "representation transfer" to distinguish it from data integration
toolkits; the benchmark here measures how well an adapted foundation model
transfers its representation to a held-out dataset, not whether batch effects
are removed.

## Current state

- baseline representation workflows exist (`TaskRunner` with `task="representation"`)
- representation transfer benchmarking does not yet exist as a formal task
  pipeline with the regime and efficiency structure established in M1
- no foundation-model path for representation transfer exists yet

## Exit artifacts

- representation transfer task spec
- metric pipeline (kNN accuracy, silhouette, transfer accuracy on held-out data)
- dataset shortlist spanning a range of distribution shift magnitudes
- regime registry (full-label, low-label, cross-study) reusing M1 formal regime
  objects
- efficiency metrics recorded per run (reusing M1 pattern)
- first representation transfer tutorial
- first representation transfer figure plan

## Checklist

- [ ] lock the representation transfer task definition (distinguish from generic
  embedding: task is transfer accuracy on held-out data)
- [ ] define kNN accuracy usage
- [ ] define silhouette score usage
- [ ] define transfer accuracy on held-out splits
- [ ] choose initial representation transfer datasets spanning a range of shift
  magnitudes
- [ ] define full-label, low-label, and cross-study regimes for representation
  transfer using the formal regime objects from M1
- [ ] add efficiency metric recording to representation transfer benchmark runs
  (reuse M1 pattern)
- [ ] define the first representation transfer tutorial
- [ ] define the first representation transfer figure inventory

## Risks / blockers

- representation transfer can become vague if the metric pipeline is not
  explicit and distinguished from plain embedding visualization
- the self-supervised path (unlabeled data) is most relevant here but is M5 work

## Dependencies

- publication operating system
- annotation milestone conventions for regimes, efficiency metrics, and
  figure tracking

## Acceptance criteria

- representation transfer has a concrete metric plan, dataset shortlist,
  regime-aware evaluation structure, efficiency recording, tutorial plan, and
  figure plan
- the regime and efficiency patterns from M1 are demonstrably reused, not
  reinvented
