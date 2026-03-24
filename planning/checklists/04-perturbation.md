# Milestone 3: perturbation pillar

Status: planned

## Objective

Define perturbation-response prediction as the third paper task, inheriting
the regime and efficiency recording patterns from M1 and M2.

## Why this matters for the paper

The abstract names perturbation-response prediction as the third downstream
task. The paper claim that scDLKit covers this task cannot be defended without
a real task pipeline with datasets, metrics, regimes, and efficiency recording.

Note: this milestone was previously numbered M4. It is now M3 because the
abstract dev order groups perturbation with spatial (both come after annotation
and representation transfer), and perturbation is independent of the spatial
model requirements.

## Current state

- no perturbation task pipeline exists yet
- no perturbation tutorial exists yet

## Exit artifacts

- perturbation task spec (distinguish: predict gene expression response to a
  genetic or chemical perturbation)
- perturbation dataset shortlist (e.g. Norman, Replogle)
- perturbation metric pipeline (MSE, DEG overlap / recovery, Pearson, Spearman)
- regime registry (full-label, low-label, cross-study) reusing formal regime
  objects
- efficiency metrics recorded per run (reusing M1 pattern)
- first perturbation tutorial
- first perturbation figure plan

## Checklist

- [ ] lock the perturbation task definition
- [ ] choose initial perturbation datasets
- [ ] define Pearson and Spearman usage
- [ ] define MSE / MAE usage
- [ ] define DE recovery metrics where relevant
- [ ] define pseudo-bulk correlation where relevant
- [ ] define full-label, low-label, and cross-study regimes for perturbation
  using formal regime objects from M1
- [ ] add efficiency metric recording to perturbation benchmark runs (reuse M1
  pattern)
- [ ] define the first perturbation tutorial
- [ ] define the first perturbation figure inventory

## Risks / blockers

- perturbation can drift into vague future work if the task spec is not pinned early
- perturbation prediction may require model capabilities (e.g. masked gene
  modeling or contrastive objectives) that are not yet implemented on any
  foundation model path

## Dependencies

- publication operating system
- annotation and representation transfer milestone conventions for regimes,
  efficiency metrics, and figure tracking

## Acceptance criteria

- perturbation has a concrete task spec, dataset shortlist, metric plan,
  regime-aware evaluation structure, efficiency recording, tutorial plan, and
  figure plan
- the regime and efficiency patterns from M1 are demonstrably reused
