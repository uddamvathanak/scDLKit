# Milestone 4: spatial pillar

Status: planned

## Objective

Make spatial domain or niche classification a first-class paper pillar with
task-specific metrics, standardized datasets, formal evaluation regimes, and
efficiency recording.

## Why this matters for the paper

The abstract explicitly names "spatial niche classification" as the fourth
downstream task. Nicheformer is part of the intended paper identity. Without a
real spatial pillar, the paper claim to cover spatial omics is not defensible.

Note: this milestone was previously numbered M2. It is now M4 because the
abstract's recommended development order places spatial after annotation and
representation transfer, and groups it with perturbation as the "extend to
remaining tasks" step. Spatial depends on Nicheformer integration, which is the
hardest model requirement in M5, so spatial is done last among the four tasks.

## Current state

- no spatial task pipeline exists yet
- no Nicheformer wrapper exists yet
- no spatial tutorial exists yet

## Exit artifacts

- spatial task spec (distinguish: niche classification vs. spatial domain
  identification)
- spatial dataset shortlist (e.g. DLPFC, Visium heart)
- spatial metric pipeline (ARI, accuracy, neighborhood coherence where needed)
- regime registry (full-label, low-label, cross-study) reusing formal regime
  objects
- efficiency metrics recorded per run (reusing M1 pattern)
- first Nicheformer integration plan
- first spatial tutorial
- first spatial qualitative figure plan

## Checklist

- [ ] lock the spatial task definition (niche classification metric target)
- [ ] choose the first spatial datasets spanning a range of shift magnitudes
- [ ] define spatial domain or niche metrics (ARI, accuracy)
- [ ] define neighborhood or coherence metrics if required
- [ ] define full-label, low-label, and cross-study regimes for spatial using
  formal regime objects from M1
- [ ] add efficiency metric recording to spatial benchmark runs (reuse M1
  pattern)
- [ ] define Nicheformer wrapper requirements
- [ ] define the first spatial tutorial scope
- [ ] define the first spatial figure inventory

## Risks / blockers

- spatial work is currently outside the implemented foundation path
- Nicheformer wrapper is the most complex new model integration in the full
  milestone sequence
- the paper will overclaim if spatial remains a placeholder without a real task
  spec and metric pipeline

## Dependencies

- publication operating system
- annotation milestone conventions for regimes, efficiency metrics, and figure
  tracking
- Nicheformer integration (Milestone 5 cross-model work may partially overlap)

## Acceptance criteria

- spatial has a real task spec, dataset shortlist, metric plan, regime-aware
  evaluation structure, efficiency recording, Nicheformer integration plan,
  tutorial plan, and figure plan
- the regime and efficiency patterns from M1 are demonstrably reused
