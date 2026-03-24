# Current focus

## Paper framing

The abstract commits scDLKit to four concrete pillars:

- Pillar 1 — Unified adaptation interface: single training interface across
  multiple foundation models and multiple PEFT strategies, including a
  self-supervised adaptation path for unlabeled data
- Pillar 2 — Distribution shift benchmarking: three formal evaluation regimes
  (full-label, low-label, cross-study) with measurable shift between train and
  eval; this is the most novel claim in the abstract
- Pillar 3 — Reproducible benchmark suite: four downstream tasks (annotation,
  representation transfer, perturbation, spatial) with standardized datasets,
  preprocessing, splits, and metrics
- Pillar 4 — Efficiency-aware evaluation: every benchmark run records trainable
  parameter count, total parameter count, wall-clock training time, peak memory,
  and checkpoint size alongside performance metrics

## Milestone sequence

The recommended development order from the abstract specification:

1. Formalize evaluation regimes as benchmark objects (Pillar 2) — already
   partially done in M1 but not yet a formal cross-task standard
2. Complete annotation benchmark artifact bundle and start representation
   transfer benchmark (Pillar 3, partial)
3. Add efficiency recording to all benchmark runs (Pillar 4) — low effort, high
   payoff
4. Ship spatial and perturbation task modules (Pillar 3, complete)
5. Add multi-model support and self-supervised adaptation path (Pillar 1,
   complete)

Milestone numbers:
- M0: Publication OS — complete
- M1: Annotation pillar — blocked (benchmark runtime)
- M2: Representation transfer pillar — planned (moved before spatial per dev
  order)
- M3: Perturbation pillar — planned (grouped with spatial in abstract dev order)
- M4: Spatial pillar — planned (deferred from previous M2 position)
- M5: Cross-model PEFT expansion + self-supervised adaptation path — planned
- M6: Paper assets and manuscript — planned

## Current milestone

M1: Annotation pillar

Status:
- blocked by benchmark runtime

Why now:
- annotation is the strongest implemented research-facing capability
- M1 must also land efficiency recording (Pillar 4) and formal regime objects
  (Pillar 2) as part of its exit criteria, so subsequent milestones inherit
  both patterns
- the generic PEFT layer and annotation benchmark code already exist; remaining
  work is evidence freeze, efficiency metrics, and benchmark feasibility

Top 3 deliverables:

1. make the full annotation benchmark execution path feasible enough to produce
   the first complete artifact bundle from `scripts/run_annotation_benchmark.py`
2. rerun and review the first full annotation benchmark artifact bundle
3. verify the Pareto figure is generated as part of the artifact bundle

Blockers:

- the current full benchmark matrix did not complete within the local 4-hour
  GPU runtime budget; the main runtime fix (pre-compute scGPT tokenization once
  per seed instead of once per strategy) is implemented but not yet validated
- the existing GitHub workflow timeout is `240` minutes on CPU, which is
  unlikely to be sufficient without a more feasible execution path
- the public annotation pillar should not be promoted from `Pilot` until the
  first benchmark artifact bundle is reviewed

Exit criteria:

- annotation benchmark artifacts exist for full-label, low-label, cross-study,
  and Pareto reporting
- each benchmark run artifact includes efficiency metrics (trainable params,
  total params, runtime, peak memory, checkpoint size)
- the main annotation tutorial remains the audited static human-pancreas
  notebook
- annotation can be described as `Implemented` only with benchmark artifacts,
  tests, docs, and tutorial evidence in place

Next milestone after this one:
- M2: Representation transfer pillar
