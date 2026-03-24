# Current focus

Current milestone:
- Milestone 1: annotation pillar

Why now:
- annotation is the strongest research-facing capability already present in the
  repo
- the generic PEFT layer and annotation benchmark code now exist, so the next
  work is evidence freeze rather than more speculative API design
- it sets the benchmark, artifact, and PEFT comparison conventions for later
  spatial, integration, and perturbation pillars

Top 3 deliverables:

1. run and audit the first full annotation benchmark artifact bundle from
   `scripts/run_annotation_benchmark.py`
2. keep the human-pancreas notebook as the main static executed annotation
   tutorial with visible last-run metadata
3. freeze the annotation pillar docs and figures so the task can be promoted
   from `Pilot` only when benchmark artifacts are actually in hand

Blockers:

- the full annotation matrix is heavier than the default CI lane and must be
  validated through dedicated workflows and cached datasets
- the public annotation pillar should not be promoted from `Pilot` until the
  first benchmark artifact bundle is reviewed
- cross-model breadth is still future work, so the docs need to stay precise
  about the current scGPT-only implementation

Exit criteria:

- annotation benchmark artifacts exist for full-label, low-label, cross-study,
  and Pareto reporting
- the main annotation tutorial remains the audited static human-pancreas
  notebook
- annotation can be described as `Implemented` only with benchmark artifacts,
  tests, docs, and tutorial evidence in place
- the next milestone after annotation is clearly ready to be spatial

Next milestone after this one:
- Milestone 2: spatial pillar
