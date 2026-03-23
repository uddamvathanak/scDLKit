# Current focus

Current milestone:
- Milestone 1: annotation pillar

Why now:
- annotation is the strongest research-facing capability already present in the
  repo
- it is the fastest path from current code to paper-grade evidence
- it sets the benchmark, artifact, and PEFT comparison conventions for the
  later spatial, integration, and perturbation pillars

Top 3 deliverables:

1. lock an explicit annotation task spec with full-label, low-label, and
   cross-study regimes
2. define the annotation benchmark matrix across frozen, full-FT, LoRA,
   adapters, prefix tuning, and IA3-style scaling
3. promote one annotation tutorial to the main research-facing surface and turn
   its outputs into figure-ready artifacts

Blockers:

- full fine-tuning baseline is not yet implemented in the current foundation path
- non-LoRA PEFT methods are still paper targets rather than repo features
- low-label and cross-study annotation regimes are not yet encoded into the
  benchmark workflow

Exit criteria:

- annotation checklist has a locked task spec and dataset shortlist
- annotation figure inventory is defined
- current annotation evidence is tracked as `Pilot` rather than hand-waved
- the next milestone after annotation is clearly ready to be spatial

Next milestone after this one:
- Milestone 2: spatial pillar
