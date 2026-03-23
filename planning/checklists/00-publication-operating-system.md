# Milestone 0: publication operating system

Status: complete

## Objective

Install the planning and tutorial structure that lets scDLKit operate like a
paper program instead of a loose feature collection.

## Why this matters for the paper

Without a planning layer, the repo keeps mixing:

- current implementation truth
- experimental pilot scope
- paper-target ambition

That makes the publication story harder to trust and harder to execute.

## Current state

This milestone is complete when the roadmap, tutorial map, and checklist system
exist and use consistent language.

## Exit artifacts

- 3-layer roadmap structure
- implementation map
- current-focus file
- milestone checklist files
- refocused tutorial surface
- public docs that separate available scope from paper target

## Checklist

- [x] add a maintainer-facing `planning/` directory
- [x] add a one-screen current-focus file
- [x] add an implementation map that distinguishes pilot from planned work
- [x] add milestone checklist files with a shared template
- [x] refocus the public tutorial index around four research task tracks
- [x] separate supporting and appendix workflows from the main task tracks
- [x] rewrite the public roadmap into paper vision, current truth, and current objective
- [x] add a lightweight planning-structure checker for CI

## Risks / blockers

- the public paper target can still be overclaimed if future docs edits ignore the status vocabulary

## Dependencies

- existing docs build and docs-contract check

## Acceptance criteria

- roadmap, implementation map, current focus, and checklist files all exist
- the tutorial index has four main task cards
- the planning checker runs in CI and docs workflows
