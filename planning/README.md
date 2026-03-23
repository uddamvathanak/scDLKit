# Publication planning

This directory is the maintainer-facing execution layer for the publication
program.

Use the planning files like this:

- [current-focus.md](./current-focus.md): the one-screen answer to what we do
  next
- [implementation-map.md](./implementation-map.md): current truth versus target
  state
- [checklists/](./checklists): milestone checklists with explicit exit criteria

The public high-level story stays in [docs/roadmap.md](../docs/roadmap.md).
This directory is where execution detail lives.

## Status vocabulary

Use only these top-line status labels in checklist files:

- `Status: planned`
- `Status: active`
- `Status: blocked`
- `Status: complete`

Use these public maturity labels in roadmap and tutorial maps:

- `Implemented`
- `Pilot`
- `Planned`
- `Target`

## Update rules

After each milestone-affecting merge, update:

1. [current-focus.md](./current-focus.md)
2. the relevant checklist file
3. [implementation-map.md](./implementation-map.md) if implementation status changed
4. [docs/roadmap.md](../docs/roadmap.md) if the public current objective changed
