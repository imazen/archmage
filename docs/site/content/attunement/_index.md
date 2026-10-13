+++
title = "Path of Attunement"
description = "A short route from an ordinary function to a portable SIMD kernel"
sort_by = "weight"
weight = 0

[extra]
sidebar = true
+++

Write a function, let Archmage select its CPU context, and keep the work inside
that context. This path teaches the **0.9.31-beta.1** unified macros. You can start
here without learning `#[arcane]`, `#[rite]`, or `incant!` first.

1. [Attune a kernel](@/attunement/first-kernel.md): an ordinary public function
   that selects a CPU tier for its whole loop.
2. [Stay in context](@/attunement/helpers.md): helpers that follow the caller's
   selected tier without another CPU probe.
3. [Add vectors](@/attunement/vectors.md): Magetypes operations, complete chunks,
   and the remaining elements.

Keep the [complete reference](@/attunement/reference.md) nearby for explicit tiers,
proofs, generics, methods, inlining, visibility, and legacy mappings.
Use [See the Expansion](@/attunement/expansions.md) to compare source and generated
code side by side, with target selection and mobile tabs.

You need two spellings for this route:

| Spelling | Purpose |
| --- | --- |
| `#[attune(...)]` | Choose which implementations and callable functions to generate |
| `attuned!(helper(...))` | Call the helper appropriate to the enclosing context |

Start with `#[attune(dispatch)]` for a kernel called from ordinary code and
`#[attune(_*)]` for its internal helpers. Add other forms when the application
needs them. `reattune!` is an explicit request to reconsider the CPU tier inside
an existing context; it is not needed for the basic path.

## Choose your guide

The [Arcane Spellcraft guide](@/archmage/_index.md) covers the existing attributes
and production examples. Those APIs remain supported. The
[full attunement reference and migration table](@/attunement/reference.md)
covers explicit proofs, visibility, inline policies, named tiers, and method and
generic restrictions.

Both paths use the same capability proofs and target-feature safety model.
Attunement puts the routine selection and forwarding in generated code.
