# Archmage SIMD Reference

This is the searchable reference companion to the [Archmage & Magetypes tutorial book](../book/). Use the search bar (or press `s`) to find intrinsics, tokens, patterns, or platform-specific details.

## What's in this reference

- **Tokens** — Complete reference for all capability tokens, their features, CPU requirements, and detection behavior.
- **Patterns** — Quick-reference for loading, storing, dispatch, and the safety model.
- **ASM-Verified Patterns** — Common code patterns with `cargo asm` proof that they compile to the expected instructions.
- **Intrinsics Reference** — Auto-generated listings of every intrinsic mapped to its required token, with safety annotations. Updated on each `just generate`.
- **Cross-Platform** — Semantic differences between x86/ARM/WASM and what gets polyfilled.

## How to use this reference

**Looking up a specific intrinsic?** Use the [Interactive Intrinsics Browser](../../intrinsics/) — search 12,000+ intrinsics with filters for token, safety, and stability.

**Already know what you're looking for?** Use search. Every intrinsic name, token name, and concept is indexed.

**Writing a new SIMD kernel?** Start with [Loading Data](patterns/loading.md) and [Safety Model](patterns/safety.md).

**Choosing a token tier?** See [Token Overview](tokens/overview.md).

**Porting x86 code to ARM?** See [Behavioral Differences](cross-platform/differences.md) and [Naming Conventions](intrinsics/naming.md).

## Conventions

- `#[attune(wrap)]` / `#[attune] fn work_v3_t(...)` = proof wrapper (boundary function called from ordinary code after `summon()`)
- `#[attune(v3)]` / `#[attune] fn work_v3(...)` = direct feature body (internal helper called from a matching feature context)
- `#[attune(_*_t, define(f32x8))]` + `attuned!` / `reattune!` / `#[attune(dispatch)]` = multi-tier family generation and runtime dispatch
- `*_t(token, ...)` = token-first magetypes constructors (`splat_t`, `load_t`, `zero_t`, `from_array_t`, `from_slice_t`)
- `Token::summon()` = runtime CPU feature detection (returns `Option<Token>`)
- `Arm64` = alias for `NeonToken`
- Legacy `#[arcane]`, `#[rite]`, `#[magetypes]`, `#[autoversion]`, and `incant!` remain supported; see [`docs/ATTUNE.md`](../../ATTUNE.md) for the full reference and migration table.
