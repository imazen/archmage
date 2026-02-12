# Archmage SIMD Reference

This is the searchable reference companion to the [Archmage & Magetypes tutorial book](../book/). Use the search bar (or press `s`) to find intrinsics, tokens, patterns, or platform-specific details.

## What's in this reference

- **Tokens** — Complete reference for all capability tokens, their features, CPU requirements, and detection behavior.
- **Patterns** — Quick-reference for loading, storing, dispatch, and the safety model.
- **ASM-Verified Patterns** — Common code patterns with `cargo asm` proof that they compile to the expected instructions. No hand-waving.
- **Intrinsics Reference** — Auto-generated listings of every intrinsic mapped to its required token, with safety annotations. Updated on each `just generate`.
- **Cross-Platform** — Semantic differences between x86/ARM/WASM and what gets polyfilled.

## How to use this reference

**Already know what you're looking for?** Use search. Every intrinsic name, token name, and concept is indexed.

**Writing a new SIMD kernel?** Start with [Loading Data](patterns/loading.md) and [Safety Model](patterns/safety.md).

**Choosing a token tier?** See [Token Overview](tokens/overview.md).

**Porting x86 code to ARM?** See [Behavioral Differences](cross-platform/differences.md) and [Naming Conventions](intrinsics/naming.md).

## Conventions

- `#[arcane]` = boundary function (called from non-SIMD code after `summon()`)
- `#[rite]` = internal helper (called from within `#[arcane]` or other `#[rite]` functions)
- `Token::summon()` = runtime CPU feature detection (returns `Option<Token>`)
- `Desktop64` = friendly alias for `X64V3Token` (AVX2 + FMA)
- `Arm64` = friendly alias for `NeonToken`
