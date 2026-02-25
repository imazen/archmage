+++
title = "Archmage Documentation"
description = "Safe SIMD via capability tokens for Rust"
template = "landing.html"

[extra]
landing = true
+++

# Archmage

> Safely invoke your intrinsic power, using the tokens granted to you by the CPU.

Two crates, two stability levels, one documentation site.

## [Archmage](/archmage) — Stable

Safe SIMD through capability tokens. Prove CPU features exist at the type level, write SIMD code without `unsafe`. Zero overhead — generates identical assembly to hand-written `#[target_feature]` + `unsafe`.

- Tokens, `#[arcane]`, `#[rite]`, `incant!`, `#[magetypes]`
- x86-64, AArch64, WASM
- `#![forbid(unsafe_code)]` compatible

## [Magetypes](/magetypes) — Experimental

SIMD vector types with natural Rust operators. `f32x8`, `i32x4`, and friends — wrapping platform intrinsics with `+`, `-`, `*`, `/`, FMA, comparisons, reductions, transcendentals.

- Token-gated construction (no CPU mismatch possible)
- Cross-platform polyfills (f32x8 works on ARM via two f32x4)
- Full API parity across x86/ARM/WASM

## Tools

- **[Intrinsics Browser](/intrinsics/)** — Search 12,000+ SIMD intrinsics by token, architecture, safety, and stability
- **[docs.rs (archmage)](https://docs.rs/archmage)** | **[docs.rs (magetypes)](https://docs.rs/magetypes)**
- **[GitHub](https://github.com/imazen/archmage)** | **[crates.io](https://crates.io/crates/archmage)**
