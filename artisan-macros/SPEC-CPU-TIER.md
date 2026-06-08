# SPEC: `#[cpu_tier]`

Normative specification for the `#[cpu_tier]` attribute macro in artisan-macros.

Status: **draft** — implementation exists in `src/lib.rs`, awaiting review.

## Synopsis

```rust
#[cpu_tier(enable = "<features>" [, arch = "<target_arch>"])]
fn <name>(<params>) <return> { <body> }
```

Attaches `#[target_feature(enable = "<features>")]`, `#[cfg(target_arch = "<target_arch>")]`, and `#[inline]` to the annotated function. Does not otherwise modify the function.

## Input grammar

Attribute arguments are a comma-separated list of `name = "string"` pairs. Required and optional args:

| Arg | Required | Type | Description |
|---|---|---|---|
| `enable` | yes | string literal | Comma-separated feature names, same grammar as `#[target_feature(enable = "...")]` |
| `arch` | no | string literal | Explicit `target_arch` override. One of `"x86_64"`, `"x86"`, `"aarch64"`, `"arm"`, `"wasm32"` |

Both args may appear in either order. Neither may appear more than once.

The attribute must be applied to an `ItemFn` (a top-level `fn` or one in an inherent impl). The function must be a safe `fn`, not `unsafe fn`.

## Output grammar

Given

```rust
#[cpu_tier(enable = "<F>", arch = "<A>")]
<vis> fn <name><generics>(<params>) <return> <where>? { <body> }
```

the macro emits

```rust
#[cfg(target_arch = "<A>")]
#[target_feature(enable = "<F>")]
#[inline]
<vis> fn <name><generics>(<params>) <return> <where>? { <body> }
```

Existing attributes on the input function are preserved; the three macro-added attributes are prepended.

## Arch inference

When `arch` is omitted, the macro scans `enable` left-to-right for a feature listed in the unambiguous-features table (`src/lib.rs` § `UNAMBIGUOUS_FEATURES`) and selects that feature's `target_arch`. The first match wins.

The table covers:

- **x86_64**: every documented `stdarch` feature that is exclusive to x86 (SSE1–4.2, POPCNT, AVX, AVX2, FMA, BMI1/2, LZCNT, F16C, MOVBE, all AVX-512 variants, VPCLMULQDQ, VAES, GFNI, PCLMULQDQ, CMPXCHG16B).
- **aarch64**: NEON, RDM, DotProd, FHM, FCMA, I8MM, BF16, SVE, SVE2, PMULL.
- **wasm32**: SIMD128, relaxed-simd.

**Ambiguous features**: `aes`, `sha2`, `sha3`, `crc`, `fp16`. These are present on more than one arch under the same stdarch feature string. They do **not** drive inference. If every feature in `enable` is ambiguous, inference fails with a compile error directing the user to add explicit `arch = "..."`.

Arch inference never silently defaults to a wrong arch. Either a known-arch feature is present (inference succeeds) or the user explicitly says which arch (explicit wins) or compilation fails (safe default).

## Validation rules

Errors emitted at macro expansion time (all as `compile_error!`):

| Condition | Error |
|---|---|
| No `enable` arg | `missing required \`enable = "..."\` argument` |
| `enable` specified twice | `\`enable\` specified twice` |
| `arch` specified twice | `\`arch\` specified twice` |
| Unknown arg name | `unknown argument \`<name>\`; expected \`enable\` or \`arch\`` |
| Non-literal arg value | `expected string literal (e.g. "avx2,fma")` |
| Function is `unsafe fn` | `#[cpu_tier] expects a safe \`fn\`, not \`unsafe fn\`. Rust 2024 edition allows \`#[target_feature]\` on safe fns; \`unsafe fn\` would break \`#![forbid(unsafe_code)]\` for downstream crates.` |
| Inference fails and no explicit `arch` | `cannot infer target_arch from features \`<features>\`: all listed features are ambiguous or unknown to artisan-macros. Add explicit \`arch = "x86_64" | "aarch64" | "wasm32"\`.` |
| Explicit `arch` outside supported set | `arch \`<arch>\` is not one of: x86_64, x86, aarch64, arm, wasm32. (artisan-macros supports these five target_arch values.)` |

The macro does **not** validate feature names against a known table. Misspelled features (e.g., `avx22`) reach `#[target_feature(enable = "...")]`, where rustc raises its own error.

## Invariants

1. **Generated function is always `fn`, never `unsafe fn`.** Rust 2024 edition permits safe fns with `#[target_feature]`; emitting `unsafe fn` would break `#![forbid(unsafe_code)]` in downstream crates.
2. **`#![forbid(unsafe_code)]` downstream.** Generated code contains no `unsafe` keyword. Downstream users with the forbid lint compile the output without modification.
3. **Feature string is verbatim.** `enable = "avx2,fma"` becomes `#[target_feature(enable = "avx2,fma")]`. No whitespace trimming, sorting, or deduplication.
4. **Arch gating is absolute.** The `#[cfg(target_arch = "...")]` is emitted on the function. Compilation for any other arch excludes the function entirely; references must be `#[cfg]`-gated or routed through `#[chain]` (which cfg-gates its trampolines).
5. **`#[inline]` is always added.** Non-inlining of tier functions breaks `#[chain]`'s compile-time elision; every tier function is a candidate for inlining into callers with matching target features.

## Examples

### Canonical x86_64 v3 tier

```rust
#[cpu_tier(enable = "avx2,fma,bmi1,bmi2,f16c,lzcnt,popcnt,movbe")]
fn dot_v3(a: &[f32; 8], b: &[f32; 8]) -> f32 {
    // hand-written AVX2+FMA intrinsics
}
```

Arch inferred as `x86_64` via `avx2`.

### Ambiguous features → explicit arch required

```rust
// This would fail inference: aes, sha2 are both on x86 and ARM:
// #[cpu_tier(enable = "aes,sha2")]
// fn crypto_step(...) { ... }

// Correct: say which arch.
#[cpu_tier(enable = "aes,sha2", arch = "aarch64")]
fn crypto_step(...) { ... }
```

### NEON tier

```rust
#[cpu_tier(enable = "neon")]
fn dot_neon(a: &[f32; 4], b: &[f32; 4]) -> f32 {
    // hand-written NEON
}
```

## Non-goals

- **Tier name inference.** `#[cpu_tier]` knows nothing about `_v3` / `_v4` / `_neon` suffixes. Those are a convention for the `#[chain]` reader, not this macro.
- **Feature-set validation against stdarch.** The macro does not maintain a list of known valid feature strings; rustc does that in its `#[target_feature]` implementation.
- **Token types.** No `X64V3Token`, `NeonToken`, etc. Users who want archmage-style tokens pass them as function parameters themselves.

## Open questions (for review)

- Should inference accept `fp16` as ambiguous-strict (current), or aarch64-biased (since x86 uses `f16c`, not `fp16`)? Current: ambiguous-strict. Trades a user-friendly inference against consistency.
- Should the error message for inference failure list the ambiguous features? Current: no, generic message. Could help debugging but adds implementation complexity.
- Support for `enable_if = "..."` or other future `target_feature` args? Not planned; add only on concrete need.
