# magetypes: opportunities for above-NEON-baseline aarch64 features

**Survey date:** 2026-04-17. Audit of `magetypes::simd::backends::*` against the
new tokens (`Arm64RdmToken`, `Arm64Sve2Token`) and the `Arm64V2`/`Arm64V3` tiers
already in the registry.

## Current op surface (per backend trait)

Every numeric backend (`I16x8Backend`, `I32x4Backend`, `F32x4Backend`, etc.)
exposes the same shape:

```
splat zero load store from_array to_array
add sub mul neg min max abs
simd_eq simd_ne simd_lt simd_le simd_gt simd_ge blend
reduce_add not bitand bitor bitxor
shl_const shr_logical_const shr_arithmetic_const
all_true any_true bitmask clamp
```

Floats add `mul_add`, `mul_sub`, `recip` family, `sqrt`, `floor`/`ceil`/`round`,
`trunc`. Block-ops files add memory-layout helpers (transpose, interleave,
deinterleave, byte/array views) — no compute.

**Total compute surface:** ~32 ops per backend, all expressible in pure NEON
(or SSE2 baseline). Nothing in the current trait surface needs above-NEON
features. Adding new tokens doesn't speed up *existing* ops — there's nothing
above the baseline to dispatch to.

## Where above-baseline features would unlock real wins

Below: ops that would be worth adding to magetypes' backend traits, the feature
they require, the tier that ships them, and the kernels that benefit.

### `mul_high` / `mul_high_round` (saturating, signed) — needs `rdm`

ARMv8.0 NEON has `sqdmulh` (signed saturating doubling multiply, returning the
high 16 bits). ARMv8.1 RDM adds `sqrdmulh` — same op but with rounding instead
of truncation. The rounding variant is materially better for Q15/Q7 fixed-point
multiplication: cuts bias from ~0.5 LSB to ~0 LSB.

**Token:** new `Arm64RdmToken`.

**Kernels in zen-family that benefit:**
- **zenresize** (i16 fixed-point resampler kernels) — rounding mul-high is
  the workhorse for Q15 filter coefficients × i16 pixel data.
- **zenquant** (palette quantization, error diffusion) — Q15 weights × pixel
  channels.
- **zenfilters** (Oklab tone curves on i16) — fixed-point gamma/curve LUTs.

**Current scalar fallback cost:** without `mul_high_round`, code currently
uses `mul` followed by `shr` + manual rounding bias add — ~3 ops vs 1.

**Suggested trait method:**

```rust
fn mul_high_round_sat(a: Self::Repr, b: Self::Repr) -> Self::Repr;
```

On `i16x8` only (matches NEON `vqrdmulhq_s16`); also `i16x4` if magetypes
exposes that width.

### `dot_i8` (signed/unsigned int8 dot product → i32) — needs `dotprod`

NEON's `sdot` / `udot` (FEAT_DotProd, ARMv8.2 optional / ARMv8.4 mandatory)
takes two `i8x16` inputs and produces an `i32x4` of pairwise group dot products
(4 lanes per output). Roughly 4× the throughput of unrolled `mla` chains.

**Token:** `Arm64V2Token` (already includes `dotprod`).

**Kernels that benefit:**
- **zenquant** Viterbi distance metric (i8 palette → i8 source diff,
  squared)
- **zensim** SAD/SSD-style local windows over i8 channels
- **zenresize** i8 channel resampling (rare but exists for some YUV pipelines)

**Suggested trait methods on `i8x16` / `u8x16`:**

```rust
fn dot_pairwise_i8_to_i32(a: i8x16, b: i8x16) -> i32x4;
fn dot_pairwise_u8_to_u32(a: u8x16, b: u8x16) -> u32x4;
```

### `mmla_i8` / `mmla_bf16` (matrix multiply accumulate) — needs `i8mm` / `bf16`

NEON's `smmla` / `ummla` (FEAT_I8MM) computes a 2×2 i32 result from two 2×8 i8
input matrices in a single instruction. `bfmmla` (FEAT_BF16) does the same for
bf16. These are the dense GEMM primitives ARM added for ML inference.

**Token:** `Arm64V3Token` (already includes both).

**Kernels that benefit:**
- **zentract** ONNX inference (the obvious one — every dense layer)
- **zensally** saliency map convolutions (small-kernel 2D convs)
- **zensim** patch comparison (SSIM-like local windows)
- **fast-ssim2** structural similarity (Gaussian-weighted patch dot products)

These are the highest-impact additions. A 4×4 bf16 matmul kernel using `bfmmla`
runs ~4× faster than the same in fp32. ML-ish workloads care a lot.

**Suggested trait methods (separate types):**

```rust
fn mmla_i8_to_i32(acc: i32x4, a: i8x16, b: i8x16) -> i32x4;
fn mmla_bf16_to_f32(acc: f32x4, a: bf16x8, b: bf16x8) -> f32x4;
```

The bf16 form needs a `bf16x8` type that magetypes doesn't have today —
requires either a `u16` view or a real `bf16` element type (Rust `f16`/`bf16`
are stable since 1.84).

### `sad_u8_to_u16` / `abs_diff` — needs nothing above NEON baseline

NEON's `vabal` / `vsabd` are baseline. Worth adding to magetypes for
**zensim** (block matching) and **zenquant** (color-distance metrics) but
this isn't a token-tier question — they should land on `NeonToken`.

### `cnt` / `popcnt_u8` — needs nothing above NEON baseline

NEON `cnt` is baseline. Useful for bit-image masks. `NeonToken` again.

### Saturating arithmetic bundle (`add_sat`, `sub_sat`, `shl_sat`)

NEON baseline (`vqaddq_*` etc.) — no token tier required. But magetypes
doesn't expose them. Useful in **zenblend**, **zenfilters** (quantize-back
clamps), and any image kernel that converts wider-precision to u8/u16 output.

### Reciprocal estimates (`recip_estimate`, `rsqrt_estimate`)

NEON `vrecpeq_f32` / `vrsqrteq_f32` — baseline. Not in magetypes today.
Useful for **zenresize** Lanczos kernel normalization.

## What above-baseline tokens would *not* help

- **All compare / blend / bitwise ops** — already optimal at NEON baseline
- **Reduce sums / horizontal ops** — baseline NEON has `vaddvq_*`
- **Min/max** — baseline has `vminq_*` / `vmaxq_*`
- **FP add/sub/mul/fma** — baseline has fmla/fmls
- **All memory-layout helpers** (transpose, interleave) — baseline NEON
- **Float rounding (`floor`/`ceil`/`round`/`trunc`)** — baseline NEON

These are the bulk of the current trait surface. They don't benefit from
`Arm64V2`/`V3`/`Sve2` — the tiers buy you new ops, not faster versions of
existing ops.

## Recommended adds, in priority order

| Op | Type(s) | Token tier | Effort | Impact |
|---|---|---|---|---|
| 1. `mul_high_round_sat` | `i16x8` (+`i16x16` w512) | `Arm64RdmToken` | small | high — used by zenresize, zenquant, zenfilters |
| 2. `dot_pairwise_i8_to_i32` | `i8x16` (+u8 variant) | `Arm64V2Token` | medium | high — quantization, similarity, AI |
| 3. `mmla_i8_to_i32` | `i8x16` matmul block | `Arm64V3Token` | larger | very high — zentract, ML inference |
| 4. `mmla_bf16_to_f32` | needs new `bf16x8` type | `Arm64V3Token` | larger (new type) | very high — same as 3 |
| 5. `add_sat`/`sub_sat` family | all int widths | `NeonToken` (baseline) | small | medium — zenblend |
| 6. `recip_estimate`/`rsqrt_estimate` | float widths | `NeonToken` | small | medium — Lanczos |
| 7. `popcnt` | u8/u16 widths | `NeonToken` | small | low — niche |

## SVE2 considerations

`Arm64Sve2Token` is research-preview today — `repr(scalable)` types aren't
accepted, stdarch SVE intrinsics are nightly-only. Until that lands, magetypes
can't expose SVE-shaped types in its public API (the trait surface assumes
fixed-width vectors with `Self::Repr`).

When the type system catches up, the largest SVE2 win for image work is
**predicated execution** of edge/boundary ops — handling non-multiple-of-8
strides without the explicit "remainder loop" that NEON code currently needs.
That's an ergonomic win as much as a perf one. Best target is a future
magetypes "scalable" backend that lives alongside `f32x4`/`f32x8` rather
than replacing them.

## Concrete next step

Land **`mul_high_round_sat` on `i16x8`** as the first above-baseline op. It's:
- the smallest scope (one method, one type, one platform path)
- the lowest-risk (well-defined NEON intrinsic, no type-system gymnastics)
- exercises the new `Arm64RdmToken` end-to-end so the dispatch matrix gets
  validated before bigger additions land

If that pattern works cleanly, follow with `dot_pairwise_i8_to_i32` on `i8x16`
(Arm64V2 tier) — same shape, broader hardware coverage, used by more kernels.
