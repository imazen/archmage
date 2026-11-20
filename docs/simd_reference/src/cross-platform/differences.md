# Behavioral Differences

Cross-architecture semantic differences that affect correctness. Magetypes handles these internally, but if you're writing raw intrinsics, you need to know them.

## FMA Argument Order

The same conceptual operation (`self * a + b`) maps to different argument orders:

```rust
// x86: fmadd(a, b, c) = a*b + c
_mm256_fmadd_ps(self.0, a.0, b.0)

// NEON: fma(a, b, c) = a + b*c  ← DIFFERENT
vfmaq_f32(b.0, self.0, a.0)  // Rearranged to get self*a + b

// WASM: relaxed_madd(a, b, c) = a*b + c  (same as x86)
f32x4_relaxed_madd(self.0, a.0, b.0)
```

**Magetypes `mul_add_relaxed` and `mul_add_fused`:** Both normalize argument order so `v.mul_add_relaxed(a, b)` and `v.mul_add_fused(a, b)` mean `v * a + b` (`mul_add` is deprecated). `mul_add_relaxed` rounds once where the hardware fuses (x86 v3/v4, NEON) and twice on the scalar backend and strict WASM; `mul_add_fused` (alias `mul_add_portable`) rounds once on every backend.

## Blend/Select Argument Order

```rust
// x86: blendv(false_val, true_val, mask) — mask LAST
_mm256_blendv_ps(false_val, true_val, mask)

// NEON: bsl(mask, true_val, false_val) — mask FIRST
vbslq_f32(mask, true_val, false_val)

// WASM: bitselect(true_val, false_val, mask) — mask LAST
v128_bitselect(true_val, false_val, mask)
```

**Magetypes `blend`:** Normalizes to `blend(mask, true_val, false_val)` across platforms.

## Comparison Return Types

```rust
// x86: Returns same float type (all-1s or 0s bit pattern)
let mask: __m256 = _mm256_cmpgt_ps(a, b);

// NEON: Returns unsigned int type — need reinterpret for float blend
let mask: uint32x4_t = vcgtq_f32(a, b);
let float_mask = vreinterpretq_f32_u32(mask);

// WASM: Returns v128 (same generic type as everything)
let mask: v128 = f32x4_gt(a, b);
```

**Magetypes:** Comparison methods return the same vector type, handling reinterprets internally.

## Bitwise on Floats

```rust
// x86: Direct operations exist
_mm256_and_ps(a, b)
_mm256_xor_ps(a, b)

// NEON: Must reinterpret through integers
let a_u = vreinterpretq_u32_f32(a);
let b_u = vreinterpretq_u32_f32(b);
vreinterpretq_f32_u32(vandq_u32(a_u, b_u))

// WASM: v128_and works on any v128
v128_and(a, b)
```

**Magetypes:** Both float and integer generic vector types implement `&`, `|`, `^` (and `&=` / `|=` / `^=`) across all backends, handling integer reinterprets internally on NEON.

## Shift Semantics (signed integers)

```rust
// x86 i32x4 logical right shift (zero-fill) — does not preserve sign
_mm_srli_epi32(v, n)  // Shifts in zeros; use _mm_srai_epi32 for arithmetic shift

// ARM i32x4 shr: arithmetic shift (sign-extending)
vshrq_n_s32(v, n)     // Preserves sign bit

// WASM i32x4 shr: arithmetic shift (sign-extending)
i32x4_shr(v, n)       // Preserves sign bit
```

**Magetypes:** Use `shr_arithmetic_const::<N>()` / `shr_arithmetic_uniform(count)` for sign-extending right shifts and `shr_logical_const::<N>()` / `shr_logical_uniform(count)` for zero-filling right shifts across all platforms.

## Reciprocal/Rsqrt Precision

| Operation | x86 Precision | NEON Precision |
|-----------|--------------|----------------|
| `rcp` (reciprocal estimate) | ~12 bits | ~8 bits |
| `rsqrt` (inverse sqrt estimate) | ~12 bits | ~8 bits |

NEON estimates are less precise. For equivalent accuracy, NEON needs two Newton-Raphson refinement steps vs one for x86:

```rust
// NEON: 2 Newton-Raphson steps for ~23-bit precision
let est = vrecpeq_f32(v);
let est = vmulq_f32(est, vrecpsq_f32(v, est));  // Step 1
let est = vmulq_f32(est, vrecpsq_f32(v, est));  // Step 2
```

**Magetypes:** `rcp_approx()` uses the native estimate (different precision per platform). `recip()` is within 4 ULP with exact results at ±0, ±inf and NaN; `recip_portable()` is exact and gives the same bits on every backend.

## 64-bit Integer Min/Max

| Platform | Native min/max for i64? |
|----------|------------------------|
| x86 (V3) | No (use comparison + blend) |
| x86 (V4) | Yes (`_mm_min_epi64`) |
| ARM | No (use comparison + blend) |
| WASM | No (use comparison + blend) |

**Magetypes:** `min()` and `max()` work on all platforms via polyfill. `min_fast()` and `max_fast()` take `X64V4Token` for the native instruction when available.

## Horizontal Operations

NEON and WASM require multi-step reductions. x86 has `haddps` but it's slow (multiple micro-ops):

```rust
// x86 reduce_add (fast path via shuffle + add, NOT hadd):
// movshdup + addps + movhlps + addss

// NEON reduce_add:
// vpaddq + vpaddq + lane extract

// WASM reduce_add:
// shuffle + add + shuffle + add + extract
```

Reduction associativity may differ between backends (tree vs left-fold), producing small relative errors (~1e-6) for inputs with large magnitude differences.

## Negation and Signed Zero

The float `-v` operator flips exactly the sign bit on every backend, matching
Rust's unary negation. `-(+0.0)` is `-0.0`, and `-(-0.0)` is `+0.0`.
NaN payloads and signalling bits are preserved; only their sign changes.

x86 uses sign-bit XOR, NEON and WASM use their native sign-flip operations,
and the scalar backend uses Rust negation. This changes the earlier x86
`0.0 - x` behavior for positive zero and NaNs. No explicit sign-mask workaround
is needed.

## Min/Max NaN Propagation

Raw hardware instructions differ in how they treat `NaN` and `±0.0`:

```rust
// x86: _mm_min_ps returns the SECOND operand when either operand is NaN
_mm_min_ps(NaN, x)    // → x   (second operand)
_mm_min_ps(x, NaN)    // → NaN (second operand)

// NEON: two instruction families
vminnmq_f32(NaN, x)   // → x   (returns numeric lane for quiet NaN)
vminq_f32(x, NaN)     // → NaN (IEEE 754-2019 minimum, propagates NaN)

// WASM: two instruction families
f32x4_min(x, NaN)     // → NaN (propagates NaN, orders -0.0 < +0.0)
f32x4_pmin(b, a)      // → b < a ? b : a
```

**Magetypes float vectors** provide three explicit pairs so callers pick the contract they need:

- `min()` / `max()` match `f32::min` / `f32::max` (`f64::min` / `f64::max`): if one lane is quiet `NaN` and the other is numeric (including `±inf`), they return the numeric lane across backends. `(-0.0, +0.0)` tie-breaking is unspecified, and signalling-`NaN` handling differs by backend (NEON `fminnm` and AVX-512 `f32` `vrangeps` quiet the signalling `NaN`; x86 V3, AVX-512 `f64`, WASM, and `ScalarToken` return the numeric lane).
- `minimum()` / `maximum()` implement IEEE 754-2019 semantics on every backend: they propagate `NaN` and order `-0.0 < +0.0`.
- `min_fast()` / `max_fast()` emit the single native instruction ignoring `NaN` and `±0.0` (`vminps`/`vminpd` on x86, `fminnm`/`fmaxnm` on NEON, `pmin(other, self)`/`pmax(other, self)` on WASM, and `a < b ? a : b` on `ScalarToken`).

## FMA vs Separate Multiply-Add

`mul_add_relaxed` uses hardware FMA (one rounding) on x86 V3/V4 and AArch64 NEON, and two roundings (`a * b` then `+ c`) on `ScalarToken` and strict WASM (with `relaxed-simd`, WASM engines may round once or twice). Near-zero cancellation (when `a * b ≈ -c`) can differ by many ULPs between one-rounding and two-rounding backends. Use `mul_add_fused` when every backend must round once like `f32::mul_add`.

## Rounding Consistency (Fixed in 0.9.16)

Prior to 0.9.16, the scalar backend's `round()` used ties-away-from-zero (`f32::round()` semantics), while all hardware backends used ties-to-even (IEEE 754 default). This caused dispatch parity failures. Fixed by implementing `roundevenf` using the 2^23 magic-number trick. All backends now produce identical rounding results.

**Magetypes:** `reduce_add()`, `reduce_max()`, `reduce_min()` use the native sequence for each platform.

## Summary of Intentional Cross-Backend Differences

API surface parity across x86, ARM, and WASM generic types is enforced by `cargo xtask parity` (0 surface differences). The remaining runtime differences are explicit in method contracts:

| Method | Contract across backends | Exact / portable alternative |
|--------|--------------------------|------------------------------|
| `mul_add_relaxed` / `mul_sub_relaxed` | Fused on x86 V3/V4 and NEON; two roundings on scalar and strict WASM; engine-defined on relaxed WASM | `mul_add_fused` / `mul_sub_fused` |
| `rcp_approx` / `rsqrt_approx` | Platform-fastest estimate (~12–17 bits; unspecified at `±0`, `±inf`, `NaN`) | `recip` / `rsqrt` (≤4 ULP) or `recip_portable` / `rsqrt_portable` (0 ULP) |
| `min_fast` / `max_fast` | Single native instruction; `NaN` and `±0.0` behavior differs per ISA | `min` / `max` (numeric over quiet `NaN`) or `minimum` / `maximum` (propagates `NaN`) |
| `min` / `max` on signalling `NaN` | Quiet `NaN` on NEON and AVX-512 `f32`; numeric lane on x86 V3, AVX-512 `f64`, WASM, scalar | `minimum` / `maximum` if `NaN` propagation is required |
| `reduce_add` (floats) | Tree vs sequential order can differ by ~1e-6 relative error on mixed-magnitude inputs | Explicit pairwise or Kahan accumulation |
