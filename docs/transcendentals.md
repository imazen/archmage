# Transcendental Functions for SIMD Types

This document describes the algorithms, accuracy, and benchmark results for transcendental function implementations in archmage's SIMD types.

## Accuracy Summary

Measured over every f32 input in each band against an f64 `std` reference, on
the scalar backend (`mul_add` rounds twice) and AVX2 (`mul_add` is one FMA), on
2026-10-05. Methods, per-band tables and the probe source:
[`benchmarks/transcendental_precision_2026-10-05.md`](../benchmarks/transcendental_precision_2026-10-05.md).
Figures are maxima over both backends, rounded up.

The two backends return different bits for 0.3–0.8% of exp/log results (by 1–3
ULP) and for about 11% of `pow_midp` results (by up to 178 ULP for n = 2.4 on
[2^-50, 2^50]). Their maxima against the true value differ by at most 0.43 ULP.
`cbrt` uses no `mul_add` and returned the same bits on both. NEON fuses like
AVX2 and strict WASM rounds twice like the scalar backend; neither was measured.

### midp tier

| Function | Domain | Max ULP | Mean ULP | Max rel. error |
|----------|--------|---------|----------|----------------|
| log2_midp | positive normals | 4.5 | 0.25 | 3.3e-7 |
| ln_midp | positive normals | 4.1 | 0.35 | 3.5e-7 |
| log10_midp | positive normals | 4.5 | 0.64 | 4.1e-7 |
| exp2_midp | x in [-126, 127.5) | 1.9 | 0.066 | 1.6e-7 |
| exp2_midp | x in [127.5, 128) | 134.1 | 32.2 | 8.0e-6 |
| exp_midp | \|x\| <= 1 | 2.0 | 0.055 | 1.7e-7 |
| exp_midp | \|x\| <= 10 | 8.2 | 0.083 | 5.9e-7 |
| exp_midp | \|x\| <= 40 | 31.3 | 0.17 | 2.0e-6 |
| exp_midp | x in [-87, 88.5] | 64.1 | 0.31 | 4.4e-6 |
| exp_midp | x in (88.5, ln(MAX)] | 197.1 | 66.3 | 1.2e-5 |
| cbrt_midp | x in [2^-126, MAX/3] | 3.2 | 0.53 | 2.5e-7 |
| cbrt_midp_precise | all normals and subnormals | 3.2 | 0.53 | 2.5e-7 |
| pow_midp(2.4) | x in [2^-8, 2^8] | 27.8 | 3.7 | 1.7e-6 |
| pow_midp(2.4) | x in [2^-20, 2^20] | 64.8 | 9.0 | 4.0e-6 |
| pow_midp(2.4) | x in [2^-50, 2^50] | 144.9 | 22.4 | 8.8e-6 |

`exp2_midp` splits x at the nearest integer so its polynomial sees |frac| <= 0.5,
but it clamps the integer part to 127, so for x in [127.5, 128) the polynomial
runs on frac up to 1. `exp_midp` multiplies by log2(e) first, so the rounding of
that product grows with |x|, and above 88.5 it lands in `exp2_midp`'s weak band.
`pow_midp` computes `exp2(n * log2(x))`: a one-ULP error in `log2` is scaled by
`n * log2(x)` before `exp2`, so its error grows with that product.

`cbrt_midp` and `cbrt_lowp` overflow an intermediate for magnitudes above
`f32::MAX / 3` (1.13e38) and return NaN, or ±inf for some inputs near the limit;
`cbrt_midp_precise` covers the whole range, subnormals included.

### lowp tier

| Function | Domain | Max rel. error | Max abs. error |
|----------|--------|----------------|----------------|
| exp2_lowp | x in [-126, 127.99] | 5.57e-3 | |
| exp2_lowp | x in (127.99, 128) | 1.23e-2 | |
| exp_lowp | x in [-87, 88.5] | 5.57e-3 | |
| pow_lowp(2.4) | x in [2^-20, 2^20] | 5.57e-3 | |
| log2_lowp | positive normals | unbounded near x = 1 | 6.4e-6 |
| ln_lowp | positive normals | unbounded near x = 1 | 8.5e-6 |
| log10_lowp | positive normals | unbounded near x = 1 | 5.8e-6 |
| cbrt_lowp | x in [2^-126, MAX/3] | 2.98e-5 | |

The lowp logarithms bound absolute error, not relative error: near x = 1 the
true value approaches zero while the error does not (`log2_lowp(1.0)` returns
about -1.87e-6). `exp2_lowp` clamps its input to 127.99, so results above
2^127.99 are up to 1.23% low; its cubic is 0.56% short of 2 at the top of each
unit interval, so the result steps by that much at every integer.

## Benchmark Results (x86-64, Zen 3)

### Single f32x8 (8 values)

| Function | Time | vs scalar |
|----------|------|-----------|
| exp2_lowp | 3.2 ns | 4.8× faster |
| exp2_midp | 3.7 ns | 4.1× faster |
| scalar exp2 | 15.3 ns | baseline |
| pow_lowp | 5.5 ns | 4.8× faster |
| pow_midp | 7.9 ns | 3.3× faster |
| scalar powf | 26.1 ns | baseline |
| cbrt_lowp | 2.3 ns | 7.0× faster |
| cbrt_midp | 3.5 ns | 4.7× faster |
| scalar cbrt | 16.4 ns | baseline |

### Bulk (1024 values, amortized per 8)

| Function | Time/1024 | Per-8 amortized |
|----------|-----------|-----------------|
| exp2_lowp | 289 ns | 2.3 ns |
| exp2_midp | 358 ns | 2.8 ns |
| scalar exp2 | 306 ns | 2.4 ns |

Bulk scalar is faster than single scalar because the compiler auto-vectorizes the loop. SIMD still wins at single-call latency.

## Accuracy Analysis for Color Processing

### Round-trip Accuracy (pow(x, 2.4) → pow(x, 1/2.4))

Testing all quantization levels for sRGB gamma round-trip:

| Bit Depth | Levels | lowp Exact | lowp Max Err | midp Exact | std::f32 |
|-----------|--------|------------|--------------|------------|----------|
| **8-bit** | 256 | 81.2% | **2 levels** | **100%** | 100% exact |
| **10-bit** | 1,024 | 45.7% | **8 levels** | **100%** | 100% exact |
| **12-bit** | 4,096 | 24.3% | **32 levels** | **100%** | 100% exact |
| **16-bit** | 65,536 | 5.2% | **512 levels** | 97%, 3% off-by-1 | 100% exact |

### Implementation Tiers and Suffixes

archmage provides two accuracy tiers. Every function has a plain form; the exp,
log and pow functions also have `_unchecked`; the midp tier adds `_precise`.

| Suffix | Edge Cases | Denormals | Use Case |
|--------|------------|-----------|----------|
| `_unchecked` | No | No | Hot loops with known-valid inputs |
| (none) | Yes | No | General use |
| `_precise` | Yes | Yes | Inputs that may be subnormal |

**Edge cases**: 0 → -inf, negative → NaN, +inf → +inf, NaN → NaN (for log functions)

**Denormals**: Very small numbers (< 1.17e-38 for f32) handled via 2^24 scale-up trick

#### Low-Precision Tier (`_lowp`)

Functions: `log2_lowp`, `exp2_lowp`, `ln_lowp`, `exp_lowp`, `log10_lowp`, `pow_lowp`, `cbrt_lowp`

Unchecked: `log2_lowp_unchecked`, `exp2_lowp_unchecked`, `ln_lowp_unchecked`, `exp_lowp_unchecked`, `log10_lowp_unchecked`, `pow_lowp_unchecked`

**NOT SUITABLE for color-accurate work:**
- up to 0.56% relative error (about 93,000 ULP) for exp2/exp/pow
- 8-bit round-trip: Only 81% exact
- 10-bit+: <50% exact

**Suitable for:**
- Preview/thumbnail generation
- Real-time effects where artifacts are acceptable
- Non-color-critical computations

#### Mid-Precision Tier (`_midp`) — RECOMMENDED

Functions: `log2_midp`, `exp2_midp`, `ln_midp`, `exp_midp`, `log10_midp`, `pow_midp`, `cbrt_midp`

Unchecked: `log2_midp_unchecked`, `exp2_midp_unchecked`, `ln_midp_unchecked`, `exp_midp_unchecked`, `log10_midp_unchecked`, `pow_midp_unchecked`

Precise (denormal-safe): `log2_midp_precise`, `ln_midp_precise`, `log10_midp_precise`, `pow_midp_precise`, `cbrt_midp_precise`; `exp2_midp_precise` and `exp_midp_precise` are the plain forms under the same name, since a subnormal input to an exponential is just close to zero

Activations: `sigmoid_midp` and `silu_midp`, built on `exp_midp` with an exact division, so saturated inputs give 0 or 1 rather than NaN

**SUITABLE for production color processing:**
- log2/ln/log10: at most 4.5 ULP; cbrt: 3.2 ULP
- exp2: 1.9 ULP below x = 127.5 (134.1 above); exp: 2.0 ULP for |x| <= 1, 64.1 on [-87, 88.5]
- pow: grows with |n * log2(x)|; for n = 2.4, 27.8 ULP on [2^-8, 2^8]
- 8-bit round-trip: **100% exact**
- 10-bit round-trip: **100% exact**
- 12-bit round-trip: **100% exact**
- 16-bit round-trip: 97% exact, 3% off-by-1

#### Platform Availability

Every f32 width (`f32x4`, `f32x8`, `f32x16`) on every backend has the same
functions: they are written once against the generic vector API in
`xtask/src/simd_types/generic_gen/transcendentals.rs`. The f64 types have no
transcendentals.

### Algorithm Implementation Status

| Use Case | Algorithm | Target | Status |
|----------|-----------|--------|--------|
| Preview/speed | lowp | <1% rel error | `pow_lowp`, `exp2_lowp`, `log2_lowp`, `ln_lowp`, `exp_lowp`, `log10_lowp` |
| Hot loops (valid inputs) | _unchecked | same as tier | `*_lowp_unchecked`, `*_midp_unchecked` |
| 8-bit sRGB | midp | 100% exact round-trip | `pow_midp`, `exp2_midp`, `log2_midp` |
| 10-bit HDR | midp | 100% exact round-trip | `pow_midp`, `exp2_midp`, `log2_midp` |
| 12-bit | midp | 100% exact round-trip | `pow_midp`, `exp2_midp`, `log2_midp` |
| 16-bit | midp | 97% exact, 3% off-by-1 | `pow_midp`, `exp2_midp`, `log2_midp` |
| Subnormal inputs | _precise | midp accuracy, subnormals included | `log2_midp_precise`, `ln_midp_precise`, `log10_midp_precise`, `pow_midp_precise`, `cbrt_midp_precise` |

**Recommendations:**
- Use midp functions for all color processing work
- Use `_unchecked` variants in hot loops when inputs are guaranteed valid (e.g., already clamped to [0, 1])
- Use `_precise` variants only when processing may include denormal values (~50% slower)
- For perfect precision, use std::f32 (scalar) at ~4-5x slower throughput

## Algorithms

### log2_lowp — Rational Polynomial

Uses bit manipulation for range reduction + (2,2) rational polynomial from butteraugli/jpegli.

```rust
// Range reduction: extract exponent and normalize mantissa to [2/3, 4/3]
let x_bits = x.to_bits() as i32;
let exp_bits = x_bits.wrapping_sub(0x3f2aaaab); // subtract 2/3
let exp_shifted = exp_bits >> 23;
let mantissa_bits = (x_bits - (exp_shifted << 23)) as u32;
let mantissa = f32::from_bits(mantissa_bits);
let exp_val = exp_shifted as f32;

// Evaluate rational polynomial on (mantissa - 1.0)
let m = mantissa - 1.0;

// Numerator: P2*m^2 + P1*m + P0
const P0: f32 = -1.850_383_34e-6;
const P1: f32 = 1.428_716_05;
const P2: f32 = 0.742_458_73;
let yp = P2.mul_add(m, P1).mul_add(m, P0);

// Denominator: Q2*m^2 + Q1*m + Q0
const Q0: f32 = 0.990_328_14;
const Q1: f32 = 1.009_671_86;
const Q2: f32 = 0.174_093_43;
let yq = Q2.mul_add(m, Q1).mul_add(m, Q0);

yp / yq + exp_val
```

**Precision**: absolute error at most 6.4e-6; relative error unbounded near x = 1
**Source**: butteraugli (libjxl), MIT licensed

### log2_midp — High Precision

Uses sqrt(2)/2 normalization + degree-6 odd polynomial on `y = (a-1)/(a+1)`.

```rust
const SQRT2_OVER_2: u32 = 0x3f3504f3;
const ONE: u32 = 0x3f800000;

let bits = x.to_bits();
let offset = ONE - SQRT2_OVER_2;
let adjusted = bits + offset;

let exp_raw = adjusted >> 23;
let n = (exp_raw - 0x7f) as f32;

let mantissa_mask = 0x007fffff;
let mantissa_bits = (adjusted & mantissa_mask) + SQRT2_OVER_2;
let a = f32::from_bits(mantissa_bits);

// Transform to [-1/3, 1/3] range
let y = (a - 1.0) / (a + 1.0);
let y2 = y * y;

// Polynomial: c0 + c1*y^2 + c2*y^4 + c3*y^6
const C0: f32 = 2.885_390_08;  // 2/ln(2)
const C1: f32 = 0.961_800_76;
const C2: f32 = 0.576_974_45;
const C3: f32 = 0.434_411_97;

let poly = C3.mul_add(y2, C2).mul_add(y2, C1).mul_add(y2, C0);
y * poly + n
```

**Precision**: at most 4.5 ULP over all positive normals

### exp2_lowp — Degree-3 Polynomial

Split into integer and fractional parts (floor-based), polynomial for 2^frac.

```rust
// floor(x) <= 127 keeps the exponent bit trick in range
let x = x.clamp(-126.0, 127.99);
let xi = x.floor();
let xf = x - xi;

// Degree-3 minimax polynomial for 2^x on [0, 1]
const C0: f32 = 1.0;
const C1: f32 = 0.693_147_18; // ln(2)
const C2: f32 = 0.240_226_5;
const C3: f32 = 0.055_504_11;

let poly = C3.mul_add(xf, C2).mul_add(xf, C1).mul_add(xf, C0);

// Scale by 2^integer using bit manipulation
let scale_bits = ((xi as i32 + 127) << 23) as u32;
let scale = f32::from_bits(scale_bits);

poly * scale
```

**Precision**: at most 0.56% relative error for x up to 127.99, 1.23% above

### exp2_midp — Degree-6 Polynomial with Round-to-Nearest Split

Uses round-to-nearest (not floor) to split into integer and fractional parts, keeping |frac| ≤ 0.5 instead of [0, 1). This reduces polynomial truncation error by ~1000× for near-integer inputs. The integer part is clamped to 127 max to prevent bit-trick overflow.

```rust
const C0: f32 = 1.0;
const C1: f32 = 0.693_147_18;  // ln(2)
// Fitted coefficients, close to the Taylor terms ln(2)^k / k!
const C2: f32 = 0.240_226_46;
const C3: f32 = 0.055_504_545;
const C4: f32 = 0.009_618_055;
const C5: f32 = 0.001_333_37;
const C6: f32 = 0.000_154_47;

// Round-to-nearest: |frac| ≤ 0.5, not [0, 1) from floor
// Clamp to 127 so (n+127)<<23 doesn't overflow
let xi = x.round().min(127.0);
let xf = x - xi;

let poly = C6.mul_add(xf, C5)
    .mul_add(xf, C4)
    .mul_add(xf, C3)
    .mul_add(xf, C2)
    .mul_add(xf, C1)
    .mul_add(xf, C0);

// Scale by 2^integer using IEEE 754 bit trick
let scale_bits = ((xi as i32 + 127) << 23) as u32;
let scale = f32::from_bits(scale_bits);
poly * scale
```

**Precision**: at most 1.9 ULP for x below 127.5; up to 134.1 ULP (8e-6 relative) in [127.5, 128), where frac reaches 1
**Edge cases**: exp2_midp returns 0 for x < -126, inf for x >= 128

### pow — Composition

All pow implementations use: `pow(x, n) = exp2(n * log2(x))`

- **lowp**: lowp log2 + lowp exp2 → at most 0.56% relative error
- **midp**: midp log2 + midp exp2 → error grows with |n * log2(x)|; for n = 2.4, 1.7e-6 relative on [2^-8, 2^8] up to 8.8e-6 on [2^-50, 2^50]

### ln (Natural Log)

`ln(x) = log2(x) * ln(2)`

### exp (Natural Exp)

`exp(x) = exp2(x * log2(e))`

## Square Root and Cube Root

archmage provides hardware-accelerated square root and software cube root:

### sqrt — Hardware SIMD

Uses hardware `sqrtps`/`vsqrtps` instruction. Full precision, fast on modern CPUs.

**Precision**: Full IEEE-754 precision
**Latency**: ~10-14 cycles on Zen2+/Skylake+

### rsqrt — Fast Reciprocal Square Root

For 1/sqrt(x), archmage provides `rsqrt_approx()` (the cheapest path per platform, about 12 bits), `rsqrt()` (within 4 ULP, exact at ±0, +inf and NaN) and `rsqrt_portable()` (exact, the same bits on every backend).

### cbrt_lowp — Cube Root (Fast)

Kahan bit-hack initial guess + 1 Halley iteration.

**Precision**: at most 259 ULP, 3.0e-5 relative error, below `f32::MAX / 3`; larger magnitudes return NaN or ±inf
**Performance**: ~2.3 ns / 8 values

### cbrt_midp — Cube Root (Accurate)

Kahan bit-hack initial guess + 2 Halley iterations.

**Precision**: at most 3.2 ULP, 2.5e-7 relative error, below `f32::MAX / 3`; larger magnitudes return NaN or ±inf (`cbrt_midp_precise` covers the whole range)
**Performance**: ~3.5 ns / 8 values
**Use case**: XYB color space (SSIMULACRA2, butteraugli), production color processing

## SIMD Intrinsics Used

### AVX2 log2
- `_mm256_castps_si256` / `_mm256_castsi256_ps` — reinterpret casts
- `_mm256_sub_epi32`, `_mm256_srai_epi32`, `_mm256_slli_epi32` — integer ops
- `_mm256_cvtepi32_ps` — int to float conversion
- `_mm256_fmadd_ps` — fused multiply-add
- `_mm256_div_ps` — division

### AVX2 exp2
- `_mm256_round_ps` — round-to-nearest
- `_mm256_min_ps` — clamp xi to 127
- `_mm256_cvtps_epi32` — float to int conversion
- `_mm256_fmadd_ps` — fused multiply-add
- `_mm256_mul_ps` — multiplication

## Comparison with sleef-rs

Benchmarked using `examples/sleef_comparison.rs` (required nightly for `portable_simd`), which was removed with the sleef feature in 337bbdd (2026-02-01). These figures predate the accuracy measurements at the top of this page, which supersede the archmage columns below.

### Performance (AVX2, 32K elements, 1000 iterations)

| Function | scalar std | sleef u10 | archmage lowp | vs sleef |
|----------|------------|-----------|---------------|----------|
| exp2 | 58 us (566 M/s) | 61 us (539 M/s) | **3 us (10,852 M/s)** | **20× faster** |
| log2 | 65 us (501 M/s) | 115 us (284 M/s) | **5.5 us (5,929 M/s)** | **21× faster** |
| pow(x, 2.4) | 123 us (267 M/s) | 248 us (132 M/s) | **10 us (3,294 M/s)** | **25× faster** |

### Accuracy (vs scalar std)

| Function | sleef u10 max err | archmage lowp max err | archmage midp max err |
|----------|-------------------|----------------------|----------------------|
| exp2 | 1.19e-7 | 5.56e-3 | 4.02e-6 |
| log2 | 1.14e-7 | 9.57e-4 | 1.3e-7 |
| pow(x, 2.4) | 1.13e-7 | 5.56e-3 | ~1.2e-6 |

### Analysis

**archmage lowp advantages:**
- 10-25× faster than sleef due to simpler polynomial approximations
- No external dependencies (pure Rust intrinsics)

**archmage midp advantages:**
- 4-5× faster than scalar std::f32
- 100% exact round-trips for 8-12 bit color processing
- Good balance of speed and accuracy for production use

**sleef advantages:**
- Higher accuracy (~1 ULP; midp exp2 is 1.9 ULP below x = 127.5 and 134.1 above)
- Required for scientific computing or highest-precision work

**CRITICAL: Use `#[arcane]` for proper inlining**

archmage SIMD types **must** be used within functions annotated with `#[arcane]` (at the entry point) or `#[rite]` (for internal helpers). Without this, every vector operation becomes a call across a `#[target_feature]` boundary (7.4× slower for the small kernel measured in [PERFORMANCE.md](PERFORMANCE.md)).

```rust
use archmage::{arcane, X64V3Token, SimdToken};
use magetypes::simd::generic::f32x8;

// WRONG - intrinsics won't inline
fn slow_version(token: X64V3Token, data: &[f32; 8]) -> [f32; 8] {
    let v = f32x8::load_t(token, data);  // Function call overhead!
    v.exp2_lowp().to_array()
}

// CORRECT - use #[arcane] macro
#[arcane(import_intrinsics)]
fn fast_version(token: X64V3Token, data: &[f32; 8]) -> [f32; 8] {
    let v = f32x8::load_t(token, data);  // Inline SIMD instructions!
    v.exp2_lowp().to_array()
}

// Usage
if let Some(token) = X64V3Token::summon() {
    let result = fast_version(token, &input);
}
```

**Recommendations:**
- For preview/thumbnails: use `_lowp` functions (fastest)
- For 8-12 bit color processing: use `_midp` functions (fast + accurate)
- For 16-bit+ or scientific: consider sleef-rs or std::f32
- Always wrap archmage code in `#[arcane]` functions

## References

- butteraugli (libjxl): fast_log2f rational polynomial
- wide crate: MIT-licensed polynomial implementations
- [sleef-rs](https://github.com/burrbull/sleef-rs): high-precision vectorized math functions
