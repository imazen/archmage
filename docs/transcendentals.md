# Transcendental Functions for SIMD Types

This document describes the algorithms, accuracy, and benchmark results for transcendental function implementations in archmage's SIMD types.

## Accuracy Summary

All numbers measured against `std` (libm) on x86-64 with AVX2+FMA. ARM and WASM produce identical results (0 ULP divergence from generic path).

### midp tier — per-function accuracy

| Function | Max ULP | Max Rel Error | Notes |
|----------|---------|---------------|-------|
| log2_midp | 2 | 1.34e-7 | Degree-6 odd polynomial on (a-1)/(a+1) |
| ln_midp | 2 | 1.94e-7 | log2_midp × ln(2) |
| log10_midp | 3 | 2.19e-7 | log2_midp × log10(2) |
| exp2_midp | 63 | 4.02e-6 | Degree-6 Taylor, round-to-nearest split |
| exp_midp | 58 | 3.74e-6 | exp2_midp(x × log2(e)) |
| cbrt_midp | 2 | 1.57e-7 | Kahan bit-hack + 2 Halley iterations |
| pow_midp (n=0.5) | 9 | 6.35e-7 | Compound: log2 × n → exp2 |
| pow_midp (n=1) | 15 | 1.13e-6 | |
| pow_midp (n=2) | 16 | 1.07e-6 | |
| pow_midp (n=3) | 55 | 4.16e-6 | |
| pow_midp (n=7) | 31 | 2.12e-6 | |

### exp2_midp — accuracy by input region

The round-to-nearest split gives uniform accuracy across all regions. No hot spots near integer boundaries.

| Region | Max ULP | Max Rel Err | Count |
|--------|---------|-------------|-------|
| near-zero [-0.01, 0.01] | 0 | 0 | 201 |
| small [-1, 1] | 1 | 8.6e-8 | 201 |
| medium [-10, 10] | 1 | 8.4e-8 | 201 |
| large [-50, 50] | 1 | 8.4e-8 | 1001 |
| near-overflow [100, 127] | 1 | 8.4e-8 | 271 |
| near-underflow [-126, -100] | 1 | 8.4e-8 | 261 |
| fractional near .0 (N+0.001) | 0 | 0 | 100 |
| fractional near .5 (N+0.499) | 1 | 8.4e-8 | 100 |
| fractional near .5+ (N+0.501) | 1 | 8.4e-8 | 100 |
| fractional near 1.0 (N+0.999) | 0 | 0 | 100 |

The 63 ULP max in the overall test comes from dense non-integer inputs near overflow (e.g., 127.9) where the polynomial evaluates at frac ≈ 0.4 across a very large scale factor.

### pow_midp — error scales with exponent

Error grows roughly linearly with |exponent| because log2 error (~2 ULP) gets multiplied by n before entering exp2.

| Exponent | Max ULP | Max Rel Err |
|----------|---------|-------------|
| n = 0.333 | 2 | 1.7e-7 |
| n = 0.5 | 2 | 1.6e-7 |
| n = 1 | 4 | 3.0e-7 |
| n = 2 | 7 | 5.1e-7 |
| n = 2.5 | 16 | 1.2e-6 |
| n = 3 | 16 | 1.5e-6 |
| n = 7 | 31 | 2.1e-6 |

### lowp tier — relative error only

ULP is meaningless for lowp near zero (log2_lowp(1) returns ~-1.87e-6 which is billions of ULP from 0). Use relative error instead.

| Function | Max Rel Error | Notes |
|----------|---------------|-------|
| log2_lowp | ~1.5e-4 | Rational polynomial |
| exp2_lowp | ~5.5e-3 | Degree-3 polynomial |
| ln_lowp | ~1.5e-4 | log2_lowp × ln(2) |
| exp_lowp | ~5.5e-3 | exp2_lowp(x × log2(e)) |
| log10_lowp | ~1.5e-4 | log2_lowp × log10(2) |
| pow_lowp | ~5.6e-3 | Compound: lowp log2 + lowp exp2 |
| cbrt_lowp | ~2.9e-5 | Kahan bit-hack + 1 Halley iteration |

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

archmage provides two accuracy tiers with three suffix variants each:

| Suffix | Edge Cases | Denormals | Use Case |
|--------|------------|-----------|----------|
| `_unchecked` | No | No | Hot loops with known-valid inputs |
| (none) | Yes | No | General use |
| `_precise` | Yes | Yes | Full IEEE compliance |

**Edge cases**: 0 → -inf, negative → NaN, +inf → +inf, NaN → NaN (for log functions)

**Denormals**: Very small numbers (< 1.17e-38 for f32) handled via 2^24 scale-up trick

#### Low-Precision Tier (`_lowp`)

Functions: `log2_lowp`, `exp2_lowp`, `ln_lowp`, `exp_lowp`, `log10_lowp`, `pow_lowp`

Unchecked: `log2_lowp_unchecked`, `exp2_lowp_unchecked`, `ln_lowp_unchecked`, `exp_lowp_unchecked`, `log10_lowp_unchecked`, `pow_lowp_unchecked`

**NOT SUITABLE for color-accurate work:**
- ~90,000 ULP max error, ~0.5% relative error
- 8-bit round-trip: Only 81% exact
- 10-bit+: <50% exact

**Suitable for:**
- Preview/thumbnail generation
- Real-time effects where artifacts are acceptable
- Non-color-critical computations

#### Mid-Precision Tier (`_midp`) — RECOMMENDED

Functions: `log2_midp`, `exp2_midp`, `ln_midp`, `exp_midp`, `log10_midp`, `pow_midp`, `cbrt_midp`

Unchecked: `log2_midp_unchecked`, `exp2_midp_unchecked`, `ln_midp_unchecked`, `exp_midp_unchecked`, `log10_midp_unchecked`, `pow_midp_unchecked`, `cbrt_midp_unchecked`

Precise (denormal-safe): `log2_midp_precise`, `ln_midp_precise`, `log10_midp_precise`, `pow_midp_precise`, `cbrt_midp_precise`

**SUITABLE for production color processing:**
- log/cbrt: ~2-3 ULP max error
- exp2/exp: ~63 ULP max error
- pow: ~6-55 ULP depending on exponent
- 8-bit round-trip: **100% exact**
- 10-bit round-trip: **100% exact**
- 12-bit round-trip: **100% exact**
- 16-bit round-trip: 97% exact, 3% off-by-1

#### Platform Availability

| Platform | lowp/midp | _unchecked | _precise |
|----------|-----------|------------|----------|
| x86-64 (AVX2+) | ✓ | ✓ | cbrt only |
| AArch64 (NEON) | ✓ | ✓ | cbrt only |
| WASM SIMD128 | ✓ | ✓ | ✓ (full) |

WASM has complete `_precise` variants because polynomial approximations are used (no hardware transcendentals).
x86/ARM use different algorithms where denormal handling is only implemented for cbrt.

### Algorithm Implementation Status

| Use Case | Algorithm | Target | Status |
|----------|-----------|--------|--------|
| Preview/speed | lowp | <1% rel error | `pow_lowp`, `exp2_lowp`, `log2_lowp`, `ln_lowp`, `exp_lowp`, `log10_lowp` |
| Hot loops (valid inputs) | _unchecked | same as tier | `*_lowp_unchecked`, `*_midp_unchecked` |
| 8-bit sRGB | midp | 100% exact round-trip | `pow_midp`, `exp2_midp`, `log2_midp` |
| 10-bit HDR | midp | 100% exact round-trip | `pow_midp`, `exp2_midp`, `log2_midp` |
| 12-bit | midp | 100% exact round-trip | `pow_midp`, `exp2_midp`, `log2_midp` |
| 16-bit | midp | 97% exact, 3% off-by-1 | `pow_midp`, `exp2_midp`, `log2_midp` |
| Denormal inputs | _precise | Full IEEE | `log2_midp_precise`, `ln_midp_precise`, `log10_midp_precise`, `pow_midp_precise` |

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

**Precision**: ~7.7e-5 max relative error
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

**Precision**: ~2 ULP max error

### exp2_lowp — Degree-3 Polynomial

Split into integer and fractional parts (floor-based), polynomial for 2^frac.

```rust
let x = x.clamp(-126.0, 126.0);
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

**Precision**: ~5.5e-3 max relative error

### exp2_midp — Degree-6 Polynomial with Round-to-Nearest Split

Uses round-to-nearest (not floor) to split into integer and fractional parts, keeping |frac| ≤ 0.5 instead of [0, 1). This reduces polynomial truncation error by ~1000× for near-integer inputs. The integer part is clamped to 127 max to prevent bit-trick overflow.

```rust
const C0: f32 = 1.0;
const C1: f32 = 0.693_147_18;  // ln(2)
const C2: f32 = 0.240_226_51;  // ln(2)^2 / 2
const C3: f32 = 0.055_504_11;  // ln(2)^3 / 6
const C4: f32 = 0.009_618_13;  // ln(2)^4 / 24
const C5: f32 = 0.001_333_55;  // ln(2)^5 / 120
const C6: f32 = 0.000_154_04;  // ln(2)^6 / 720

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

**Precision**: ~63 ULP max error, ~4e-6 max relative error
**Edge cases**: exp2_midp returns 0 for x < -126, inf for x >= 128

### pow — Composition

All pow implementations use: `pow(x, n) = exp2(n * log2(x))`

- **lowp**: lowp log2 + lowp exp2 → ~5.5e-3 error
- **midp**: midp log2 + midp exp2 → ~1e-6 to ~4e-6 error depending on exponent

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

For 1/sqrt(x), archmage provides `rsqrt_approx()` (raw ~12-bit precision) and `rsqrt()` (refined with Newton-Raphson).

### cbrt_lowp — Cube Root (Fast)

Kahan bit-hack initial guess + 1 Halley iteration.

**Precision**: ~254 ULP max error, ~2.9e-5 max relative error
**Performance**: ~2.3 ns / 8 values

### cbrt_midp — Cube Root (Accurate)

Kahan bit-hack initial guess + 2 Halley iterations.

**Precision**: ~2 ULP max error, ~1.6e-7 max relative error
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

Benchmarked using `examples/sleef_comparison.rs` (requires nightly for `portable_simd`).

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
- Higher accuracy (~1 ULP vs ~63 ULP for midp exp2)
- Required for scientific computing or highest-precision work

**CRITICAL: Use `#[arcane]` for proper inlining**

archmage SIMD types **must** be used within functions annotated with `#[arcane]` (at the entry point) or `#[rite]` (for internal helpers). Without this, each call crosses a `#[target_feature]` boundary, costing 4-6× (see [PERFORMANCE.md](PERFORMANCE.md)).

```rust
use archmage::{arcane, X64V3Token, SimdToken};
use magetypes::simd::f32x8;

// WRONG - intrinsics won't inline
fn slow_version(token: X64V3Token, data: &[f32; 8]) -> [f32; 8] {
    let v = f32x8::load(token, data);  // Function call overhead!
    v.exp2_lowp().to_array()
}

// CORRECT - use #[arcane] macro
#[arcane(import_intrinsics)]
fn fast_version(token: X64V3Token, data: &[f32; 8]) -> [f32; 8] {
    let v = f32x8::load(token, data);  // Inline SIMD instructions!
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
