# Transcendental Functions for SIMD Types

This document describes the algorithms and benchmark results for transcendental function implementations in archmage's SIMD types.

## Benchmark Results (AVX2+FMA, 32K elements)

### exp2 Performance

| Implementation | ns/iter | Throughput | Max Error | Avg Error |
|----------------|---------|------------|-----------|-----------|
| scalar std | 55,360 | 592 M/s | baseline | baseline |
| **simd lowp** | **3,106** | **10,550 M/s** | 5.5e-3 | 1.2e-3 |
| simd midp | 3,829 | 8,557 M/s | 8.3e-5 | 1.3e-5 |

### log2 Performance

| Implementation | ns/iter | Throughput | Max Error | Avg Error |
|----------------|---------|------------|-----------|-----------|
| scalar std | 62,278 | 526 M/s | baseline | baseline |
| **simd lowp** | **3,646** | **8,986 M/s** | 7.7e-5 | 4.5e-7 |
| simd midp | 4,584 | 7,148 M/s | **1.3e-7** | 4.0e-9 |

### pow(x, 2.4) Performance (sRGB decode)

| Implementation | ns/iter | Throughput | Max Error | Avg Error |
|----------------|---------|------------|-----------|-----------|
| scalar std | 96,941 | 338 M/s | baseline | baseline |
| **simd lowp** | **9,756** | **3,359 M/s** | 5.5e-3 | 1.3e-3 |
| simd midp | 14,558 | 2,251 M/s | 8.5e-5 | 1.5e-5 |

## Accuracy Analysis for Color Processing

### Round-trip Accuracy (pow(x, 2.4) -> pow(x, 1/2.4))

Testing all quantization levels for sRGB gamma round-trip:

| Bit Depth | Levels | lowp Exact | lowp Max Err | midp Exact | std::f32 |
|-----------|--------|------------|--------------|------------|----------|
| **8-bit** | 256 | 81.2% | **2 levels** | **100%** | 100% exact |
| **10-bit** | 1,024 | 45.7% | **8 levels** | **100%** | 100% exact |
| **12-bit** | 4,096 | 24.3% | **32 levels** | **100%** | 100% exact |
| **16-bit** | 65,536 | 5.2% | **512 levels** | 97%, 3% off-by-1 | 100% exact |

### ULP Error by Function (lowp tier)

| Function | Range | Max ULP | Avg ULP | Max Rel Error |
|----------|-------|---------|---------|---------------|
| exp2_lowp | [0, 1] | 93,261 | 18,198 | 5.56e-3 |
| exp2_lowp | [-10, 10] | 92,913 | 18,155 | 5.54e-3 |
| log2_lowp | (0, 1] | varies* | 305,431 | 1.29e-2 |
| log2_lowp | [1, 255] | varies* | 11.97M | 1.85e-6 |
| pow_lowp(x, 2.4) | (0, 1] | 93,350 | 20,037 | 5.56e-3 |
| pow_lowp(x, 1/2.4) | (0, 1] | 93,308 | 29,154 | 5.56e-3 |

*log2 ULP varies greatly near 0 due to floating-point representation

### ULP Error by Function (midp tier)

| Function | Range | Max ULP | Avg ULP | Max Rel Error |
|----------|-------|---------|---------|---------------|
| pow_midp(x, 2.4) | (0, 1] | 145 | 19.7 | 8.65e-6 |
| pow_midp(x, 1/2.4) | (0, 1] | 141 | 30.0 | 8.40e-6 |

### Two Implementation Tiers

archmage provides two accuracy tiers for transcendental functions:

#### Low-Precision Tier (`_lowp` suffix)

Functions: `pow_lowp()`, `exp2_lowp()`, `log2_lowp()`, `ln_lowp()`, `exp_lowp()`

**NOT SUITABLE for color-accurate work:**
- ~90,000 ULP max error, ~0.5% relative error
- 8-bit round-trip: Only 81% exact
- 10-bit+: <50% exact

**Suitable for:**
- Preview/thumbnail generation
- Real-time effects where artifacts are acceptable
- Non-color-critical computations

#### Mid-Precision Tier (`_midp` suffix) - RECOMMENDED

Functions: `pow_midp()`, `exp2_midp()`, `log2_midp()`, `ln_midp()`, `exp_midp()`

**SUITABLE for production color processing:**
- ~145 ULP max error, ~8e-6 relative error
- 8-bit round-trip: **100% exact**
- 10-bit round-trip: **100% exact**
- 12-bit round-trip: **100% exact**
- 16-bit round-trip: 97% exact, 3% off-by-1

### Algorithm Implementation Status

| Use Case | Algorithm | Target | Status |
|----------|-----------|--------|--------|
| Preview/speed | lowp | <1% rel error | `pow_lowp()`, `exp2_lowp()`, `log2_lowp()` |
| 8-bit sRGB | midp | 100% exact round-trip | `pow_midp()`, `exp2_midp()`, `log2_midp()` |
| 10-bit HDR | midp | 100% exact round-trip | `pow_midp()`, `exp2_midp()`, `log2_midp()` |
| 12-bit | midp | 100% exact round-trip | `pow_midp()`, `exp2_midp()`, `log2_midp()` |
| 16-bit | midp | 97% exact, 3% off-by-1 | `pow_midp()`, `exp2_midp()`, `log2_midp()` |

**Recommendation**: Use midp functions (`pow_midp`, `exp2_midp`, `log2_midp`) for all color processing work.
For perfect precision, use std::f32 (scalar) at ~7-15x slower throughput.

## Algorithms

### log2_lowp - Rational Polynomial

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

### log2_midp - High Precision

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

**Precision**: ~3 ULP max error

### exp2_lowp - Polynomial

Split into integer and fractional parts, polynomial for 2^frac.

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

### exp2_midp - High Precision

Uses degree-6 minimax polynomial.

```rust
const C0: f32 = 1.0;
const C1: f32 = 0.693_147_18;  // ln(2)
const C2: f32 = 0.240_226_51;  // ln(2)^2 / 2
const C3: f32 = 0.055_504_11;  // ln(2)^3 / 6
const C4: f32 = 0.009_618_13;  // ln(2)^4 / 24
const C5: f32 = 0.001_333_55;  // ln(2)^5 / 120
const C6: f32 = 0.000_154_04;  // ln(2)^6 / 720

let poly = C6.mul_add(xf, C5)
    .mul_add(xf, C4)
    .mul_add(xf, C3)
    .mul_add(xf, C2)
    .mul_add(xf, C1)
    .mul_add(xf, C0);
```

**Precision**: ~140 ULP max error, ~8e-6 max relative error

### pow - Composition

All pow implementations use: `pow(x, n) = exp2(n * log2(x))`

- **lowp**: lowp log2 + lowp exp2 -> ~5.5e-3 error
- **midp**: midp log2 + midp exp2 -> ~8.5e-6 error

### ln (Natural Log)

`ln(x) = log2(x) * ln(2)`

```rust
const LN2: f32 = 0.693_147_18;
let result = log2_lowp(x) * LN2;  // or log2_midp for midp variant
```

### exp (Natural Exp)

`exp(x) = exp2(x * log2(e))`

```rust
const LOG2_E: f32 = 1.442_695_04;
let result = exp2_lowp(x * LOG2_E);  // or exp2_midp for midp variant
```

## SIMD Intrinsics Used

### AVX2 log2
- `_mm256_castps_si256` / `_mm256_castsi256_ps` - reinterpret casts
- `_mm256_sub_epi32`, `_mm256_srai_epi32`, `_mm256_slli_epi32` - integer ops
- `_mm256_cvtepi32_ps` - int to float conversion
- `_mm256_fmadd_ps` - fused multiply-add
- `_mm256_div_ps` - division

### AVX2 exp2
- `_mm256_floor_ps` - floor
- `_mm256_cvtps_epi32` - float to int conversion
- `_mm256_fmadd_ps` - fused multiply-add
- `_mm256_mul_ps` - multiplication

## Comparison with sleef-rs

Benchmarked using `examples/sleef_comparison.rs` (requires nightly for `portable_simd`).

### Performance (AVX2, 32K elements, 1000 iterations)

| Function | scalar std | sleef u10 | archmage lowp | vs sleef |
|----------|------------|-----------|---------------|----------|
| exp2 | 58 us (566 M/s) | 61 us (539 M/s) | **3 us (10,852 M/s)** | **20x faster** |
| log2 | 65 us (501 M/s) | 115 us (284 M/s) | **5.5 us (5,929 M/s)** | **21x faster** |
| pow(x, 2.4) | 123 us (267 M/s) | 248 us (132 M/s) | **10 us (3,294 M/s)** | **25x faster** |

### Accuracy (vs scalar std)

| Function | sleef u10 max err | archmage lowp max err | archmage midp max err |
|----------|-------------------|----------------------|----------------------|
| exp2 | 1.19e-7 | 5.56e-3 | 8.3e-5 |
| log2 | 1.14e-7 | 9.57e-4 | 1.3e-7 |
| pow(x, 2.4) | 1.13e-7 | 5.56e-3 | 8.5e-6 |

### Analysis

**archmage lowp advantages:**
- 10-25x faster than sleef due to simpler polynomial approximations
- No external dependencies (pure Rust intrinsics)

**archmage midp advantages:**
- 7-15x faster than scalar std::f32
- 100% exact round-trips for 8-12 bit color processing
- Good balance of speed and accuracy for production use

**sleef advantages:**
- Higher accuracy (~1 ULP vs ~140 ULP for midp)
- Required for scientific computing or highest-precision work

**CRITICAL: #[target_feature] requirement**

archmage SIMD types **must** be used within functions annotated with `#[target_feature]` or the `#[arcane]` macro. Without this, intrinsics are called as functions instead of inlined, causing 50-100x slowdowns.

```rust
// WRONG - intrinsics won't inline
fn slow_version(token: Avx2FmaToken, data: &[f32]) {
    let v = f32x8::load(token, ...);  // Function call overhead!
    v.exp2_lowp();
}

// CORRECT - intrinsics inline properly
#[target_feature(enable = "avx2,fma")]
unsafe fn fast_version(token: Avx2FmaToken, data: &[f32]) {
    let v = f32x8::load(token, ...);  // Inline SIMD instructions!
    v.exp2_lowp();
}
```

**Recommendations:**
- For preview/thumbnails: use `_lowp` functions (fastest)
- For 8-12 bit color processing: use `_midp` functions (fast + accurate)
- For 16-bit+ or scientific: consider sleef-rs or std::f32
- Always wrap archmage code in `#[arcane]` or `#[target_feature]` functions

## References

- butteraugli (libjxl): fast_log2f rational polynomial
- wide crate: MIT-licensed polynomial implementations
- [sleef-rs](https://github.com/burrbull/sleef-rs): high-precision vectorized math functions
