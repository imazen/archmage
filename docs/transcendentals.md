# Transcendental Functions for SIMD Types

This document describes the algorithms and benchmark results for transcendental function implementations in archmage's SIMD types.

## Benchmark Results (AVX2+FMA, 32K elements)

### exp2 Performance

| Implementation | ns/iter | Throughput | Max Error | Avg Error |
|----------------|---------|------------|-----------|-----------|
| scalar std | 55,360 | 592 M/s | baseline | baseline |
| scalar fast | 191,368 | 171 M/s | 5.5e-3 | 1.2e-3 |
| **simd basic** | **3,106** | **10,550 M/s** | 5.5e-3 | 1.2e-3 |
| simd hp | 3,829 | 8,557 M/s | 8.3e-5 | 1.3e-5 |
| simd lut | 8,271 | 3,962 M/s | **1.2e-7** | 2.0e-8 |

### log2 Performance

| Implementation | ns/iter | Throughput | Max Error | Avg Error |
|----------------|---------|------------|-----------|-----------|
| scalar std | 62,278 | 526 M/s | baseline | baseline |
| scalar fast | 200,706 | 163 M/s | 7.7e-5 | 4.5e-7 |
| **simd basic** | **3,646** | **8,986 M/s** | 7.7e-5 | 4.5e-7 |
| simd hp | 4,584 | 7,148 M/s | **1.3e-7** | 4.0e-9 |

### pow(x, 2.4) Performance (sRGB decode)

| Implementation | ns/iter | Throughput | Max Error | Avg Error |
|----------------|---------|------------|-----------|-----------|
| scalar std | 96,941 | 338 M/s | baseline | baseline |
| scalar fast | 393,571 | 83 M/s | 5.5e-3 | 1.3e-3 |
| **simd basic** | **9,756** | **3,359 M/s** | 5.5e-3 | 1.3e-3 |
| simd hp | 14,558 | 2,251 M/s | 8.5e-5 | 1.5e-5 |
| simd lut | 20,843 | 1,572 M/s | **7.9e-7** | 8.5e-8 |

## Accuracy Analysis for Color Processing

### Round-trip Accuracy (pow(x, 2.4) → pow(x, 1/2.4))

Testing all quantization levels for sRGB gamma round-trip:

| Bit Depth | Levels | archmage Exact | Off by 1 | Off by >1 | Max Error | std::f32 |
|-----------|--------|----------------|----------|-----------|-----------|----------|
| **8-bit** | 256 | 81.2% | 15.2% | 3.5% | **2 levels** | 100% exact |
| **10-bit** | 1,024 | 45.7% | 28.2% | 26.1% | **8 levels** | 100% exact |
| **12-bit** | 4,096 | 24.3% | 16.8% | 58.9% | **32 levels** | 100% exact |
| **16-bit** | 65,536 | 5.2% | 5.3% | 89.5% | **512 levels** | 100% exact |

### ULP Error by Function

| Function | Range | Max ULP | Avg ULP | Max Rel Error |
|----------|-------|---------|---------|---------------|
| exp2 | [0, 1] | 93,261 | 18,198 | 5.56e-3 |
| exp2 | [-10, 10] | 92,913 | 18,155 | 5.54e-3 |
| log2 | (0, 1] | varies* | 305,431 | 1.29e-2 |
| log2 | [1, 255] | varies* | 11.97M | 1.85e-6 |
| pow(x, 2.4) | (0, 1] | 93,350 | 20,037 | 5.56e-3 |
| pow(x, 1/2.4) | (0, 1] | 93,308 | 29,154 | 5.56e-3 |

*log2 ULP varies greatly near 0 due to floating-point representation

### Two Implementation Tiers

archmage provides two accuracy tiers for transcendental functions:

#### Basic Tier (pow, exp2, log2, ln, exp)

**⚠️ NOT SUITABLE for color-accurate work:**
- ~90,000 ULP max error, ~0.5% relative error
- 8-bit round-trip: Only 81% exact
- 10-bit+: <50% exact

**Suitable for:**
- Preview/thumbnail generation
- Real-time effects where artifacts are acceptable
- Non-color-critical computations

#### HP Tier (pow_hp, exp2_hp, log2_hp, ln_hp, exp_hp) ✅ RECOMMENDED

**✅ SUITABLE for production color processing:**
- ~145 ULP max error, ~8e-6 relative error
- 8-bit round-trip: **100% exact**
- 10-bit round-trip: **100% exact**
- 12-bit round-trip: **100% exact**
- 16-bit round-trip: 97% exact, 3% off-by-1

### Algorithm Implementation Status

| Use Case | Algorithm | Target | Status |
|----------|-----------|--------|--------|
| Preview/speed | basic | <1% rel error | ✅ `pow()`, `exp2()`, `log2()` |
| 8-bit sRGB | HP | 100% exact round-trip | ✅ `pow_hp()`, `exp2_hp()`, `log2_hp()` |
| 10-bit HDR | HP | 100% exact round-trip | ✅ `pow_hp()`, `exp2_hp()`, `log2_hp()` |
| 12-bit | HP | 100% exact round-trip | ✅ `pow_hp()`, `exp2_hp()`, `log2_hp()` |
| 16-bit | HP | 97% exact, 3% off-by-1 | ✅ `pow_hp()`, `exp2_hp()`, `log2_hp()` |

**Recommendation**: Use HP functions (`pow_hp`, `exp2_hp`, `log2_hp`) for all color processing work.

## Algorithms

### log2 - Basic (Rational Polynomial)

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

### log2 - High Precision

Uses sqrt(2)/2 normalization + degree-3 polynomial on `x = (a-1)/(a+1)`.

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
let x = (a - 1.0) / (a + 1.0);
let x2 = x * x;

// Polynomial: c0 + c1*x^2 + c2*x^4
const C0: f32 = 0.961_796_7;
const C1: f32 = 0.577_078_04;
const C2: f32 = 0.412_198_57;

let u = C2.mul_add(x2, C1).mul_add(x2, C0);
let log2_scale = 2.885_39; // 2/ln(2)

x2 * x * u + x * log2_scale + n
```

**Precision**: ~1.3e-7 max relative error

### exp2 - Basic (Polynomial)

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

### exp2 - High Precision

Uses degree-5 minimax polynomial.

```rust
const C0: f32 = 1.0;
const C1: f32 = 0.693_147_18;  // ln(2)
const C2: f32 = 0.240_226_51;  // ln(2)^2 / 2
const C3: f32 = 0.055_504_11;  // ln(2)^3 / 6
const C4: f32 = 0.009_618_13;  // ln(2)^4 / 24
const C5: f32 = 0.001_333_55;  // ln(2)^5 / 120

let poly = C5.mul_add(xf, C4)
    .mul_add(xf, C3)
    .mul_add(xf, C2)
    .mul_add(xf, C1)
    .mul_add(xf, C0);
```

**Precision**: ~8.3e-5 max relative error

### exp2 - LUT (from linear-srgb)

64-entry lookup table + polynomial refinement.

```rust
// 64-entry table: 2^(i/64) for i in 0..64
static EXP2_TABLE: [u32; 64] = [...];

// Magic constant for rounding to nearest 1/64
let redux = f32::from_bits(0x4b400000) / 64.0;
let sum = x + redux;
let ui = sum.to_bits();

// Get table index (low 6 bits)
let i0 = (ui + 32) & 63;

// Get exponent adjustment
let k = (ui + 32) >> 6;

// Get fractional part for refinement
let uf = sum - redux;
let f = x - uf;

// Table lookup
let z0 = f32::from_bits(EXP2_TABLE[i0]);

// Polynomial refinement
const C0: f32 = 0.240_226_5;
const C1: f32 = 0.693_147_2;
let u = C0.mul_add(f, C1) * f;
let result = u.mul_add(z0, z0);

// Scale by 2^k
let scale = f32::from_bits((k + 127) << 23);
result * scale
```

**Precision**: ~1.2e-7 max relative error
**Source**: linear-srgb crate

### pow - Composition

All pow implementations use: `pow(x, n) = exp2(n * log2(x))`

- **basic**: basic log2 + basic exp2 → ~5.5e-3 error
- **hp**: hp log2 + hp exp2 → ~8.5e-5 error
- **lut**: hp log2 + lut exp2 → ~7.9e-7 error

### ln (Natural Log)

`ln(x) = log2(x) * ln(2)`

```rust
const LN2: f32 = 0.693_147_18;
let result = log2(x) * LN2;
```

### exp (Natural Exp)

`exp(x) = exp2(x * log2(e))`

```rust
const LOG2_E: f32 = 1.442_695_04;
let result = exp2(x * LOG2_E);
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

### AVX2 exp2 LUT
- `_mm256_set_ps` - scalar gather (manual)
- Or `_mm256_i32gather_epi32` - hardware gather (if available)

## Comparison with sleef-rs

Benchmarked using `examples/sleef_comparison.rs` (requires nightly for `portable_simd`).

### Performance (AVX2, 32K elements, 1000 iterations)

| Function | scalar std | sleef u10 | archmage | Archmage vs sleef |
|----------|------------|-----------|----------|-------------------|
| exp2 | 58 µs (566 M/s) | 61 µs (539 M/s) | **3 µs (10,852 M/s)** | **20x faster** |
| log2 | 65 µs (501 M/s) | 115 µs (284 M/s) | **5.5 µs (5,929 M/s)** | **21x faster** |
| pow(x, 2.4) | 123 µs (267 M/s) | 248 µs (132 M/s) | **10 µs (3,294 M/s)** | **25x faster** |

### Accuracy (vs scalar std)

| Function | sleef u10 max err | archmage max err | Sleef advantage |
|----------|-------------------|------------------|-----------------|
| exp2 | 1.19e-7 | 5.56e-3 | ~47,000x more accurate |
| log2 | 1.14e-7 | 9.57e-4 | ~8,400x more accurate |
| pow(x, 2.4) | 1.13e-7 | 5.56e-3 | ~49,000x more accurate |

### Analysis

**archmage advantages:**
- 10-25x faster than sleef due to simpler polynomial approximations
- No external dependencies (pure Rust intrinsics)
- Suitable for performance-critical 8-bit image processing

**sleef advantages:**
- Much higher accuracy (~1 ULP vs ~0.5% max error)
- Required for scientific computing or high-bit-depth processing
- Uses higher-degree polynomials and LUT-based implementations

**CRITICAL: #[target_feature] requirement**

archmage SIMD types **must** be used within functions annotated with `#[target_feature]` or the `#[arcane]` macro. Without this, intrinsics are called as functions instead of inlined, causing 50-100x slowdowns.

```rust
// WRONG - intrinsics won't inline
fn slow_version(token: Avx2FmaToken, data: &[f32]) {
    let v = f32x8::load(token, ...);  // Function call overhead!
    v.exp2();
}

// CORRECT - intrinsics inline properly
#[target_feature(enable = "avx2,fma")]
unsafe fn fast_version(token: Avx2FmaToken, data: &[f32]) {
    let v = f32x8::load(token, ...);  // Inline SIMD instructions!
    v.exp2();
}
```

**Recommendations:**
- For 8-bit image processing (sRGB, gamma): use archmage (fastest, sufficient accuracy)
- For 10-16 bit HDR or scientific use: consider sleef-rs or higher-precision implementations
- Always wrap archmage code in `#[arcane]` or `#[target_feature]` functions

## References

- butteraugli (libjxl): fast_log2f rational polynomial
- linear-srgb crate: LUT-based exp2 with polynomial refinement
- wide crate: MIT-licensed polynomial implementations
- [sleef-rs](https://github.com/burrbull/sleef-rs): high-precision vectorized math functions
