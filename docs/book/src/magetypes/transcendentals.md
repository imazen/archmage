# Transcendental Functions

magetypes provides SIMD implementations of common mathematical functions. These are polynomial approximations optimized for speed.

## Precision Levels

Functions come in multiple precision variants:

| Suffix | Precision | Speed | Use Case |
|--------|-----------|-------|----------|
| `_lowp` | ~12 bits | Fastest | Graphics, audio |
| `_midp` | ~20 bits | Balanced | General use |
| (none) | Full | Slowest | Scientific |

```rust
let v = f32x8::splat(token, 2.0);

let fast = v.exp2_lowp();      // ~12-bit precision, fastest
let balanced = v.exp2_midp();  // ~20-bit precision
let precise = v.exp2();        // Full precision
```

## Exponential Functions

```rust
// Base-2 exponential: 2^x
let v = f32x8::splat(token, 3.0);
let result = v.exp2();  // [8.0; 8]

// Natural exponential: e^x
let result = v.exp();   // [e³; 8] ≈ [20.09; 8]
```

## Logarithms

```rust
let v = f32x8::splat(token, 8.0);

// Base-2 logarithm: log₂(x)
let result = v.log2();  // [3.0; 8]

// Natural logarithm: ln(x)
let result = v.ln();    // [ln(8); 8] ≈ [2.08; 8]

// Base-10 logarithm: log₁₀(x)
let result = v.log10(); // [log₁₀(8); 8] ≈ [0.90; 8]
```

## Power Functions

```rust
let base = f32x8::splat(token, 2.0);
let exp = f32x8::splat(token, 3.0);

// x^y (uses exp2(y * log2(x)))
let result = base.pow(exp);  // [8.0; 8]
```

## Root Functions

```rust
let v = f32x8::splat(token, 9.0);

// Square root
let result = v.sqrt();  // [3.0; 8]

// Cube root
let result = v.cbrt();  // [∛9; 8] ≈ [2.08; 8]

// Reciprocal square root: 1/√x
let result = v.rsqrt();  // [1/3; 8] ≈ [0.33; 8]
```

## Approximations

Fast approximations for graphics/games:

```rust
// Reciprocal: 1/x (approximate)
let v = f32x8::splat(token, 4.0);
let result = v.rcp();  // ≈ [0.25; 8]

// Reciprocal square root (approximate)
let result = v.rsqrt();  // ≈ [0.5; 8]
```

For higher precision, use Newton-Raphson refinement:

```rust
// One Newton-Raphson iteration for rsqrt
let approx = v.rsqrt();
let refined = approx * (f32x8::splat(token, 1.5) - v * approx * approx * f32x8::splat(token, 0.5));
```

## Special Handling

### Domain Errors

```rust
let v = f32x8::from_array(token, [-1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

// sqrt of negative → NaN
let sqrt = v.sqrt();  // [NaN, 0.0, 1.0, 1.41, 1.73, 2.0, 2.24, 2.45]

// log of non-positive → NaN or -inf
let log = v.ln();     // [NaN, -inf, 0.0, 0.69, 1.10, 1.39, 1.61, 1.79]
```

### Unchecked Variants

Some functions have `_unchecked` variants that skip domain validation:

```rust
// Assumes all inputs are valid (positive for sqrt/log)
let result = v.sqrt_unchecked();  // Faster, UB if negative
let result = v.ln_unchecked();    // Faster, UB if ≤ 0
```

## Example: Gaussian Function

```rust
#[arcane]
fn gaussian(token: Desktop64, x: &[f32; 8], sigma: f32) -> [f32; 8] {
    let v = f32x8::from_array(token, *x);
    let sigma_v = f32x8::splat(token, sigma);
    let two = f32x8::splat(token, 2.0);

    // exp(-x² / (2σ²))
    let x_sq = v * v;
    let two_sigma_sq = two * sigma_v * sigma_v;
    let exponent = -(x_sq / two_sigma_sq);
    let result = exponent.exp_midp();  // Good precision, fast

    result.to_array()
}
```

## Example: Softmax

```rust
#[arcane]
fn softmax(token: Desktop64, logits: &[f32; 8]) -> [f32; 8] {
    let v = f32x8::from_array(token, *logits);

    // Subtract max for numerical stability
    let max = v.reduce_max();
    let shifted = v - f32x8::splat(token, max);

    // exp(x - max)
    let exp = shifted.exp_midp();

    // Normalize
    let sum = exp.reduce_add();
    let result = exp / f32x8::splat(token, sum);

    result.to_array()
}
```

## Platform Notes

- **x86-64**: All functions available for f32x4, f32x8, f64x2, f64x4
- **AArch64**: Full support via NEON
- **WASM**: Most functions available, some via scalar fallback

The implementation uses polynomial approximations tuned per platform for best performance.
