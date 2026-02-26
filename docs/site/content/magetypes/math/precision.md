+++
title = "Precision Levels"
weight = 2
+++

Transcendental functions come in multiple precision variants. Pick the level that matches your accuracy needs.

## Precision Tiers

| Suffix | Precision | Relative Speed | Typical Use |
|--------|-----------|----------------|-------------|
| `_lowp` | ~12 bits (~3.5 decimal digits) | Fastest | Graphics, audio, game physics |
| `_midp` | ~20 bits (~6 decimal digits) | Balanced | General compute, ML inference |

```rust
use magetypes::simd::{
    generic::f32x8,
    backends::F32x8Convert,
};

fn example<T: F32x8Convert>(token: T) {
    let v = f32x8::<T>::splat(token, 2.0);

    let fast     = v.exp2_lowp();   // ~12-bit precision
    let balanced = v.exp2_midp();   // ~20-bit precision
}
```

There are no unsuffixed "full precision" transcendentals — `_midp` is the highest level. For exact results, use `sqrt()` (a hardware instruction).

## When to Use Each

**`_lowp` (~12 bits):** Use when visual or perceptual quality is what matters. 12 bits of precision means errors below 1 part in 4000 — invisible in pixel colors, inaudible in audio samples, and unnoticeable in physics simulations. The speed advantage compounds when you're calling these functions millions of times per frame.

**`_midp` (~20 bits):** The default choice when you don't have a strong reason to pick another. Accurate enough for ML inference, signal processing, and most numerical work. The `_midp` variants use the same polynomial families as `_lowp` but with more terms.

## Precise Variants

Some `_midp` functions have a `_precise` variant that adds a correction step:

```rust
use magetypes::simd::{
    generic::f32x8,
    backends::F32x8Convert,
};

fn example<T: F32x8Convert>(token: T) {
    let v = f32x8::<T>::splat(token, 2.0);

    let result = v.log2_midp();          // ~20-bit precision
    let result = v.log2_midp_precise();  // Slightly more accurate, slightly slower
}
```

Available `_precise` variants: `log2_midp_precise`, `ln_midp_precise`, `pow_midp_precise`, `log10_midp_precise`, `cbrt_midp_precise`.

## Unchecked Variants

Functions with domain restrictions (log requires positive input, etc.) have `_unchecked` variants that skip validation:

```rust
// given token: T where T: F32x8Convert
// let v = f32x8::<T>::splat(token, 2.0);

// Checked: returns NaN for non-positive inputs
let result = v.ln_lowp();

// Unchecked: undefined for non-positive inputs, slightly faster
let result = v.ln_lowp_unchecked();
```

Use `_unchecked` only when you've already validated inputs or your algorithm guarantees valid ranges. The speed difference is small — the check is usually a single comparison.

Available `_unchecked` variants exist for both `_lowp` and `_midp` tiers: `log2`, `exp2`, `ln`, `exp`, `log10`, `pow`.

## Available Combinations

| Function | `_lowp` | `_midp` | `_midp_precise` |
|----------|---------|---------|-----------------|
| `exp2` | yes | yes | — |
| `exp` | yes | yes | — |
| `log2` | yes | yes | yes |
| `ln` | yes | yes | yes |
| `log10` | yes | yes | yes |
| `pow` | yes | yes | yes |
| `cbrt` | — | yes | yes |
| `sqrt` | exact hardware instruction | — | — |

`sqrt` is a hardware instruction, not an approximation — there's no precision variant because it's already fast and exact.
