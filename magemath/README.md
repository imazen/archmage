# Magemath

Configurable f32 power, exponential and logarithm approximations, built on
[Magetypes](../magetypes) vectors and [Archmage](..) feature proofs.
`no_std` by default; opt in to `std` for runtime CPU detection and native fallbacks.

```rust
use magemath::powf::{PowfOptions, try_powf_array_with};
const CURVE: u32 = PowfOptions::PORTABLE_GAMMA_MIDP.bits();
let result = try_powf_array_with::<CURVE, 4>([0.0, 0.125, 0.5, 1.0], 2.2).unwrap();
assert_eq!(result[0], 0.0);
assert_eq!(result[3], 1.0);
```

Const builders end in `.bits()` so the configuration specializes through the
caller and kernel. Each family documents its domain, precision and validation.
These are bounded approximations, not universal correctly rounded libm functions.

Use `try_*_f32x4_with` / `try_*_f32x8_with` (and x16 with `w512`) in typed kernels,
passing the existing feature token explicitly. Use the corresponding scalar
`*_with_token` call for tails with the same flags and token.

See [migration and scope](../docs/MAGEMATH-MIGRATION.md). Existing Magetypes
methods remain available during deprecation. Primitive `sqrt` stays in Magetypes;
reciprocal and reciprocal-square-root policy families are possible later work.
