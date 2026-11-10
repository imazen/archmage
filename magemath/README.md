# Magemath

Configurable power, exponential, logarithm, min/max and reciprocal families, built on
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
passing the existing feature token explicitly. Transcendentals use the corresponding
`*_with_token` tail. Reciprocal/minmax use `*_f32xN_lane_with` (or f64 for minmax)
to retain both the token and vector width for native estimate parity.

See [migration and scope](../docs/MAGEMATH-MIGRATION.md). Published flat Magetypes methods retain their arithmetic during deprecation;
the unpublished configured modules and delegates live only in Magemath. Primitive `sqrt` stays in Magetypes;
see [primitive contracts and measured costs](../docs/PRIMITIVE-CONTRACTS.md) for
configurable f32 reciprocal/rsqrt and f32/f64 min/max. Native estimates can vary
by ISA and width; portable profiles retain matching scalar/vector bits.
