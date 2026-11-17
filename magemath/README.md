# Magemath

Configurable f32 power, exponential, logarithm and cube-root approximations, built on
[Magetypes](../magetypes) vectors and [Archmage](..) feature proofs.
`no_std` by default; opt in to `std` for runtime CPU detection and native fallbacks.

```rust
use magemath::powf::{presets::GammaMidp, try_powf_array_with, try_powf_slice_with};

let result = try_powf_array_with::<GammaMidp, 4>([0.0, 0.125, 0.5, 1.0], 2.2).unwrap();
assert_eq!(result[0], 0.0);
assert_eq!(result[3], 1.0);

let src = [0.0, 0.125, 0.5, 1.0, 2.0];
let mut dst = [0.0; 5];
try_powf_slice_with::<GammaMidp>(&src, &mut dst, 2.2).unwrap();
```

Profiles implement [`PowfProfile`](powf::PowfProfile) / [`ExpfProfile`](expf::ExpfProfile) /
[`LogfProfile`](logf::LogfProfile) / [`CbrtfProfile`](cbrtf::CbrtfProfile) with declared
range bounds (or use built-in [`powf::presets`], root modifiers [`Caller`], [`Unchecked`],
[`Backend`], or the legacy `Options<BITS>` adapter). These are bounded approximations,
not universal correctly rounded libm functions.

Use `try_*_f32x4_with` / `try_*_f32x8_with` (and x16 with `w512`) inside an
`#[attune(_*_t, define(f32x8))]` kernel, passing the caller's token and using
`*_with_token` for the scalar tail with the same profile and token proof:

```rust
use archmage::prelude::*;
use magemath::Caller;
use magemath::powf::{
    PowfError, presets::GammaMidp, try_powf_f32x8_with, try_powf_with_token,
};

#[attune(_*_t, define(f32x8))]
fn powf_slice_impl(token: Token, src: &[f32], dst: &mut [f32], exp: f32) -> Result<(), PowfError> {
    let (sc, st) = f32x8::partition_slice_t(token, src);
    let (dc, dt) = f32x8::partition_slice_mut_t(token, dst);
    for (s, d) in sc.iter().zip(dc.iter_mut()) {
        try_powf_f32x8_with::<Caller<GammaMidp>, _>(token, f32x8::load_t(token, s), exp)?.store(d);
    }
    for (&x, y) in st.iter().zip(dt.iter_mut()) {
        *y = try_powf_with_token::<Caller<GammaMidp>, _>(token, x, exp)?;
    }
    Ok(())
}

pub fn powf_slice(src: &[f32], dst: &mut [f32], exp: f32) -> Result<(), PowfError> {
    attuned!(powf_slice_impl(src, dst, exp))
}
```

See [choosing and migrating](../docs/MAGEMATH-MIGRATION.md). The flat Magetypes
approximation methods (`pow_midp`, `exp2_lowp`, `cbrt_midp`, ...) are unchanged and
not deprecated; Magemath adds explicit domains, validation and scalar tails beside them.
Primitive `sqrt` stays in Magetypes.
