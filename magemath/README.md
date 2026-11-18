# Magemath

Configurable `f32` power, exponential, logarithm and cube-root approximations, built on
[Magetypes](../magetypes) vectors and [Archmage](..) feature proofs.
`no_std` by default; opt in to `std` for runtime CPU detection and native fallbacks.

```rust
use magemath::powf::{Powf, PowfOptions, try_powf_slice};

const GAMMA: u128 = PowfOptions::new().gamma().midp().bits();

let result = Powf::<GAMMA>::array([0.0, 0.125, 0.5, 1.0], 2.2).unwrap();
assert_eq!(result[0], 0.0);
assert_eq!(result[3], 1.0);

let src = [0.0, 0.125, 0.5, 1.0, 2.0];
let mut dst = [0.0; 5];
Powf::<GAMMA>::slice(&src, &mut dst, 2.2).unwrap();

// Default positive-normal Midp convenience helpers:
let mut pos_dst = [0.0; 3];
try_powf_slice(&[0.25, 0.5, 2.0], &mut pos_dst, 2.2).unwrap();
```

Each family has one fluent `const` options builder ([`PowfOptions`](powf::PowfOptions),
[`ExpfOptions`](expf::ExpfOptions), [`LogfOptions`](logf::LogfOptions),
[`CbrtfOptions`](cbrtf::CbrtfOptions)) ending in `.bits()` for compile-time
specialization (`Powf<CFG>`, `Expf<CFG>`, `Logf<CFG>`, `Cbrtf<CFG>`, and inline
vector methods via [`SimdMath`]) or `.runtime()` for dynamic options. These are
bounded approximations, not universal correctly rounded libm functions.

Inside an `#[attune(_*_t, define(f32x8))]` kernel, import [`SimdMath`] to call
configured operations inline on any Magetypes `f32` vector (`self.token()` is
reused automatically), and use `*_with_token` for scalar tails:

```rust
use archmage::prelude::*;
use magemath::SimdMath;
use magemath::powf::{Powf, PowfError, PowfOptions};

const GAMMA_CALLER: u128 = PowfOptions::new().gamma().midp().caller().bits();

#[attune(_*_t, define(f32x8))]
fn powf_slice_impl(token: Token, src: &[f32], dst: &mut [f32], exp: f32) -> Result<(), PowfError> {
    assert_eq!(src.len(), dst.len(), "powf slice lengths differ");
    let (sc, st) = f32x8::partition_slice_t(token, src);
    let (dc, dt) = f32x8::partition_slice_mut_t(token, dst);
    for (s, d) in sc.iter().zip(dc.iter_mut()) {
        f32x8::load_t(token, s).try_powf_with::<GAMMA_CALLER>(exp)?.store(d);
    }
    for (&x, y) in st.iter().zip(dt.iter_mut()) {
        *y = Powf::<GAMMA_CALLER>::scalar_with_token(token, x, exp)?;
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
