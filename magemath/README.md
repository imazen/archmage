# magemath

Vector and fixed-array math on `magetypes`, with the same bits on every CPU.

**Status: draft, unpublished.** `f32x8` only; `f32x4`, `f32x16` and f64 wait on a
caller.

```rust
use magemath::portable::{array, slice};

let mut plane = vec![0.5f32; 1_000];
slice::pow_midp_f32(&mut plane, 2.4); // runtime dispatch, once per call

let lanes = array::exp2_midp_f32x8([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]);
assert_eq!(lanes[3], 8.0);
```

Every result in `portable` equals the matching `zenmath::scalar` function on
x86 v3 and v4x, NEON, WASM (strict and relaxed), and the scalar tier. NaN results
agree in being NaN, with payload and sign unspecified.

magetypes' `f32x8::exp2_midp`-style methods do not promise that: their polynomial
steps round twice on the scalar backend and strict WASM, and NaN input gives a
different answer per ISA. `portable` uses the same polynomials with one rounding
everywhere and NaN out for NaN in.

On x86 and NEON the midp kernels are at least as fast as those methods, and the lowp
kernels take 12-55% longer because they handle specials the methods don't. Where the
hardware can't fuse the price is software FMA: 2-5x on the x86-64 scalar tier (CPUs without
V3), 10-25x on strict WASM, 1.1-3.7x on relaxed WASM with an engine that fuses
([measurements](docs/DESIGN.md#what-it-costs)). Use the magetypes methods when you want
the fastest result per tier and don't need matching bits.

## Modules

| module | contents |
|---|---|
| `portable::kernels` | one token-generic `f32x8` function per op, to chain inside your own `#[attune]` body, plus `*_unchecked` forms for hot loops with known-valid inputs (18-38% less time on x86 and NEON) |
| `portable::slice` | `&mut [f32]` bulk ops |
| `portable::array` | `[f32; 8]` ops |
| `native` | reserved for kernels whose bits differ by tier; empty |

Each `slice` and `array` op is an `#[attune]` family: a dispatcher, `_v3_t`-style
proof wrappers, and `_v3`-style direct variants. Dispatch costs a cached check per
call, so pass whole buffers. To run several ops per element, chain `kernels` in
one `#[attune]` body. Values then stay in vector registers across the chain:

```rust
use archmage::prelude::*;
use magemath::portable::kernels;

#[attune(_*_t, define(f32x8))]
fn gamma(token: Token, a: [f32; 8]) -> [f32; 8] {
    let x = f32x8::from_array_t(token, a);
    kernels::exp2_midp(token, kernels::log2_midp(token, x) * f32x8::splat_t(token, 2.4))
        .to_array()
}

let out = attuned!(gamma([0.5; 8]));
assert_eq!(out[0], zenmath::scalar::pow_midp_f32(0.5, 2.4));
```

Ops: `sqrt`, `floor`, `ceil`, `roundeven`; `log2`, `exp2`, `ln`, `exp`, `log10`,
`pow`, `cbrt` at `_lowp` and `_midp`; `cbrt_midp_precise`, `sigmoid_midp`,
`silu_midp`. Special cases are tabled in the `portable` module docs. Accuracy per op
and input band is in [`docs/transcendentals.md`](../docs/transcendentals.md).

## Features

| feature | contents |
|---|---|
| `std` (default) | runtime CPU detection; without it, dispatch sees only compile-time features |
| `avx512` | `v4x` tier, which pulls magetypes' 512-bit backends |

The crate contains no `unsafe` (`#![forbid(unsafe_code)]`). Everything generated is
`#[inline]`, so a build compiles machine code only for the ops it calls.

## Testing

`tests/parity.rs` compares every compiled tier against the mirrors bit for bit,
with no tolerance. `tests/golden.rs` pins the mirrors across commits and
architectures, `tests/reference.rs` holds accuracy to the published maxima against
an f64 reference, and `tests/exhaustive.rs` covers all 2^32 inputs. Recipes:
`just test-magemath`, `test-magemath-wasm`, `test-magemath-no-fma`,
`test-magemath-i686`, `test-magemath-v3-build`, `magemath-exhaustive`, and
`magemath-golden-regen` after an intentional change.

A tier your CPU lacks fails the run rather than vanishing from it. Name it in
`MAGEMATH_SKIP_TIERS` (for example `MAGEMATH_SKIP_TIERS=v4x`) to skip it.
