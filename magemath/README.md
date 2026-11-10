# magemath

Vector and fixed-array math on `magetypes`, with portable and relaxed surfaces.

**Status: draft, unpublished.** Approximation families support `f32x4` and
`f32x8`, plus `f32x16` with `w512`. f64 vector transcendentals remain outside
this implementation. See [the migration inventory](docs/MIGRATION.md).

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

The correctly rounded `portable::array::{exp_f32x8,ln_f32x8,log2_f32x8,pow_f32x8}`
and `portable::slice::{exp_f32,ln_f32,log2_f32,pow_f32}` use CORE-MATH algorithms.
`pow` accepts one exponent per input. They handle subnormals and IEEE special
values, assuming nearest-even rounding and gradual underflow. Common paths
use four binary64 lanes; difficult lanes use the complete `zenmath` fallback.
See [validation and performance](../benchmarks/core_math_vector_2026-11-06/README.md).

magetypes' `f32x8::exp2_midp`-style methods do not promise that: their polynomial
steps round twice on the scalar backend and strict WASM, and NaN input gives a
different answer per ISA. The lowp/midp `portable` functions use the same polynomials with one rounding
everywhere and NaN out for NaN in.

`relaxed` exposes the same operations and special-value handling, with backend
rounding: fused on V3/V4x and NEON, separate multiply/add on the scalar backend
and strict WASM, and the engine's choice on relaxed WASM. NaN inputs still return
NaN. Both surfaces have the same documented accuracy bounds.

`unfused` is a draft experiment with separate multiply/add on every tier. Its
final name and inclusion are pending the portable-rounding decision. The
[design record](docs/DESIGN.md) tracks the contracts and measurements.

## Modules

| module | contents |
|---|---|
| `portable::kernels` | one token-generic `f32x8` function per op, to chain inside your own `#[attune]` body, plus `*_unchecked` forms for hot loops with known-valid inputs (18-38% less time on x86 and NEON) |
| `portable::slice` | `&mut [f32]` bulk ops |
| `portable::array` | `[f32; 8]` ops |
| `relaxed::{kernels,slice,array}` | the same API with backend-dependent rounding |
| `unfused::{kernels,slice,array}` | prototype with two roundings on every tier |
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

## Draft compile-time power behavior

`PowBehavior` selects per-operation power semantics through validated const bits.
Existing power calls keep their results. Correctly rounded, Midp and Lowp policies
share scalar/packed behavior; the near-one-ULP prototype is not a supported preset.
See [power behavior](docs/POW_BEHAVIOR.md) for mode order, preconditions and examples.

## Math family features

Default features preserve the existing APIs and bits. `correct-rounding` exposes
the precise unsuffixed functions, `normal` exposes `magemath::normal::{array,slice}`
using the same precise implementation today, and `configurable` exposes named
approximations and power policies. Disabling defaults leaves primitives until a
family is selected. See [the family and dependency mapping](docs/FEATURES.md).

The [transcendental migration inventory](docs/MIGRATION.md) covers all 35
legacy names at widths four/eight and optional sixteen, including exact legacy
and existing portable/relaxed/unfused contract choices.
