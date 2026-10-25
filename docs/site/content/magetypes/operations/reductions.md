+++
title = "Reductions"
weight = 3
+++

`reduce_add`, `reduce_min`, and `reduce_max` collapse vector lanes to a scalar.
The [generic luma example](@/magetypes/dispatch/types-and-dispatch.md) shows a
complete reduction loop, f64 outer accumulator, and scalar tail.

Float addition order can differ by backend and vector width. Cancellation can
make the discrepancy large; it is not always a tiny fixed error. Define the
acceptable domain and tolerance for the algorithm, and test long inputs as well
as one vector. Min/max reductions also need an explicit NaN policy.

For comparison-generated integer masks:

| Method | Result |
|---|---|
| `any_true()` | At least one true lane |
| `all_true()` | Every lane true |
| `bitmask()` | Packed lane sign bits in lane order |

The return integer type of `bitmask()` is shape-specific; an eight-lane vector
need not return `u8`. Use the API's actual result type. The fused predicate
scans in `zenpixels-convert/src/scan.rs` and `zenpng/src/simd/scan.rs` combine
violations across blocks before reducing, avoiding a serial scalar decision for
every small vector. Const generics remove checks no longer needed by a scan.

Terminal byte reductions such as sum and absolute-difference sums avoid
materializing intermediate widened vectors when the caller only needs a scalar.
Use widening when later vector arithmetic actually needs those lanes.

## Multiply-add in loops: which form is faster

Whether a fused multiply-add or a separate multiply and add runs faster
depends on where the loop's dependency chain runs. Measured on V3 code
(`f32x8`, `f64x4`, scalar `f64`), interleaved in one binary on an Intel Core
Ultra 7 265K and an AMD Ryzen 9 7900X; ratios are separate time ÷ fused time
([record](https://github.com/imazen/archmage/blob/main/benchmarks/fused_vs_separate_2026-10-25.md)):

| Loop shape | Example | Separate ÷ fused | Faster |
|---|---|---:|---|
| Chain through the multiplicand | Horner polynomial `p = p * x + c` | 1.50–1.61× | fused |
| One accumulator | dot product `acc = a * b + acc` | 0.49–0.58× (265K), 0.78–0.80× (7900X) | separate |
| Four vector accumulators | `acc[k] = a * b + acc[k]` | 1.02× | either |
| Four scalar `f64` accumulators | same, on scalars | 0.58× (265K), 0.84× (7900X) | separate |
| Independent results | `out = a * b + c` | 0.81–1.04× | either |

In a Horner chain each step needs the previous result as a factor, so both
halves of a separate multiply-add sit on the chain, and one fused instruction
is shorter. In an accumulation the multiplies are independent of each other;
only the add depends on the previous step, and an add has lower latency than
a fused multiply-add on both cores. More accumulators hide that latency until
the two forms run at the same speed.

Choosing a form:

- **`a * b + acc`** for reductions with one or a few accumulators. Rust never
  contracts it into an FMA, so it rounds twice and gives the same bits on
  every backend.
- **`mul_add_relaxed`** for Horner-style chains when speed matters most: one
  FMA on x86 v3/v4 and NEON, a multiply then an add on the scalar backend and
  strict WASM, so it is fast everywhere but its bits differ by backend.
- **`mul_add_fused`** when results must match bit for bit and round once. It is
  the same FMA on x86 v3/v4 and NEON, but the scalar backend without FMA and
  strict WASM compute it in software, 1.7–27× (scalar) and 7.9–13.5× (strict
  WASM) the time of `a * b + c`; relaxed WASM uses the engine's madd when the
  engine fuses
  ([costs](@/magetypes/isa-quirks.md#fused-arithmetic-and-reductions)).
- Several accumulators help more than either form: four vector accumulators
  made the choice irrelevant above. Their partial sums associate differently,
  so they change the result's rounding too.

```rust
use archmage::prelude::*;

/// One accumulator: the chain runs through the addend, so a separate
/// multiply and add is the faster form.
#[magetypes(define(f32x8), v3, neon, wasm128, scalar)]
fn dot_one_impl(token: Token, a: &[f32], b: &[f32]) -> f32 {
    let (ac, at) = a.as_chunks::<8>();
    let (bc, bt) = b.as_chunks::<8>();
    let mut acc = f32x8::zero_t(token);
    for (x, y) in ac.iter().zip(bc) {
        acc = f32x8::load_t(token, x) * f32x8::load_t(token, y) + acc;
    }
    let mut sum = acc.reduce_add();
    for (x, y) in at.iter().zip(bt) {
        sum += x * y;
    }
    sum
}

/// Four independent accumulators hide the add latency; either form is then
/// equally fast.
#[magetypes(define(f32x8), v3, neon, wasm128, scalar)]
fn dot_four_impl(token: Token, a: &[f32], b: &[f32]) -> f32 {
    let (ac, at) = a.as_chunks::<32>();
    let (bc, bt) = b.as_chunks::<32>();
    let mut acc = [f32x8::zero_t(token); 4];
    for (x, y) in ac.iter().zip(bc) {
        let (x, y) = (x.as_chunks::<8>().0, y.as_chunks::<8>().0);
        for k in 0..4 {
            acc[k] = f32x8::load_t(token, &x[k]) * f32x8::load_t(token, &y[k]) + acc[k];
        }
    }
    let mut sum = ((acc[0] + acc[1]) + (acc[2] + acc[3])).reduce_add();
    for (x, y) in at.iter().zip(bt) {
        sum += x * y;
    }
    sum
}

/// Horner chain: each step multiplies the previous result, so one fused
/// instruction per step is the faster form. `mul_add_relaxed` fuses where the
/// hardware does; use `mul_add_fused` if every backend must give the same bits.
#[magetypes(define(f32x8), v3, neon, wasm128, scalar)]
fn horner_impl(token: Token, x: &[f32; 8], coefficients: &[f32]) -> [f32; 8] {
    let x = f32x8::load_t(token, x);
    let mut p = f32x8::zero_t(token);
    for &c in coefficients {
        p = p.mul_add_relaxed(x, f32x8::splat_t(token, c));
    }
    p.to_array()
}

pub fn dot_one(a: &[f32], b: &[f32]) -> f32 {
    incant!(dot_one_impl(a, b), [v3, neon, wasm128, scalar])
}
pub fn dot_four(a: &[f32], b: &[f32]) -> f32 {
    incant!(dot_four_impl(a, b), [v3, neon, wasm128, scalar])
}
pub fn horner(x: &[f32; 8], coefficients: &[f32]) -> [f32; 8] {
    incant!(horner_impl(x, coefficients), [v3, neon, wasm128, scalar])
}

// Small integers keep every product and sum exact, so all forms agree here.
let a: Vec<f32> = (0..100).map(|i| (i % 7) as f32).collect();
let b: Vec<f32> = (0..100).map(|i| (i % 5) as f32).collect();
let expected: f32 = a.iter().zip(&b).map(|(x, y)| x * y).sum();
assert_eq!(dot_one(&a, &b), expected);
assert_eq!(dot_four(&a, &b), expected);
// 2x^2 + 3x + 1
let p = horner(&[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], &[2.0, 3.0, 1.0]);
assert_eq!(p, [1.0, 6.0, 15.0, 28.0, 45.0, 66.0, 91.0, 120.0]);
```
