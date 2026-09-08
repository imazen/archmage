+++
title = "Float / Integer"
weight = 1
+++

Choose the conversion's numerical contract before choosing the instruction.
`zenjpeg` quantization and zen color kernels depend on rounding, clamping, and
finite input ranges; those are algorithm decisions, not incidental details.

| Call | Contract |
|---|---|
| `v.to_i32()` | Truncates toward zero for in-range finite lanes; exceptional results vary by backend |
| `v.to_i32_saturating()` | Rust `as`-style truncation and saturation, NaN → 0 |
| `v.to_i32_round()` | Rounded conversion; consult ISA tables for rounding-mode and exceptional cases |
| `v.to_f32()` on integer vectors | Numeric conversion; large integers can lose precision |
| `v.bitcast_to_i32()` | Bit reinterpretation, not numeric conversion |

This reference check exercises values for which the distinction matters:

```rust
use archmage::prelude::*;
#[magetypes(define(f32x8), v3, neon, wasm128, scalar)]
fn convert_impl(token: Token, values: [f32; 8]) -> [i32; 8] {
    f32x8::from_array(token, values).to_i32_saturating().to_array()
}
pub fn convert(values: [f32; 8]) -> [i32; 8] {
    incant!(convert_impl(values), [v3, neon, wasm128, scalar])
}
let values = [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, 3e9, -3e9, 1.9, -1.9, 0.0];
assert_eq!(convert(values), values.map(|x| x as i32));
```

Bare x86 truncation can return `i32::MIN` for positive overflow and NaN; other
backends saturate with NaN → 0. `to_i32_saturating` repairs that divergence on
x86. See the exact [ISA fixup table](@/magetypes/isa-quirks.md) for instructions
and measured costs. Do not pay for a fixup unnecessarily when a proved input
domain makes the native operation equivalent, but document and test that proof.

Width-specific names such as `to_i32x8` remain aliases. They do not change the
exceptional-value contract. For quantization, also define tie handling and the
range before narrowing; saturating the final store cannot repair earlier overflow.

## Half-precision storage

`zenresize/src/simd/wide_kernels.rs` uses `F16Convert` for row/slice conversion.
Half-precision storage is a numeric format, not a `u16 as f32` cast. The u16
input holds IEEE binary16 bits.

```rust
use archmage::prelude::*;
use magetypes::simd::F16Convert;
#[magetypes(v3, neon, wasm128, scalar)]
fn decode_half_impl(token: Token, input: &[u16], output: &mut [f32]) {
    token.f16_to_f32_slice(input, output);
}
pub fn decode_half(input: &[u16], output: &mut [f32]) {
    incant!(decode_half_impl(input, output), [v3, neon, wasm128, scalar])
}
let mut output = [0.0; 3];
decode_half(&[0x0000, 0x3c00, 0xc000], &mut output);
assert_eq!(output, [0.0, 1.0, -2.0]);
```

The whole-buffer converters require equal lengths and can select a stronger
hardware conversion path once per buffer. They are an intentional exception to
“never detect inside a helper”: the unit of work is an entire slice. In-register
`i32x4::f16_to_f32` and `f32x4::to_f16` do not perform that whole-buffer dispatch.
A V4 holder can extract `.v3()` to reach the supported slice converter surface.
NaN bit handling and hardware availability vary with target and Rust version;
consult the [F16Convert API](https://docs.rs/magetypes/latest/magetypes/simd/trait.F16Convert.html)
and exhaustive conversion tests for that contract.
