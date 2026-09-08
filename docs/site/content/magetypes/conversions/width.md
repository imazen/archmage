+++
title = "Width Conversions"
weight = 2
+++

Widen lanes before arithmetic that needs extra range. Narrow with saturation
only when clamping is the intended output contract. This is central to byte
filters and codec quantization; [zenresize's convolution](@/magetypes/examples/convolution.md)
provides the surrounding row/strip design.

The following API exercise preserves lane order through unsigned widening and
signed-to-unsigned saturating narrowing. It deliberately uses a bounded input
range so the intermediate i16 addition cannot overflow.

```rust
use archmage::prelude::*;
#[magetypes(define(u8x16, i16x8), v3, neon, wasm128, scalar)]
fn brighten_impl(token: Token, input: [u8; 16], amount: u8) -> [u8; 16] {
    let bytes = u8x16::from_array(token, input);
    let lo = bytes.widen_low().bitcast_i16x8();
    let hi = bytes.widen_high().bitcast_i16x8();
    let offset = i16x8::splat(token, i16::from(amount));
    (lo + offset).narrow_saturating_u8(hi + offset).to_array()
}
pub fn brighten(input: [u8; 16], amount: u8) -> [u8; 16] {
    incant!(brighten_impl(input, amount), [v3, neon, wasm128, scalar])
}
let input = [0, 1, 2, 3, 4, 5, 6, 7, 240, 241, 242, 243, 252, 253, 254, 255];
assert_eq!(brighten(input, 10), input.map(|v| v.saturating_add(10)));
```

Unsigned input widens to values in 0..255, which can be reinterpreted as positive
i16. Adding an unsigned byte gives 0..510, safely within i16. The final narrowing
clamps to 0..255. This proof would fail for an unrestricted i16 offset.

| Method | Lane arrangement |
|---|---|
| `widen_low()` / `widen_high()` | Low/high input halves, preserving order; signed input sign-extends |
| `narrow_saturating_i8(high)` / `narrow_saturating_u8(high)` | All clamped lanes of self, then all clamped lanes of high |
| Corresponding i32→i16 / u16 methods | Same concatenate-and-saturate model |
| `pairwise_widen_add()` | Adds adjacent source lanes into wider result lanes |

A pairwise sum is not a dot product or a terminal reduction. Use it when the
next stage needs adjacent-pair vector sums. ISA lane-local packing instructions
may need a shuffle to satisfy the public whole-vector ordering; that fixup is
part of the [ISA contract](@/magetypes/isa-quirks.md), not optional overhead.
