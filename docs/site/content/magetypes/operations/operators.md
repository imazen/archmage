+++
title = "Arithmetic & Comparisons"
weight = 2
+++

Magetypes supports natural arithmetic inside the feature-enabled kernels shown
in [gain](@/magetypes/examples/generic-kernels.md) and
[SrcOver blending](@/magetypes/examples/pixel-blending.md).

| Operation | Calls / syntax | Notes |
|---|---|---|
| Arithmetic | `a + b`, `a - b`, `a * b`, `a / b`, `-a` | Availability depends on element type |
| Assignment | `+=`, `-=`, `*=` | Same vector semantics |
| Fused form | `a.mul_add(b, c)`, `a.mul_sub(b, c)` | FMA availability and fallback rounding differ by ISA |
| Comparisons | `simd_eq`, `simd_ne`, `simd_lt`, `simd_le`, `simd_gt`, `simd_ge` | Produce vector lane masks, not Rust scalar booleans |
| Selection | `f32x8::blend(mask, yes, no)` | Use comparison-generated canonical masks |
| Min/max | `min`, `max` | NaN and signed-zero behavior needs the ISA contract |
| Magnitude | `abs`, `sqrt` | `sqrt` is rounded floating-point square root, not exact real arithmetic |

This reference exercise shows compare/select in the same generated call-chain
shape used by zen color kernels. It is not a complete tone-mapping algorithm.

```rust
use archmage::prelude::*;
#[magetypes(define(f32x8), v3, neon, wasm128, scalar)]
fn positive_impl(token: Token, values: [f32; 8]) -> [f32; 8] {
    let v = f32x8::from_array(token, values);
    let zero = f32x8::zero(token);
    f32x8::blend(v.simd_gt(zero), v, zero).to_array()
}
pub fn positive(values: [f32; 8]) -> [f32; 8] {
    incant!(positive_impl(values), [v3, neon, wasm128, scalar])
}
assert_eq!(positive([-1.0, 2.0, 0.0, 3.0, -4.0, 5.0, 6.0, 7.0]),
           [0.0, 2.0, 0.0, 3.0, 0.0, 5.0, 6.0, 7.0]);
```

Do not assume all boolean identities hold for NaNs: current `simd_ne` semantics
vary. Do not infer bit-identical results from replacing multiply-plus-add with
`mul_add`. The [ISA tables](@/magetypes/isa-quirks.md) specify these differences
and the fixups the library does apply.
