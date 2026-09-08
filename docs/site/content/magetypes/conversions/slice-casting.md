+++
title = "Slice Casting"
weight = 4
+++

Prefer array chunks plus `load`/`store` for ordinary image rows. They accept
normal scalar alignment and make tails explicit, as in
[zenfilters gain](@/magetypes/examples/generic-kernels.md).

`cast_slice(token, slice)` and `cast_slice_mut` instead borrow a slice of vector
objects. They return `None` when the scalar count or alignment is incompatible.
Alignment can differ by backend, so a cast that works for one tier can fail for
another. Do not unwrap merely because the input length is a multiple of lanes.

This is a reference-only API exercise; the reviewed zen image loops mostly use
array chunks and value loads rather than vector-reference slice casts.

```rust
use archmage::prelude::*;
#[magetypes(define(f32x8), v3, neon, wasm128, scalar)]
fn roundtrip_impl(token: Token) -> [f32; 8] {
    let v = f32x8::splat(token, 1.0);
    let bytes = *v.as_bytes();
    f32x8::from_bytes(token, &bytes).to_array()
}
pub fn roundtrip() -> [f32; 8] {
    incant!(roundtrip_impl(), [v3, neon, wasm128, scalar])
}
assert_eq!(roundtrip(), [1.0; 8]);
```

`as_bytes` / `as_bytes_mut` borrow the native representation; `from_bytes` and
`from_bytes_owned` construct a value with a token. These are native-endian bit
views, not a portable serialized format. Arbitrary float bits include NaNs.

Magetypes vectors do not expose unrestricted `Pod`/`Zeroable` construction,
which would bypass the token requirement. This does not prohibit using bytemuck
for ordinary scalar pixel buffers under that crate's own validity rules.
