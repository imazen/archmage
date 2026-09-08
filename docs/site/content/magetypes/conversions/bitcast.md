+++
title = "Bitcast"
weight = 3
+++

A bitcast preserves bits; a numeric conversion preserves a numeric value as far
as the destination can represent it. They are not interchangeable. For example,
bitcasting `1.0f32` to i32 yields its IEEE bit encoding, not integer one.

This reference exercise checks the distinction within a complete feature context.
Integer/float bit manipulation is used by zen color and approximation kernels;
this roundtrip is an API test rather than a production algorithm.

```rust
use archmage::prelude::*;
#[magetypes(define(f32x8), v3, neon, wasm128, scalar)]
fn bits_impl(token: Token, input: [f32; 8]) -> ([i32; 8], [f32; 8]) {
    let v = f32x8::from_array(token, input);
    let bits = v.bitcast_to_i32();
    (bits.to_array(), bits.bitcast_to_f32().to_array())
}
pub fn bits(input: [f32; 8]) -> ([i32; 8], [f32; 8]) {
    incant!(bits_impl(input), [v3, neon, wasm128, scalar])
}
let (encoded, roundtrip) = bits([1.0; 8]);
assert_eq!(encoded, [1.0f32.to_bits() as i32; 8]);
assert_eq!(roundtrip, [1.0; 8]);
```

Use `bitcast_to_i32` / `bitcast_to_f32` for value casts. Older width-specific
names such as `bitcast_i32x8` remain compatibility aliases. Signed/unsigned byte
casts have shape-specific names; follow their method reference.

Reference casts preserve borrowing and require compatible size/alignment and
valid bit patterns; use only the library's provided methods. They do not make
arbitrary user structs safe to reinterpret. Prefer value operations unless a
measured caller needs a borrowed view.
