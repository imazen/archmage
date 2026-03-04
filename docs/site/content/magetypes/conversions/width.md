+++
title = "Width Conversions"
weight = 2
+++

Change the width of vector elements: narrow wide values to fit in fewer bits, or widen narrow values for more headroom.

## Narrowing (Wider to Narrower)

### Integer packing with saturation

```rust
use magetypes::simd::{
    generic::{i16x8, i32x4},
    backends::{I16x8Backend, I32x4Backend},
};

#[inline(always)]
fn pack_examples<T: I16x8Backend>(token: T) {
    // i16x8 -> i8x16 (two i16x8 vectors packed into one i8x16, clamped to i8 range)
    let a = i16x8::<T>::from_array(token, [1, 2, 3, 4, 5, 6, 7, 8]);
    let b = i16x8::<T>::from_array(token, [9, 10, 11, 12, 13, 14, 15, 16]);
    let narrow = a.pack_i8(b);   // [1, 2, ..., 16] as i8x16

    // i16x8 -> u8x16 (unsigned, saturated)
    let narrow = a.pack_u8(b);   // Values clamped to 0..255
}

#[inline(always)]
fn pack_i32<T: I32x4Backend>(token: T) {
    // i32x4 -> i16x8
    let c = i32x4::<T>::from_array(token, [1, 2, 3, 4]);
    let d = i32x4::<T>::from_array(token, [5, 6, 7, 8]);
    let narrow = c.pack_i16(d);  // [1, 2, 3, 4, 5, 6, 7, 8] as i16x8
}
```

Saturation means values outside the target range are clamped to the target type's min/max rather than wrapping.

## Widening (Narrower to Wider)

### Integer extension

```rust
use magetypes::simd::{
    generic::i16x8,
    backends::I16x8Backend,
};

#[inline(always)]
fn extend_examples<T: I16x8Backend>(token: T) {
    // i16x8 -> two i32x4 halves
    let narrow = i16x8::<T>::from_array(token, [1, 2, 3, 4, 5, 6, 7, 8]);
    let (lo, hi) = narrow.extend_i32();   // lo = [1, 2, 3, 4], hi = [5, 6, 7, 8]

    // Or just the low half
    let lo = narrow.extend_lo_i32();      // [1, 2, 3, 4] as i32x4
}
```

`extend_i32()` returns a tuple of `(lo, hi)` — the full vector split into two halves at the wider element size. `extend_lo_i32()` returns only the lower half.

### Float conversion from integer

```rust
// given token: T where T: I16x8Backend
// i16x8 -> f32x4 (lower half, via integer extension + float conversion)
let narrow = i16x8::<T>::from_array(token, [1, 2, 3, 4, 5, 6, 7, 8]);
let floats = narrow.extend_lo_f32();  // [1.0, 2.0, 3.0, 4.0] as f32x4
```

## Float-Integer Conversions

See [Float / Integer](@/magetypes/conversions/float-int.md) for `to_i32x4()`, `to_i32x4_round()`, and `to_f32x4()`.

## Example: Image Brightening

Widening to a larger type for arithmetic, then narrowing back:

```rust
use archmage::{arcane, SimdToken};
use magetypes::simd::{
    generic::i16x8,
    backends::I16x8Backend,
};

#[arcane(import_intrinsics)]
fn brighten<T: I16x8Backend>(token: T, pixels: &[u8; 16], amount: i16) -> [u8; 16] {
    let v = i16x8::<T>::from_array(token, [
        pixels[0] as i16, pixels[1] as i16, pixels[2] as i16, pixels[3] as i16,
        pixels[4] as i16, pixels[5] as i16, pixels[6] as i16, pixels[7] as i16,
    ]);
    let v2 = i16x8::<T>::from_array(token, [
        pixels[8] as i16,  pixels[9] as i16,  pixels[10] as i16, pixels[11] as i16,
        pixels[12] as i16, pixels[13] as i16, pixels[14] as i16, pixels[15] as i16,
    ]);

    let brightness = i16x8::<T>::splat(token, amount);
    let lo_bright = v + brightness;
    let hi_bright = v2 + brightness;

    // Pack back to u8 with saturation (clamps to 0..255)
    let result = lo_bright.pack_u8(hi_bright);
    result.to_array()
}
```
