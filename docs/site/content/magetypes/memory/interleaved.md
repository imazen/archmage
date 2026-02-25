+++
title = "Interleaved Data"
weight = 3
+++

Image pixels, audio samples, and sensor data are often interleaved: RGBARGBA... or LRLRLR... SIMD works best on separate channels, so you deinterleave before processing and reinterleave after.

## 4-Channel Deinterleave (RGBA)

Separate interleaved RGBA data into individual channels:

```rust
// Input: 4 f32x4 vectors of interleaved RGBA data
let input = [
    f32x4::from_array(token, [r0, g0, b0, a0]),
    f32x4::from_array(token, [r1, g1, b1, a1]),
    f32x4::from_array(token, [r2, g2, b2, a2]),
    f32x4::from_array(token, [r3, g3, b3, a3]),
];

let [r, g, b, a] = f32x4::deinterleave_4ch(input);
// r = [R0, R1, R2, R3]
// g = [G0, G1, G2, G3]
// b = [B0, B1, B2, B3]
// a = [A0, A1, A2, A3]
```

## Process Channels

Once deinterleaved, operations on each channel are straightforward:

```rust
// Brighten the red channel
let r_bright = r + f32x4::splat(token, 0.1);

// Apply gamma to green
let g_gamma = g.pow_midp(1.0 / 2.2);
```

## 4-Channel Reinterleave

Pack channels back into interleaved format:

```rust
let output = f32x4::interleave_4ch([r_bright, g_gamma, b, a]);
// output[0..4] contain RGBARGBA... interleaved data
```

## Low/High Interleave

For simpler cases, interleave two vectors element by element:

```rust
let a = f32x4::from_array(token, [1.0, 2.0, 3.0, 4.0]);
let b = f32x4::from_array(token, [5.0, 6.0, 7.0, 8.0]);

let lo = a.interleave_lo(b);  // [1.0, 5.0, 2.0, 6.0]
let hi = a.interleave_hi(b);  // [3.0, 7.0, 4.0, 8.0]
```

`interleave_lo` and `interleave_hi` are available on `f32x4` across all platforms. They interleave the lower or upper halves of two vectors.

## Transpose

For matrix-like data, transpose 4x4 blocks:

```rust
let mut rows = [row0, row1, row2, row3];
f32x4::transpose_4x4(&mut rows);
// rows are now transposed in-place

// Or use the non-mutating version:
let [r0, r1, r2, r3] = f32x4::transpose_4x4_copy([row0, row1, row2, row3]);
```

## Platform Notes

- **x86-64**: Uses `vunpcklps`/`vunpckhps` and shuffle instructions
- **AArch64**: Uses native `vzip1q`/`vzip2q`
- **WASM**: Uses `i32x4_shuffle`

The API is identical across platforms. Performance is comparable.
