# Type Conversions

magetypes provides conversions between SIMD types and between SIMD and scalar types.

## Float ↔ Integer Conversions

### Float to Integer

```rust
let floats = f32x8::from_array(token, [1.5, 2.7, -3.2, 4.0, 5.9, 6.1, 7.0, 8.5]);

// Truncate toward zero (like `as i32`)
let ints = floats.to_i32x8();  // [1, 2, -3, 4, 5, 6, 7, 8]

// Round to nearest
let rounded = floats.to_i32x8_round();  // [2, 3, -3, 4, 6, 6, 7, 8]
```

### Integer to Float

```rust
let ints = i32x8::from_array(token, [1, 2, 3, 4, 5, 6, 7, 8]);
let floats = ints.to_f32x8();  // [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
```

## Width Conversions

### Narrowing (Wider → Narrower)

```rust
// f64x4 → f32x4 (lose precision)
let doubles = f64x4::from_array(token, [1.0, 2.0, 3.0, 4.0]);
let floats = doubles.to_f32x4();

// i32x8 → i16x8 (pack with saturation)
let wide = i32x8::from_array(token, [1, 2, 3, 4, 5, 6, 7, 8]);
let narrow = wide.pack_i16();
```

### Widening (Narrower → Wider)

```rust
// f32x4 → f64x4 (extend precision, lower half)
let floats = f32x4::from_array(token, [1.0, 2.0, 3.0, 4.0]);
let doubles = floats.to_f64x4_low();  // Converts first 2 elements

// i16x8 → i32x8 (sign-extend lower half)
let narrow = i16x8::from_array(token, [1, 2, 3, 4, 5, 6, 7, 8]);
let wide = narrow.extend_i32_low();  // [1, 2, 3, 4]
```

## Bitcast (Reinterpret)

Reinterpret bits as a different type (same size):

```rust
// f32x8 → i32x8 (view float bits as integers)
let floats = f32x8::splat(token, 1.0);
let bits = floats.bitcast_i32x8();

// i32x8 → f32x8
let ints = i32x8::splat(token, 0x3f800000);  // IEEE 754 for 1.0
let floats = ints.bitcast_f32x8();
```

**Warning**: Bitcast doesn't convert values—it reinterprets the raw bits.

## Signed ↔ Unsigned

```rust
// i32x8 → u32x8 (reinterpret, no conversion)
let signed = i32x8::from_array(token, [-1, 0, 1, 2, 3, 4, 5, 6]);
let unsigned = signed.to_u32x8();  // [0xFFFFFFFF, 0, 1, 2, 3, 4, 5, 6]

// u32x8 → i32x8
let unsigned = u32x8::splat(token, 0xFFFFFFFF);
let signed = unsigned.to_i32x8();  // [-1; 8]
```

## Lane Extraction and Insertion

```rust
// Extract single lane
let v = f32x8::from_array(token, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
let third = v.extract::<2>();  // 3.0

// Insert single lane
let v = v.insert::<2>(99.0);  // [1.0, 2.0, 99.0, 4.0, 5.0, 6.0, 7.0, 8.0]
```

## Half-Width Operations

Split or combine vectors:

```rust
// Split f32x8 into two f32x4
let full = f32x8::from_array(token, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
let (low, high) = full.split();
// low  = [1.0, 2.0, 3.0, 4.0]
// high = [5.0, 6.0, 7.0, 8.0]

// Combine two f32x4 into f32x8
let combined = f32x8::from_halves(token, low, high);
```

## Slice Casting (Token-Gated)

magetypes provides safe, token-gated slice casting as an alternative to `bytemuck`:

```rust
// Cast aligned &[f32] to &[f32x8]
let data: &[f32] = &[1.0; 64];
if let Some(chunks) = f32x8::cast_slice(token, data) {
    // chunks: &[f32x8] with 8 elements
    for chunk in chunks {
        // ...
    }
}

// Mutable version
let data: &mut [f32] = &mut [0.0; 64];
if let Some(chunks) = f32x8::cast_slice_mut(token, data) {
    // chunks: &mut [f32x8]
}
```

**Why not `bytemuck`?** Implementing `Pod`/`Zeroable` would let users bypass token-gated construction:

```rust
// bytemuck would allow this (BAD):
let v: f32x8 = bytemuck::Zeroable::zeroed();  // No token check!

// magetypes requires token (GOOD):
let v = f32x8::zero(token);  // Token proves CPU support
```

The token-gated `cast_slice` returns `None` if alignment or length is wrong—no UB possible.

## Byte-Level Access

View vectors as bytes (no token needed—you already have the vector):

```rust
let v = f32x8::splat(token, 1.0);

// View as bytes (zero-cost)
let bytes: &[u8; 32] = v.as_bytes();

// Mutable view
let mut v = f32x8::splat(token, 0.0);
let bytes: &mut [u8; 32] = v.as_bytes_mut();

// Create from bytes (token-gated)
let bytes = [0u8; 32];
let v = f32x8::from_bytes(token, &bytes);
```

## Conversion Example: Image Processing

```rust
#[arcane]
fn brighten(token: Desktop64, pixels: &mut [u8]) {
    // Process 32 bytes at a time
    for chunk in pixels.chunks_exact_mut(32) {
        let v = u8x32::from_slice(token, chunk);

        // Convert to wider type for arithmetic
        let (lo, hi) = v.split();
        let lo_wide = lo.extend_u16_low();
        let hi_wide = hi.extend_u16_low();

        // Add brightness (with saturation)
        let brightness = u16x16::splat(token, 20);
        let lo_bright = lo_wide.saturating_add(brightness);
        let hi_bright = hi_wide.saturating_add(brightness);

        // Pack back to u8 with saturation
        let result = u8x32::from_halves(
            token,
            lo_bright.pack_u8_saturate(),
            hi_bright.pack_u8_saturate()
        );

        result.store_slice(chunk);
    }
}
```
