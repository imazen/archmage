+++
title = "Bitcast"
weight = 3
+++

Bitcast reinterprets the raw bits of a vector as a different type without converting the values. The total bit width must match.

## Float / Integer Bitcast

```rust
// View float bits as integers
let floats = f32x8::splat(token, 1.0);
let bits = floats.bitcast_i32x8();
// Each lane is 0x3f800000 (IEEE 754 representation of 1.0)

// View integer bits as floats
let ints = i32x8::splat(token, 0x3f800000);
let floats = ints.bitcast_f32x8();
// Each lane is 1.0
```

Bitcast does not convert values. `1.0f32` as bits is `0x3f800000`; bitcasting that integer back gives `1.0f32`. But bitcasting an integer `1` to float gives `1.4e-45` (the IEEE 754 float with that bit pattern), not `1.0`.

## Signed / Unsigned

Reinterpret between signed and unsigned with the same element width:

```rust
// i32x8 -> u32x8 (no conversion, just reinterpretation)
let signed = i32x8::from_array(token, [-1, 0, 1, 2, 3, 4, 5, 6]);
let unsigned = signed.bitcast_u32x8();
// [0xFFFFFFFF, 0, 1, 2, 3, 4, 5, 6]

// u32x8 -> i32x8
let unsigned = u32x8::splat(token, 0xFFFFFFFF);
let signed = unsigned.bitcast_i32x8();
// [-1; 8]
```

This is the same as `as u32` or `as i32` in Rust — no data changes, just the type's interpretation of the bit pattern.

## When to Use Bitcast

Bitcasts are common in SIMD programming for:

- **Bit manipulation of floats** — extract exponents, manipulate sign bits, fast absolute value via AND with a mask
- **Type punning between same-width types** — view `f32x8` as `i32x8` for integer comparisons
- **Implementing higher-level operations** — many SIMD algorithms mix float and integer views of the same data
