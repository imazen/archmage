+++
title = "Integer & Bitwise"
weight = 4
+++

Integer types (`i32x8<T>`, `u8x16<T>`, `i16x8<T>`, etc.) support arithmetic, bitwise operations, and shifts.

## Arithmetic

```rust
use magetypes::simd::{generic::i32x8, backends::I32x8Backend};

#[inline(always)]
fn arithmetic_example<T: I32x8Backend>(token: T) {
    let a = i32x8::<T>::splat(token, 10);
    let b = i32x8::<T>::splat(token, 3);

    let sum  = a + b;   // [13; 8]
    let diff = a - b;   // [7; 8]
    let prod = a * b;   // [30; 8]
}
```

## Bitwise Operations

Generic integer types support the standard Rust bitwise operators `&`, `|`, `^`, and the `.not()` method. These work identically on all platforms:

```rust
use magetypes::simd::{generic::i32x8, backends::I32x8Backend};

#[inline(always)]
fn bitwise_example<T: I32x8Backend>(token: T) {
    let a = i32x8::<T>::splat(token, 0b1010);
    let b = i32x8::<T>::splat(token, 0b1100);

    let and = a & b;        // 0b1000
    let or  = a | b;        // 0b1110
    let xor = a ^ b;        // 0b0110
    let not = a.not();      // bitwise NOT
}
```

## Shifts

Shifts use const generics for the shift amount:

```rust
let shl = a.shl::<2>();               // Shift left by 2 bits
let shr = a.shr::<1>();               // Shift right by 1 bit
let shr_a = a.shr_arithmetic::<1>();  // Sign-extending shift right
```

`shr` does a logical (zero-filling) shift on x86-64. On ARM and WASM, `shr` for signed types does an arithmetic (sign-extending) shift. If you need portable sign-extending behavior, use `shr_arithmetic` explicitly. See [Behavioral Differences](@/magetypes/cross-platform/differences.md).

## Packing and Extension

See [Width Conversions](@/magetypes/conversions/width.md) for `pack_u8`, `pack_i8`, `extend_i32`, etc.

## Example: Byte Threshold

```rust
use archmage::{X64V3Token, SimdToken, arcane};
use magetypes::simd::{
    generic::{u8x16, i16x8},
    backends::U8x16Backend,
};

#[arcane]
fn threshold(token: X64V3Token, pixels: &mut [u8; 16], cutoff: u8) {
    let v = u8x16::<X64V3Token>::from_array(token, *pixels);
    let threshold = u8x16::<X64V3Token>::splat(token, cutoff);
    let white = u8x16::<X64V3Token>::splat(token, 255);
    let black = u8x16::<X64V3Token>::zero(token);

    let mask = v.simd_gt(threshold);
    let result = mask.blend(white, black);
    result.store(pixels);
}
```
