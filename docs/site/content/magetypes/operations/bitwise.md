+++
title = "Integer & Bitwise"
weight = 4
+++

Integer types (`i32x8`, `u8x16`, `i16x8`, etc.) support arithmetic, bitwise operations, and shifts.

## Arithmetic

```rust
let a = i32x8::splat(token, 10);
let b = i32x8::splat(token, 3);

let sum  = a + b;   // [13; 8]
let diff = a - b;   // [7; 8]
let prod = a * b;   // [30; 8]
```

## Bitwise Operations

**Use methods for portable code.** Methods work identically on all platforms:

```rust
let and = a.and(b);
let or  = a.or(b);
let xor = a.xor(b);
let not = a.not();
```

On x86-64, bitwise trait operators (`&`, `|`, `^`, `!`) are also implemented:

```rust
// x86-64 only — these are operator overloads, not available on ARM/WASM
let and = a & b;
let or  = a | b;
let xor = a ^ b;
let not = !a;
```

On ARM and WASM, use the `.and()`, `.or()`, `.xor()`, `.not()` methods. This is a known [behavioral difference](@/magetypes/cross-platform/differences.md) — the methods are the portable choice.

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
use archmage::{Desktop64, SimdToken, arcane};
use magetypes::simd::{u8x16, i16x8};

#[arcane]
fn threshold(token: Desktop64, pixels: &mut [u8; 16], cutoff: u8) {
    let v = u8x16::from_array(token, *pixels);
    let threshold = u8x16::splat(token, cutoff);
    let white = u8x16::splat(token, 255);
    let black = u8x16::zero(token);

    let mask = v.simd_gt(threshold);
    let result = mask.blend(white, black);
    result.store(pixels);
}
```
