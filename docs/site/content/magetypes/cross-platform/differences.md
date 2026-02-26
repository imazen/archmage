+++
title = "Behavioral Differences"
weight = 2
+++

Magetypes maintains full API parity across x86-64, AArch64, and WASM — the same methods exist everywhere. But a few operations have different semantics between architectures. These are not bugs; they reflect real hardware differences.

## Bitwise Operators

| | x86-64 | AArch64 | WASM |
|---|--------|---------|------|
| `a & b`, `a \| b`, `a ^ b`, `!a` | Trait impls (operators work) | **Methods only** | **Methods only** |
| `a.and(b)`, `a.or(b)`, `a.xor(b)`, `a.not()` | Works | Works | Works |

**Portable choice:** Always use `.and()`, `.or()`, `.xor()`, `.not()` methods. They work on all platforms.

```rust
use magetypes::simd::{
    generic::f32x8,
    backends::F32x8Backend,
};

fn bitwise_example<T: F32x8Backend>(a: f32x8<T>, b: f32x8<T>) {
    // Portable — works on all platforms
    let result = a.and(b);

    // x86-64 only — won't compile on ARM/WASM
    // let result = a & b;
}
```

## Shift Right for Signed Integers

| | x86-64 | AArch64 | WASM |
|---|--------|---------|------|
| `shr` on signed types (i8, i16, i32) | Logical (zero-fill) | Arithmetic (sign-extend) | Arithmetic (sign-extend) |
| `shr_arithmetic` | Sign-extending | Sign-extending | Sign-extending |

**Portable choice:** Use `shr_arithmetic` when you want sign-extending behavior. Use `shr` when you want zero-filling behavior (and be aware it sign-extends on ARM/WASM).

```rust
use magetypes::simd::{
    generic::i32x4,
    backends::I32x4Backend,
};

fn shift_example<T: I32x4Backend>(token: T) {
    let v = i32x4::<T>::splat(token, -8);

    // shr: behavior differs by platform
    let shifted = v.shr::<1>();
    // x86-64:  [-8 >> 1] with zero-fill = some large positive number
    // ARM/WASM: [-8 >> 1] with sign-extend = -4

    // shr_arithmetic: consistent everywhere
    let shifted = v.shr_arithmetic::<1>();
    // All platforms: -4
}
```

This difference exists because x86-64's SSE/AVX shift instructions are logical for all types, while ARM NEON and WASM use arithmetic shifts for signed types.

## Blend Signature

| | x86-64 / AArch64 | WASM |
|---|-------------------|------|
| `blend(mask, true_val, false_val)` | `blend(true_val, false_val)` on mask | `self.blend(other, mask)` on value |

The method exists on all platforms but the calling convention differs. For portable code that uses blend, test on all target platforms.

## interleave_lo / interleave_hi

Available on `f32x4` across all platforms. Not available on integer types.

```rust
use magetypes::simd::{
    generic::f32x4,
    backends::F32x4Backend,
};

fn interleave_example<T: F32x4Backend>(a: f32x4<T>, b: f32x4<T>) {
    // Works everywhere
    let lo = a.interleave_lo(b);
    let hi = a.interleave_hi(b);
}
```

## Summary: Safe Portable Patterns

| Operation | Portable Method |
|-----------|----------------|
| Bitwise AND | `.and()` |
| Bitwise OR | `.or()` |
| Bitwise XOR | `.xor()` |
| Bitwise NOT | `.not()` |
| Sign-extending right shift | `.shr_arithmetic::<N>()` |
| Arithmetic, FMA, comparisons | All operators and methods |
| Reductions | All methods |
| Transcendentals | All methods |

The vast majority of magetypes operations are fully portable with identical semantics. The differences listed above are the complete set.
