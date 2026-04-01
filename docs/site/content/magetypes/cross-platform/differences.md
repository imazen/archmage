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

#[inline(always)]
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

#[inline(always)]
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

#[inline(always)]
fn interleave_example<T: F32x4Backend>(a: f32x4<T>, b: f32x4<T>) {
    // Works everywhere
    let lo = a.interleave_lo(b);
    let hi = a.interleave_hi(b);
}
```

## Floating-Point Behavioral Differences

These differences arise from how hardware implements IEEE 754 operations. They affect specific edge cases (NaN, signed zero, near-zero cancellation) but not normal arithmetic.

### Negation and Signed Zero

| | x86-64 | AArch64 | WASM |
|---|--------|---------|------|
| `neg(0.0)` | `+0.0` (uses `sub(0, x)`) | `-0.0` (uses `vneg`) | `-0.0` (uses `f32x4_neg`) |

x86 implements negation as `0 - x`, which produces `+0.0` for zero inputs (IEEE 754: `+0 - +0 = +0`). ARM and WASM flip the sign bit directly, preserving `-0.0`. If the sign of zero matters for your algorithm, use bitwise XOR with a sign mask instead of the `-` operator.

### Min/Max NaN Propagation

| | x86-64 | AArch64 / WASM / Scalar |
|---|--------|--------------------------|
| `min(NaN, x)` | Returns `x` (second operand) | Returns `x` (non-NaN value) |
| `min(x, NaN)` | Returns `NaN` | Returns `x` (non-NaN value) |

SSE `minps`/`maxps` always returns the second operand when the first is NaN, and the first when the second is NaN — it doesn't distinguish "propagate NaN" from "return non-NaN." ARM and WASM (and scalar `f32::min`) always return the non-NaN value regardless of operand order. If your inputs may contain NaN, filter them first or use comparison + blend for consistent behavior.

### FMA vs Separate Multiply-Add

| | x86-64 (AVX2+) / AArch64 | WASM / Scalar fallback |
|---|---------------------------|------------------------|
| `mul_add(a, b, c)` | Fused multiply-add (one rounding) | `a * b + c` (two roundings) |

Hardware FMA computes `a × b + c` with a single rounding step, while the scalar/WASM fallback rounds the intermediate product before adding `c`. This matters most when `a × b` nearly cancels with `c` — the results can differ by many ULPs near zero. For most inputs the difference is sub-ULP. Accept small differences in dispatch parity tests.

### Comparison NaN Semantics (simd_ne)

| | x86-64 | AArch64 / WASM |
|---|--------|----------------|
| `simd_ne(NaN, x)` | True (unordered: NaN ≠ anything) | May vary by implementation |

The `simd_ne` operation uses the hardware's not-equal comparison, which may be "ordered" or "unordered" depending on platform. For portable NaN-aware inequality, use `simd_eq` + `not`.

### Reduction Associativity (reduce_add)

All backends compute `reduce_add` using tree reduction, but the exact grouping may differ between scalar (left-fold) and hardware (pairwise tree). For inputs with large magnitude differences, floating-point associativity causes small relative errors (~1e-6). This is inherent to IEEE 754 and not a bug.

### Rounding Consistency (Fixed in 0.9.16)

As of version 0.9.16, `round()`, `floor()`, `ceil()`, and `to_i32_round()` produce identical results across all backends including the scalar fallback. Previously, the scalar backend used ties-away-from-zero for rounding while all hardware used ties-to-even (IEEE 754 default). This was fixed by implementing `roundevenf` in the scalar math library.

## Summary: Safe Portable Patterns

| Operation | Portable Method |
|-----------|----------------|
| Bitwise AND | `.and()` |
| Bitwise OR | `.or()` |
| Bitwise XOR | `.xor()` |
| Bitwise NOT | `.not()` |
| Sign-extending right shift | `.shr_arithmetic::<N>()` |
| Arithmetic, comparisons, rounding | All operators and methods (bit-exact across backends) |
| Reductions | All methods (tiny FP associativity differences possible) |
| FMA (`mul_add`/`mul_sub`) | All methods (±1 ULP difference scalar vs hardware) |
| Transcendentals | All methods (tolerance-based parity) |

The vast majority of magetypes operations are fully portable with identical semantics. The floating-point edge cases listed above affect only NaN, signed zero, and near-zero cancellation scenarios.
