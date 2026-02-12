# Behavioral Differences

Cross-architecture semantic differences that affect correctness. Magetypes handles these internally, but if you're writing raw intrinsics, you need to know them.

## FMA Argument Order

The same conceptual operation (`self * a + b`) maps to different argument orders:

```rust
// x86: fmadd(a, b, c) = a*b + c
_mm256_fmadd_ps(self.0, a.0, b.0)

// NEON: fma(a, b, c) = a + b*c  ← DIFFERENT
vfmaq_f32(b.0, self.0, a.0)  // Rearranged to get self*a + b

// WASM: relaxed_madd(a, b, c) = a*b + c  (same as x86)
f32x4_relaxed_madd(self.0, a.0, b.0)
```

**Magetypes `mul_add`:** Handles this automatically. `v.mul_add(a, b)` always means `v*a + b` regardless of platform.

## Blend/Select Argument Order

```rust
// x86: blendv(false_val, true_val, mask) — mask LAST
_mm256_blendv_ps(false_val, true_val, mask)

// NEON: bsl(mask, true_val, false_val) — mask FIRST
vbslq_f32(mask, true_val, false_val)

// WASM: bitselect(true_val, false_val, mask) — mask LAST
v128_bitselect(true_val, false_val, mask)
```

**Magetypes `blend`:** Normalizes to `blend(mask, true_val, false_val)` across platforms.

## Comparison Return Types

```rust
// x86: Returns same float type (all-1s or 0s bit pattern)
let mask: __m256 = _mm256_cmpgt_ps(a, b);

// NEON: Returns unsigned int type — need reinterpret for float blend
let mask: uint32x4_t = vcgtq_f32(a, b);
let float_mask = vreinterpretq_f32_u32(mask);

// WASM: Returns v128 (same generic type as everything)
let mask: v128 = f32x4_gt(a, b);
```

**Magetypes:** Comparison methods return the same vector type, handling reinterprets internally.

## Bitwise on Floats

```rust
// x86: Direct operations exist
_mm256_and_ps(a, b)
_mm256_xor_ps(a, b)

// NEON: Must reinterpret through integers
let a_u = vreinterpretq_u32_f32(a);
let b_u = vreinterpretq_u32_f32(b);
vreinterpretq_f32_u32(vandq_u32(a_u, b_u))

// WASM: v128_and works on any v128
v128_and(a, b)
```

**Magetypes integer types:** Use `.and()`, `.or()`, `.xor()` methods for portable bitwise ops. On x86, operator traits (`&`, `|`, `^`) also work for integers. On ARM and WASM, only methods are available.

## Shift Semantics (signed integers)

```rust
// x86 i32x4 shr: logical shift (zero-fill) — WRONG for negative numbers
_mm_srli_epi32(v, n)  // Shifts in zeros regardless of sign

// ARM i32x4 shr: arithmetic shift (sign-extending)
vshrq_n_s32(v, n)     // Preserves sign bit

// WASM i32x4 shr: arithmetic shift (sign-extending)
i32x4_shr(v, n)       // Preserves sign bit
```

**Magetypes:** Use `shr_arithmetic()` for portable sign-extending right shift across all platforms.

## Reciprocal/Rsqrt Precision

| Operation | x86 Precision | NEON Precision |
|-----------|--------------|----------------|
| `rcp` (reciprocal estimate) | ~12 bits | ~8 bits |
| `rsqrt` (inverse sqrt estimate) | ~12 bits | ~8 bits |

NEON estimates are less precise. For equivalent accuracy, NEON needs two Newton-Raphson refinement steps vs one for x86:

```rust
// NEON: 2 Newton-Raphson steps for ~23-bit precision
let est = vrecpeq_f32(v);
let est = vmulq_f32(est, vrecpsq_f32(v, est));  // Step 1
let est = vmulq_f32(est, vrecpsq_f32(v, est));  // Step 2
```

**Magetypes:** `rcp_approx()` uses the native estimate (different precision per platform). `recip()` provides full precision.

## 64-bit Integer Min/Max

| Platform | Native min/max for i64? |
|----------|------------------------|
| x86 (V3) | No (use comparison + blend) |
| x86 (V4) | Yes (`_mm_min_epi64`) |
| ARM | No (use comparison + blend) |
| WASM | No (use comparison + blend) |

**Magetypes:** `min()` and `max()` work on all platforms via polyfill. `min_fast()` and `max_fast()` take `X64V4Token` for the native instruction when available.

## Horizontal Operations

NEON and WASM require multi-step reductions. x86 has `haddps` but it's slow (multiple micro-ops):

```rust
// x86 reduce_add (fast path via shuffle + add, NOT hadd):
// movshdup + addps + movhlps + addss

// NEON reduce_add:
// vpaddq + vpaddq + lane extract

// WASM reduce_add:
// shuffle + add + shuffle + add + extract
```

**Magetypes:** `reduce_add()`, `reduce_max()`, `reduce_min()` use the optimal sequence for each platform.

## Known Behavioral Differences Table

From the project CLAUDE.md, these are documented and tests account for them:

| Issue | x86 | ARM | WASM |
|-------|-----|-----|------|
| Integer bitwise operators (`&`, `\|`, `^`) | Trait impls (operators) | Methods only | Methods only |
| `shr` for signed integers | Logical (zero-fill) | Arithmetic (sign-extend) | Arithmetic (sign-extend) |
| `blend` signature | `(mask, true, false)` | `(mask, true, false)` | `(self, other, mask)` |
| `interleave_lo/hi` | f32x4 only | f32x4 only | f32x4 only |
