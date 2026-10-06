# Fallback Strategy Design

This document explains how archmage handles operations that lack direct hardware intrinsics.

## Fallback Hierarchy

Operations fall into three categories based on performance requirements:

### Category A: Lane-Independent Operations (No Fallback Needed)

These operations work identically on each lane and can be delegated to narrower types:

```
add, sub, mul, div          → lo.op() + hi.op()
min, max, abs, neg          → lo.op() + hi.op()
sqrt, floor, ceil, round    → lo.op() + hi.op()
bitwise and, or, xor, not   → lo.op() + hi.op()
comparisons (simd_eq, etc.) → lo.op() + hi.op()
```

**Polyfill implementation pattern** (from the generated `x86_v3.rs`, where
`f32x16` on `X64V3Token` is two `__m256` halves):
```rust
#[inline(always)]
fn abs(self, a: [__m256; 2]) -> [__m256; 2] {
    [
        <archmage::X64V3Token as F32x8Backend>::abs(self, a[0]),
        <archmage::X64V3Token as F32x8Backend>::abs(self, a[1]),
    ]
}
```

### Category B: Cross-Lane Reductions (Prefer SIMD, Accept Scalar)

Horizontal operations that combine lanes. Every type has `reduce_add` (integer
types wrap); the float types also have `reduce_min` and `reduce_max`. There are
no integer min/max or bitwise reductions. How the generated backends implement
`reduce_add`:

| Lanes | x86 V3 | x86 V4 (512-bit) | NEON | WASM |
|-------|--------|------------------|------|------|
| f32, f64 | extract + shuffle + add | `_mm512_reduce_add_ps`/`_pd` | `vpaddq`/`vaddvq` | lane extracts |
| i32, u32, i64 | extract + shuffle + add | scalar fold | `vaddvq` | lane extracts |
| i8, u8, i16, u16, u64 | scalar fold | scalar fold | `vaddvq` | scalar fold |

`_mm512_reduce_add_ps` is a compiler-provided sequence, not one instruction.

**Integer scalar fallback pattern** (from the generated `x86_v3.rs`):
```rust
#[arcane(suppress_const_test, _self = X64V3Token)]
fn reduce_add(self, a: __m128i) -> i8 {
    let arr = <Self as I8x16Backend>::to_array(_self, a);
    arr.iter().copied().fold(0i8, i8::wrapping_add)
}
```

**Polyfill composition pattern:**
```rust
#[inline(always)]
fn reduce_add(self, a: [__m256; 2]) -> f32 {
    <archmage::X64V3Token as F32x8Backend>::reduce_add(self, a[0])
        + <archmage::X64V3Token as F32x8Backend>::reduce_add(self, a[1])
}
```

### Category C: Transcendental Functions (Polynomial Approximation)

Operations with no hardware support requiring mathematical computation:

| Function | Implementation | Max Error |
|----------|---------------|-----------|
| exp2_lowp | Degree-3 polynomial | 5.57e-3 relative (x ≤ 127.99) |
| exp2_midp | Degree-6 polynomial | 1.9 ULP for x < 127.5; up to 134.1 ULP in [127.5, 128) |
| log2_lowp | Mantissa polynomial | 6.4e-6 absolute |
| ln_lowp | log2_lowp * LN_2 | 8.5e-6 absolute |

There are no trigonometric functions. [transcendentals.md](transcendentals.md)
has the measured accuracy for every function and domain.

**Pattern:** the transcendentals are written once, in
`xtask/src/simd_types/generic_gen/transcendentals.rs`, against the generic
vector API (`mul_add`, `round`, `shl_const`, bit casts), so every backend gets
the same polynomial.

## Platform-Specific Considerations

### x86-64 SSE (w128)
- Horizontal adds (`hadd_ps`, SSSE3 `hadd_epi32`) exist but are slow; the
  backends reduce with shuffles and adds instead
- Has floor/ceil via SSE4.1

### x86-64 AVX2 (w256)
- 256-bit horizontal adds work within each 128-bit lane, so reductions extract
  the high half and continue at 128 bits
- Has FMA for polynomial evaluation

### x86-64 AVX-512 (w512)
- `_mm512_reduce_add_ps`, `_mm512_reduce_min_ps` and `_mm512_reduce_max_ps`
  are compiler-provided sequences, used for the float types
- Without `X64V4Token`, 512-bit types are two 256-bit halves (see Category A)

### ARM NEON (w128) — Implemented
- Has vaddvq_f32 for float horizontal add (single instruction)
- Has vaddvq_s32 for integer horizontal add
- Has native FMA (vfmaq_f32)

### ARM NEON Polyfill (256-bit) — Implemented
- Uses two 128-bit NEON vectors
- Delegates to efficient 128-bit intrinsics
- compose with scalar: `lo.reduce_add() + hi.reduce_add()`

### WASM SIMD128 (w128) — Implemented
- Uses `v128` type for all element types
- Has `f32x4_add`, `i32x4_add`, etc.
- No FMA in SIMD128: `mul_add` is a multiply then an add (two roundings), and
  `mul_add_portable` fuses in software. With relaxed SIMD enabled, `mul_add`
  emits `f32x4_relaxed_madd`, which an engine may or may not fuse.

### WASM SIMD128 Polyfill (256-bit) — Implemented
- Uses two 128-bit WASM vectors
- Same polyfill pattern as ARM/SSE

## Generator Strategy

The xtask generator should:

1. **Check for native intrinsic** → Use it
2. **Check for efficient shuffle sequence** → Generate it
3. **Fall back to scalar** → Generate `to_array().iter()` pattern
4. **For polyfills** → Compose from underlying type operations

### Fallback Selection in Generator

Current implementations are emitted by the backend generators inside
token-matched `#[arcane]` contexts. Use the generic vector operation from a
`#[magetypes]` kernel; the old manually wrapped implementation is obsolete.

## Adding New Operations

When adding an operation that may lack intrinsics:

1. **Document the fallback strategy** in this file
2. **Add to generator** with architecture-specific selection
3. **Add polyfill support** that composes from narrower type
4. **Add verification test** comparing polyfill to native results

## Performance Expectations

| Category | Expected Overhead |
|----------|------------------|
| Lane-independent polyfill | 2x (two ops instead of one) |
| Shuffle reduction | ~10-20 cycles |
| Scalar reduction | O(lanes) cycles |
| Transcendental polynomial | ~20-50 cycles depending on precision |

For hot loops, prefer:
- Keeping data in SIMD until final reduction
- Using lower precision if acceptable
- Batching reductions across multiple vectors

## Performance: `#[arcane]` and `#[rite]`

### Use `#[arcane]` at the entry point, `#[rite]` for everything inside

The `#[arcane]` macro generates functions with `#[target_feature]` attributes.
Operators like `a + b` inside `#[arcane]` or `#[rite]` compile to single SIMD instructions.

**The cost is the `#[target_feature]` boundary**, not the wrapper itself. Each `#[arcane]` call
from non-SIMD code creates an optimization boundary LLVM can't inline across.

Measured overhead (see [PERFORMANCE.md](PERFORMANCE.md)):
- Simple vector add: 4x slower per-iteration `#[arcane]` vs loop-inside-`#[arcane]`
- DCT-8: 6.2x slower per-row `#[arcane]` vs loop-inside-`#[arcane]`
- `#[rite]` inside `#[arcane]`: 0x overhead (fully inlined)

```rust
use archmage::{arcane, rite, X64V3Token, SimdToken};
use magetypes::simd::generic::f32x8;

// Entry point — called from non-SIMD code
#[arcane]
fn process_vectors(token: X64V3Token, input: &[[f32; 8]]) -> f32 {
    let mut sum = f32x8::zero_t(token);
    for arr in input {
        sum = add_chunk(token, sum, arr);  // #[rite] inlines here
    }
    sum.reduce_add()
}

// Internal helper — inlines into #[arcane] caller
#[rite]
fn add_chunk(token: X64V3Token, acc: f32x8<X64V3Token>, arr: &[f32; 8]) -> f32x8<X64V3Token> {
    acc + f32x8::load_t(token, arr)  // a load and one vaddps
}
```

Value-based intrinsics have been safe inside `#[target_feature]` functions
since Rust 1.87; only raw-pointer memory operations still need `unsafe`, and
`import_intrinsics` replaces those with reference-based versions.
