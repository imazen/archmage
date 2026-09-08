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

**Polyfill implementation pattern:**
```rust
pub fn abs(self) -> Self {
    Self { lo: self.lo.abs(), hi: self.hi.abs() }
}
```

### Category B: Cross-Lane Reductions (Prefer SIMD, Accept Scalar)

Horizontal operations that combine lanes:

| Operation | Floats | Integers |
|-----------|--------|----------|
| reduce_add | SIMD shuffle tree | Scalar fallback |
| reduce_min | SIMD shuffle tree | Scalar fallback |
| reduce_max | SIMD shuffle tree | Scalar fallback |
| reduce_and | SIMD available | SIMD available |
| reduce_or  | SIMD available | SIMD available |

**Float SIMD pattern (AVX2):**
Current implementations are emitted by the backend generators inside
token-matched `#[arcane]` contexts. Use the generic vector operation from a
`#[magetypes]` kernel; the old manually wrapped implementation is obsolete.

**Integer scalar fallback pattern:**
```rust
pub fn reduce_add(self) -> i8 {
    // Uses as_array() for zero-copy access (not to_array() which copies)
    self.as_array().iter().copied().fold(0_i8, i8::wrapping_add)
}
```

**Polyfill composition pattern:**
```rust
pub fn reduce_add(self) -> f32 {
    self.lo.reduce_add() + self.hi.reduce_add()
}
```

### Category C: Transcendental Functions (Polynomial Approximation)

Operations with no hardware support requiring mathematical computation:

| Function | Implementation | Max Error |
|----------|---------------|-----------|
| exp2_lowp | Degree-3 polynomial | ~5.5e-3 |
| exp2_midp | Degree-6 polynomial | ~1e-6 |
| log2_lowp | Mantissa polynomial | ~3e-4 |
| ln_lowp | log2_lowp * LN_2 | ~3e-4 |
| sin_lowp | Range-reduced Chebyshev | varies |

**Pattern:**
Current implementations are emitted by the backend generators inside
token-matched `#[arcane]` contexts. Use the generic vector operation from a
`#[magetypes]` kernel; the old manually wrapped implementation is obsolete.

## Platform-Specific Considerations

### x86-64 SSE (w128)
- Has hadd_ps for float reductions
- No hadd for integer types → scalar fallback
- Has floor/ceil via SSE4.1

### x86-64 AVX2 (w256)
- Has _mm256_hadd_ps for floats
- No efficient 256-bit integer hadd → extract + 128-bit or scalar
- Has FMA for polynomial evaluation

### x86-64 AVX-512 (w512)
- Has _mm512_reduce_add_ps (single instruction!)
- Has _mm512_reduce_min_ps, _mm512_reduce_max_ps
- Use these when available, polyfill with extract + 256-bit otherwise

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
- Relaxed SIMD for FMA (`f32x4_relaxed_madd`)

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
#[arcane(import_intrinsics)]
fn process_vectors(token: X64V3Token, input: &[[f32; 8]]) -> f32 {
    let mut sum = f32x8::zero(token);
    for arr in input {
        sum = add_chunk(token, sum, arr);  // #[rite] inlines here
    }
    sum.reduce_add()
}

// Internal helper — inlines into #[arcane] caller
#[rite(import_intrinsics)]
fn add_chunk(token: X64V3Token, acc: f32x8<X64V3Token>, arr: &[f32; 8]) -> f32x8<X64V3Token> {
    let v: f32x8<X64V3Token> = (*arr).into();
    acc + v  // Compiles to a single vaddps
}
```

### Note on `unsafe` in intrinsic examples

As of Rust 1.87+, value-based intrinsics (arithmetic, comparison, shuffle, etc.) are safe
inside `#[target_feature]` functions. The `unsafe` blocks in the examples above (Category C)
reflect the pre-1.87 style. Inside `#[arcane]`/`#[rite]` functions, only memory operations
(raw pointers) still require `unsafe`. Use `import_intrinsics` for safe memory ops.
