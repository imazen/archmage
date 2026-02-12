# Polyfills

Magetypes provides consistent APIs across platforms. When a native instruction doesn't exist, a polyfill emulates it using available instructions. This page documents what gets polyfilled and the performance implications.

## Width Polyfills

NEON and WASM only have 128-bit registers. Wider types use pairs of narrower operations:

| Type | x86 (V3) | ARM (NEON) | WASM |
|------|----------|------------|------|
| `f32x4` | Native SSE | Native NEON | Native SIMD128 |
| `f32x8` | Native AVX2 | 2× `f32x4` polyfill | 2× `f32x4` polyfill |
| `f32x16` | Native AVX-512 | 4× `f32x4` polyfill | 4× `f32x4` polyfill |

Width polyfills are element-wise: `f32x8::add` on ARM does two `vaddq_f32` calls. The overhead is minimal — you're doing the same number of operations, just in two instructions instead of one.

### Width polyfill structure

```rust
// Polyfill f32x8 on ARM = two NEON f32x4
pub struct f32x8 {
    lo: f32x4,  // Lanes 0-3
    hi: f32x4,  // Lanes 4-7
}

impl Add for f32x8 {
    fn add(self, rhs: Self) -> Self {
        Self {
            lo: self.lo + rhs.lo,
            hi: self.hi + rhs.hi,
        }
    }
}
```

### Horizontal operations with polyfills

Reductions across a polyfilled type need to reduce each half, then combine:

```rust
// f32x8::reduce_add on ARM
fn reduce_add(self) -> f32 {
    self.lo.reduce_add() + self.hi.reduce_add()
}
```

This is slightly more work than a native 256-bit reduction, but the difference is negligible compared to the data processing typically surrounding it.

## Operation Polyfills

Some operations don't have direct hardware support on all platforms:

### 64-bit integer min/max

| Platform | Implementation |
|----------|---------------|
| x86 V3 | Compare + blend (no native `_mm256_min_epi64`) |
| x86 V4 | Native `_mm256_min_epi64` |
| ARM | Compare + blend |
| WASM | Compare + blend |

### Byte shifts (i8x16/u8x16 shl/shr)

No platform has native per-element byte shifts. All use:
```
shift = widen to 16-bit → shift → mask to 8-bit
```
This is ~2 instructions and works identically everywhere.

### Cubic root (cbrt)

No SIMD instruction exists for cbrt on any platform. All use:
```
scalar initial guess → Newton-Raphson refinement
```

### Transcendentals (exp, log, sin, cos, etc.)

Implemented as polynomial approximations on all platforms. The polynomials are the same; only the intrinsics used differ:

| Precision | Suffix | Accuracy |
|-----------|--------|----------|
| Low | `_lowp` | ~3-4 decimal digits |
| Medium | `_midp` | ~6-7 decimal digits |
| Precise | `_midp_precise` | Additional Newton-Raphson step |

## Performance Matrix

Approximate relative cost of polyfilled operations compared to native:

| Operation | Native | Polyfill overhead |
|-----------|--------|-------------------|
| Element-wise (add, mul, etc.) | 1× | 2× for width polyfill (2 ops instead of 1) |
| Reductions (reduce_add, etc.) | 1× | ~1.5× (reduce each half + combine) |
| Shuffles | 1× | Varies (some need creative instruction sequences) |
| Transcendentals | 1× | Same cost (polynomial approximation everywhere) |
| 64-bit min/max | 1× (V4) | ~3× (compare + blend) |

The 2× overhead for width polyfills is the theoretical maximum. In practice, the CPU's out-of-order execution and instruction-level parallelism often hide it, especially when the polyfilled operations are interleaved with other work.

## Parity Verification

All polyfills are verified by `cargo xtask parity`:

```bash
# Check that all types have the same API across x86/ARM/WASM
just parity
```

Current status: **0 parity issues**. All W128 types have identical APIs across x86, ARM, and WASM.

Additionally, `magetypes/tests/polyfill_parity.rs` compares polyfill results against native implementations on x86 to verify numerical correctness.
