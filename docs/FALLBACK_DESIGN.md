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
```rust
pub fn reduce_add(self) -> f32 {
    unsafe {
        let hi = _mm256_extractf128_ps::<1>(self.0);
        let lo = _mm256_castps256_ps128(self.0);
        let sum = _mm_add_ps(lo, hi);
        let h1 = _mm_hadd_ps(sum, sum);
        let h2 = _mm_hadd_ps(h1, h1);
        _mm_cvtss_f32(h2)
    }
}
```

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
```rust
pub fn exp2_lowp(self) -> Self {
    const C0: f32 = 1.0;
    const C1: f32 = core::f32::consts::LN_2;
    const C2: f32 = 0.240_226_5;
    const C3: f32 = 0.055_504_11;

    unsafe {
        // Split into integer and fractional parts
        let floor = _mm256_floor_ps(self.0);
        let fract = _mm256_sub_ps(self.0, floor);

        // Horner's method: ((C3*x + C2)*x + C1)*x + C0
        let c3 = _mm256_set1_ps(C3);
        let c2 = _mm256_set1_ps(C2);
        let c1 = _mm256_set1_ps(C1);
        let c0 = _mm256_set1_ps(C0);

        let y = _mm256_fmadd_ps(c3, fract, c2);
        let y = _mm256_fmadd_ps(y, fract, c1);
        let y = _mm256_fmadd_ps(y, fract, c0);

        // Scale by 2^floor using integer manipulation
        let exp_int = _mm256_cvtps_epi32(floor);
        let exp_int = _mm256_add_epi32(exp_int, _mm256_set1_epi32(127));
        let exp_int = _mm256_slli_epi32::<23>(exp_int);
        let scale = _mm256_castsi256_ps(exp_int);

        Self(_mm256_mul_ps(y, scale))
    }
}
```

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

### ARM NEON (w128)
- Has vaddvq_f32 for float horizontal add (single instruction)
- Has vaddvq_s32 for integer horizontal add
- Has native FMA (vfmaq_f32)

### ARM NEON Polyfill (256-bit)
- Uses two 128-bit NEON vectors
- Delegates to efficient 128-bit intrinsics
- compose with scalar: `lo.reduce_add() + hi.reduce_add()`

## Generator Strategy

The xtask generator should:

1. **Check for native intrinsic** → Use it
2. **Check for efficient shuffle sequence** → Generate it
3. **Fall back to scalar** → Generate `to_array().iter()` pattern
4. **For polyfills** → Compose from underlying type operations

### Fallback Selection in Generator

```rust
fn generate_reduce_add(arch: &dyn Arch, elem: ElementType) -> String {
    if let Some(intrinsic) = arch.reduce_add_intrinsic(elem) {
        // Native instruction available
        format!("unsafe {{ {} }}", intrinsic)
    } else if elem.is_float() && arch.has_hadd(elem) {
        // Use shuffle tree reduction
        arch.generate_shuffle_reduce_add(elem)
    } else {
        // Scalar fallback
        format!(
            "self.to_array().iter().copied().fold({}, {}::wrapping_add)",
            elem.zero_literal(),
            elem.type_name()
        )
    }
}
```

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
