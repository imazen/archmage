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

## archmage vs wide: Design Tradeoffs

### CRITICAL: Use `#[arcane]` for Performance-Critical Code

**archmage operators inline properly when called from `#[target_feature]` contexts.**

The `#[arcane]` macro generates functions with `#[target_feature]` attributes.
When you use operators like `a + b` INSIDE an `#[arcane]` function, the intrinsics
inline properly into AVX/NEON instructions.

```rust
use archmage::{arcane, Avx2FmaToken, SimdToken};
use archmage::simd::f32x8;

// CORRECT - operators inline properly inside #[arcane]
#[arcane]
fn process_vectors(token: Avx2FmaToken, input: &[[f32; 8]]) -> f32 {
    let mut sum = f32x8::zero(token);
    for arr in input {
        let v: f32x8 = (*arr).into();
        sum = sum + v;  // This + compiles to a single vaddps instruction!
    }
    sum.reduce_add()
}
```

**Without `#[arcane]` or `#[target_feature]`:** Operators still work correctly, but
intrinsics are called as separate functions rather than being inlined. This is ~1.3x
slower but still uses proper SIMD instructions.

**Alternative: Use `-C target-cpu=native`** for benchmarking or when you can't use
`#[arcane]` everywhere:

```bash
# For benchmarking
RUSTFLAGS="-C target-cpu=native" cargo bench
just bench  # Automatically sets the flag
```

### Benchmark Results

**Scenario 1: Using `#[arcane]` / `#[target_feature]` (recommended)**

Without `-C target-cpu=native`, but using proper `#[target_feature]` contexts:

| Operation | archmage | wide | Winner |
|-----------|----------|------|--------|
| batch add (1024 elements) | 83ns | 65ns | wide 1.28x |
| batch fma (1024 elements) | 10ns | varies | competitive |

This is the expected use case - archmage with runtime detection inside `#[arcane]` functions.
Wide is slightly faster because it uses compile-time feature detection (`#[cfg(target_feature)]`).

**Scenario 2: With `-C target-cpu=native` (all code benefits)**

With compile-time CPU targeting, both archmage and wide compile optimally:

| Operation | archmage | wide | Winner |
|-----------|----------|------|--------|
| f32x8 add | 788ps | 1002ps | **archmage 1.27x** |
| f32x8 mul | 796ps | 1070ps | **archmage 1.34x** |
| f32x8 div | 3.9ns | 3.9ns | tie |
| f32x8 fma | 998ps | 1264ps | **archmage 1.27x** |
| f32x8 sqrt | 2.0ns | 2.0ns | tie |
| f32x8 floor | 622ps | 718ps | **archmage 1.15x** |
| f32x8 ceil | 622ps | 718ps | **archmage 1.15x** |
| f32x8 round | 622ps | 722ps | **archmage 1.16x** |
| f32x8 min | 787ps | 1004ps | **archmage 1.28x** |
| f32x8 max | 785ps | 1003ps | **archmage 1.28x** |
| f32x8 abs | 626ps | 720ps | **archmage 1.15x** |
| f32x8 reduce_add | 1.6ns | 1.4ns | wide 1.14x |
| f32x8 load | 619ps | 713ps | **archmage 1.15x** |
| f32x8 store | 779ps | 1.3ns | **archmage 1.67x** |
| batch add (128) | 118ns | 131ns | **archmage 1.11x** |

**Summary:** With `#[arcane]` alone, archmage is ~1.3x slower than wide.
With `-C target-cpu=native`, archmage is 15-35% faster than wide on most operations.

### Design Approaches

**wide's approach:**
- Compile-time feature detection (`#[cfg(target_feature="avx")]`)
- Zero-cost transmutes via bytemuck (`Pod` trait)
- Direct struct initialization from arrays
- No runtime checks

**archmage's approach:**
- Runtime token verification (safe construction at program start)
- Zero-cost `From<[T; N]>` via bytemuck (same as wide, as of recent changes)
- Token parameter passing (zero-size, no runtime cost)
- `#[target_feature]` enables safe intrinsic calls within function bodies

### Why archmage is Faster

1. **Direct intrinsic access**: archmage types wrap raw `__m256` directly with minimal abstraction
2. **Native floor/ceil/round**: Uses SSE4.1/AVX instructions directly
3. **Efficient min/max**: Direct `_mm256_min_ps`/`_mm256_max_ps`
4. **Zero-copy conversions**: `from_array()` uses `transmute`, matching wide

### When wide Might Win

1. **Horizontal reductions**: wide's `reduce_add` is slightly faster (~14%)
2. **Very old code**: Code written before archmage added bytemuck support

### Recommendations

For maximum performance with archmage:
1. **Always compile with `-C target-cpu=native`** (or at least `-C target-cpu=haswell` for AVX2)
2. Use `just bench` for benchmarking (automatically sets correct flags)
3. For production, ensure your build system passes appropriate RUSTFLAGS
4. Both archmage and wide can coexist - archmage tokens can gate code sections
