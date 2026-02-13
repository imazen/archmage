# Intrinsics Guide for archmage/magetypes Implementers

> **For users**: Use `#[arcane]`/`#[rite]` instead of manual `#[target_feature]`. This guide is an implementer reference for raw intrinsics.

This guide covers platform differences, naming conventions, and performance pitfalls when implementing SIMD types using raw intrinsics.

## Naming Conventions

| Platform | Pattern | Example |
|----------|---------|---------|
| x86 | `_mm[width]_op_suffix` | `_mm256_add_ps`, `_mm_mul_epi32` |
| NEON | `v[op][mod]q_type` | `vaddq_f32`, `vmulq_s32` |
| WASM | `[type]x[lanes]_[op]` | `f32x4_add`, `i32x4_mul` |

### x86 Suffixes

| Suffix | Meaning |
|--------|---------|
| `_ps` | Packed single (f32) |
| `_pd` | Packed double (f64) |
| `_epi8/16/32/64` | Packed signed integer |
| `_epu8/16/32/64` | Packed unsigned integer |
| `_si128/256/512` | Bitwise on full register |

### NEON Modifiers

| Modifier | Meaning | Example |
|----------|---------|---------|
| `q` | Quadword (128-bit) | `vaddq_f32` |
| `l` | Long (widen result) | `vaddl_s16` → i32 |
| `n` | Narrow (shrink result) | `vaddhn_s32` → i16 |
| `h` | High half | `vaddhn_high_s32` |
| `p` | Pairwise | `vpaddq_f32` |
| `a` | Accumulate | `vmlaq_f32` (FMA) |

## Critical Semantic Differences

### FMA Argument Order

```rust
// Semantic: self * a + b

// x86: fmadd(a, b, c) = a*b + c
_mm256_fmadd_ps(self.0, a.0, b.0)

// NEON: fmaq(a, b, c) = a + b*c  ← DIFFERENT!
vfmaq_f32(b.0, self.0, a.0)  // Args swapped to get self*a + b

// WASM: relaxed_madd(a, b, c) = a*b + c
f32x4_relaxed_madd(self.0, a.0, b.0)
```

### Blend/Select Argument Order

```rust
// x86: blendv(a, b, mask) = mask ? b : a
_mm256_blendv_ps(false_val, true_val, mask)

// NEON: bsl(mask, a, b) = mask ? a : b  ← mask is FIRST
vbslq_f32(mask, true_val, false_val)

// WASM: bitselect(a, b, mask) = mask ? a : b
v128_bitselect(true_val, false_val, mask)
```

### Comparison Return Types

```rust
// x86: Returns same float type (all-1s or 0s bit pattern)
let mask: __m256 = _mm256_cmpgt_ps(a, b);

// NEON: Returns uint type - need reinterpret for float mask
let mask: uint32x4_t = vcgtq_f32(a, b);
let float_mask = vreinterpretq_f32_u32(mask);

// WASM: Returns v128 with -1 or 0 per lane
let mask: v128 = f32x4_gt(a, b);
```

### Bitwise on Floats

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

## Type Systems

| Platform | Float | Integer |
|----------|-------|---------|
| x86 | `__m128`, `__m256`, `__m512` | `__m128i`, `__m256i`, `__m512i` (one type for all) |
| NEON | `float32x4_t`, `float64x2_t` | **Separate types:** `int32x4_t`, `uint8x16_t`, etc. |
| WASM | `v128` | `v128` (one type for all) |

## Missing/Different Operations

| Operation | x86 | NEON | Notes |
|-----------|-----|------|-------|
| 64-bit int min/max | AVX-512VL only | None | Use comparison + blend |
| Horizontal add | `_mm_hadd_ps` | `vpaddq_f32` | NEON needs multiple steps |
| Reciprocal estimate | 12-bit (`_mm_rcp_ps`) | 8-bit (`vrecpeq_f32`) | NEON needs more Newton-Raphson |
| Rsqrt estimate | 12-bit (`_mm_rsqrt_ps`) | 8-bit (`vrsqrteq_f32`) | NEON needs more Newton-Raphson |

## Interleaved Loads (NEON specialty)

```rust
// NEON has native structure-of-arrays loads
vld2q_f32(ptr) -> float32x4x2_t  // Load 8 floats, deinterleave to 2 vectors
vld3q_f32(ptr) -> float32x4x3_t  // Load 12 floats, deinterleave to 3 vectors
vld4q_f32(ptr) -> float32x4x4_t  // Load 16 floats, deinterleave to 4 vectors

// Great for: RGB→planar, complex numbers, vertex data
// x86 equivalent requires multiple shuffles
```

---

# Penalized Instructions & Performance Pitfalls

## x86: AVX-512 Frequency Throttling

**Intel Alder Lake (12th gen) and later consumer CPUs** removed AVX-512 entirely. On server/HEDT chips that have it:

- **License 1 (L1)**: Light 512-bit usage → minor downclock (~100-200 MHz)
- **License 2 (L2)**: Heavy 512-bit or heavy FP → moderate downclock
- **Turbo loss**: Can lose 10-20% clock speed during sustained 512-bit workloads

**Mitigation**: For short bursts, AVX-512 wins. For sustained compute, 256-bit may be faster due to higher clocks. Profile your actual workload.

**AMD Zen 4+**: No significant throttling. AVX-512 is implemented as 2×256-bit but clocks stay high.

## x86: AVX/SSE Transition Penalty

On **pre-Skylake Intel** and **pre-Zen AMD**:

```rust
// BAD: Mixing AVX and SSE without VZEROUPPER causes ~70 cycle penalty
fn mixed_bad() {
    let v256 = _mm256_add_ps(a, b);  // Uses upper 128 bits
    let v128 = _mm_add_ps(c, d);      // SSE - penalty!
}

// Rust handles this automatically in most cases, but beware:
// - Calling non-inlined SSE functions from AVX code
// - FFI boundaries
// - Hand-written assembly
```

**Modern CPUs** (Intel Ice Lake+, AMD Zen): No transition penalty. Still good practice to avoid mixing for code clarity.

## x86: Slow Instructions to Avoid

| Instruction | Latency | Throughput | Why Slow |
|-------------|---------|------------|----------|
| `HADDPS/HSUBPS` | 5-7 | 2 | Horizontal ops = many µops |
| `DPPS/DPPD` | 9-15 | 1-2 | Dot product = many µops |
| `GATHER` | 12-25 | varies | Multiple memory accesses |
| `SCATTER` | 18-40 | varies | Even worse than gather |
| `PCMPISTRI/M` | 10-18 | 3 | String ops, complex µops |
| `MASKMOVDQU` | 200+ | 1 | Serializing, non-temporal |
| `MXCSR` writes | ~20 | 1 | Pipeline serialization |

### Horizontal Operations

```rust
// Slow: HADDPS (multiple µops)
let sum = _mm_hadd_ps(v, v);
let sum = _mm_hadd_ps(sum, sum);

// Often faster: Shuffle + add
let shuf = _mm_movehdup_ps(v);     // [1,1,3,3]
let sum1 = _mm_add_ps(v, shuf);    // [0+1,1+1,2+3,3+3]
let shuf = _mm_movehl_ps(sum1, sum1);
let sum2 = _mm_add_ss(sum1, shuf);
```

### Gather/Scatter

```rust
// Gather: ~4-7 µops, variable latency based on cache behavior
let gathered = _mm256_i32gather_ps(base, indices, 4);

// Often faster for small, known patterns: explicit loads + shuffle
let v0 = _mm_load_ss(&data[i0]);
let v1 = _mm_load_ss(&data[i1]);
// ... combine with shuffles
```

**When gather IS worth it**: Random access patterns, large index ranges, or when the alternative is scalar loads.

### Cross-Lane Shuffles (AVX2)

```rust
// Slow: VPERM2F128 crosses 128-bit lanes
let swapped = _mm256_permute2f128_ps(v, v, 0x01);  // ~3 cycles

// Slow: VPERMPD/VPERMPS cross lanes
let perm = _mm256_permutevar8x32_ps(v, idx);  // ~3 cycles

// Fast: In-lane shuffles
let shuf = _mm256_shuffle_ps(v, v, 0b10_11_00_01);  // 1 cycle
```

## x86: Denormals

**Denormal numbers** (very small floats near zero) cause **100x+ slowdown** when they appear in computations.

```rust
// Set DAZ+FTZ flags at program start for SIMD-heavy code
unsafe {
    let mut mxcsr = _mm_getcsr();
    mxcsr |= 0x8040;  // DAZ (bit 6) + FTZ (bit 15)
    _mm_setcsr(mxcsr);
}
```

**Trade-off**: Loses IEEE compliance for denormal handling. Usually fine for graphics/audio/ML.

## NEON: Penalties

### GP ↔ NEON Register Moves

```rust
// Slow: Moving between general-purpose and NEON registers
let lane: i32 = vgetq_lane_s32(v, 0);  // NEON → GP: ~3-6 cycles
let v = vdupq_n_s32(scalar);           // GP → NEON: ~3-6 cycles

// Fast: Keep data in NEON registers, use NEON operations
let broadcasted = vdupq_laneq_s32(v, 0);  // NEON → NEON: 1 cycle
```

**Pattern**: Avoid extracting lanes to scalars mid-computation. Do all SIMD work, then extract at the end.

### Lower Precision Estimates

| Operation | x86 Precision | NEON Precision | Newton-Raphson Steps |
|-----------|---------------|----------------|----------------------|
| `recpe` | ~12 bits | ~8 bits | x86: 1, NEON: 2 |
| `rsqrte` | ~12 bits | ~8 bits | x86: 1, NEON: 2 |

```rust
// NEON reciprocal with 2 Newton-Raphson steps for ~23 bits
fn recip_precise(v: float32x4_t) -> float32x4_t {
    let est = vrecpeq_f32(v);
    let est = vmulq_f32(est, vrecpsq_f32(v, est));  // Step 1
    vmulq_f32(est, vrecpsq_f32(v, est))             // Step 2
}
```

### No Native 64-bit Integer Min/Max

```rust
// NEON: No vminq_s64 / vmaxq_s64!
// Must use comparison + blend
fn min_i64(a: int64x2_t, b: int64x2_t) -> int64x2_t {
    let mask = vcltq_s64(a, b);  // a < b
    vbslq_s64(mask, a, b)        // mask ? a : b
}
```

### Horizontal Operations Need Multiple Steps

```rust
// NEON horizontal sum: O(log N) pairwise additions
fn reduce_add_f32x4(v: float32x4_t) -> f32 {
    let sum = vpaddq_f32(v, v);      // [a+b, c+d, a+b, c+d]
    let sum = vpaddq_f32(sum, sum);  // [a+b+c+d, ...]
    vgetq_lane_f32(sum, 0)
}
```

### SVE/SVE2: Do Not Use

**SVE (Scalable Vector Extension)** is prohibited in archmage:

- Not shipped in consumer hardware (as of 2025)
- Variable vector length (128-2048 bits) complicates codegen
- Only available on: AWS Graviton 3+, Fujitsu A64FX, some Arm Neoverse server chips
- When it ships widely, we'll add tokens. Until then, use NEON.

## WASM: Considerations

- **Relaxed SIMD**: Behavior varies by runtime (browser/engine). Use for performance, not correctness.
- **Runtime detection**: `Wasm128Token::summon()` works like other tokens. Also available via compile-time `#[cfg(target_feature = "simd128")]`.
- **Alignment**: Less penalty than native, but aligned access still preferred.

## General: Memory Performance

### Cache Line Splits

```rust
// BAD: Unaligned access crossing 64-byte cache line boundary
let ptr = base.add(60) as *const __m256;  // 32-byte load at offset 60
let v = _mm256_loadu_ps(ptr);  // Crosses into next cache line!

// BETTER: Align your data or process in aligned chunks
#[repr(align(32))]
struct AlignedData([f32; 8]);
```

### Store Forwarding

```rust
// BAD: Store then load with different size/alignment
_mm_storeu_ps(ptr, narrow);           // Store 16 bytes
let wide = _mm256_loadu_ps(ptr);      // Load 32 bytes overlapping - stall!

// GOOD: Match store and load sizes, or add computation between
```

### Prefetching

```rust
// Usually unnecessary - hardware prefetchers are good
// But for irregular access patterns:
_mm_prefetch(ptr.add(512), _MM_HINT_T0);  // Prefetch 512 bytes ahead
```

## Summary: Default Recommendations

1. **Use 256-bit (AVX2) as the sweet spot** for x86 - good perf, no throttling concerns
2. **Avoid horizontal operations in hot loops** - restructure data if possible
3. **Set DAZ+FTZ** for SIMD-heavy applications
4. **Keep data in SIMD registers** - minimize GP↔SIMD transfers
5. **Use aligned data** when possible (32-byte for AVX2, 64-byte for AVX-512)
6. **Profile gather/scatter vs alternatives** - not always faster
7. **On NEON, use 2 Newton-Raphson steps** for reciprocal/rsqrt
8. **Test on actual target hardware** - µarch differences are real
