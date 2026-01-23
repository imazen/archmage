# Context Handoff: SIMD Kernel Examples for Archmage

## Current State

The `examples/fast_dct.rs` example is complete and working:
- Vectorized matrix multiplication DCT with FMA
- AVX2 in-register 8x8 transpose
- ~6-8x faster than scalar (37-49M blocks/sec)
- Uses `f32x8::raw()` and `from_raw()` for intrinsic access

## Task: Port SIMD Kernels from Related Codebases

Port example kernels from `image-webp`, `jpegli-rs`, and `zenimage` to demonstrate archmage's capabilities.

---

## Source 1: ~/work/image-webp

### Key Files

| File | Operations | Priority |
|------|------------|----------|
| `src/yuv_simd.rs` (767 lines) | YUV→RGB color conversion, 14-bit fixed-point, 32 pixels/iteration | HIGH |
| `src/transform_simd_intrinsics.rs` (661 lines) | 4x4 DCT/IDCT, SSE2/AVX2/AVX-512 variants | HIGH |
| `src/loop_filter_avx2.rs` (1341 lines) | VP8 deblocking filter, 16 pixels/iteration | MEDIUM |
| `src/simd_sse.rs` (838 lines) | SSE for 4x4 blocks, distortion metrics | LOW |

### YUV→RGB Kernel (from yuv_simd.rs)

```rust
// Fixed-point YUV to RGB (libwebp compatible)
// R = (19077 * y             + 26149 * v - 14234) >> 6
// G = (19077 * y -  6419 * u - 13320 * v +  8708) >> 6
// B = (19077 * y + 33050 * u             - 17685) >> 6

// Key intrinsics:
// - _mm_mulhi_epu16(): High 16 bits of unsigned multiply
// - _mm_packus_epi16(): Pack with unsigned saturation
// - _mm_unpacklo/hi_epi8(): Interleave for RGBA packing
```

### 4x4 DCT (from transform_simd_intrinsics.rs)

- Uses runtime dispatch via `OnceLock` function pointers
- SSE2, AVX2, AVX-512 variants
- Processes 4x4 blocks (VP8 format)

---

## Source 2: ~/work/jpegli-rs/jpegli-rs/src

### Key Files

| File | Operations | Priority |
|------|------------|----------|
| `encode_simd.rs` (200+ lines) | RGB→YCbCr, chroma downsampling 2x2/2x1/1x2 | HIGH |
| `encode/mage_simd.rs` (150+ lines) | Archmage-based DCT, transpose | HIGH (already uses archmage!) |
| `decode/upsample.rs` | Triangle filter upsampling | MEDIUM |
| `decode/idct_int.rs` | Integer IDCT for decode | MEDIUM |
| `quant/aq/simd.rs` | Adaptive quantization masking | LOW |

### RGB→YCbCr Kernel (from encode_simd.rs)

```rust
// Uses wide::f32x8 + multiversed for dispatch
// FMA for color matrix: Y = R*0.299 + G*0.587 + B*0.114
// Chroma downsampling with box filter

// Key patterns:
// - gather_even_odd_x8(): Deinterleave for downsampling
// - _mm_shuffle_epi8: RGB channel extraction
// - _mm256_permutevar8x32_ps: Variable permute for gather
```

### Archmage DCT (from encode/mage_simd.rs)

**Already uses archmage tokens!** Can be directly adapted:

```rust
use archmage::{arcane, HasAvx, HasAvx2, HasFma};

#[arcane]
fn mage_transpose_8x8_inplace_inner(_token: impl HasAvx, r: &mut [__m256; 8]) {
    // 3-phase transpose: unpack → unpack → permute2f128
}

#[arcane]
fn mage_dct1d_4_inner<T: HasAvx2 + HasFma>(token: T, m: &mut [__m256; 4]) {
    // DCT butterfly with FMA
}
```

---

## Source 3: ~/work/zenimage/src

### Key Files

| File | Operations | Priority |
|------|------------|----------|
| `simd/fast.rs` (962 lines) | Vertical/horizontal reduction, convolution, LUT | HIGH |
| `graphics/alpha_simd.rs` (1409 lines) | Premultiply/unpremultiply, compositing, sRGB↔linear | HIGH |
| `simd/neon.rs` (598 lines) | ARM NEON variants of fast.rs | MEDIUM |
| `pipeline/ops/resize.rs` | SIMD resize operations | MEDIUM |

### Vertical Reduction Kernel (from simd/fast.rs)

```rust
// ~12x speedup over scalar
// Fixed-point i32 accumulators, 15-bit shift

#[target_feature(enable = "avx2")]
unsafe fn reduce_vertical_u8_avx2(inputs: &[&[u8]], output: &mut [u8], weights: &[i16]) {
    // Load 16 bytes → extend u8→i16→i32
    // _mm256_cvtepu8_epi16, _mm256_cvtepi16_epi32
    // Multiply-accumulate with _mm256_mullo_epi32
    // Pack i32→i16→u8 with saturation
}
```

### Alpha Operations (from graphics/alpha_simd.rs)

```rust
// Uses multiversed for dispatch: x86-64-v3, x86-64-v2, aarch64-basic

#[multiversed("x86-64-v3", "x86-64-v2", "aarch64-basic")]
pub fn premultiply_alpha_f32_simd(data: &mut [f32]) {
    // AVX2: _mm256_permutevar8x32_ps to broadcast alpha
    // SSE4.1: _mm_shuffle_ps + _mm_blend_ps
    // Scalar fallback
}

pub fn unpremultiply_alpha_f32_simd(data: &mut [f32]) {
    // Division with epsilon protection
    // _mm256_div_ps, _mm256_cmp_ps, _mm256_blendv_ps
}
```

---

## Recommended Example Structure

Create these examples in `examples/`:

1. **`color_convert.rs`** - YUV↔RGB, RGB→YCbCr with archmage tokens
2. **`alpha_blend.rs`** - Premultiply/unpremultiply/compositing
3. **`convolution.rs`** - Separable filters (box, gaussian)
4. **`resize_kernel.rs`** - Lanczos/bilinear with SIMD reduction

Each should demonstrate:
- Token-gated construction (`Avx2FmaToken::try_new()`)
- `#[arcane]` for intrinsic functions
- `f32x8`/`i32x8` operations with `wide`-like ergonomics
- Scalar fallback for correctness testing

---

## Key Archmage APIs to Use

```rust
// Tokens
use archmage::{Avx2FmaToken, Desktop64, SimdToken, arcane};
use archmage::simd::{f32x8, i32x8};

// Load/store
let v = f32x8::load(token, &arr);
v.store(&mut out);

// Raw access (for intrinsics)
let raw: __m256 = v.raw();
unsafe { f32x8::from_raw(result) }

// Operations
v.mul_add(a, b)  // FMA
v.reduce_add()   // Horizontal sum
v.min(other), v.max(other), v.clamp(lo, hi)
```

---

## Files to Read First

1. `src/simd/x86/w256.rs` - f32x8 implementation (raw/from_raw methods)
2. `examples/fast_dct.rs` - Working example with transpose
3. `examples/cross_platform.rs` - Token dispatch pattern

---

## Performance Targets (from source benchmarks)

| Kernel | Source | Speedup | Notes |
|--------|--------|---------|-------|
| DCT-8x8 | archmage | 6-8x | Current fast_dct.rs |
| YUV→RGB | image-webp | 10-15x | 32 pixels/iteration |
| Vertical reduction | zenimage | ~12x | u8 with i32 accumulators |
| Horizontal conv | zenimage | 7-8x | Strided access limits gain |
| Separable 5x5 blur | zenimage | 15-20x | Combines V+H reductions |
| Alpha premultiply | zenimage | 2-3x | Memory-bound |

---

## Git Status

```
On branch main (ahead of origin/main by 60 commits)
Clean working tree after fast_dct.rs commit
```

Last commit: `70de91f feat: add fast DCT-8x8 example with AVX2+FMA`

---

## Next Steps

1. Create `examples/color_convert.rs` with YUV↔RGB kernel
2. Create `examples/alpha_blend.rs` with premultiply/unpremultiply
3. Create `examples/convolution.rs` with vertical reduction
4. Add `#[multiwidth]` variants where applicable
5. Benchmark against scalar and document speedups
