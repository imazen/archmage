# Context Handoff: SIMD Kernel Examples for Archmage

## Completed Examples

### 1. `examples/fast_dct.rs` - 8x8 DCT with Matrix Multiply
- Vectorized matrix multiplication DCT with FMA
- AVX2 in-register 8x8 transpose (3-phase unpack/shuffle/permute)
- ~6-8x faster than scalar (37-49M blocks/sec)
- Uses `f32x8::raw()` and `from_raw()` for intrinsic access

### 2. `examples/color_convert.rs` - YUV↔RGB Color Space
- Float f32x8 API: 3.0x speedup (clean, readable)
- Fixed-point SSE2: 9.9x speedup (bit-exact with libwebp)
- BT.601 coefficients for both directions
- Round-trip correctness testing

### 3. `examples/alpha_blend.rs` - Alpha Channel Operations
- Premultiply/unpremultiply alpha
- Porter-Duff "over" compositing
- Alpha broadcast with `_mm256_permutevar8x32_ps`
- Memory-bound (~1x speedup, shows correct patterns)

### 4. `examples/convolution.rs` - Vertical/Horizontal Reduction
- f32 vertical reduction: 5.2x speedup
- u8 fixed-point vertical: 6.8x speedup
- Box filter 3x3: 1.4x speedup
- Shows separable kernel optimization

### 5. `examples/simd_kernels.rs` - Top 8 Image Processing Hotspots
Comprehensive collection of kernels from image-webp, jpegli-rs, zenimage:

| # | Kernel | Source | Speedup |
|---|--------|--------|---------|
| 1 | 4x4 DCT (VP8) | image-webp | Integer transform |
| 2 | 8x8 DCT butterfly | jpegli-rs | FMA chains |
| 3 | Chroma downsample 2x2 | jpegli-rs | Gather pattern |
| 4 | RGB→YCbCr | jpegli-rs | 1128 Mpix/s |
| 5 | Horizontal convolution | zenimage | Strided filter |
| 6 | sRGB↔Linear | zenimage | 1260 Mpix/s |
| 7 | Multiply/Screen/Overlay | zenimage | 526 Mpix/s |
| 8 | Horizontal reduction | zenimage | Strided sum |

---

## Key Archmage Patterns Demonstrated

### Token-Gated Dispatch
```rust
if let Some(token) = Avx2FmaToken::try_new() {
    fast_kernel(token, &mut data);
}
```

### FMA Chains for Matrix Operations
```rust
// RGB→YCbCr with 3 FMA operations
let y = r.mul_add(ky_r, g.mul_add(ky_g, b * ky_b));
```

### Alpha Broadcast
```rust
// Replicate alpha to all RGBA channels
let alpha = _mm256_permutevar8x32_ps(pixels,
    _mm256_set_epi32(7,7,7,7, 3,3,3,3));
```

### Gamma Approximation
```rust
// x^2.4 ≈ x^2 * x^0.4 using sqrt chains
let sqrt_x = adjusted.sqrt();
let sqrt_sqrt_x = sqrt_x.sqrt();      // x^0.25
let x_0125 = sqrt_sqrt_x.sqrt();      // x^0.125
let x_04_approx = sqrt_sqrt_x * x_0125; // x^0.375 ≈ x^0.4
let gamma_result = x2 * x_04_approx;
```

### Fixed-Point for u8 Data
```rust
// 15-bit fixed-point with proper rounding
const HALF_SCALE: i32 = 1 << 14;
acc = _mm256_add_epi32(acc, _mm256_mullo_epi32(data, weight));
result = _mm256_srai_epi32::<15>(acc);
```

---

## Performance Summary

| Example | Kernel | Speedup | Notes |
|---------|--------|---------|-------|
| fast_dct.rs | DCT-8x8 | 6-8x | FMA chains |
| color_convert.rs | YUV→RGB float | 3x | Clean API |
| color_convert.rs | YUV→RGB fixed | 10x | SSE2, bit-exact |
| convolution.rs | Vertical f32 | 5x | Contiguous access |
| convolution.rs | Vertical u8 | 7x | Fixed-point |
| simd_kernels.rs | sRGB→Linear | 1260 Mpix/s | Sqrt approximation |
| simd_kernels.rs | RGB→YCbCr | 1128 Mpix/s | FMA matrix |

---

## Source References

### image-webp
- `src/yuv_simd.rs` - Fixed-point YUV→RGB
- `src/transform_simd_intrinsics.rs` - 4x4 DCT/IDCT

### jpegli-rs
- `src/encode_simd.rs` - RGB→YCbCr, chroma downsampling
- `src/encode/mage_simd.rs` - Archmage-based DCT (reference implementation)

### zenimage
- `src/simd/fast.rs` - Vertical/horizontal reduction
- `src/graphics/alpha_simd.rs` - Alpha ops, sRGB↔linear, blend modes

---

## Git Status

```
On branch main
Examples added:
- e12c374 feat: add SIMD kernel examples for image processing
- 0ecd363 feat: add simd_kernels.rs with top 8 image processing hotspots
```
