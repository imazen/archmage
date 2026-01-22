# Archmage Context Handoff

## Current State

The `archmage` crate provides token-gated SIMD types with `wide`-like ergonomics. Auto-generated via `cargo run -p xtask -- generate`.

### Completed Features (30 types: f32/f64/i8-i64/u8-u64 × 128/256/512-bit)

| Feature | Status | Notes |
|---------|--------|-------|
| Token-gated construction | ✅ | `load()`, `splat()`, `zero()`, `from_array()` |
| Arithmetic ops | ✅ | `+`, `-`, `*`, `/` (float), `*` (i16/i32/u16/u32) |
| Bitwise ops | ✅ | `&`, `\|`, `^`, assignments |
| Comparisons | ✅ | `simd_eq`, `simd_lt`, `simd_le`, `simd_gt`, `simd_ge`, `simd_ne` |
| Blend/select | ✅ | `Type::blend(mask, if_true, if_false)` |
| Scalar broadcast | ✅ | `v + 2.0`, `v * 3.0` (no splat needed) |
| Horizontal reduce | ✅ | `reduce_add()`, `reduce_min()`, `reduce_max()` (float) |
| Type conversions | ✅ | `f32x8::to_i32x8()`, `f32x8::from_i32x8()` |
| Math: min/max/clamp | ✅ | |
| Math: sqrt/abs/floor/ceil/round | ✅ | Float types |
| Math: mul_add/mul_sub (FMA) | ✅ | Float types |

### Recently Added (Session 2)

| Feature | Status | Notes |
|---------|--------|-------|
| `rcp_approx()` | ✅ | Fast reciprocal approximation |
| `recip()` | ✅ | Newton-Raphson refined reciprocal |
| `rsqrt_approx()` | ✅ | Fast reciprocal sqrt approximation |
| `rsqrt()` | ✅ | Newton-Raphson refined reciprocal sqrt |
| `not()` | ✅ | Bitwise complement (all types) |
| `shl<N>()`, `shr<N>()`, `shr_arithmetic<N>()` | ✅ | Integer shifts (16/32/64-bit only, no 8-bit intrinsics) |

### TODO: Medium (Intrinsics Exist, Complex API)

- Shuffles/permutes (many variants)
- Widening conversions (i16→i32, etc.)
- Narrowing with saturation
- Gather/scatter (AVX2+)
- f32↔f64 conversions

### TODO: Transcendentals (Polynomial Approximations)

**Repository Scan Results** (zenimage, jpegli-rs, image-webp, heic, linear-srgb, yuv, yuvxyb):

| Function | Usage Count | Primary Use Cases |
|----------|-------------|-------------------|
| `powf(x, const)` | 50+ | sRGB gamma (2.4, 1/2.4), simple gamma (2.2, 1/2.2) |
| `log2` | ~20 | UltraHDR gain maps, quantization, bit depth |
| `exp2` | ~10 | Inverse of log2 for gain maps, quant fields |
| `ln` | ~5 | HLG transfer function, gamma modulation |
| `exp` | ~5 | Gaussian kernel generation |
| `log10` | ~5 | PSNR calculation, legacy transfer functions |

**Priority Order:**
1. **exp2** + **log2** - These together give `pow(x, n) = exp2(n * log2(x))`
2. **powf(x, const)** - Uses exp2 + log2, covers 90% of use cases
3. **ln** - Needed for HLG transfer function
4. **exp** - Gaussian kernels (less critical)
5. **log10** - Can be `log2(x) / log2(10)` or dedicated

**Reference Implementations:**
- `~/work/linear-srgb/src/fast_math.rs` - Working SIMD `log2_x8`, `exp2_x8`, `pow_x8` using `wide` crate
- `~/work/butteraugli/butteraugli/src/opsin.rs:53` - `fast_log2f` rational polynomial
- `wide-1.1.1/src/f32x8_.rs` - MIT-licensed polynomial implementations

**Transfer Functions:**
- **sRGB**: `1.055 * x.powf(1/2.4) - 0.055` / `((x + 0.055) / 1.055).powf(2.4)`
- **PQ (HDR)**: Complex, uses pow with m1=0.1593, m2=78.84, n=0.1593, c1=0.8359, c2=18.85, c3=18.69
- **HLG (HDR)**: Uses ln: `A * (12*x - B).ln() + C`

### TODO: AArch64/NEON

Generator structure exists but no NEON codegen yet.

## Next Steps

1. **Implement exp2/log2** in generator using polynomial approximations from linear-srgb
2. **Add pow(x, const)** as `exp2(const * log2(x))`
3. **Consider LUT vs polynomial tradeoffs** for precision vs code size

## Key Files

- `xtask/src/simd_types.rs` - SIMD type generator (1200+ lines)
- `src/simd/mod.rs` - Generated output (~245KB)
- `tests/generated_simd_types.rs` - Generated tests

## Generator Pattern

Add new operations by creating a function like:
```rust
fn generate_reciprocal_ops(ty: &SimdType) -> String {
    // Generate method code
}
```

Then call it from `generate_type()`:
```rust
code.push_str(&generate_reciprocal_ops(ty));
```

Regenerate with: `cargo run -p xtask -- generate`

## Testing

```bash
cargo test --test generated_simd_types  # SIMD type tests
cargo clippy -- -D warnings             # Lint check
```
