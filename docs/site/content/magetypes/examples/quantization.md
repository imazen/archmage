+++
title = "Quantization with Masks"
description = "JPEG block quantization using f32x8 and i32x8 with comparisons, blends, and type conversion"
weight = 4
+++

This example from `zenjpeg` shows a more complex SIMD pattern: using two different vector types together (`f32x8` and `i32x8`), performing comparisons to generate masks, and blending results based on those masks. This is the core of JPEG adaptive quantization — deciding which DCT coefficients to zero out based on perceptual thresholds.

## Block quantization with zero-biasing

Each 8x8 DCT block has 64 coefficients. We process them in rows of 8 — a natural fit for `f32x8`. The algorithm:

1. Multiply each coefficient by the quantization multiplier
2. Compare the absolute value against a perceptual threshold
3. Round to integer
4. Zero out coefficients below the threshold (via mask blend)

```rust
use archmage::prelude::*;
use magetypes::simd::generic::f32x8 as GenericF32x8;
use magetypes::simd::generic::i32x8 as GenericI32x8;

/// Quantize an 8x8 f32 block with adaptive zero-biasing.
///
/// Generates one function per tier: _v3 (AVX2), _neon, _wasm128, _scalar.
#[magetypes(v3, neon, wasm128, scalar)]
fn quantize_block(
    token: Token,
    block: &[[f32; 8]; 8],          // 8x8 DCT coefficients
    mul_rows: &[[f32; 8]; 8],       // quantization multipliers
    zero_bias: &ZeroBiasSimd,       // perceptual thresholds
    aq_strength: f32,               // adaptive quantization strength
) -> [i16; 64] {
    #[allow(non_camel_case_types)]
    type f32x8 = GenericF32x8<Token>;
    #[allow(non_camel_case_types)]
    type i32x8 = GenericI32x8<Token>;

    let aq_m = f32x8::splat(token, aq_strength);
    let zero_i32 = i32x8::zero(token);
    let mut result = [0i16; 64];

    for row in 0..8 {
        // Load one row of 8 coefficients
        let block_m = f32x8::from_array(token, block[row]);
        let mul_m = f32x8::from_array(token, mul_rows[row]);
        let offset_m = f32x8::from_array(token, zero_bias.offset_rows[row]);
        let bias_mul_m = f32x8::from_array(token, zero_bias.mul_rows[row]);

        // Quantize: coefficient * multiplier
        let qval = block_m * mul_m;

        // Compute perceptual threshold: bias_mul * aq_strength + offset
        let threshold = bias_mul_m.mul_add(aq_m, offset_m);

        // Compare |qval| >= threshold → mask of all-1s or all-0s per lane
        let abs_qval = qval.abs();
        let mask = abs_qval.simd_ge(threshold);

        // Round to nearest integer (f32 → i32)
        let rounded = qval.to_i32_round();

        // Cross-type blend: use the f32 comparison mask to select i32 values.
        // bitcast_to_i32 reinterprets the f32 mask bits as i32.
        let mask_i32 = mask.bitcast_to_i32();
        let blended = i32x8::blend(mask_i32, rounded, zero_i32);

        // Extract to scalar array for output
        let arr = blended.to_array();
        let k = row * 8;
        result[k] = arr[0] as i16;
        result[k + 1] = arr[1] as i16;
        result[k + 2] = arr[2] as i16;
        result[k + 3] = arr[3] as i16;
        result[k + 4] = arr[4] as i16;
        result[k + 5] = arr[5] as i16;
        result[k + 6] = arr[6] as i16;
        result[k + 7] = arr[7] as i16;
    }

    result
}
```

## What's happening in the SIMD

The interesting part is the mask-based blend:

```
abs_qval:    [0.3, 5.2, 0.1, 8.7, 0.0, 3.1, 0.4, 12.0]
threshold:   [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,  1.0]
mask (ge):   [  0, -1,   0,  -1,   0,  -1,   0,   -1 ]  (all-0 or all-1 bits)
rounded:     [  0,  5,   0,   9,   0,   3,   0,   12 ]
blended:     [  0,  5,   0,   9,   0,   3,   0,   12 ]  (zero where mask is 0)
```

On AVX2, `simd_ge` compiles to `_mm256_cmp_ps(..., _CMP_GE_OQ)`. `bitcast_to_i32` is `_mm256_castps_si256` (zero-cost register reinterpretation). `blend` is `_mm256_blendv_epi8` or equivalent.

On NEON, the same operations compile to `vcgeq_f32` (comparison) and `vbslq_s32` (bitwise select). The polyfill for `f32x8` runs two `f32x4` operations — the API is identical.

## Multiple SIMD types in one function

This pattern — using `f32x8` and `i32x8` together — works because both types are parameterized by the same `Token`. The `#[magetypes]` macro substitutes `Token` in both type aliases simultaneously, so the `f32x8` and `i32x8` use compatible representations on each platform.

The `bitcast_to_i32()` method handles the type conversion: it reinterprets the float comparison mask's bit pattern as an integer mask. This is a register reinterpretation (zero-cost on x86 and ARM), not a numeric conversion.

## Dispatch

```rust
pub fn quantize_block(
    block: &[[f32; 8]; 8],
    mul_rows: &[[f32; 8]; 8],
    zero_bias: &ZeroBiasSimd,
    aq_strength: f32,
) -> [i16; 64] {
    incant!(quantize_block(block, mul_rows, zero_bias, aq_strength))
}
```

The `v3` tier (AVX2) uses native 256-bit `f32x8` and `i32x8`. NEON and WASM128 use the polyfilled 2x128-bit versions. Scalar uses array-based emulation. All produce identical results.
