//! Ergonomics examples for the generic SIMD API.
//!
//! These tests demonstrate how to write platform-generic SIMD code using
//! the strategy-pattern types: `f32x8<T>`, `i32x8<T>`, etc.
//!
//! The key idea: write ONE function generic over `T: F32x8Backend`, and it
//! works with any backend — AVX2, NEON, WASM SIMD128, or scalar fallback.

#![allow(dead_code)]

use archmage::{ScalarToken, SimdToken};
use magetypes::simd::backends::{F32x4Backend, F32x8Backend, F32x8Convert};
use magetypes::simd::generic::{f32x4, f32x8, i32x8};

#[cfg(target_arch = "x86_64")]
use archmage::X64V3Token;

#[cfg(target_arch = "x86_64")]
fn get_x86() -> X64V3Token {
    X64V3Token::summon().expect("tests require AVX2+FMA")
}

// ============================================================================
// Example 1: Basic vector math — one function, any backend
// ============================================================================

/// Sum a slice of f32 values using 8-lane SIMD.
///
/// This function works identically with X64V3Token (AVX2), NeonToken (polyfill),
/// or ScalarToken (array ops). The compiler monomorphizes per backend.
fn sum_f32x8<T: F32x8Backend>(token: T, data: &[f32]) -> f32 {
    let mut acc = f32x8::<T>::zero(token);
    let (chunks, remainder) = data.split_at(data.len() - data.len() % 8);

    for chunk in chunks.chunks_exact(8) {
        let v = f32x8::<T>::load(token, chunk.try_into().unwrap());
        acc = acc + v;
    }

    let mut total = acc.reduce_add();
    for &x in remainder {
        total += x;
    }
    total
}

#[test]
fn example_sum_scalar() {
    let data: Vec<f32> = (1..=100).map(|i| i as f32).collect();
    let result = sum_f32x8(ScalarToken, &data);
    assert!((result - 5050.0).abs() < 0.01);
}

#[cfg(target_arch = "x86_64")]
#[test]
fn example_sum_x86() {
    let data: Vec<f32> = (1..=100).map(|i| i as f32).collect();
    let result = sum_f32x8(get_x86(), &data);
    assert!((result - 5050.0).abs() < 0.01);
}

#[cfg(target_arch = "x86_64")]
#[test]
fn example_sum_cross_backend() {
    let data: Vec<f32> = (1..=100).map(|i| i as f32).collect();
    let x86 = sum_f32x8(get_x86(), &data);
    let scalar = sum_f32x8(ScalarToken, &data);
    assert!((x86 - scalar).abs() < 0.01, "backends must agree");
}

// ============================================================================
// Example 2: Dot product — natural operators
// ============================================================================

fn dot_product<T: F32x8Backend>(token: T, a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    let n = a.len();
    let mut acc = f32x8::<T>::zero(token);
    let chunks = n / 8;

    for i in 0..chunks {
        let va = f32x8::<T>::load(token, a[i * 8..][..8].try_into().unwrap());
        let vb = f32x8::<T>::load(token, b[i * 8..][..8].try_into().unwrap());
        acc = va.mul_add(vb, acc); // FMA: acc += va * vb
    }

    let mut total = acc.reduce_add();
    for i in (chunks * 8)..n {
        total += a[i] * b[i];
    }
    total
}

#[test]
fn example_dot_product() {
    let a: Vec<f32> = (0..64).map(|i| i as f32).collect();
    let b: Vec<f32> = (0..64).map(|i| (64 - i) as f32).collect();
    let expected: f32 = a.iter().zip(&b).map(|(x, y)| x * y).sum();

    let result = dot_product(ScalarToken, &a, &b);
    assert!((result - expected).abs() < 1.0, "{result} vs {expected}");

    #[cfg(target_arch = "x86_64")]
    {
        let result = dot_product(get_x86(), &a, &b);
        assert!((result - expected).abs() < 1.0, "{result} vs {expected}");
    }
}

// ============================================================================
// Example 3: Image brightness — scalar broadcast operators
// ============================================================================

fn adjust_brightness<T: F32x8Backend>(token: T, pixels: &mut [f32], amount: f32) {
    let zero = f32x8::<T>::zero(token);
    let max = f32x8::<T>::splat(token, 255.0);

    let chunks = pixels.len() / 8;
    for i in 0..chunks {
        let chunk: &mut [f32; 8] = (&mut pixels[i * 8..i * 8 + 8]).try_into().unwrap();
        let v = f32x8::<T>::load(token, chunk);
        let adjusted = (v + amount).clamp(zero, max); // scalar broadcast!
        adjusted.store(chunk);
    }
    // Handle remainder
    for p in &mut pixels[chunks * 8..] {
        *p = (*p + amount).clamp(0.0, 255.0);
    }
}

#[test]
fn example_brightness() {
    let mut pixels = vec![100.0_f32, 200.0, 50.0, 250.0, 10.0, 128.0, 0.0, 255.0];
    adjust_brightness(ScalarToken, &mut pixels, 30.0);
    assert_eq!(
        pixels,
        [130.0, 230.0, 80.0, 255.0, 40.0, 158.0, 30.0, 255.0]
    );
}

// ============================================================================
// Example 4: SoA color processing — deinterleave + process + reinterleave
// ============================================================================

fn apply_gamma_correction<T: F32x8Backend + F32x8Convert>(token: T, rgba_pixels: &mut [u8; 32]) {
    let (r, g, b, a) = f32x8::<T>::load_8_rgba_u8(rgba_pixels);

    // Normalize to [0, 1]
    let inv255 = f32x8::<T>::splat(token, 1.0 / 255.0);
    let r = r * inv255;
    let g = g * inv255;
    let b = b * inv255;

    // Apply gamma 2.2 (approximate: square then sqrt for ~1.5 power)
    // Just demonstrating the operations — not real gamma correction
    let r = r.sqrt();
    let g = g.sqrt();
    let b = b.sqrt();

    // Back to [0, 255]
    let scale = f32x8::<T>::splat(token, 255.0);
    let r = r * scale;
    let g = g * scale;
    let b = b * scale;
    let a = a; // Alpha unchanged

    *rgba_pixels = f32x8::<T>::store_8_rgba_u8(r, g, b, a);
}

#[test]
fn example_gamma_correction() {
    let mut pixels = [0u8; 32];
    // Set pixel 0: RGBA = (64, 128, 196, 255)
    pixels[0] = 64;
    pixels[1] = 128;
    pixels[2] = 196;
    pixels[3] = 255;

    apply_gamma_correction(ScalarToken, &mut pixels);

    // After sqrt gamma: sqrt(64/255)*255 ≈ 128, sqrt(128/255)*255 ≈ 181, etc.
    assert!((pixels[0] as f32 - 128.0).abs() < 2.0);
    assert!((pixels[1] as f32 - 181.0).abs() < 2.0);
    assert!((pixels[2] as f32 - 223.0).abs() < 2.0);
    assert_eq!(pixels[3], 255); // Alpha unchanged
}

// ============================================================================
// Example 5: Transpose — matrix operations
// ============================================================================

fn transpose_8x8_generic<T: F32x8Backend>(_token: T, matrix: &[f32; 64]) -> [f32; 64] {
    let rows = f32x8::<T>::load_8x8(matrix);
    let transposed = f32x8::<T>::transpose_8x8_copy(rows);
    let mut out = [0.0f32; 64];
    f32x8::<T>::store_8x8(&transposed, &mut out);
    out
}

#[test]
fn example_transpose() {
    // Identity-like: row i has value (i+1) in every column
    let mut matrix = [0.0f32; 64];
    for i in 0..8 {
        for j in 0..8 {
            matrix[i * 8 + j] = (i * 8 + j) as f32;
        }
    }

    let result = transpose_8x8_generic(ScalarToken, &matrix);

    // After transpose, result[i][j] == original[j][i]
    for i in 0..8 {
        for j in 0..8 {
            assert_eq!(result[i * 8 + j], matrix[j * 8 + i]);
        }
    }
}

// ============================================================================
// Example 6: Conditional operations — comparisons and blending
// ============================================================================

fn clamp_negatives_to_zero<T: F32x8Backend>(token: T, data: &[f32; 8]) -> [f32; 8] {
    let v = f32x8::<T>::load(token, data);
    let zero = f32x8::<T>::zero(token);
    let mask = v.simd_lt(zero); // true where v < 0
    let result = f32x8::<T>::blend(mask, zero, v); // select zero where negative
    result.to_array()
}

#[test]
fn example_clamp_negatives() {
    let data = [-1.0, 2.0, -3.0, 4.0, -5.0, 6.0, -7.0, 8.0];
    let result = clamp_negatives_to_zero(ScalarToken, &data);
    assert_eq!(result, [0.0, 2.0, 0.0, 4.0, 0.0, 6.0, 0.0, 8.0]);

    #[cfg(target_arch = "x86_64")]
    {
        let result = clamp_negatives_to_zero(get_x86(), &data);
        assert_eq!(result, [0.0, 2.0, 0.0, 4.0, 0.0, 6.0, 0.0, 8.0]);
    }
}

// ============================================================================
// Example 7: Cross-type conversion — f32 ↔ i32
// ============================================================================

fn quantize_to_int<T: F32x8Convert>(token: T, values: &[f32; 8], scale: f32) -> [i32; 8] {
    let v = f32x8::<T>::load(token, values);
    let scaled = v * scale;
    scaled.to_i32_round().to_array()
}

fn dequantize_from_int<T: F32x8Convert>(token: T, values: &[i32; 8], scale: f32) -> [f32; 8] {
    let v = i32x8::<T>::load(token, values);
    let f = f32x8::<T>::from_i32(token, v);
    (f / scale).to_array()
}

#[test]
fn example_quantize() {
    let values = [1.5, -2.3, 3.7, -4.1, 5.5, -6.9, 7.2, -8.8];
    let quantized = quantize_to_int(ScalarToken, &values, 100.0);
    assert_eq!(quantized, [150, -230, 370, -410, 550, -690, 720, -880]);

    let back = dequantize_from_int(ScalarToken, &quantized, 100.0);
    for i in 0..8 {
        assert!((back[i] - values[i]).abs() < 0.01);
    }
}

// ============================================================================
// Example 8: 4-lane operations — f32x4
// ============================================================================

fn normalize_vec4<T: F32x4Backend>(token: T, v: &[f32; 4]) -> [f32; 4] {
    let vec = f32x4::<T>::load(token, v);
    let sq = vec * vec;
    let len_sq = sq.reduce_add();
    if len_sq < 1e-10 {
        return [0.0; 4];
    }
    let inv_len = 1.0 / len_sq.sqrt();
    (vec * inv_len).to_array()
}

#[test]
fn example_normalize() {
    let v = [3.0, 4.0, 0.0, 0.0];
    let result = normalize_vec4(ScalarToken, &v);
    assert!((result[0] - 0.6).abs() < 1e-6);
    assert!((result[1] - 0.8).abs() < 1e-6);
    assert!(result[2].abs() < 1e-6);
    assert!(result[3].abs() < 1e-6);
}

// ============================================================================
// Example 9: Interleave — AoS ↔ SoA conversion
// ============================================================================

fn separate_channels<T: F32x8Backend>(token: T, interleaved: &[[f32; 8]; 4]) -> [[f32; 8]; 4] {
    let vecs: [f32x8<T>; 4] = core::array::from_fn(|i| f32x8::<T>::load(token, &interleaved[i]));
    let channels = f32x8::<T>::deinterleave_4ch(vecs);
    core::array::from_fn(|i| channels[i].to_array())
}

#[test]
fn example_deinterleave() {
    // 8 RGBA pixels packed as 4 vectors of 8 (2 pixels per vector)
    let interleaved = [
        [1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0], // R0,G0,B0,A0,R1,G1,B1,A1
        [0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0], // R2,G2,B2,A2,R3,G3,B3,A3
        [0.5, 0.5, 0.5, 1.0, 0.0, 0.0, 0.0, 1.0], // R4,G4,B4,A4,R5,G5,B5,A5
        [1.0, 1.0, 1.0, 1.0, 0.5, 0.0, 0.5, 1.0], // R6,G6,B6,A6,R7,G7,B7,A7
    ];

    let [r, g, b, a] = separate_channels(ScalarToken, &interleaved);

    // R channel: R0..R7
    assert_eq!(r, [1.0, 0.0, 0.0, 1.0, 0.5, 0.0, 1.0, 0.5]);
    // G channel
    assert_eq!(g, [0.0, 1.0, 0.0, 1.0, 0.5, 0.0, 1.0, 0.0]);
    // B channel
    assert_eq!(b, [0.0, 0.0, 1.0, 0.0, 0.5, 0.0, 1.0, 0.5]);
    // A channel (all 1.0)
    assert_eq!(a, [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]);
}

// ============================================================================
// Example 10: Dispatch pattern — entry point → generic kernel
// ============================================================================

/// The dispatch pattern: public API calls into a generic kernel.
///
/// This is the recommended way to use the generic types:
/// 1. Public function does runtime detection
/// 2. Calls generic kernel with the appropriate token
/// 3. Generic kernel is monomorphized per backend
fn sum_dispatch(data: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    if let Some(token) = X64V3Token::summon() {
        return sum_f32x8(token, data);
    }

    // Scalar fallback (always available)
    sum_f32x8(ScalarToken, data)
}

#[test]
fn example_dispatch() {
    let data: Vec<f32> = (1..=100).map(|i| i as f32).collect();
    let result = sum_dispatch(&data);
    assert!((result - 5050.0).abs() < 0.01);
}

// ============================================================================
// Example 11: Platform-specific raw access (escape hatch)
// ============================================================================

#[cfg(target_arch = "x86_64")]
#[test]
fn example_raw_access() {
    let t = get_x86();
    let v = f32x8::<X64V3Token>::splat(t, 42.0);

    // Get the raw __m256 for manual intrinsic use
    let raw: core::arch::x86_64::__m256 = v.raw();

    // Create from raw __m256
    let back = f32x8::from_m256(t, raw);
    assert_eq!(back.to_array(), [42.0; 8]);
}

// ============================================================================
// Example 12: Bitcast — reinterpret float bits as integers
// ============================================================================

fn float_sign_bits<T: F32x8Convert>(token: T, values: &[f32; 8]) -> [bool; 8] {
    let v = f32x8::<T>::load(token, values);
    let bits = v.bitcast_to_i32();
    let arr = bits.to_array();
    core::array::from_fn(|i| arr[i] < 0)
}

#[test]
fn example_bitcast() {
    let values = [
        1.0,
        -2.0,
        3.0,
        -4.0,
        0.0,
        -0.0,
        f32::INFINITY,
        f32::NEG_INFINITY,
    ];
    let signs = float_sign_bits(ScalarToken, &values);
    assert_eq!(signs, [false, true, false, true, false, true, false, true]);
}

// ============================================================================
// Example 13: Byte-level access and serialization
// ============================================================================

fn serialize_vector<T: F32x8Backend>(v: f32x8<T>) -> [u8; 32] {
    *v.as_bytes()
}

fn deserialize_vector<T: F32x8Backend>(token: T, bytes: &[u8; 32]) -> f32x8<T> {
    f32x8::<T>::from_bytes(token, bytes)
}

#[test]
fn example_serialization() {
    let original = [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let v = f32x8::<ScalarToken>::load(ScalarToken, &original);
    let bytes = serialize_vector(v);
    let restored = deserialize_vector(ScalarToken, &bytes);
    assert_eq!(restored.to_array(), original);
}

// ============================================================================
// Example 14: Slice casting — zero-copy reinterpret of aligned data
// ============================================================================

#[test]
fn example_cast_slice() {
    // Aligned allocation (Vec guarantees alignment for primitive types)
    let data: Vec<f32> = (0..64).map(|i| i as f32).collect();

    // Try to cast the slice as a slice of f32x8 vectors
    if let Some(vectors) = f32x8::<ScalarToken>::cast_slice(ScalarToken, &data) {
        assert_eq!(vectors.len(), 8); // 64 / 8 = 8 vectors
        assert_eq!(
            vectors[0].to_array(),
            [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
        );
        assert_eq!(
            vectors[7].to_array(),
            [56.0, 57.0, 58.0, 59.0, 60.0, 61.0, 62.0, 63.0]
        );
    }
    // Note: cast_slice returns None if alignment is wrong or length not multiple of 8
}

// ============================================================================
// Example 15: 4x4 matrix multiply (f32x4)
// ============================================================================

fn mat4_mul<T: F32x4Backend>(token: T, a: &[f32; 16], b: &[f32; 16]) -> [f32; 16] {
    // Load B as column vectors
    let b_cols: [f32x4<T>; 4] = core::array::from_fn(|j| {
        f32x4::<T>::from_array(token, [b[j], b[4 + j], b[8 + j], b[12 + j]])
    });

    let mut result = [0.0f32; 16];
    for i in 0..4 {
        let row = f32x4::<T>::load(token, a[i * 4..][..4].try_into().unwrap());
        for j in 0..4 {
            result[i * 4 + j] = (row * b_cols[j]).reduce_add();
        }
    }
    result
}

#[test]
fn example_mat4_mul() {
    #[rustfmt::skip]
    let identity = [
        1.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 1.0,
    ];
    #[rustfmt::skip]
    let m = [
        1.0, 2.0, 3.0, 4.0,
        5.0, 6.0, 7.0, 8.0,
        9.0, 10.0, 11.0, 12.0,
        13.0, 14.0, 15.0, 16.0,
    ];

    let result = mat4_mul(ScalarToken, &identity, &m);
    assert_eq!(result, m, "identity * m == m");

    let result = mat4_mul(ScalarToken, &m, &identity);
    assert_eq!(result, m, "m * identity == m");
}

// ============================================================================
// Example 16: Transcendental math
// ============================================================================

fn softmax_f32x8<T: F32x8Convert>(token: T, logits: &[f32; 8]) -> [f32; 8] {
    let v = f32x8::<T>::load(token, logits);

    // Subtract max for numerical stability
    let max_val = f32x8::<T>::splat(token, v.reduce_max());
    let shifted = v - max_val;

    // exp(shifted)
    let exp_vals = shifted.exp_lowp();

    // Normalize by sum
    let sum = exp_vals.reduce_add();
    let result = exp_vals / sum;

    result.to_array()
}

#[test]
fn example_softmax() {
    let logits = [1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0];
    let probs = softmax_f32x8(ScalarToken, &logits);

    // All probabilities should be positive
    for &p in &probs {
        assert!(p > 0.0, "probability must be positive");
    }

    // Sum should be ~1.0
    let sum: f32 = probs.iter().sum();
    assert!(
        (sum - 1.0).abs() < 0.01,
        "probabilities must sum to 1, got {sum}"
    );

    // Symmetric: first 4 == last 4
    for i in 0..4 {
        assert!((probs[i] - probs[i + 4]).abs() < 0.01);
    }
}

// ============================================================================
// Example 17: u8 conversion — image pixel processing
// ============================================================================

fn invert_pixels<T: F32x8Backend>(token: T, pixels: &[u8; 8]) -> [u8; 8] {
    let v = f32x8::<T>::from_u8(pixels);
    let max = f32x8::<T>::splat(token, 255.0);
    let inverted = max - v;
    inverted.to_u8()
}

#[test]
fn example_invert_pixels() {
    let pixels = [0, 64, 128, 192, 255, 1, 127, 254];
    let inverted = invert_pixels(ScalarToken, &pixels);
    assert_eq!(inverted, [255, 191, 127, 63, 0, 254, 128, 1]);
}

// ============================================================================
// Example 18: Index access — per-lane read/write
// ============================================================================

#[test]
fn example_indexing() {
    let t = ScalarToken;
    let mut v =
        f32x8::<ScalarToken>::from_array(t, [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0]);

    // Read individual lanes
    assert_eq!(v[0], 10.0);
    assert_eq!(v[7], 80.0);

    // Write individual lanes
    v[3] = 999.0;
    assert_eq!(v[3], 999.0);
    assert_eq!(
        v.to_array(),
        [10.0, 20.0, 30.0, 999.0, 50.0, 60.0, 70.0, 80.0]
    );
}

// ============================================================================
// Example 19: Debug display
// ============================================================================

#[test]
fn example_debug() {
    let v = f32x8::<ScalarToken>::from_array(ScalarToken, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    let dbg = format!("{v:?}");
    assert!(dbg.contains("f32x8"));
    assert!(dbg.contains("1.0"));
    assert!(dbg.contains("8.0"));
}

// ============================================================================
// Example 20: as_array — zero-copy reference access
// ============================================================================

#[test]
fn example_as_array() {
    let mut v =
        f32x8::<ScalarToken>::from_array(ScalarToken, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);

    // Borrow as slice — no copy
    let arr: &[f32; 8] = v.as_array();
    assert_eq!(arr[0], 1.0);

    // Mutable borrow — modify in-place
    let arr_mut: &mut [f32; 8] = v.as_array_mut();
    arr_mut[0] = 99.0;
    assert_eq!(v[0], 99.0);
}

// ============================================================================
// Example 21: Reductions — min, max, add
// ============================================================================

fn stats<T: F32x8Backend>(token: T, data: &[f32; 8]) -> (f32, f32, f32) {
    let v = f32x8::<T>::load(token, data);
    (v.reduce_min(), v.reduce_max(), v.reduce_add())
}

#[test]
fn example_reductions() {
    let data = [3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0];
    let (min, max, sum) = stats(ScalarToken, &data);
    assert_eq!(min, 1.0);
    assert_eq!(max, 9.0);
    assert!((sum - 31.0).abs() < 0.01);
}

// ============================================================================
// Example 22: Approximations — fast reciprocal and inverse sqrt
// ============================================================================

fn fast_normalize<T: F32x8Backend>(token: T, data: &[f32; 8]) -> [f32; 8] {
    let v = f32x8::<T>::load(token, data);
    let sq = v * v;
    let sum = f32x8::<T>::splat(token, sq.reduce_add());
    let inv_len = sum.rsqrt_approx(); // ~12-bit accuracy
    (v * inv_len).to_array()
}

#[test]
fn example_fast_normalize() {
    let data = [3.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    let result = fast_normalize(ScalarToken, &data);
    // Should be approximately [0.6, 0.8, 0, 0, ...]
    assert!((result[0] - 0.6).abs() < 0.01);
    assert!((result[1] - 0.8).abs() < 0.01);
}
