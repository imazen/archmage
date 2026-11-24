//! Pinned behavior tests for polyfilled float reductions (`sub_reduce_tree!`).
//!
//! Polyfilled wide reductions fold sub-vectors lane-wise using the narrower
//! backend's `add`/`min`/`max` before a single horizontal reduction — matching
//! wider native hardware folding order (e.g., AVX-512) and keeping `min`/`max`
//! NaN behavior consistent across widths on each architecture.

/// V3 `f32x16::reduce_add` folds the two 256-bit halves lane-wise (`a[i] + a[i+8]`)
/// before one horizontal sum, matching AVX-512 folding order.
#[cfg(all(target_arch = "x86_64", target_feature = "sse2", feature = "w512"))]
#[test]
fn v3_f32x16_reduce_add_lane_fold() {
    use archmage::SimdToken;
    if let Some(token) = archmage::X64V3Token::summon() {
        let mut lanes = [0.0f32; 16];
        lanes[0] = 1e20;
        lanes[1] = 1.0;
        lanes[8] = -1e20;
        let v = magetypes::simd::generic::f32x16::from_array_t(token, lanes);
        assert_eq!(v.reduce_add().to_bits(), 1.0f32.to_bits());
    }
}

/// NEON `f32x8::reduce_min`/`max` folds halves with number-preferring `min`/`max`
/// before `vminnmvq_f32`/`vmaxnmvq_f32`, matching NEON `f32x4::reduce_min`/`max`.
#[cfg(target_arch = "aarch64")]
#[test]
fn neon_f32x8_reduce_extrema_number_preferring() {
    use archmage::SimdToken;
    if let Some(token) = archmage::NeonToken::summon() {
        let v = magetypes::simd::generic::f32x8::from_array_t(
            token,
            [f32::NAN, 2.0, 3.0, 4.0, 1.0, 6.0, 7.0, 8.0],
        );
        assert_eq!(v.reduce_min().to_bits(), 1.0f32.to_bits());
        assert_eq!(v.reduce_max().to_bits(), 8.0f32.to_bits());
    }
}

/// WASM `f64x4::reduce_add` folds the two 128-bit halves lane-wise (`f64x2_add`)
/// before a single scalar addition.
#[cfg(target_arch = "wasm32")]
#[test]
fn wasm_f64x4_reduce_add_lane_fold() {
    use archmage::SimdToken;
    if let Some(token) = archmage::Wasm128Token::summon() {
        let v = magetypes::simd::generic::f64x4::from_array_t(token, [1e100, 1.0, -1e100, 0.0]);
        assert_eq!(v.reduce_add().to_bits(), 1.0f64.to_bits());
    }
}
