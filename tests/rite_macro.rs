//! Tests for the #[rite] macro - inner SIMD helpers that inline into matching callers.

#![cfg(target_arch = "x86_64")]

use archmage::{Desktop64, SimdToken, X64V3Token, arcane, rite};
use std::arch::x86_64::*;

// Helper function using #[rite] - no wrapper, just target_feature annotation
#[rite]
fn add_vectors(_token: X64V3Token, a: &[f32; 8], b: &[f32; 8]) -> [f32; 8] {
    unsafe {
        let va = _mm256_loadu_ps(a.as_ptr());
        let vb = _mm256_loadu_ps(b.as_ptr());
        let sum = _mm256_add_ps(va, vb);
        let mut out = [0.0f32; 8];
        _mm256_storeu_ps(out.as_mut_ptr(), sum);
        out
    }
}

#[rite]
fn mul_vectors(_token: X64V3Token, a: &[f32; 8], b: &[f32; 8]) -> [f32; 8] {
    unsafe {
        let va = _mm256_loadu_ps(a.as_ptr());
        let vb = _mm256_loadu_ps(b.as_ptr());
        let prod = _mm256_mul_ps(va, vb);
        let mut out = [0.0f32; 8];
        _mm256_storeu_ps(out.as_mut_ptr(), prod);
        out
    }
}

#[rite]
fn horizontal_sum(_token: X64V3Token, v: __m256) -> f32 {
    // No unsafe needed - value-based intrinsics are safe inside #[target_feature] (Rust 1.85+)
    let sum = _mm256_hadd_ps(v, v);
    let sum = _mm256_hadd_ps(sum, sum);
    let low = _mm256_castps256_ps128(sum);
    let high = _mm256_extractf128_ps::<1>(sum);
    _mm_cvtss_f32(_mm_add_ss(low, high))
}

// Entry point using #[arcane] calls #[rite] helpers
#[arcane]
fn dot_product(token: X64V3Token, a: &[f32; 8], b: &[f32; 8]) -> f32 {
    // These calls should inline directly - same target features, no boundary
    let products = mul_vectors(token, a, b);
    unsafe {
        let v = _mm256_loadu_ps(products.as_ptr());
        horizontal_sum(token, v)
    }
}

// Complex example with multiple #[rite] calls
#[arcane]
fn weighted_sum(
    token: X64V3Token,
    a: &[f32; 8],
    b: &[f32; 8],
    weight_a: f32,
    weight_b: f32,
) -> f32 {
    // Scale a
    let scaled_a = {
        let weights = [weight_a; 8];
        mul_vectors(token, a, &weights)
    };
    // Scale b
    let scaled_b = {
        let weights = [weight_b; 8];
        mul_vectors(token, b, &weights)
    };
    // Add and sum
    let sum = add_vectors(token, &scaled_a, &scaled_b);
    unsafe {
        let v = _mm256_loadu_ps(sum.as_ptr());
        horizontal_sum(token, v)
    }
}

#[test]
fn test_rite_basic() {
    if let Some(token) = X64V3Token::summon() {
        let a = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let b = [1.0f32; 8];

        // Direct call to #[rite] function requires unsafe
        // (Safe when called from #[arcane] context)
        let sum = unsafe { add_vectors(token, &a, &b) };
        assert_eq!(sum, [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
    }
}

#[test]
fn test_rite_from_arcane() {
    if let Some(token) = X64V3Token::summon() {
        let a = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let b = [2.0f32; 8];

        let result = dot_product(token, &a, &b);
        // 1*2 + 2*2 + 3*2 + 4*2 + 5*2 + 6*2 + 7*2 + 8*2 = 2*(1+2+3+4+5+6+7+8) = 2*36 = 72
        assert_eq!(result, 72.0);
    }
}

#[test]
fn test_rite_complex() {
    if let Some(token) = X64V3Token::summon() {
        let a = [1.0f32; 8];
        let b = [2.0f32; 8];

        let result = weighted_sum(token, &a, &b, 0.5, 0.5);
        // 0.5 * 1.0 * 8 + 0.5 * 2.0 * 8 = 4 + 8 = 12
        assert_eq!(result, 12.0);
    }
}

// ============================================================================
// Tier-based #[rite(v3)] — no token parameter needed
// ============================================================================

// Helper with tier name instead of token parameter
#[rite(v3)]
fn add_vectors_tierless(a: &[f32; 8], b: &[f32; 8]) -> [f32; 8] {
    unsafe {
        let va = _mm256_loadu_ps(a.as_ptr());
        let vb = _mm256_loadu_ps(b.as_ptr());
        let sum = _mm256_add_ps(va, vb);
        let mut out = [0.0f32; 8];
        _mm256_storeu_ps(out.as_mut_ptr(), sum);
        out
    }
}

// Tier + import_intrinsics — safe memory ops, no token param
#[rite(v3, import_intrinsics)]
fn mul_vectors_tierless(a: &[f32; 8], b: &[f32; 8]) -> [f32; 8] {
    let va = _mm256_loadu_ps(a);
    let vb = _mm256_loadu_ps(b);
    let prod = _mm256_mul_ps(va, vb);
    let mut out = [0.0f32; 8];
    _mm256_storeu_ps(&mut out, prod);
    out
}

// Entry point calling tier-based #[rite] helpers
#[arcane]
fn dot_product_tierless(token: X64V3Token, a: &[f32; 8], b: &[f32; 8]) -> f32 {
    let products = mul_vectors_tierless(a, b); // no token needed!
    unsafe {
        let v = _mm256_loadu_ps(products.as_ptr());
        horizontal_sum(token, v)
    }
}

#[test]
fn test_rite_tier_basic() {
    if X64V3Token::summon().is_some() {
        let a = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let b = [1.0f32; 8];

        let sum = unsafe { add_vectors_tierless(&a, &b) };
        assert_eq!(sum, [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
    }
}

#[test]
fn test_rite_tier_import_intrinsics() {
    if X64V3Token::summon().is_some() {
        let a = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let b = [2.0f32; 8];

        let products = unsafe { mul_vectors_tierless(&a, &b) };
        assert_eq!(products, [2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0]);
    }
}

#[test]
fn test_rite_tier_from_arcane() {
    if let Some(token) = X64V3Token::summon() {
        let a = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let b = [2.0f32; 8];

        let result = dot_product_tierless(token, &a, &b);
        assert_eq!(result, 72.0);
    }
}

// Tier with stub — generates unreachable stub on wrong arch
#[rite(v3, stub)]
fn negate_tierless(a: &[f32; 8]) -> [f32; 8] {
    let zero = _mm256_setzero_ps();
    unsafe {
        let va = _mm256_loadu_ps(a.as_ptr());
        let neg = _mm256_sub_ps(zero, va);
        let mut out = [0.0f32; 8];
        _mm256_storeu_ps(out.as_mut_ptr(), neg);
        out
    }
}

// V2 tier — SSE4.2 level (128-bit)
#[rite(v2)]
fn popcount_tierless(val: i32) -> i32 {
    // POPCNT is available at v2 — safe inside #[target_feature] (Rust 1.85+)
    core::arch::x86_64::_popcnt32(val)
}

#[test]
fn test_rite_tier_v2() {
    if archmage::X64V2Token::summon().is_some() {
        let result = unsafe { popcount_tierless(0b1010_1010) };
        assert_eq!(result, 4);
    }
}

#[test]
fn test_rite_tier_stub() {
    if X64V3Token::summon().is_some() {
        let a = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let result = unsafe { negate_tierless(&a) };
        assert_eq!(result, [-1.0, -2.0, -3.0, -4.0, -5.0, -6.0, -7.0, -8.0]);
    }
}

// Wildcard token parameter: `_: TokenType` should be accepted
#[rite]
fn scale_vector(_: X64V3Token, a: &[f32; 8], factor: f32) -> [f32; 8] {
    unsafe {
        let va = _mm256_loadu_ps(a.as_ptr());
        let vf = _mm256_set1_ps(factor);
        let result = _mm256_mul_ps(va, vf);
        let mut out = [0.0f32; 8];
        _mm256_storeu_ps(out.as_mut_ptr(), result);
        out
    }
}

#[test]
fn test_rite_wildcard_token() {
    if let Some(token) = X64V3Token::summon() {
        let a = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let result = unsafe { scale_vector(token, &a, 3.0) };
        assert_eq!(result, [3.0, 6.0, 9.0, 12.0, 15.0, 18.0, 21.0, 24.0]);
    }
}

#[test]
fn test_rite_with_desktop64_alias() {
    if let Some(token) = Desktop64::summon() {
        let a = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let b = [1.0f32; 8];

        // Desktop64 = X64V3Token, so this works
        // Direct call requires unsafe (safe from #[arcane] context)
        let sum = unsafe { add_vectors(token, &a, &b) };
        assert_eq!(sum, [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
    }
}

// ============================================================================
// Extended tier-based #[rite] tests
// ============================================================================

// --- V1 tier (SSE2 baseline) ---
// V1 is always available on x86_64 — SSE2 is in the baseline spec.

#[rite(v1)]
fn add_i32x4_v1(a: &[i32; 4], b: &[i32; 4]) -> [i32; 4] {
    unsafe {
        let va = _mm_loadu_si128(a.as_ptr().cast());
        let vb = _mm_loadu_si128(b.as_ptr().cast());
        let sum = _mm_add_epi32(va, vb);
        let mut out = [0i32; 4];
        _mm_storeu_si128(out.as_mut_ptr().cast(), sum);
        out
    }
}

#[rite(v1)]
fn f64_add_v1(a: &[f64; 2], b: &[f64; 2]) -> [f64; 2] {
    unsafe {
        let va = _mm_loadu_pd(a.as_ptr());
        let vb = _mm_loadu_pd(b.as_ptr());
        let sum = _mm_add_pd(va, vb);
        let mut out = [0.0f64; 2];
        _mm_storeu_pd(out.as_mut_ptr(), sum);
        out
    }
}

#[test]
fn test_rite_tier_v1_i32_add() {
    use archmage::X64V1Token;
    // V1 (SSE2) is always available on x86_64
    if X64V1Token::summon().is_some() {
        let a = [10i32, 20, 30, 40];
        let b = [1, 2, 3, 4];
        let result = unsafe { add_i32x4_v1(&a, &b) };
        assert_eq!(result, [11, 22, 33, 44]);
    }
}

#[test]
fn test_rite_tier_v1_f64_add() {
    use archmage::X64V1Token;
    if X64V1Token::summon().is_some() {
        let a = [1.5f64, 2.5];
        let b = [3.0, 4.0];
        let result = unsafe { f64_add_v1(&a, &b) };
        assert_eq!(result, [4.5, 6.5]);
    }
}

// --- V2 tier (SSE4.2 + POPCNT) ---

#[rite(v2)]
fn blend_i16_v2(a: &[i16; 8], b: &[i16; 8]) -> [i16; 8] {
    // _mm_blendv_epi8 is SSE4.1 (included in v2)
    unsafe {
        let va = _mm_loadu_si128(a.as_ptr().cast());
        let vb = _mm_loadu_si128(b.as_ptr().cast());
        // Use a constant blend mask: select alternating from a and b
        // _mm_blend_epi16 takes an imm8 — pick elements 0,2,4,6 from a, 1,3,5,7 from b
        let result = _mm_blend_epi16::<0b1010_1010>(va, vb);
        let mut out = [0i16; 8];
        _mm_storeu_si128(out.as_mut_ptr().cast(), result);
        out
    }
}

#[rite(v2, import_intrinsics)]
fn crc32_step_v2(crc: u32, data: u8) -> u32 {
    // CRC32 is available at SSE4.2 (part of v2)
    // import_intrinsics makes this available directly
    _mm_crc32_u8(crc, data)
}

#[test]
fn test_rite_tier_v2_blend() {
    use archmage::X64V2Token;
    if X64V2Token::summon().is_some() {
        let a = [1i16, 2, 3, 4, 5, 6, 7, 8];
        let b = [10i16, 20, 30, 40, 50, 60, 70, 80];
        let result = unsafe { blend_i16_v2(&a, &b) };
        // Even indices from a, odd indices from b (_mm_blend_epi16 mask 0xAA)
        assert_eq!(result, [1, 20, 3, 40, 5, 60, 7, 80]);
    }
}

#[test]
fn test_rite_tier_v2_crc32_import_intrinsics() {
    use archmage::X64V2Token;
    if X64V2Token::summon().is_some() {
        // CRC32 of 0 with byte 0x01 should produce a known value
        let result = unsafe { crc32_step_v2(0, 1) };
        assert_ne!(result, 0); // CRC32 of non-zero input is non-zero
        // Check determinism
        let result2 = unsafe { crc32_step_v2(0, 1) };
        assert_eq!(result, result2);
    }
}

// --- V3 tier: FMA (fused multiply-add) ---

#[rite(v3)]
fn fma_f32x8(a: &[f32; 8], b: &[f32; 8], c: &[f32; 8]) -> [f32; 8] {
    // FMA is v3-specific — not available at v2
    unsafe {
        let va = _mm256_loadu_ps(a.as_ptr());
        let vb = _mm256_loadu_ps(b.as_ptr());
        let vc = _mm256_loadu_ps(c.as_ptr());
        let result = _mm256_fmadd_ps(va, vb, vc); // a*b + c
        let mut out = [0.0f32; 8];
        _mm256_storeu_ps(out.as_mut_ptr(), result);
        out
    }
}

#[test]
fn test_rite_tier_v3_fma() {
    if X64V3Token::summon().is_some() {
        let a = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let b = [2.0f32; 8];
        let c = [10.0f32; 8];
        let result = unsafe { fma_f32x8(&a, &b, &c) };
        // a*b + c = [1*2+10, 2*2+10, 3*2+10, ...] = [12, 14, 16, 18, 20, 22, 24, 26]
        assert_eq!(result, [12.0, 14.0, 16.0, 18.0, 20.0, 22.0, 24.0, 26.0]);
    }
}

// --- Tier + import_intrinsics + stub combo ---

#[rite(v3, import_intrinsics, stub)]
fn abs_f32x8_all_options(a: &[f32; 8]) -> [f32; 8] {
    // import_intrinsics: safe memory ops (reference-based loads/stores)
    // stub: unreachable stub on non-x86_64
    // v3: no token parameter needed
    let va = _mm256_loadu_ps(a);
    let sign_mask = _mm256_set1_ps(-0.0);
    let abs = _mm256_andnot_ps(sign_mask, va);
    let mut out = [0.0f32; 8];
    _mm256_storeu_ps(&mut out, abs);
    out
}

#[test]
fn test_rite_tier_import_intrinsics_stub_combo() {
    if X64V3Token::summon().is_some() {
        let a = [-1.0f32, 2.0, -3.0, 4.0, -5.0, 6.0, -7.0, 8.0];
        let result = unsafe { abs_f32x8_all_options(&a) };
        assert_eq!(result, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    }
}

// --- Tier-based #[rite] called from #[arcane] (primary use case) ---

#[rite(v3, import_intrinsics)]
fn sum_f32x8_tierless(data: &[f32; 8]) -> f32 {
    let v = _mm256_loadu_ps(data);
    let sum = _mm256_hadd_ps(v, v);
    let sum = _mm256_hadd_ps(sum, sum);
    let low = _mm256_castps256_ps128(sum);
    let high = _mm256_extractf128_ps::<1>(sum);
    _mm_cvtss_f32(_mm_add_ss(low, high))
}

#[rite(v3, import_intrinsics)]
fn scale_f32x8_tierless(data: &[f32; 8], factor: f32) -> [f32; 8] {
    let v = _mm256_loadu_ps(data);
    let f = _mm256_set1_ps(factor);
    let result = _mm256_mul_ps(v, f);
    let mut out = [0.0f32; 8];
    _mm256_storeu_ps(&mut out, result);
    out
}

// Entry point with token; calls tierless helpers inside
#[arcane(import_intrinsics)]
fn normalize_f32x8(_token: X64V3Token, data: &[f32; 8]) -> [f32; 8] {
    let total = sum_f32x8_tierless(data);
    if total == 0.0 {
        return *data;
    }
    let inv = 1.0 / total;
    scale_f32x8_tierless(data, inv)
}

#[test]
fn test_rite_tier_called_from_arcane_normalize() {
    if let Some(token) = X64V3Token::summon() {
        let data = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let result = normalize_f32x8(token, &data);
        // Sum = 36.0, each element / 36.0
        let expected: [f32; 8] = core::array::from_fn(|i| (i as f32 + 1.0) / 36.0);
        for (r, e) in result.iter().zip(expected.iter()) {
            assert!((r - e).abs() < 1e-6, "got {r}, expected {e}");
        }
    }
}

// --- Tier-based #[rite] calling other tier-based #[rite] ---

#[rite(v3, import_intrinsics)]
fn square_f32x8(data: &[f32; 8]) -> [f32; 8] {
    let v = _mm256_loadu_ps(data);
    let sq = _mm256_mul_ps(v, v);
    let mut out = [0.0f32; 8];
    _mm256_storeu_ps(&mut out, sq);
    out
}

#[rite(v3, import_intrinsics)]
fn sum_of_squares_tierless(data: &[f32; 8]) -> f32 {
    // Tier-based rite calling another tier-based rite
    let squared = square_f32x8(data);
    sum_f32x8_tierless(&squared)
}

// Top-level arcane entry point composing two tierless rite helpers
#[arcane]
fn l2_norm_squared(_token: X64V3Token, data: &[f32; 8]) -> f32 {
    sum_of_squares_tierless(data)
}

#[test]
fn test_rite_tier_calling_tier_rite() {
    if let Some(token) = X64V3Token::summon() {
        let data = [1.0f32, 2.0, 3.0, 4.0, 0.0, 0.0, 0.0, 0.0];
        let result = l2_norm_squared(token, &data);
        // 1^2 + 2^2 + 3^2 + 4^2 = 1 + 4 + 9 + 16 = 30
        assert_eq!(result, 30.0);
    }
}

// --- Mixed: token-based #[rite] calling tier-based #[rite] ---

#[rite]
fn mixed_caller_token_based(_token: X64V3Token, data: &[f32; 8]) -> f32 {
    // Token-based rite calling a tier-based rite helper
    let scaled = scale_f32x8_tierless(data, 2.0);
    sum_f32x8_tierless(&scaled)
}

#[test]
fn test_mixed_token_rite_calls_tier_rite() {
    if let Some(token) = X64V3Token::summon() {
        let data = [1.0f32; 8];
        let result = unsafe { mixed_caller_token_based(token, &data) };
        // 2.0 * 1.0 * 8 = 16.0
        assert_eq!(result, 16.0);
    }
}

// --- Mixed: tier-based #[rite] calling token-based #[rite] ---
// NOTE: tier-based rite cannot pass a token since it doesn't have one.
// But from inside an #[arcane] context, a tier-based rite can be called,
// and separately the token-based rite can also be called from #[arcane].
// This tests composing both flavors from the same #[arcane] entry point.

#[arcane(import_intrinsics)]
fn compose_mixed_flavors(token: X64V3Token, a: &[f32; 8], b: &[f32; 8]) -> f32 {
    // Call tier-based rite (no token needed)
    let sum_a = sum_f32x8_tierless(a);
    // Call token-based rite (needs token)
    let products = mul_vectors(token, a, b);
    let v = _mm256_loadu_ps(&products);
    let sum_prod = horizontal_sum(token, v);
    // Combine results
    sum_a + sum_prod
}

#[test]
fn test_mixed_arcane_calls_both_rite_flavors() {
    if let Some(token) = X64V3Token::summon() {
        let a = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let b = [1.0f32; 8];
        let result = compose_mixed_flavors(token, &a, &b);
        // sum_a = 36.0, products = a*b = a (since b=1), sum_prod = 36.0
        // total = 36.0 + 36.0 = 72.0
        assert_eq!(result, 72.0);
    }
}

// --- Tier with self receiver in impl block ---

struct SimdProcessor {
    scale: f32,
    offset: f32,
}

impl SimdProcessor {
    #[rite(v3, import_intrinsics)]
    fn process_chunk(&self, data: &[f32; 8]) -> [f32; 8] {
        let v = _mm256_loadu_ps(data);
        let scale = _mm256_set1_ps(self.scale);
        let offset = _mm256_set1_ps(self.offset);
        let result = _mm256_fmadd_ps(v, scale, offset); // v * scale + offset
        let mut out = [0.0f32; 8];
        _mm256_storeu_ps(&mut out, result);
        out
    }

    #[rite(v3, import_intrinsics)]
    fn reduce_sum(&self, data: &[f32; 8]) -> f32 {
        let v = _mm256_loadu_ps(data);
        let sum = _mm256_hadd_ps(v, v);
        let sum = _mm256_hadd_ps(sum, sum);
        let low = _mm256_castps256_ps128(sum);
        let high = _mm256_extractf128_ps::<1>(sum);
        _mm_cvtss_f32(_mm_add_ss(low, high))
    }
}

#[test]
fn test_rite_tier_self_receiver() {
    if X64V3Token::summon().is_some() {
        let processor = SimdProcessor {
            scale: 2.0,
            offset: 10.0,
        };
        let data = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let result = unsafe { processor.process_chunk(&data) };
        // v * 2.0 + 10.0 = [12, 14, 16, 18, 20, 22, 24, 26]
        assert_eq!(result, [12.0, 14.0, 16.0, 18.0, 20.0, 22.0, 24.0, 26.0]);
    }
}

#[test]
fn test_rite_tier_self_reduce() {
    if X64V3Token::summon().is_some() {
        let processor = SimdProcessor {
            scale: 1.0,
            offset: 0.0,
        };
        let data = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let result = unsafe { processor.reduce_sum(&data) };
        assert_eq!(result, 36.0);
    }
}

// --- Multiple arguments / return types ---

#[rite(v3, import_intrinsics)]
fn minmax_f32x8(data: &[f32; 8]) -> (f32, f32) {
    let v = _mm256_loadu_ps(data);

    // Horizontal min: compare halves, then pairs
    let hi128 = _mm256_extractf128_ps::<1>(v);
    let lo128 = _mm256_castps256_ps128(v);

    let min128 = _mm_min_ps(lo128, hi128);
    let min64 = _mm_min_ps(min128, _mm_movehl_ps(min128, min128));
    let min32 = _mm_min_ps(min64, _mm_shuffle_ps::<0x01>(min64, min64));
    let min_val = _mm_cvtss_f32(min32);

    let max128 = _mm_max_ps(lo128, hi128);
    let max64 = _mm_max_ps(max128, _mm_movehl_ps(max128, max128));
    let max32 = _mm_max_ps(max64, _mm_shuffle_ps::<0x01>(max64, max64));
    let max_val = _mm_cvtss_f32(max32);

    (min_val, max_val)
}

#[test]
fn test_rite_tier_tuple_return() {
    if X64V3Token::summon().is_some() {
        let data = [5.0f32, -3.0, 8.0, 1.0, -7.0, 4.0, 2.0, 6.0];
        let (min, max) = unsafe { minmax_f32x8(&data) };
        assert_eq!(min, -7.0);
        assert_eq!(max, 8.0);
    }
}

#[rite(v3, import_intrinsics)]
fn dot_with_offset(a: &[f32; 8], b: &[f32; 8], offset_a: f32, offset_b: f32, scale: f32) -> f32 {
    let va = _mm256_loadu_ps(a);
    let vb = _mm256_loadu_ps(b);
    let off_a = _mm256_set1_ps(offset_a);
    let off_b = _mm256_set1_ps(offset_b);
    let adjusted_a = _mm256_add_ps(va, off_a);
    let adjusted_b = _mm256_add_ps(vb, off_b);
    let products = _mm256_mul_ps(adjusted_a, adjusted_b);
    let scaled = _mm256_mul_ps(products, _mm256_set1_ps(scale));

    // Horizontal sum
    let sum = _mm256_hadd_ps(scaled, scaled);
    let sum = _mm256_hadd_ps(sum, sum);
    let low = _mm256_castps256_ps128(sum);
    let high = _mm256_extractf128_ps::<1>(sum);
    _mm_cvtss_f32(_mm_add_ss(low, high))
}

#[test]
fn test_rite_tier_many_args() {
    if X64V3Token::summon().is_some() {
        let a = [1.0f32; 8];
        let b = [1.0f32; 8];
        // (1+2) * (1+3) * 0.5 = 3 * 4 * 0.5 = 6.0, times 8 lanes = 48.0
        let result = unsafe { dot_with_offset(&a, &b, 2.0, 3.0, 0.5) };
        assert_eq!(result, 48.0);
    }
}

// --- Tier-based with const generics ---

#[rite(v3, import_intrinsics)]
fn sum_first_n<const N: usize>(data: &[f32; 8]) -> f32 {
    // Load all 8, but only sum the first N by masking
    let v = _mm256_loadu_ps(data);
    let mut mask_arr = [0.0f32; 8];
    let limit = if N > 8 { 8 } else { N };
    for slot in mask_arr.iter_mut().take(limit) {
        *slot = f32::from_bits(0xFFFF_FFFF);
    }
    let mask = _mm256_loadu_ps(&mask_arr);
    let masked = _mm256_and_ps(v, mask);

    let sum = _mm256_hadd_ps(masked, masked);
    let sum = _mm256_hadd_ps(sum, sum);
    let low = _mm256_castps256_ps128(sum);
    let high = _mm256_extractf128_ps::<1>(sum);
    _mm_cvtss_f32(_mm_add_ss(low, high))
}

#[test]
fn test_rite_tier_const_generic() {
    if X64V3Token::summon().is_some() {
        let data = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let sum4 = unsafe { sum_first_n::<4>(&data) };
        assert_eq!(sum4, 10.0); // 1+2+3+4

        let sum2 = unsafe { sum_first_n::<2>(&data) };
        assert_eq!(sum2, 3.0); // 1+2

        let sum8 = unsafe { sum_first_n::<8>(&data) };
        assert_eq!(sum8, 36.0); // 1+2+...+8
    }
}

// --- Composition: multiple tier-based helpers chained in arcane entry point ---

#[rite(v3, import_intrinsics)]
fn clamp_f32x8(data: &[f32; 8], lo: f32, hi: f32) -> [f32; 8] {
    let v = _mm256_loadu_ps(data);
    let vlo = _mm256_set1_ps(lo);
    let vhi = _mm256_set1_ps(hi);
    let clamped = _mm256_min_ps(_mm256_max_ps(v, vlo), vhi);
    let mut out = [0.0f32; 8];
    _mm256_storeu_ps(&mut out, clamped);
    out
}

#[rite(v3, import_intrinsics)]
fn subtract_f32x8(a: &[f32; 8], b: &[f32; 8]) -> [f32; 8] {
    let va = _mm256_loadu_ps(a);
    let vb = _mm256_loadu_ps(b);
    let result = _mm256_sub_ps(va, vb);
    let mut out = [0.0f32; 8];
    _mm256_storeu_ps(&mut out, result);
    out
}

/// Full pipeline: clamp -> subtract mean -> scale -> sum of squares
/// Exercises chaining 4 different tier-based #[rite] helpers from one #[arcane].
#[arcane]
fn pipeline_norm(_token: X64V3Token, data: &[f32; 8]) -> f32 {
    // Step 1: clamp to [0, 10]
    let clamped = clamp_f32x8(data, 0.0, 10.0);
    // Step 2: compute mean
    let total = sum_f32x8_tierless(&clamped);
    let mean = total / 8.0;
    // Step 3: subtract mean
    let mean_arr = [mean; 8];
    let centered = subtract_f32x8(&clamped, &mean_arr);
    // Step 4: sum of squares of centered values
    let sq = square_f32x8(&centered);
    sum_f32x8_tierless(&sq)
}

#[test]
fn test_rite_tier_pipeline_composition() {
    if let Some(token) = X64V3Token::summon() {
        let data = [2.0f32, 4.0, 6.0, 8.0, 2.0, 4.0, 6.0, 8.0];
        let result = pipeline_norm(token, &data);
        // After clamp [0,10]: same. Mean = (2+4+6+8)*2/8 = 40/8 = 5.0
        // Centered: [-3, -1, 1, 3, -3, -1, 1, 3]
        // Squares: [9, 1, 1, 9, 9, 1, 1, 9]
        // Sum = 40.0
        assert_eq!(result, 40.0);
    }
}

// --- V1 tier always available + more SSE2 intrinsics ---

#[rite(v1, import_intrinsics)]
fn shuffle_f32x4_v1(data: &[f32; 4]) -> [f32; 4] {
    // Reverse the 4 floats using SSE2 shuffle
    let v = _mm_loadu_ps(data);
    // Shuffle: 3,2,1,0
    let reversed = _mm_shuffle_ps::<0b00_01_10_11>(v, v);
    let mut out = [0.0f32; 4];
    _mm_storeu_ps(&mut out, reversed);
    out
}

#[test]
fn test_rite_tier_v1_shuffle_always_available() {
    use archmage::X64V1Token;
    // SSE2 is always available on x86_64 — no token check needed,
    // but we check anyway for consistency.
    if X64V1Token::summon().is_some() {
        let data = [1.0f32, 2.0, 3.0, 4.0];
        let result = unsafe { shuffle_f32x4_v1(&data) };
        assert_eq!(result, [4.0, 3.0, 2.0, 1.0]);
    }
}

// --- V2 tier + import_intrinsics: SSE4.1 rounding ---

#[rite(v2, import_intrinsics)]
fn round_f32x4_v2(data: &[f32; 4]) -> [f32; 4] {
    // _mm_round_ps is SSE4.1 (part of v2)
    let v = _mm_loadu_ps(data);
    // Round to nearest integer
    let rounded = _mm_round_ps::<{ _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC }>(v);
    let mut out = [0.0f32; 4];
    _mm_storeu_ps(&mut out, rounded);
    out
}

#[test]
fn test_rite_tier_v2_rounding() {
    use archmage::X64V2Token;
    if X64V2Token::summon().is_some() {
        let data = [1.3f32, 2.7, -0.5, 3.5];
        let result = unsafe { round_f32x4_v2(&data) };
        // Banker's rounding: 1.3->1, 2.7->3, -0.5->0, 3.5->4
        assert_eq!(result, [1.0, 3.0, 0.0, 4.0]);
    }
}

// --- V3 FMA in a more complex operation: polynomial evaluation ---

#[rite(v3, import_intrinsics)]
fn poly_eval_v3(x: &[f32; 8], a: f32, b: f32, c: f32) -> [f32; 8] {
    // Evaluate a*x^2 + b*x + c using FMA
    let vx = _mm256_loadu_ps(x);
    let va = _mm256_set1_ps(a);
    let vb = _mm256_set1_ps(b);
    let vc = _mm256_set1_ps(c);

    // a*x + b
    let ax_plus_b = _mm256_fmadd_ps(va, vx, vb);
    // (a*x + b)*x + c = a*x^2 + b*x + c
    let result = _mm256_fmadd_ps(ax_plus_b, vx, vc);

    let mut out = [0.0f32; 8];
    _mm256_storeu_ps(&mut out, result);
    out
}

#[test]
fn test_rite_tier_v3_polynomial_fma() {
    if X64V3Token::summon().is_some() {
        // f(x) = 2x^2 + 3x + 1
        let x = [0.0f32, 1.0, 2.0, 3.0, -1.0, -2.0, 0.5, 10.0];
        let result = unsafe { poly_eval_v3(&x, 2.0, 3.0, 1.0) };
        // f(0)=1, f(1)=6, f(2)=15, f(3)=28, f(-1)=0, f(-2)=3, f(0.5)=3, f(10)=231
        let expected = [1.0, 6.0, 15.0, 28.0, 0.0, 3.0, 3.0, 231.0];
        for (r, e) in result.iter().zip(expected.iter()) {
            assert!((r - e).abs() < 1e-4, "poly_eval: got {r}, expected {e}");
        }
    }
}

// --- Struct method: tier-based with &mut self ---

struct Accumulator {
    buffer: [f32; 8],
}

impl Accumulator {
    fn new() -> Self {
        Self { buffer: [0.0; 8] }
    }

    #[rite(v3, import_intrinsics)]
    fn accumulate(&mut self, data: &[f32; 8]) {
        let current = _mm256_loadu_ps(&self.buffer);
        let incoming = _mm256_loadu_ps(data);
        let sum = _mm256_add_ps(current, incoming);
        _mm256_storeu_ps(&mut self.buffer, sum);
    }

    #[rite(v3, import_intrinsics)]
    fn result(&self) -> f32 {
        let v = _mm256_loadu_ps(&self.buffer);
        let sum = _mm256_hadd_ps(v, v);
        let sum = _mm256_hadd_ps(sum, sum);
        let low = _mm256_castps256_ps128(sum);
        let high = _mm256_extractf128_ps::<1>(sum);
        _mm_cvtss_f32(_mm_add_ss(low, high))
    }
}

#[test]
fn test_rite_tier_mut_self_accumulator() {
    if X64V3Token::summon().is_some() {
        let mut acc = Accumulator::new();
        let batch1 = [1.0f32; 8];
        let batch2 = [2.0f32; 8];
        let batch3 = [3.0f32; 8];

        unsafe {
            acc.accumulate(&batch1);
            acc.accumulate(&batch2);
            acc.accumulate(&batch3);
        }

        let total = unsafe { acc.result() };
        // (1+2+3) * 8 = 48.0
        assert_eq!(total, 48.0);
    }
}

// --- Two different tiers used in the same #[arcane] entry point ---

#[rite(v2)]
fn popcnt_u32_v2(val: u32) -> u32 {
    core::arch::x86_64::_popcnt32(val as i32) as u32
}

#[arcane(import_intrinsics)]
fn count_nonzero_lanes(_token: X64V3Token, data: &[f32; 8]) -> u32 {
    // Use v3-level intrinsics for comparison
    let v = _mm256_loadu_ps(data);
    let zero = _mm256_setzero_ps();
    let cmp = _mm256_cmp_ps::<_CMP_NEQ_OQ>(v, zero);
    let mask = _mm256_movemask_ps(cmp) as u32;
    // Use v2-level popcount to count set bits
    popcnt_u32_v2(mask)
}

#[test]
fn test_rite_two_tiers_from_arcane() {
    if let Some(token) = X64V3Token::summon() {
        let data = [1.0f32, 0.0, 3.0, 0.0, 5.0, 6.0, 0.0, 8.0];
        let count = count_nonzero_lanes(token, &data);
        assert_eq!(count, 5); // 5 non-zero lanes
    }
}

// --- Tier-based with bool return ---

#[rite(v3, import_intrinsics)]
fn all_positive(data: &[f32; 8]) -> bool {
    let v = _mm256_loadu_ps(data);
    let zero = _mm256_setzero_ps();
    let cmp = _mm256_cmp_ps::<_CMP_GT_OQ>(v, zero);
    let mask = _mm256_movemask_ps(cmp);
    mask == 0xFF // all 8 lanes > 0
}

#[rite(v3, import_intrinsics)]
fn any_negative(data: &[f32; 8]) -> bool {
    let v = _mm256_loadu_ps(data);
    let zero = _mm256_setzero_ps();
    let cmp = _mm256_cmp_ps::<_CMP_LT_OQ>(v, zero);
    let mask = _mm256_movemask_ps(cmp);
    mask != 0 // any lane < 0
}

#[test]
fn test_rite_tier_bool_return() {
    if X64V3Token::summon().is_some() {
        let all_pos = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        assert!(unsafe { all_positive(&all_pos) });
        assert!(!unsafe { any_negative(&all_pos) });

        let has_neg = [1.0f32, -2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        assert!(!unsafe { all_positive(&has_neg) });
        assert!(unsafe { any_negative(&has_neg) });

        let all_neg = [-1.0f32; 8];
        assert!(!unsafe { all_positive(&all_neg) });
        assert!(unsafe { any_negative(&all_neg) });
    }
}

// --- V1 import_intrinsics: safe 128-bit memory ops ---

#[rite(v1, import_intrinsics)]
fn reverse_pairs_v1(data: &[f64; 2]) -> [f64; 2] {
    // _mm_shuffle_pd with SSE2 to swap the two f64 lanes
    let v = _mm_loadu_pd(data);
    let swapped = _mm_shuffle_pd::<0b01>(v, v);
    let mut out = [0.0f64; 2];
    _mm_storeu_pd(&mut out, swapped);
    out
}

#[test]
fn test_rite_tier_v1_import_intrinsics() {
    use archmage::X64V1Token;
    if X64V1Token::summon().is_some() {
        let data = [42.0f64, 99.0];
        let result = unsafe { reverse_pairs_v1(&data) };
        assert_eq!(result, [99.0, 42.0]);
    }
}

// --- Chaining 3 different tierless rite helpers with varying signatures ---

#[rite(v3, import_intrinsics)]
fn interleave_low_f32x8(a: &[f32; 8], b: &[f32; 8]) -> [f32; 8] {
    let va = _mm256_loadu_ps(a);
    let vb = _mm256_loadu_ps(b);
    let result = _mm256_unpacklo_ps(va, vb);
    let mut out = [0.0f32; 8];
    _mm256_storeu_ps(&mut out, result);
    out
}

#[rite(v3, import_intrinsics)]
fn interleave_high_f32x8(a: &[f32; 8], b: &[f32; 8]) -> [f32; 8] {
    let va = _mm256_loadu_ps(a);
    let vb = _mm256_loadu_ps(b);
    let result = _mm256_unpackhi_ps(va, vb);
    let mut out = [0.0f32; 8];
    _mm256_storeu_ps(&mut out, result);
    out
}

#[arcane]
fn interleave_and_sum(_token: X64V3Token, a: &[f32; 8], b: &[f32; 8]) -> (f32, f32) {
    let lo = interleave_low_f32x8(a, b);
    let hi = interleave_high_f32x8(a, b);
    let sum_lo = sum_f32x8_tierless(&lo);
    let sum_hi = sum_f32x8_tierless(&hi);
    (sum_lo, sum_hi)
}

#[test]
fn test_rite_tier_chain_three_helpers() {
    if let Some(token) = X64V3Token::summon() {
        let a = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let b = [10.0f32, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0];
        let (sum_lo, sum_hi) = interleave_and_sum(token, &a, &b);
        // unpacklo within 128-bit lanes: [a0,b0,a1,b1, a4,b4,a5,b5]
        // = [1,10,2,20, 5,50,6,60] => sum = 154
        assert_eq!(sum_lo, 154.0);
        // unpackhi within 128-bit lanes: [a2,b2,a3,b3, a6,b6,a7,b7]
        // = [3,30,4,40, 7,70,8,80] => sum = 242
        assert_eq!(sum_hi, 242.0);
    }
}

// --- Tier + stub on V2 ---

#[rite(v2, stub)]
fn popcount_array_v2_stub(data: &[u32; 4]) -> [u32; 4] {
    [
        core::arch::x86_64::_popcnt32(data[0] as i32) as u32,
        core::arch::x86_64::_popcnt32(data[1] as i32) as u32,
        core::arch::x86_64::_popcnt32(data[2] as i32) as u32,
        core::arch::x86_64::_popcnt32(data[3] as i32) as u32,
    ]
}

#[test]
fn test_rite_tier_v2_stub() {
    use archmage::X64V2Token;
    if X64V2Token::summon().is_some() {
        let data = [0b1111u32, 0b10101010, 0b11111111, 0b0];
        let result = unsafe { popcount_array_v2_stub(&data) };
        assert_eq!(result, [4, 4, 8, 0]);
    }
}

// --- V3 with generic lifetime parameters ---

#[rite(v3, import_intrinsics)]
fn load_and_compare<'a>(data: &'a [f32; 8], threshold: f32) -> i32 {
    let v = _mm256_loadu_ps(data);
    let t = _mm256_set1_ps(threshold);
    let cmp = _mm256_cmp_ps::<_CMP_GE_OQ>(v, t);
    _mm256_movemask_ps(cmp)
}

#[test]
fn test_rite_tier_lifetime_generic() {
    if X64V3Token::summon().is_some() {
        let data = [1.0f32, 5.0, 3.0, 7.0, 2.0, 6.0, 4.0, 8.0];
        let mask = unsafe { load_and_compare(&data, 5.0) };
        // Lanes >= 5.0: indices 1(5.0), 3(7.0), 5(6.0), 7(8.0) => mask bits 1,3,5,7
        assert_eq!(mask, 0b1010_1010_u32 as i32);
    }
}
