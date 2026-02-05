use archmage::{IntoConcreteToken, ScalarToken, SimdToken, incant, magetypes};

#[magetypes]
pub fn dot(token: Token, a: &[f32], b: &[f32]) -> f32 {
    let _ = token;
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

pub fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    incant!(dot(a, b))
}

#[magetypes]
pub fn normalize(token: Token, data: &mut [f32]) {
    let _ = token;
    let sum: f32 = data.iter().map(|x| x * x).sum();
    if sum > 0.0 {
        let inv_norm = 1.0 / sum.sqrt();
        for x in data.iter_mut() {
            *x *= inv_norm;
        }
    }
}

pub fn normalize_api(data: &mut [f32]) {
    incant!(normalize(data))
}

#[magetypes]
pub fn scale(token: Token, data: &mut [f32], factor: f32) {
    let _ = token;
    for x in data.iter_mut() {
        *x *= factor;
    }
}

fn scale_dispatch<T: IntoConcreteToken>(token: T, data: &mut [f32], factor: f32) {
    incant!(scale(data, factor) with token)
}

#[magetypes]
pub fn chunk_count(token: Token, total: usize) -> usize {
    let _ = token;
    total / LANES
}

pub fn chunk_count_api(total: usize) -> usize {
    incant!(chunk_count(total))
}

#[magetypes]
pub fn min_max(token: Token, data: &[f32]) -> (f32, f32) {
    let _ = token;
    if data.is_empty() {
        return (f32::NAN, f32::NAN);
    }
    let mut min = data[0];
    let mut max = data[0];
    for &x in &data[1..] {
        if x < min {
            min = x;
        }
        if x > max {
            max = x;
        }
    }
    (min, max)
}

pub fn min_max_api(data: &[f32]) -> (f32, f32) {
    incant!(min_max(data))
}

#[magetypes]
pub fn square(token: Token, data: &mut [f32]) {
    let _ = token;
    for x in data.iter_mut() {
        *x *= *x;
    }
}

#[magetypes]
pub fn add_const(token: Token, data: &mut [f32], val: f32) {
    let _ = token;
    for x in data.iter_mut() {
        *x += val;
    }
}

pub fn square_plus_one(data: &mut [f32]) {
    incant!(square(data));
    incant!(add_const(data, 1.0));
}

#[magetypes]
pub fn identity(token: Token, x: f32) -> f32 {
    let _ = token;
    x
}

pub fn identity_api(x: f32) -> f32 {
    incant!(identity(x))
}

// =============================================================================
// Tests
// =============================================================================

#[test]
fn dot_product_basic() {
    let a = [1.0, 2.0, 3.0, 4.0];
    let b = [4.0, 3.0, 2.0, 1.0];
    assert_eq!(dot_product(&a, &b), 20.0);
}

#[test]
fn normalize_unit_vector() {
    let mut data = [3.0f32, 4.0];
    normalize_api(&mut data);
    assert!((data[0] - 0.6).abs() < 1e-6);
    assert!((data[1] - 0.8).abs() < 1e-6);
}

#[test]
fn generic_dispatch_scalar() {
    let mut data = [1.0, 2.0, 3.0];
    scale_dispatch(ScalarToken, &mut data, 10.0);
    assert_eq!(data, [10.0, 20.0, 30.0]);
}

#[test]
fn chained_dispatch() {
    let mut data = [2.0, 3.0, 4.0];
    square_plus_one(&mut data);
    assert_eq!(data, [5.0, 10.0, 17.0]);
}

#[test]
fn min_max_basic() {
    let (min, max) = min_max_api(&[3.0, 1.0, 4.0, 1.0, 5.0, 9.0]);
    assert_eq!(min, 1.0);
    assert_eq!(max, 9.0);
}

#[test]
fn nan_passthrough() {
    assert!(identity_api(f32::NAN).is_nan());
}

mod alias_test {
    use super::*;
    use archmage::simd_route;

    pub fn dot_via_alias(a: &[f32], b: &[f32]) -> f32 {
        simd_route!(dot(a, b))
    }

    #[test]
    fn simd_route_identical_to_incant() {
        let a = [1.0, 2.0, 3.0];
        let b = [4.0, 5.0, 6.0];
        assert_eq!(dot_product(&a, &b), dot_via_alias(&a, &b));
    }
}

// =============================================================================
// Edge cases
// =============================================================================

#[test]
fn dot_product_empty() {
    assert_eq!(dot_product(&[], &[]), 0.0);
}

#[test]
fn dot_product_single() {
    assert_eq!(dot_product(&[5.0], &[3.0]), 15.0);
}

#[test]
fn dot_product_large() {
    let n = 1024;
    let a: Vec<f32> = (0..n).map(|i| i as f32).collect();
    let b: Vec<f32> = (0..n).map(|i| (n - i) as f32).collect();
    let expected: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    assert_eq!(dot_product(&a, &b), expected);
}

#[test]
fn normalize_zeros() {
    let mut data = [0.0f32, 0.0, 0.0];
    normalize_api(&mut data);
    assert_eq!(data, [0.0, 0.0, 0.0]);
}

#[test]
fn min_max_single() {
    let (min, max) = min_max_api(&[42.0]);
    assert_eq!(min, 42.0);
    assert_eq!(max, 42.0);
}

#[test]
fn infinity_passthrough() {
    assert_eq!(identity_api(f32::INFINITY), f32::INFINITY);
    assert_eq!(identity_api(f32::NEG_INFINITY), f32::NEG_INFINITY);
}

#[test]
fn negative_zero() {
    let result = identity_api(-0.0f32);
    assert!(result.is_sign_negative());
    assert_eq!(result, 0.0);
}

#[test]
fn lanes_meaningful() {
    let result = chunk_count_api(32);
    assert!(result >= 1);
    assert!(result <= 32);
}

// =============================================================================
// Platform-specific dispatch verification
// =============================================================================

#[cfg(target_arch = "x86_64")]
mod x86_dispatch {
    use super::*;

    #[test]
    fn v3_dispatch() {
        if let Some(token) = archmage::X64V3Token::summon() {
            let mut data = [1.0, 2.0, 3.0];
            scale_dispatch(token, &mut data, 2.0);
            assert_eq!(data, [2.0, 4.0, 6.0]);
        }
    }

    #[cfg(feature = "avx512")]
    #[test]
    fn v4_dispatch() {
        if let Some(token) = archmage::X64V4Token::summon() {
            let mut data = [1.0, 2.0, 3.0];
            scale_dispatch(token, &mut data, 3.0);
            assert_eq!(data, [3.0, 6.0, 9.0]);
        }
    }
}

// =============================================================================
// ScalarToken fundamentals
// =============================================================================

#[test]
fn scalar_token_always_available() {
    assert_eq!(ScalarToken::guaranteed(), Some(true));
    assert!(ScalarToken::summon().is_some());
    assert_eq!(ScalarToken::NAME, "Scalar");
}

#[test]
fn scalar_token_into_concrete() {
    let token = ScalarToken;
    assert!(token.as_scalar().is_some());
    assert!(token.as_x64v2().is_none());
    assert!(token.as_x64v3().is_none());
    assert!(token.as_neon().is_none());
    assert!(token.as_neon_aes().is_none());
    assert!(token.as_neon_sha3().is_none());
    assert!(token.as_neon_crc().is_none());
    assert!(token.as_wasm128().is_none());
}

#[cfg(target_arch = "x86_64")]
#[test]
fn x64v3_into_concrete() {
    if let Some(token) = archmage::X64V3Token::summon() {
        assert!(token.as_x64v3().is_some());
        assert!(token.as_scalar().is_none());
        assert!(token.as_x64v2().is_none());
        assert!(token.as_neon().is_none());
        assert!(token.as_neon_aes().is_none());
        assert!(token.as_neon_sha3().is_none());
        assert!(token.as_neon_crc().is_none());
        assert!(token.as_wasm128().is_none());
    }
}

// =============================================================================
// IntoConcreteToken — all tokens match only themselves
// =============================================================================

#[test]
fn x64v2_into_concrete() {
    let token = archmage::X64V2Token::summon();
    if let Some(t) = token {
        assert!(t.as_x64v2().is_some());
        assert!(t.as_x64v3().is_none());
        assert!(t.as_scalar().is_none());
        assert!(t.as_neon().is_none());
        assert!(t.as_neon_aes().is_none());
        assert!(t.as_neon_sha3().is_none());
        assert!(t.as_neon_crc().is_none());
        assert!(t.as_wasm128().is_none());
    }
}

#[test]
fn neon_into_concrete() {
    let token = archmage::NeonToken::summon();
    if let Some(t) = token {
        assert!(t.as_neon().is_some());
        assert!(t.as_neon_aes().is_none());
        assert!(t.as_neon_sha3().is_none());
        assert!(t.as_neon_crc().is_none());
        assert!(t.as_scalar().is_none());
        assert!(t.as_x64v2().is_none());
        assert!(t.as_x64v3().is_none());
        assert!(t.as_wasm128().is_none());
    }
}

#[test]
fn neon_aes_into_concrete() {
    let token = archmage::NeonAesToken::summon();
    if let Some(t) = token {
        assert!(t.as_neon_aes().is_some());
        assert!(t.as_neon().is_none());
        assert!(t.as_neon_sha3().is_none());
        assert!(t.as_neon_crc().is_none());
        assert!(t.as_scalar().is_none());
        assert!(t.as_x64v2().is_none());
        assert!(t.as_x64v3().is_none());
        assert!(t.as_wasm128().is_none());
    }
}

#[test]
fn neon_sha3_into_concrete() {
    let token = archmage::NeonSha3Token::summon();
    if let Some(t) = token {
        assert!(t.as_neon_sha3().is_some());
        assert!(t.as_neon().is_none());
        assert!(t.as_neon_aes().is_none());
        assert!(t.as_neon_crc().is_none());
        assert!(t.as_scalar().is_none());
    }
}

#[test]
fn neon_crc_into_concrete() {
    let token = archmage::NeonCrcToken::summon();
    if let Some(t) = token {
        assert!(t.as_neon_crc().is_some());
        assert!(t.as_neon().is_none());
        assert!(t.as_neon_aes().is_none());
        assert!(t.as_neon_sha3().is_none());
        assert!(t.as_scalar().is_none());
    }
}

#[test]
fn wasm128_into_concrete() {
    let token = archmage::Simd128Token::summon();
    if let Some(t) = token {
        assert!(t.as_wasm128().is_some());
        assert!(t.as_scalar().is_none());
        assert!(t.as_neon().is_none());
        assert!(t.as_neon_aes().is_none());
        assert!(t.as_neon_sha3().is_none());
        assert!(t.as_neon_crc().is_none());
        assert!(t.as_x64v2().is_none());
        assert!(t.as_x64v3().is_none());
    }
}

#[cfg(feature = "avx512")]
mod avx512_into_concrete {
    use archmage::{IntoConcreteToken, SimdToken};

    #[test]
    fn x64v4_into_concrete() {
        if let Some(t) = archmage::X64V4Token::summon() {
            assert!(t.as_x64v4().is_some());
            assert!(t.as_avx512_modern().is_none());
            assert!(t.as_avx512_fp16().is_none());
            assert!(t.as_x64v2().is_none());
            assert!(t.as_x64v3().is_none());
            assert!(t.as_scalar().is_none());
            assert!(t.as_neon().is_none());
        }
    }

    #[test]
    fn avx512_modern_into_concrete() {
        if let Some(t) = archmage::Avx512ModernToken::summon() {
            assert!(t.as_avx512_modern().is_some());
            assert!(t.as_x64v4().is_none());
            assert!(t.as_avx512_fp16().is_none());
            assert!(t.as_x64v2().is_none());
            assert!(t.as_x64v3().is_none());
            assert!(t.as_scalar().is_none());
        }
    }

    #[test]
    fn avx512_fp16_into_concrete() {
        if let Some(t) = archmage::Avx512Fp16Token::summon() {
            assert!(t.as_avx512_fp16().is_some());
            assert!(t.as_avx512_modern().is_none());
            assert!(t.as_x64v4().is_none());
            assert!(t.as_x64v2().is_none());
            assert!(t.as_x64v3().is_none());
            assert!(t.as_scalar().is_none());
        }
    }
}

// =============================================================================
// Generic dispatch with sub-tier tokens
// =============================================================================

/// Verify that IntoConcreteToken enables manual dispatch for sub-tier tokens
fn dispatch_with_sub_tiers<T: IntoConcreteToken>(token: T) -> &'static str {
    if token.as_neon_aes().is_some() {
        return "neon_aes";
    }
    if token.as_neon_sha3().is_some() {
        return "neon_sha3";
    }
    if token.as_neon_crc().is_some() {
        return "neon_crc";
    }
    if token.as_neon().is_some() {
        return "neon";
    }
    if token.as_scalar().is_some() {
        return "scalar";
    }
    "unknown"
}

#[test]
fn sub_tier_dispatch_scalar() {
    assert_eq!(dispatch_with_sub_tiers(ScalarToken), "scalar");
}

#[cfg(target_arch = "aarch64")]
#[test]
fn sub_tier_dispatch_neon() {
    if let Some(token) = archmage::NeonToken::summon() {
        assert_eq!(dispatch_with_sub_tiers(token), "neon");
    }
    if let Some(token) = archmage::NeonAesToken::summon() {
        assert_eq!(dispatch_with_sub_tiers(token), "neon_aes");
    }
}

// =============================================================================
// #[magetypes] with f32xN type - demonstrates scalar→SIMD→scalar
// =============================================================================

#[magetypes]
pub fn sum_squares(token: Token, data: &[f32]) -> f32 {
    let _ = token;
    let chunks = data.chunks_exact(LANES);
    let remainder = chunks.remainder();

    let mut acc = f32xN::splat(token, 0.0);
    for chunk in chunks {
        // Convert slice to array, then to SIMD
        let arr: [f32; LANES] = chunk.try_into().unwrap();
        let v = f32xN::from_array(token, arr);
        acc = acc + v * v;
    }

    // Reduce SIMD to scalar
    let mut sum: f32 = acc.reduce_add();

    // Handle remainder scalar
    for &x in remainder {
        sum += x * x;
    }
    sum
}

pub fn sum_squares_api(data: &[f32]) -> f32 {
    incant!(sum_squares(data))
}

#[test]
fn sum_squares_basic() {
    let data = [1.0, 2.0, 3.0, 4.0];
    let result = sum_squares_api(&data);
    assert_eq!(result, 1.0 + 4.0 + 9.0 + 16.0);
}

#[test]
fn sum_squares_with_remainder() {
    // 10 elements: 8 in SIMD chunk + 2 remainder on AVX2
    let data: Vec<f32> = (1..=10).map(|x| x as f32).collect();
    let expected: f32 = data.iter().map(|x| x * x).sum();
    assert_eq!(sum_squares_api(&data), expected);
}

#[test]
fn sum_squares_empty() {
    assert_eq!(sum_squares_api(&[]), 0.0);
}

#[test]
fn sum_squares_single() {
    assert_eq!(sum_squares_api(&[5.0]), 25.0);
}
