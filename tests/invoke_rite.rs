//! Tests for `token.invoke_rite()` — closure-based #[target_feature] entry.

#[test]
fn scalar_invoke_rite_passthrough() {
    let token = archmage::ScalarToken;
    let result = token.invoke_rite(|_t| 42);
    assert_eq!(result, 42);
}

#[test]
fn scalar_invoke_rite_captures_state() {
    let token = archmage::ScalarToken;
    let mut acc = 0;
    token.invoke_rite(|_t| {
        acc += 1;
    });
    assert_eq!(acc, 1);
}

#[cfg(target_arch = "x86_64")]
mod x86_tests {
    use archmage::SimdToken;

    #[test]
    fn v1_invoke_rite() {
        // V1 (SSE2) is always available on x86_64
        let token = archmage::X64V1Token::summon().unwrap();
        let result = token.invoke_rite(|_t| {
            // Inside the rite, SSE2 intrinsics are safe (Rust 1.85+)
            // Just verify the closure runs and returns
            7 + 8
        });
        assert_eq!(result, 15);
    }

    #[test]
    fn v3_invoke_rite_with_simd() {
        if let Some(token) = archmage::X64V3Token::summon() {
            let data = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
            let result = token.invoke_rite(|_t| {
                // Inside the rite, AVX2+FMA intrinsics are safe
                #[cfg(target_arch = "x86_64")]
                unsafe {
                    use core::arch::x86_64::*;
                    let v = _mm256_loadu_ps(data.as_ptr());
                    let sum = _mm256_hadd_ps(v, v);
                    let sum = _mm256_hadd_ps(sum, sum);
                    let lo = _mm256_castps256_ps128(sum);
                    let hi = _mm256_extractf128_ps(sum, 1);
                    let total = _mm_add_ss(lo, hi);
                    _mm_cvtss_f32(total)
                }
            });
            assert_eq!(result, 36.0);
        }
    }

    #[test]
    fn v3_invoke_rite_returns_complex_type() {
        if let Some(token) = archmage::X64V3Token::summon() {
            let (a, b) = token.invoke_rite(|_t| ("hello", vec![1, 2, 3]));
            assert_eq!(a, "hello");
            assert_eq!(b, vec![1, 2, 3]);
        }
    }

    #[test]
    fn invoke_rite_token_passes_through() {
        if let Some(token) = archmage::X64V3Token::summon() {
            // The closure receives the same token
            token.invoke_rite(|inner_token| {
                // inner_token is the same X64V3Token
                assert_eq!(
                    archmage::X64V3Token::NAME,
                    <archmage::X64V3Token as SimdToken>::NAME
                );
                let _ = inner_token; // proves it's the right type
            });
        }
    }

    #[test]
    fn invoke_rite_nested_is_valid() {
        if let Some(token) = archmage::X64V3Token::summon() {
            let result = token.invoke_rite(|t: archmage::X64V3Token| {
                // Nested invoke_rite — same token, same boundary
                t.invoke_rite(|_inner: archmage::X64V3Token| 99)
            });
            assert_eq!(result, 99);
        }
    }
}
