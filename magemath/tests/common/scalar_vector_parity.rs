//! Shared helpers for exhaustive const-generic scalar vs vector parity tests.
#![allow(dead_code)]

#[inline(always)]
pub fn same_f32(a: f32, b: f32) -> bool {
    (a.is_nan() && b.is_nan()) || a.to_bits() == b.to_bits()
}

#[track_caller]
pub fn assert_f32_eq(actual: f32, expected: f32, ctx: &str) {
    if expected.is_nan() {
        assert!(
            actual.is_nan(),
            "{ctx}: expected NaN, got {actual:?} ({:#010x})",
            actual.to_bits()
        );
    } else {
        assert_eq!(
            actual.to_bits(),
            expected.to_bits(),
            "{ctx}: got {actual:?} ({:#010x}), expected {expected:?} ({:#010x})",
            actual.to_bits(),
            expected.to_bits()
        );
    }
}

#[track_caller]
pub fn assert_scalar_res_eq<E: PartialEq + core::fmt::Debug>(
    actual: Result<f32, E>,
    expected: Result<f32, E>,
    ctx: &str,
) {
    match (actual, expected) {
        (Ok(a), Ok(e)) => assert_f32_eq(a, e, ctx),
        (Err(ae), Err(ee)) => assert_eq!(ae, ee, "{ctx}: error mismatch"),
        (a, e) => panic!("{ctx}: result variant mismatch: got {a:?}, expected {e:?}"),
    }
}

#[track_caller]
pub fn assert_array_res_eq<const N: usize, E: PartialEq + core::fmt::Debug>(
    actual: Result<[f32; N], E>,
    expected: Result<[f32; N], E>,
    ctx: &str,
) {
    match (actual, expected) {
        (Ok(aa), Ok(ea)) => {
            for i in 0..N {
                assert_f32_eq(aa[i], ea[i], ctx);
            }
        }
        (Err(ae), Err(ee)) => assert_eq!(ae, ee, "{ctx}: array error mismatch"),
        (a, e) => panic!("{ctx}: array result variant mismatch: got {a:?}, expected {e:?}"),
    }
}

macro_rules! run_all_summonable_tokens {
    () => {{
        #[cfg(any(
            target_arch = "x86_64",
            target_arch = "aarch64",
            all(target_arch = "wasm32", target_feature = "simd128")
        ))]
        use archmage::SimdToken;

        check_token_scalar(archmage::ScalarToken);

        #[cfg(target_arch = "x86_64")]
        {
            if let Some(t) = archmage::X64V3Token::summon() {
                check_token_v3(t);
            }
            #[cfg(feature = "avx512")]
            {
                if let Some(t) = archmage::X64V4Token::summon() {
                    check_token_v4(t);
                }
                if let Some(t) = archmage::X64V4xToken::summon() {
                    check_token_v4x(t);
                }
                if let Some(t) = archmage::Avx512Fp16Token::summon() {
                    check_token_fp16(t);
                }
            }
        }

        #[cfg(target_arch = "aarch64")]
        if let Some(t) = archmage::NeonToken::summon() {
            check_token_neon(t);
        }

        #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
        {
            if let Some(t) = archmage::Wasm128Token::summon() {
                check_token_wasm128(t);
            }
            #[cfg(target_feature = "relaxed-simd")]
            if let Some(t) = archmage::Wasm128RelaxedFmaToken::summon() {
                check_token_wasm128_relaxed_fma(t);
            }
        }
    }};
}
