//! Comprehensive tests for `X64V3VnniToken` (`x86-64-v3` + Crypto + GFNI + AVX-VNNI).
//!
//! Exercises:
//! - Token traits, `TARGET_FEATURES`, leaf topology (not implied by `X64V4xToken`)
//! - Zero-cost downcast extractors (`v3_gfni_crypto`, `v3_crypto`, `v3`, `x64_crypto`, `v2`, `v1`)
//! - `IntoConcreteToken::as_x64v3_vnni`
//! - `incant!` and `#[autoversion]` dispatch ordering (`v3_vnni` > `v3_gfni_crypto` > `v3` > `scalar`)
//! - Process-wide disable cascading (`X64V3GfniCryptoToken` -> `X64V3VnniToken`, leaf isolation)
//! - Hardware execution of VEX-encoded AVX-VNNI dot-product instructions (`vpdpbusd`, `vpdpbusds`,
//!   `vpdpwssd`, `vpdpwssds`) against scalar reference values when `X64V3VnniToken::summon()` succeeds.

#![forbid(unsafe_code)]

use archmage::{IntoConcreteToken, ScalarToken, SimdToken, X64V3VnniToken};

#[cfg(not(target_arch = "x86_64"))]
#[test]
fn v3_vnni_is_stub_off_x86_64() {
    assert_eq!(X64V3VnniToken::compiled_with(), Some(false));
    assert!(X64V3VnniToken::summon().is_none());
    assert!(ScalarToken.as_x64v3_vnni().is_none());
}

#[cfg(target_arch = "x86_64")]
mod x86_tests {
    use super::*;
    use archmage::{
        X64CryptoToken, X64V1Token, X64V2Token, X64V3CryptoToken, X64V3GfniCryptoToken, X64V3Token,
        attune, autoversion, incant,
    };
    use core::arch::x86_64::*;

    #[test]
    fn target_features_and_leaf_topology() {
        assert_eq!(X64V3VnniToken::NAME, "x86-64-v3 VNNI");
        let features = X64V3VnniToken::TARGET_FEATURES;
        for required in [
            "avx2",
            "fma",
            "bmi2",
            "f16c",
            "aes",
            "pclmulqdq",
            "vaes",
            "vpclmulqdq",
            "gfni",
            "avxvnni",
        ] {
            assert!(
                features.split(',').any(|f| f == required),
                "missing feature {required} in {features}"
            );
        }
        assert!(
            !features.split(',').any(|f| f.starts_with("avx512")),
            "X64V3VnniToken must be pure VEX (no AVX-512): {features}"
        );

        #[cfg(feature = "avx512")]
        {
            let v4x = archmage::X64V4xToken::TARGET_FEATURES;
            assert!(
                !v4x.split(',').any(|f| f == "avxvnni"),
                "X64V4xToken must not require VEX avxvnni (Zen 4/5 lack avxvnni): {v4x}"
            );
        }
    }

    #[test]
    fn hierarchy_implications_and_extractors() {
        let v1 = X64V1Token::summon().expect("SSE2 baseline on x86_64");
        assert!(v1.as_x64v3_vnni().is_none());
        assert!(ScalarToken.as_x64v3_vnni().is_none());

        if let Some(v3) = X64V3Token::summon() {
            assert!(v3.as_x64v3_vnni().is_none());
        }
        if let Some(gfni) = X64V3GfniCryptoToken::summon() {
            assert!(gfni.as_x64v3_vnni().is_none());
        }

        if let Some(vnni) = X64V3VnniToken::summon() {
            assert!(vnni.as_x64v3_vnni().is_some());
            assert!(X64V3GfniCryptoToken::summon().is_some());
            assert!(X64V3CryptoToken::summon().is_some());
            assert!(X64V3Token::summon().is_some());
            assert!(X64CryptoToken::summon().is_some());
            assert!(X64V2Token::summon().is_some());

            let _gfni: X64V3GfniCryptoToken = vnni.v3_gfni_crypto();
            let _v3c: X64V3CryptoToken = vnni.v3_crypto();
            let _v3: X64V3Token = vnni.v3();
            let _c: X64CryptoToken = vnni.x64_crypto();
            let _v2: X64V2Token = vnni.v2();
            let _v1: X64V1Token = vnni.v1();
        }
    }

    #[attune(wrap)]
    fn which_tier_v3_vnni(_token: X64V3VnniToken) -> &'static str {
        "v3_vnni"
    }

    #[attune(wrap)]
    fn which_tier_v3_gfni_crypto(_token: X64V3GfniCryptoToken) -> &'static str {
        "v3_gfni_crypto"
    }

    #[attune(wrap)]
    fn which_tier_v3(_token: X64V3Token) -> &'static str {
        "v3"
    }

    fn which_tier_scalar(_token: ScalarToken) -> &'static str {
        "scalar"
    }

    #[attune(_v3_vnni, _v3_gfni_crypto, _v3, _scalar, dispatch)]
    fn attune_sum(values: &[i32]) -> i32 {
        values.iter().copied().sum()
    }

    #[test]
    fn incant_and_autoversion_dispatch_order() {
        #[cfg(feature = "std")]
        let _lock = archmage::testing::lock_token_testing();

        let dispatched = incant!(which_tier(), [v3_vnni, v3_gfni_crypto, v3, scalar]);
        let expected = if X64V3VnniToken::summon().is_some() {
            "v3_vnni"
        } else if X64V3GfniCryptoToken::summon().is_some() {
            "v3_gfni_crypto"
        } else if X64V3Token::summon().is_some() {
            "v3"
        } else {
            "scalar"
        };
        assert_eq!(dispatched, expected);
        assert_eq!(attune_sum(&[1, 2, 3, 4, 5, 6, 7, 8]), 36);

        #[autoversion(v3_vnni, v3_gfni_crypto, v3, scalar)]
        fn auto_sum(values: &[i32]) -> i32 {
            values.iter().copied().sum()
        }
        assert_eq!(auto_sum(&[1, 2, 3, 4, 5, 6, 7, 8]), 36);
    }

    #[cfg(feature = "std")]
    #[test]
    fn disable_cascade_and_leaf_isolation() {
        let _lock = archmage::testing::lock_token_testing();
        if X64V3VnniToken::compiled_with() == Some(true)
            || X64V3GfniCryptoToken::compiled_with() == Some(true)
        {
            return;
        }

        let had_gfni = X64V3GfniCryptoToken::summon().is_some();
        let had_vnni = X64V3VnniToken::summon().is_some();
        #[cfg(feature = "avx512")]
        let had_v4x = archmage::X64V4xToken::summon().is_some();

        // 1. Disabling leaf X64V3VnniToken must NOT disable X64V3GfniCryptoToken or X64V4xToken.
        X64V3VnniToken::dangerously_disable_token_process_wide(true).unwrap();
        assert!(X64V3VnniToken::manually_disabled().unwrap());
        assert!(X64V3VnniToken::summon().is_none());
        assert_eq!(X64V3GfniCryptoToken::summon().is_some(), had_gfni);
        #[cfg(feature = "avx512")]
        assert_eq!(archmage::X64V4xToken::summon().is_some(), had_v4x);

        X64V3VnniToken::dangerously_disable_token_process_wide(false).unwrap();
        assert_eq!(X64V3VnniToken::summon().is_some(), had_vnni);

        // 2. Disabling parent X64V3GfniCryptoToken MUST cascade to X64V3VnniToken.
        X64V3GfniCryptoToken::dangerously_disable_token_process_wide(true).unwrap();
        assert!(X64V3GfniCryptoToken::summon().is_none());
        assert!(
            X64V3VnniToken::summon().is_none(),
            "disabling X64V3GfniCryptoToken must cascade to X64V3VnniToken"
        );

        X64V3GfniCryptoToken::dangerously_disable_token_process_wide(false).unwrap();
        assert_eq!(X64V3GfniCryptoToken::summon().is_some(), had_gfni);
        assert_eq!(X64V3VnniToken::summon().is_some(), had_vnni);
    }

    fn extract_i32x8(v: __m256i) -> [i32; 8] {
        #[attune(v3)]
        fn inner(v: __m256i) -> [i32; 8] {
            [
                _mm256_extract_epi32::<0>(v),
                _mm256_extract_epi32::<1>(v),
                _mm256_extract_epi32::<2>(v),
                _mm256_extract_epi32::<3>(v),
                _mm256_extract_epi32::<4>(v),
                _mm256_extract_epi32::<5>(v),
                _mm256_extract_epi32::<6>(v),
                _mm256_extract_epi32::<7>(v),
            ]
        }
        #[attune(wrap)]
        fn outer(_token: X64V3Token, v: __m256i) -> [i32; 8] {
            inner(v)
        }
        outer(X64V3Token::summon().unwrap(), v)
    }

    #[attune(wrap)]
    fn run_avx_vnni_intrinsics(
        _token: X64V3VnniToken,
    ) -> ([i32; 8], [i32; 8], [i32; 8], [i32; 8], i32) {
        // 1. 256-bit unsigned u8 * signed i8 dot-product accumulate (`vpdpbusd`)
        let acc = _mm256_setr_epi32(10, 20, 30, 40, 50, 60, 70, 80);
        // Each 32-bit lane holds 4 u8 bytes: [1, 2, 3, 4] = 0x04030201
        let u8s = _mm256_set1_epi32(0x04_03_02_01);
        // Signed i8 bytes: [-1, 2, -3, 4] -> dot = 1*(-1) + 2*2 + 3*(-3) + 4*4 = -1 + 4 - 9 + 16 = 10
        let i8s = _mm256_set1_epi32(i32::from_le_bytes([-1i8 as u8, 2, -3i8 as u8, 4]));
        let dpbusd = _mm256_dpbusd_avx_epi32(acc, u8s, i8s);

        // 2. Saturating `vpdpbusds` near i32::MAX
        let sat_acc = _mm256_set1_epi32(i32::MAX - 5);
        let dpbusds = _mm256_dpbusds_avx_epi32(sat_acc, u8s, i8s);

        // 3. 256-bit signed i16 * signed i16 dot-product accumulate (`vpdpwssd`)
        // Each 32-bit lane holds 2 i16s: [300, -200] * [4, 5] = 1200 - 1000 = 200
        let i16_a = _mm256_set1_epi32((300i16 as u16 as i32) | (((-200i16 as u16) as i32) << 16));
        let i16_b = _mm256_set1_epi32((4i16 as u16 as i32) | ((5i16 as u16 as i32) << 16));
        let dpwssd = _mm256_dpwssd_avx_epi32(acc, i16_a, i16_b);

        // 4. Saturating `vpdpwssds` near i32::MIN
        let min_acc = _mm256_set1_epi32(i32::MIN + 50);
        let neg_b = _mm256_set1_epi32(((-4i16 as u16) as i32) | ((5i16 as u16 as i32) << 16));
        // 300*(-4) + (-200)*5 = -2200 -> saturates at i32::MIN
        let dpwssds = _mm256_dpwssds_avx_epi32(min_acc, i16_a, neg_b);

        // 5. 128-bit `_mm_dpbusd_avx_epi32`
        let acc128 = _mm_set1_epi32(100);
        let u8s128 = _mm_set1_epi32(0x04_03_02_01);
        let i8s128 = _mm_set1_epi32(i32::from_le_bytes([-1i8 as u8, 2, -3i8 as u8, 4]));
        let dp128 = _mm_dpbusd_avx_epi32(acc128, u8s128, i8s128);
        let lane0_128 = _mm_extract_epi32::<0>(dp128);

        (
            [
                _mm256_extract_epi32::<0>(dpbusd),
                _mm256_extract_epi32::<1>(dpbusd),
                _mm256_extract_epi32::<2>(dpbusd),
                _mm256_extract_epi32::<3>(dpbusd),
                _mm256_extract_epi32::<4>(dpbusd),
                _mm256_extract_epi32::<5>(dpbusd),
                _mm256_extract_epi32::<6>(dpbusd),
                _mm256_extract_epi32::<7>(dpbusd),
            ],
            [
                _mm256_extract_epi32::<0>(dpbusds),
                _mm256_extract_epi32::<1>(dpbusds),
                _mm256_extract_epi32::<2>(dpbusds),
                _mm256_extract_epi32::<3>(dpbusds),
                _mm256_extract_epi32::<4>(dpbusds),
                _mm256_extract_epi32::<5>(dpbusds),
                _mm256_extract_epi32::<6>(dpbusds),
                _mm256_extract_epi32::<7>(dpbusds),
            ],
            [
                _mm256_extract_epi32::<0>(dpwssd),
                _mm256_extract_epi32::<1>(dpwssd),
                _mm256_extract_epi32::<2>(dpwssd),
                _mm256_extract_epi32::<3>(dpwssd),
                _mm256_extract_epi32::<4>(dpwssd),
                _mm256_extract_epi32::<5>(dpwssd),
                _mm256_extract_epi32::<6>(dpwssd),
                _mm256_extract_epi32::<7>(dpwssd),
            ],
            [
                _mm256_extract_epi32::<0>(dpwssds),
                _mm256_extract_epi32::<1>(dpwssds),
                _mm256_extract_epi32::<2>(dpwssds),
                _mm256_extract_epi32::<3>(dpwssds),
                _mm256_extract_epi32::<4>(dpwssds),
                _mm256_extract_epi32::<5>(dpwssds),
                _mm256_extract_epi32::<6>(dpwssds),
                _mm256_extract_epi32::<7>(dpwssds),
            ],
            lane0_128,
        )
    }

    #[test]
    fn hardware_avx_vnni_execution() {
        #[cfg(feature = "std")]
        let _lock = archmage::testing::lock_token_testing();

        if let Some(token) = X64V3VnniToken::summon() {
            let (dpbusd, dpbusds, dpwssd, dpwssds, dp128_0) = run_avx_vnni_intrinsics(token);
            let _ = extract_i32x8;
            assert_eq!(dpbusd, [20, 30, 40, 50, 60, 70, 80, 90]);
            assert_eq!(dpbusds, [i32::MAX; 8]);
            assert_eq!(dpwssd, [210, 220, 230, 240, 250, 260, 270, 280]);
            assert_eq!(dpwssds, [i32::MIN; 8]);
            assert_eq!(dp128_0, 110);
        }
    }
}
