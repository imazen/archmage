//! Comprehensive tests for `X64V3Vnni2Token`
//! (`x86-64-v3` + Crypto + GFNI + AVX-VNNI + ADX + SHA + AVX-VNNI-INT8 + AVX-NE-CONVERT + AVX-IFMA).
//!
//! Exercises:
//! - Token traits, `TARGET_FEATURES`, pure-VEX constraint (no AVX-512), and leaf topology
//!   (not implied by `X64V4xToken` or `X64V3VnniToken`)
//! - Zero-cost downcast extractors (`v3_vnni`, `v3_gfni_crypto`, `v3_crypto`, `v3`, `x64_crypto`, `v2`, `v1`)
//! - `IntoConcreteToken::as_x64v3_vnni2`
//! - `incant!` and `#[autoversion]` dispatch ordering (`v3_vnni2` > `v3_vnni` > `v3_gfni_crypto` > `v3` > `scalar`)
//! - Process-wide disable cascading (`X64V3GfniCryptoToken` -> `X64V3VnniToken` -> `X64V3Vnni2Token`,
//!   plus leaf isolation when disabling `X64V3Vnni2Token`)
//! - Safe hardware execution of `avxvnniint8` (`vpdpbssd`/`vpdpbssds`), `avxifma` (`vpmadd52luq`/`vpmadd52huq`),
//!   `avxneconvert` (`vcvtneps2bf16`), and `sha` (`_mm_sha256msg1_epu32`) under `#![forbid(unsafe_code)]`
//!   (`adx` `_addcarryx_u64` requires `unsafe` on Rust 1.89 and is exercised in `feature_consistency`).

#![forbid(unsafe_code)]

use archmage::{IntoConcreteToken, ScalarToken, SimdToken, X64V3Vnni2Token};

#[cfg(not(target_arch = "x86_64"))]
#[test]
fn v3_vnni2_is_stub_off_x86_64() {
    assert_eq!(X64V3Vnni2Token::compiled_with(), Some(false));
    assert!(X64V3Vnni2Token::summon().is_none());
    assert!(ScalarToken.as_x64v3_vnni2().is_none());
}

#[cfg(target_arch = "x86_64")]
mod x86_tests {
    use super::*;
    use archmage::{
        X64CryptoToken, X64V1Token, X64V2Token, X64V3CryptoToken, X64V3GfniCryptoToken, X64V3Token,
        X64V3VnniToken, attune, autoversion, incant,
    };
    use core::arch::x86_64::*;

    #[test]
    fn target_features_and_leaf_topology() {
        assert_eq!(X64V3Vnni2Token::NAME, "x86-64-v3 VNNI-2");
        let features = X64V3Vnni2Token::TARGET_FEATURES;
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
            "adx",
            "sha",
            "avxvnniint8",
            "avxneconvert",
            "avxifma",
        ] {
            assert!(
                features.split(',').any(|f| f == required),
                "missing feature {required} in {features}"
            );
        }
        assert!(
            !features.split(',').any(|f| f.starts_with("avx512")),
            "X64V3Vnni2Token must be pure VEX (no AVX-512): {features}"
        );
        for omitted in ["avxvnniint16", "sha512", "sm3", "sm4"] {
            assert!(
                !features.split(',').any(|f| f == omitted),
                "X64V3Vnni2Token must remain cross-vendor and omit {omitted}: {features}"
            );
        }

        let vnni = X64V3VnniToken::TARGET_FEATURES;
        for extra in ["avxvnniint8", "avxneconvert", "avxifma"] {
            assert!(
                !vnni.split(',').any(|f| f == extra),
                "X64V3VnniToken must not require {extra}: {vnni}"
            );
        }

        #[cfg(feature = "avx512")]
        {
            let v4x = archmage::X64V4xToken::TARGET_FEATURES;
            for v3_vnni2_only in ["avxvnni", "avxvnniint8", "avxneconvert", "avxifma"] {
                assert!(
                    !v4x.split(',').any(|f| f == v3_vnni2_only),
                    "X64V4xToken must not require VEX {v3_vnni2_only}: {v4x}"
                );
            }
        }
    }

    #[test]
    fn hierarchy_implications_and_extractors() {
        let v1 = X64V1Token::summon().expect("SSE2 baseline on x86_64");
        assert!(v1.as_x64v3_vnni2().is_none());
        assert!(ScalarToken.as_x64v3_vnni2().is_none());

        if let Some(v3) = X64V3Token::summon() {
            assert!(v3.as_x64v3_vnni2().is_none());
        }
        if let Some(gfni) = X64V3GfniCryptoToken::summon() {
            assert!(gfni.as_x64v3_vnni2().is_none());
        }
        if let Some(vnni) = X64V3VnniToken::summon() {
            let _ = vnni;
            // `X64V3VnniToken` may or may not summon `X64V3Vnni2Token` depending on host,
            // but `IntoConcreteToken::as_x64v3_vnni2()` checks concrete type only.
            assert!(vnni.as_x64v3_vnni2().is_none());
        }

        if let Some(vnni2) = X64V3Vnni2Token::summon() {
            assert!(vnni2.as_x64v3_vnni2().is_some());
            assert!(X64V3VnniToken::summon().is_some());
            assert!(X64V3GfniCryptoToken::summon().is_some());
            assert!(X64V3CryptoToken::summon().is_some());
            assert!(X64V3Token::summon().is_some());
            assert!(X64CryptoToken::summon().is_some());
            assert!(X64V2Token::summon().is_some());

            let _vnni: X64V3VnniToken = vnni2.v3_vnni();
            let _gfni: X64V3GfniCryptoToken = vnni2.v3_gfni_crypto();
            let _v3c: X64V3CryptoToken = vnni2.v3_crypto();
            let _v3: X64V3Token = vnni2.v3();
            let _c: X64CryptoToken = vnni2.x64_crypto();
            let _v2: X64V2Token = vnni2.v2();
            let _v1: X64V1Token = vnni2.v1();
        }
    }

    #[attune(wrap)]
    fn which_tier_v3_vnni2(_token: X64V3Vnni2Token) -> &'static str {
        "v3_vnni2"
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

    #[attune(_v3_vnni2, _v3_vnni, _v3_gfni_crypto, _v3, _scalar, dispatch)]
    fn attune_sum(values: &[i32]) -> i32 {
        values.iter().copied().sum()
    }

    #[test]
    fn incant_and_autoversion_dispatch_order() {
        #[cfg(feature = "std")]
        let _lock = archmage::testing::lock_token_testing();

        let dispatched = incant!(
            which_tier(),
            [v3_vnni2, v3_vnni, v3_gfni_crypto, v3, scalar]
        );
        let expected = if X64V3Vnni2Token::summon().is_some() {
            "v3_vnni2"
        } else if X64V3VnniToken::summon().is_some() {
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

        #[autoversion(v3_vnni2, v3_vnni, v3_gfni_crypto, v3, scalar)]
        fn auto_sum(values: &[i32]) -> i32 {
            values.iter().copied().sum()
        }
        assert_eq!(auto_sum(&[1, 2, 3, 4, 5, 6, 7, 8]), 36);
    }

    #[cfg(feature = "std")]
    #[test]
    fn disable_cascade_and_leaf_isolation() {
        let _lock = archmage::testing::lock_token_testing();
        if X64V3Vnni2Token::compiled_with() == Some(true)
            || X64V3VnniToken::compiled_with() == Some(true)
            || X64V3GfniCryptoToken::compiled_with() == Some(true)
        {
            return;
        }

        let had_gfni = X64V3GfniCryptoToken::summon().is_some();
        let had_vnni = X64V3VnniToken::summon().is_some();
        let had_vnni2 = X64V3Vnni2Token::summon().is_some();
        #[cfg(feature = "avx512")]
        let had_v4x = archmage::X64V4xToken::summon().is_some();

        // 1. Disabling leaf X64V3Vnni2Token must NOT disable X64V3VnniToken or X64V4xToken.
        X64V3Vnni2Token::dangerously_disable_token_process_wide(true).unwrap();
        assert!(X64V3Vnni2Token::manually_disabled().unwrap());
        assert!(X64V3Vnni2Token::summon().is_none());
        assert_eq!(X64V3VnniToken::summon().is_some(), had_vnni);
        #[cfg(feature = "avx512")]
        assert_eq!(archmage::X64V4xToken::summon().is_some(), had_v4x);

        X64V3Vnni2Token::dangerously_disable_token_process_wide(false).unwrap();
        assert_eq!(X64V3Vnni2Token::summon().is_some(), had_vnni2);

        // 2. Disabling X64V3VnniToken MUST cascade to X64V3Vnni2Token.
        X64V3VnniToken::dangerously_disable_token_process_wide(true).unwrap();
        assert!(X64V3VnniToken::summon().is_none());
        assert!(
            X64V3Vnni2Token::summon().is_none(),
            "disabling X64V3VnniToken must cascade to X64V3Vnni2Token"
        );
        X64V3VnniToken::dangerously_disable_token_process_wide(false).unwrap();
        assert_eq!(X64V3VnniToken::summon().is_some(), had_vnni);
        assert_eq!(X64V3Vnni2Token::summon().is_some(), had_vnni2);

        // 3. Disabling X64V3GfniCryptoToken MUST cascade through VNNI to X64V3Vnni2Token.
        X64V3GfniCryptoToken::dangerously_disable_token_process_wide(true).unwrap();
        assert!(X64V3GfniCryptoToken::summon().is_none());
        assert!(X64V3VnniToken::summon().is_none());
        assert!(X64V3Vnni2Token::summon().is_none());

        X64V3GfniCryptoToken::dangerously_disable_token_process_wide(false).unwrap();
        assert_eq!(X64V3GfniCryptoToken::summon().is_some(), had_gfni);
        assert_eq!(X64V3VnniToken::summon().is_some(), had_vnni);
        assert_eq!(X64V3Vnni2Token::summon().is_some(), had_vnni2);
    }

    struct Vnni2Outputs {
        dpbssd: [i32; 8],
        dpbssds: [i32; 8],
        ifma_lo: [i64; 4],
        ifma_hi: [i64; 4],
        sha: [i32; 4],
    }

    #[attune(wrap)]
    fn run_v3_vnni2_intrinsics(_token: X64V3Vnni2Token) -> Vnni2Outputs {
        // avxvnniint8: signed * signed 8-bit dot products
        let src = _mm256_set_epi32(17, 16, 15, 14, 13, 12, 11, 10);
        let a = _mm256_set_epi8(
            -8, -8, -8, -8, -7, -7, -7, -7, -6, -6, -6, -6, -5, -5, -5, -5, // hi
            -4, -4, -4, -4, -3, -3, -3, -3, -2, -2, -2, -2, -1, -1, -1, -1, // lo
        );
        let b = _mm256_set1_epi8(-2);
        let dpbssd = _mm256_dpbssd_epi32(src, a, b);
        let dpbssds = _mm256_dpbssds_epi32(
            _mm256_set1_epi32(i32::MAX - 5),
            _mm256_set1_epi8(4),
            _mm256_set1_epi8(4),
        );

        // avxifma: 52-bit unsigned multiply-add (lo and hi)
        let lo = _mm256_madd52lo_avx_epu64(
            _mm256_set_epi64x(40, 30, 20, 10),
            _mm256_set_epi64x(7, 5, 3, 2),
            _mm256_set_epi64x(11, 13, 17, 19),
        );
        let hi_factor = _mm256_set1_epi64x(1i64 << 30); // (2^30 * 2^30) >> 52 = 2^8 = 256
        let hi = _mm256_madd52hi_avx_epu64(_mm256_set_epi64x(4, 3, 2, 1), hi_factor, hi_factor);

        // avxneconvert (safe BF16 pack available since Rust 1.89)
        let f32s = _mm256_set_ps(8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0);
        let pbh = _mm256_cvtneps_avx_pbh(f32s);
        let _ = core::hint::black_box(pbh);

        // sha inherited from X64V3VnniToken (`adx` `_addcarryx_u64` requires `unsafe` in Rust 1.89
        // and is exercised in `tests/feature_consistency.rs`)
        let sha_msg = _mm_sha256msg1_epu32(_mm_set1_epi32(1), _mm_set1_epi32(2));

        let extract32 = |v: __m256i| {
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
        };
        let extract64 = |v: __m256i| {
            [
                _mm256_extract_epi64::<0>(v),
                _mm256_extract_epi64::<1>(v),
                _mm256_extract_epi64::<2>(v),
                _mm256_extract_epi64::<3>(v),
            ]
        };
        let sha = [
            _mm_extract_epi32::<0>(sha_msg),
            _mm_extract_epi32::<1>(sha_msg),
            _mm_extract_epi32::<2>(sha_msg),
            _mm_extract_epi32::<3>(sha_msg),
        ];

        Vnni2Outputs {
            dpbssd: extract32(dpbssd),
            dpbssds: extract32(dpbssds),
            ifma_lo: extract64(lo),
            ifma_hi: extract64(hi),
            sha,
        }
    }

    #[test]
    fn hardware_avx_vnni2_execution() {
        #[cfg(feature = "std")]
        let _lock = archmage::testing::lock_token_testing();

        if let Some(token) = X64V3Vnni2Token::summon() {
            let out = run_v3_vnni2_intrinsics(token);
            assert_eq!(out.dpbssd, [18, 27, 36, 45, 54, 63, 72, 81]);
            assert_eq!(out.dpbssds, [i32::MAX; 8]);
            assert_eq!(out.ifma_lo, [10 + 38, 20 + 51, 30 + 65, 40 + 77]);
            assert_eq!(out.ifma_hi, [257, 258, 259, 260]);
            assert_ne!(out.sha, [0; 4]);
        }
    }
}
