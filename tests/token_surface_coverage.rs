//! Coverage for `IntoConcreteToken` downcasts/conversions, x86 parent
//! extractors & disable/compiled_with checks, cross-arch token stubs, and
//! `testing` display/policy paths.

use archmage::{
    Arm64V2Token, Arm64V3Token, Avx512Fp16Token, IntoConcreteToken, NeonAesToken, NeonCrcToken,
    NeonSha3Token, NeonToken, ScalarToken, SimdToken, Wasm128RelaxedToken, Wasm128Token,
    X64CryptoToken, X64V1Token, X64V2Token, X64V3CryptoToken, X64V3GfniCryptoToken, X64V3Token,
    X64V3VnniToken, X64V4Token, X64V4xToken,
};

fn assert_all_into_concrete_none_except<T: IntoConcreteToken + Copy>(
    token: T,
    expected_name: &'static str,
) {
    // Every cross-arch or strictly stronger downcast must return None; exact
    // identity must return Some.
    assert_eq!(
        token.as_scalar().is_some(),
        expected_name == ScalarToken::NAME
    );
    assert_eq!(
        token.as_x64v1().is_some(),
        expected_name == X64V1Token::NAME
    );
    assert_eq!(
        token.as_x64v2().is_some(),
        expected_name == X64V2Token::NAME
    );
    assert_eq!(
        token.as_x64_crypto().is_some(),
        expected_name == X64CryptoToken::NAME
    );
    assert_eq!(
        token.as_x64v3().is_some(),
        expected_name == X64V3Token::NAME
    );
    assert_eq!(
        token.as_x64v3_crypto().is_some(),
        expected_name == X64V3CryptoToken::NAME
    );
    assert_eq!(
        token.as_x64v3_gfni_crypto().is_some(),
        expected_name == X64V3GfniCryptoToken::NAME
    );
    assert_eq!(
        token.as_x64v3_vnni().is_some(),
        expected_name == X64V3VnniToken::NAME
    );
    assert_eq!(
        token.as_x64v4().is_some(),
        expected_name == X64V4Token::NAME
    );
    assert_eq!(
        token.as_x64v4x().is_some(),
        expected_name == X64V4xToken::NAME
    );
    assert_eq!(
        token.as_avx512_fp16().is_some(),
        expected_name == Avx512Fp16Token::NAME
    );
    assert_eq!(token.as_neon().is_some(), expected_name == NeonToken::NAME);
    assert_eq!(
        token.as_neon_aes().is_some(),
        expected_name == NeonAesToken::NAME
    );
    assert_eq!(
        token.as_neon_sha3().is_some(),
        expected_name == NeonSha3Token::NAME
    );
    assert_eq!(
        token.as_neon_crc().is_some(),
        expected_name == NeonCrcToken::NAME
    );
    assert_eq!(
        token.as_arm_v2().is_some(),
        expected_name == Arm64V2Token::NAME
    );
    assert_eq!(
        token.as_arm_v3().is_some(),
        expected_name == Arm64V3Token::NAME
    );
    assert_eq!(
        token.as_wasm128().is_some(),
        expected_name == Wasm128Token::NAME
    );
    assert_eq!(
        token.as_wasm128_relaxed().is_some(),
        expected_name == Wasm128RelaxedToken::NAME
    );

    // Exercise try_* extraction helpers too (never returns cross-arch tokens)
    let _ = (
        token.try_v1(),
        token.try_v2(),
        token.try_x64_crypto(),
        token.try_v3(),
        token.try_v3_crypto(),
        token.try_v3_gfni_crypto(),
        token.try_v3_vnni(),
        token.try_v4(),
        token.try_v4x(),
        token.try_avx512_fp16(),
        token.try_neon(),
        token.try_neon_aes(),
        token.try_neon_sha3(),
        token.try_neon_crc(),
        token.try_arm_v2(),
        token.try_arm_v3(),
        token.try_wasm128(),
        token.try_wasm128_relaxed(),
    );
}

#[test]
fn scalar_and_available_x86_tokens_into_concrete_and_parent_extractors() {
    assert_all_into_concrete_none_except(ScalarToken, ScalarToken::NAME);

    #[cfg(target_arch = "x86_64")]
    {
        if let Some(t) = X64V1Token::summon() {
            assert_all_into_concrete_none_except(t, X64V1Token::NAME);
            assert!(t.try_v1().is_some());
            assert!(t.try_v2().is_none());
            let _ = X64V1Token::compiled_with();
            let _ = X64V1Token::manually_disabled();
        }
        if let Some(t) = X64V2Token::summon() {
            assert_all_into_concrete_none_except(t, X64V2Token::NAME);
            let _ = t.v1();
            assert!(t.try_v1().is_some());
            assert!(t.try_v2().is_some());
            let _ = X64V2Token::compiled_with();
            let _ = X64V2Token::manually_disabled();
        }
        if let Some(t) = X64CryptoToken::summon() {
            assert_all_into_concrete_none_except(t, X64CryptoToken::NAME);
            let _ = (t.v1(), t.v2());
            assert!(t.try_v2().is_some());
            let _ = X64CryptoToken::compiled_with();
            let _ = X64CryptoToken::manually_disabled();
        }
        if let Some(t) = X64V3Token::summon() {
            assert_all_into_concrete_none_except(t, X64V3Token::NAME);
            let _ = (t.v1(), t.v2());
            assert!(t.try_v3().is_some());
            let _ = X64V3Token::compiled_with();
            let _ = X64V3Token::manually_disabled();
        }
        if let Some(t) = X64V3CryptoToken::summon() {
            assert_all_into_concrete_none_except(t, X64V3CryptoToken::NAME);
            let _ = (t.v1(), t.v2(), t.v3(), t.x64_crypto());
            assert!(t.try_v3().is_some());
            assert!(t.try_x64_crypto().is_some());
            let _ = X64V3CryptoToken::compiled_with();
            let _ = X64V3CryptoToken::manually_disabled();
        }
        if let Some(t) = X64V3GfniCryptoToken::summon() {
            assert_all_into_concrete_none_except(t, X64V3GfniCryptoToken::NAME);
            let _ = (t.v1(), t.v2(), t.v3(), t.x64_crypto());
            let _ = X64V3GfniCryptoToken::compiled_with();
            let _ = X64V3GfniCryptoToken::manually_disabled();
        }
        if let Some(t) = X64V3VnniToken::summon() {
            assert_all_into_concrete_none_except(t, X64V3VnniToken::NAME);
            let _ = (t.v1(), t.v2(), t.v3());
            let _ = X64V3VnniToken::compiled_with();
            let _ = X64V3VnniToken::manually_disabled();
        }
        #[cfg(feature = "avx512")]
        {
            if let Some(t) = X64V4Token::summon() {
                assert_all_into_concrete_none_except(t, X64V4Token::NAME);
                let _ = (t.v1(), t.v2(), t.v3());
                assert!(t.try_v4().is_some());
                let _ = X64V4Token::compiled_with();
                let _ = X64V4Token::manually_disabled();
            }
            if let Some(t) = X64V4xToken::summon() {
                assert_all_into_concrete_none_except(t, X64V4xToken::NAME);
                let _ = (t.v1(), t.v2(), t.v3(), t.v4());
                assert!(t.try_v4x().is_some());
                assert!(t.try_v4().is_some());
                let _ = X64V4xToken::compiled_with();
                let _ = X64V4xToken::manually_disabled();
            }
            if let Some(t) = Avx512Fp16Token::summon() {
                assert_all_into_concrete_none_except(t, Avx512Fp16Token::NAME);
                let _ = (t.v1(), t.v2(), t.v3(), t.v4());
                let _ = Avx512Fp16Token::compiled_with();
                let _ = Avx512Fp16Token::manually_disabled();
            }
        }
    }
}

#[test]
fn cross_arch_stub_tokens_report_unavailable_consistently() {
    #[cfg(not(target_arch = "aarch64"))]
    {
        macro_rules! check_arm_stub {
            ($tok:ty) => {{
                assert!(<$tok>::summon().is_none());
                assert_eq!(<$tok>::compiled_with(), Some(false));
                #[allow(deprecated)]
                {
                    assert_eq!(<$tok>::guaranteed(), Some(false));
                }
                let err = <$tok>::manually_disabled().unwrap_err();
                assert!(!format!("{err}").is_empty());
                assert!(<$tok>::dangerously_disable_token_process_wide(false).is_err());
            }};
        }
        check_arm_stub!(NeonToken);
        check_arm_stub!(NeonAesToken);
        check_arm_stub!(NeonSha3Token);
        check_arm_stub!(NeonCrcToken);
        check_arm_stub!(Arm64V2Token);
        check_arm_stub!(Arm64V3Token);
    }

    #[cfg(not(target_arch = "wasm32"))]
    {
        macro_rules! check_wasm_stub {
            ($tok:ty) => {{
                assert!(<$tok>::summon().is_none());
                assert_eq!(<$tok>::compiled_with(), Some(false));
                #[allow(deprecated)]
                {
                    assert_eq!(<$tok>::guaranteed(), Some(false));
                }
                let err = <$tok>::manually_disabled().unwrap_err();
                assert!(!format!("{err}").is_empty());
                assert!(<$tok>::dangerously_disable_token_process_wide(false).is_err());
            }};
        }
        check_wasm_stub!(Wasm128Token);
        check_wasm_stub!(Wasm128RelaxedToken);
    }
}

#[cfg(feature = "std")]
#[test]
fn testing_display_and_warn_stderr_policy() {
    use archmage::testing::{CompileTimePolicy, for_each_token_permutation};

    let mut seen_labels = Vec::new();
    let report = for_each_token_permutation(CompileTimePolicy::WarnStderr, |perm| {
        let formatted = format!("{perm}");
        assert_eq!(formatted, perm.label);
        seen_labels.push(formatted);
    });
    assert!(!seen_labels.is_empty());
    assert_eq!(seen_labels.len(), report.permutations_run);
    let report_str = format!("{report}");
    assert!(
        report_str.contains(&format!("{}", report.permutations_run)),
        "report Display missing permutation count: {report_str}"
    );
}
