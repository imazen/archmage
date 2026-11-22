use archmage::tokens::IntoConcreteToken;
use archmage::{
    Arm64V2Token, Arm64V3Token, Avx512Fp16Token, NeonAesToken, NeonCrcToken, NeonSha3Token,
    NeonToken, ScalarToken, SimdToken, Wasm128RelaxedToken, Wasm128Token, X64CryptoToken,
    X64V1Token, X64V2Token, X64V3CryptoToken, X64V3GfniCryptoToken, X64V3Token, X64V3VnniToken,
    X64V4Token, X64V4xToken,
};

macro_rules! check_simd_proofs {
    ($token:ident, $parents:ident; $($target:ident: $as:ident, $try_method:ident);+ $(;)?) => {
        $(
            let exact = T::NAME == $target::NAME;
            assert_eq!($token.$as().is_some(), exact, "{} -> {}", T::NAME, stringify!($as));
            assert_eq!(
                $token.$try_method().is_some(),
                exact || $parents.contains(&$target::NAME),
                "{} -> {}",
                T::NAME,
                stringify!($try_method),
            );
        )+
    };
}

fn check<T: IntoConcreteToken>(parents: &[&str]) {
    if let Some(token) = T::summon() {
        assert_eq!(token.as_scalar().is_some(), T::NAME == ScalarToken::NAME);
        check_simd_proofs!(token, parents;
            X64V1Token: as_x64v1, try_v1;
            X64V2Token: as_x64v2, try_v2;
            X64CryptoToken: as_x64_crypto, try_x64_crypto;
            X64V3Token: as_x64v3, try_v3;
            X64V3CryptoToken: as_x64v3_crypto, try_v3_crypto;
            X64V3GfniCryptoToken: as_x64v3_gfni_crypto, try_v3_gfni_crypto;
            X64V3VnniToken: as_x64v3_vnni, try_v3_vnni;
            X64V4Token: as_x64v4, try_v4;
            X64V4xToken: as_x64v4x, try_v4x;
            Avx512Fp16Token: as_avx512_fp16, try_avx512_fp16;
            NeonToken: as_neon, try_neon;
            NeonAesToken: as_neon_aes, try_neon_aes;
            NeonSha3Token: as_neon_sha3, try_neon_sha3;
            NeonCrcToken: as_neon_crc, try_neon_crc;
            Arm64V2Token: as_arm_v2, try_arm_v2;
            Arm64V3Token: as_arm_v3, try_arm_v3;
            Wasm128Token: as_wasm128, try_wasm128;
            Wasm128RelaxedToken: as_wasm128_relaxed, try_wasm128_relaxed;
        );
        eprintln!("Checked {}", T::NAME);
    } else if T::compiled_with() == Some(true) {
        panic!("{} must be available on this target", T::NAME);
    }
}

#[test]
fn identity_and_hierarchy_matrix() {
    let v1 = X64V1Token::NAME;
    let v2 = X64V2Token::NAME;
    let crypto = X64CryptoToken::NAME;
    let v3 = X64V3Token::NAME;
    let v3_crypto = X64V3CryptoToken::NAME;
    let gfni = X64V3GfniCryptoToken::NAME;
    let v4 = X64V4Token::NAME;
    let neon = NeonToken::NAME;

    check::<ScalarToken>(&[]);
    check::<X64V1Token>(&[]);
    check::<X64V2Token>(&[v1]);
    check::<X64CryptoToken>(&[v2, v1]);
    check::<X64V3Token>(&[v2, v1]);
    check::<X64V3CryptoToken>(&[v3, crypto, v2, v1]);
    check::<X64V3GfniCryptoToken>(&[v3_crypto, v3, crypto, v2, v1]);
    check::<X64V3VnniToken>(&[gfni, v3_crypto, v3, crypto, v2, v1]);
    check::<X64V4Token>(&[v3, crypto, v2, v1]);
    check::<X64V4xToken>(&[v4, gfni, v3_crypto, v3, crypto, v2, v1]);
    check::<Avx512Fp16Token>(&[v4, v3, crypto, v2, v1]);
    check::<NeonToken>(&[]);
    check::<NeonAesToken>(&[neon]);
    check::<NeonSha3Token>(&[neon]);
    check::<NeonCrcToken>(&[neon]);
    check::<Arm64V2Token>(&[neon]);
    check::<Arm64V3Token>(&[Arm64V2Token::NAME, neon]);
    check::<Wasm128Token>(&[]);
    check::<Wasm128RelaxedToken>(&[Wasm128Token::NAME]);
}
