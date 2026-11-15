#![forbid(unsafe_code)]

use archmage::SimdToken;
#[cfg(feature = "w512")]
use magemath::{
    cbrtf::try_cbrt_f32x16_with,
    expf::{try_exp_f32x16_with, try_exp2_f32x16_with},
    logf::{
        try_ln_f32x16_with, try_log1p_f32x16_with, try_log2_f32x16_with, try_log10_f32x16_with,
    },
    powf::{try_powf_f32x16_runtime, try_powf_f32x16_with},
};
use magemath::{
    cbrtf::{CbrtfOptions, try_cbrt_f32x4_with, try_cbrt_f32x8_with, try_cbrt_with_token},
    expf::{
        ExpfOptions, try_exp_f32x4_with, try_exp_f32x8_with, try_exp_with_token,
        try_exp2_f32x4_with, try_exp2_f32x8_with, try_exp2_with_token,
    },
    logf::{
        LogfOptions, try_ln_f32x4_with, try_ln_f32x8_with, try_ln_with_token, try_log1p_f32x4_with,
        try_log1p_f32x8_with, try_log1p_with_token, try_log2_f32x4_with, try_log2_f32x8_with,
        try_log2_with_token, try_log10_f32x4_with, try_log10_f32x8_with, try_log10_with_token,
    },
    powf::{
        PowfOptions, try_powf_f32x4_runtime, try_powf_f32x4_with, try_powf_f32x8_runtime,
        try_powf_f32x8_with, try_powf_with_token,
    },
};
#[cfg(feature = "w512")]
use magetypes::simd::backends::F32x16Convert;
use magetypes::simd::backends::{F32x4Convert, F32x8Convert};
#[cfg(feature = "w512")]
use magetypes::simd::generic::f32x16;
use magetypes::simd::generic::{f32x4, f32x8};

const EXP_PORTABLE: u32 = ExpfOptions::new().bits();
const EXP_BACKEND: u32 = ExpfOptions::new()
    .with_portability(magemath::expf::ExpfPortability::BackendDependent)
    .bits();
const LOG_PORTABLE: u32 = LogfOptions::new().bits();
const CBRT_PORTABLE: u32 = CbrtfOptions::new().bits();
const POW_PORTABLE_HIGHP: u32 = PowfOptions::PORTABLE_HIGHP.bits();
const POW_LOWP: u32 = PowfOptions::new().bits();

fn check_token_math<T: F32x4Convert + F32x8Convert>(token: T) {
    let inputs4 = [0.25f32, 0.5, 1.5, 2.75];
    let inputs8 = [0.125f32, 0.25, 0.5, 0.75, 1.25, 1.75, 2.5, 3.25];
    let v4 = f32x4::from_array_t(token, inputs4);
    let v8 = f32x8::from_array_t(token, inputs8);

    // Scalar + vector exp / exp2 (both portable & backend-dependent paths)
    for &x in &inputs4 {
        let e = try_exp_with_token::<EXP_PORTABLE, _>(token, x).unwrap();
        let e2 = try_exp2_with_token::<EXP_BACKEND, _>(token, x).unwrap();
        assert!((e - x.exp()).abs() <= 1e-5, "exp({x})={e}");
        assert!((e2 - x.exp2()).abs() <= 1e-5, "exp2({x})={e2}");
    }
    let ve4 = try_exp_f32x4_with::<EXP_PORTABLE, _>(token, v4)
        .unwrap()
        .to_array();
    let ve8 = try_exp_f32x8_with::<EXP_PORTABLE, _>(token, v8)
        .unwrap()
        .to_array();
    let ve2_4 = try_exp2_f32x4_with::<EXP_BACKEND, _>(token, v4)
        .unwrap()
        .to_array();
    let ve2_8 = try_exp2_f32x8_with::<EXP_BACKEND, _>(token, v8)
        .unwrap()
        .to_array();
    for (i, &x) in inputs4.iter().enumerate() {
        assert!((ve4[i] - x.exp()).abs() <= 1e-5);
        assert!((ve2_4[i] - x.exp2()).abs() <= 1e-5);
    }
    for (i, &x) in inputs8.iter().enumerate() {
        assert!((ve8[i] - x.exp()).abs() <= 1e-5);
        assert!((ve2_8[i] - x.exp2()).abs() <= 1e-5);
    }

    // Log family
    for &x in &inputs4 {
        assert!((try_ln_with_token::<LOG_PORTABLE, _>(token, x).unwrap() - x.ln()).abs() <= 1e-5);
        assert!(
            (try_log2_with_token::<LOG_PORTABLE, _>(token, x).unwrap() - x.log2()).abs() <= 1e-5
        );
        assert!(
            (try_log10_with_token::<LOG_PORTABLE, _>(token, x).unwrap() - x.log10()).abs() <= 1e-5
        );
        assert!(
            (try_log1p_with_token::<LOG_PORTABLE, _>(token, x).unwrap() - x.ln_1p()).abs() <= 1e-5
        );
    }
    let ln4 = try_ln_f32x4_with::<LOG_PORTABLE, _>(token, v4)
        .unwrap()
        .to_array();
    let l2_4 = try_log2_f32x4_with::<LOG_PORTABLE, _>(token, v4)
        .unwrap()
        .to_array();
    let l10_4 = try_log10_f32x4_with::<LOG_PORTABLE, _>(token, v4)
        .unwrap()
        .to_array();
    let l1p_4 = try_log1p_f32x4_with::<LOG_PORTABLE, _>(token, v4)
        .unwrap()
        .to_array();
    let ln8 = try_ln_f32x8_with::<LOG_PORTABLE, _>(token, v8)
        .unwrap()
        .to_array();
    let l2_8 = try_log2_f32x8_with::<LOG_PORTABLE, _>(token, v8)
        .unwrap()
        .to_array();
    let l10_8 = try_log10_f32x8_with::<LOG_PORTABLE, _>(token, v8)
        .unwrap()
        .to_array();
    let l1p_8 = try_log1p_f32x8_with::<LOG_PORTABLE, _>(token, v8)
        .unwrap()
        .to_array();
    for (i, &x) in inputs4.iter().enumerate() {
        assert!((ln4[i] - x.ln()).abs() <= 1e-5);
        assert!((l2_4[i] - x.log2()).abs() <= 1e-5);
        assert!((l10_4[i] - x.log10()).abs() <= 1e-5);
        assert!((l1p_4[i] - x.ln_1p()).abs() <= 1e-5);
    }
    for (i, &x) in inputs8.iter().enumerate() {
        assert!((ln8[i] - x.ln()).abs() <= 1e-5);
        assert!((l2_8[i] - x.log2()).abs() <= 1e-5);
        assert!((l10_8[i] - x.log10()).abs() <= 1e-5);
        assert!((l1p_8[i] - x.ln_1p()).abs() <= 1e-5);
    }

    // Cbrt
    for &x in &inputs4 {
        assert!(
            (try_cbrt_with_token::<CBRT_PORTABLE, _>(token, x).unwrap() - x.cbrt()).abs() <= 1e-5
        );
    }
    let c4 = try_cbrt_f32x4_with::<CBRT_PORTABLE, _>(token, v4)
        .unwrap()
        .to_array();
    let c8 = try_cbrt_f32x8_with::<CBRT_PORTABLE, _>(token, v8)
        .unwrap()
        .to_array();
    for (i, &x) in inputs4.iter().enumerate() {
        assert!((c4[i] - x.cbrt()).abs() <= 1e-5);
    }
    for (i, &x) in inputs8.iter().enumerate() {
        assert!((c8[i] - x.cbrt()).abs() <= 1e-5);
    }

    // Powf (lowp + portable highp, const + runtime)
    let exp = 2.4f32;
    for &x in &inputs4 {
        assert!(
            (try_powf_with_token::<POW_PORTABLE_HIGHP, _>(token, x, exp).unwrap() - x.powf(exp))
                .abs()
                <= 1e-4
        );
    }
    let p4 = try_powf_f32x4_with::<POW_PORTABLE_HIGHP, _>(token, v4, exp)
        .unwrap()
        .to_array();
    let p8 = try_powf_f32x8_with::<POW_PORTABLE_HIGHP, _>(token, v8, exp)
        .unwrap()
        .to_array();
    let pr4 = try_powf_f32x4_runtime(token, v4, exp, PowfOptions::new())
        .unwrap()
        .to_array();
    let pr8 = try_powf_f32x8_runtime(token, v8, exp, PowfOptions::new())
        .unwrap()
        .to_array();
    let pl4 = try_powf_f32x4_with::<POW_LOWP, _>(token, v4, exp)
        .unwrap()
        .to_array();
    assert_eq!(pr4.map(f32::to_bits), pl4.map(f32::to_bits));
    for (i, &x) in inputs4.iter().enumerate() {
        assert!((p4[i] - x.powf(exp)).abs() <= 1e-4);
        assert!((pr4[i] - x.powf(exp)).abs() <= 1e-3);
    }
    for (i, &x) in inputs8.iter().enumerate() {
        assert!((p8[i] - x.powf(exp)).abs() <= 1e-4);
        assert!((pr8[i] - x.powf(exp)).abs() <= 1e-3);
    }
}

#[cfg(feature = "w512")]
fn check_token_math_w512<T: F32x16Convert>(token: T) {
    let inputs16: [f32; 16] = core::array::from_fn(|i| 0.25 + (i as f32) * 0.25);
    let v16 = f32x16::from_array_t(token, inputs16);
    let exp = 2.4f32;

    let e16 = try_exp_f32x16_with::<EXP_PORTABLE, _>(token, v16)
        .unwrap()
        .to_array();
    let e2_16 = try_exp2_f32x16_with::<EXP_BACKEND, _>(token, v16)
        .unwrap()
        .to_array();
    let ln16 = try_ln_f32x16_with::<LOG_PORTABLE, _>(token, v16)
        .unwrap()
        .to_array();
    let l2_16 = try_log2_f32x16_with::<LOG_PORTABLE, _>(token, v16)
        .unwrap()
        .to_array();
    let l10_16 = try_log10_f32x16_with::<LOG_PORTABLE, _>(token, v16)
        .unwrap()
        .to_array();
    let l1p_16 = try_log1p_f32x16_with::<LOG_PORTABLE, _>(token, v16)
        .unwrap()
        .to_array();
    let c16 = try_cbrt_f32x16_with::<CBRT_PORTABLE, _>(token, v16)
        .unwrap()
        .to_array();
    let p16 = try_powf_f32x16_with::<POW_PORTABLE_HIGHP, _>(token, v16, exp)
        .unwrap()
        .to_array();
    let pr16 = try_powf_f32x16_runtime(token, v16, exp, PowfOptions::new())
        .unwrap()
        .to_array();

    for (i, &x) in inputs16.iter().enumerate() {
        assert!((e16[i] - x.exp()).abs() <= 1e-5);
        assert!((e2_16[i] - x.exp2()).abs() <= 1e-5);
        assert!((ln16[i] - x.ln()).abs() <= 1e-5);
        assert!((l2_16[i] - x.log2()).abs() <= 1e-5);
        assert!((l10_16[i] - x.log10()).abs() <= 1e-5);
        assert!((l1p_16[i] - x.ln_1p()).abs() <= 1e-5);
        assert!((c16[i] - x.cbrt()).abs() <= 1e-5);
        assert!((p16[i] - x.powf(exp)).abs() <= 1e-4);
        assert!((pr16[i] - x.powf(exp)).abs() <= 1e-3);
    }
}

#[test]
fn all_21_tokens_route_correctly_in_magemath() {
    macro_rules! check_token {
        ($tok:ty) => {{
            let _: fn($tok) = check_token_math::<$tok>;
            #[cfg(feature = "w512")]
            {
                let _: fn($tok) = check_token_math_w512::<$tok>;
            }
            if let Some(t) = <$tok>::summon() {
                check_token_math(t);
                #[cfg(feature = "w512")]
                check_token_math_w512(t);
            }
        }};
    }

    check_token!(archmage::ScalarToken);
    #[cfg(target_arch = "x86_64")]
    {
        check_token!(archmage::X64V1Token);
        check_token!(archmage::X64V2Token);
        check_token!(archmage::X64CryptoToken);
        check_token!(archmage::X64V3Token);
        check_token!(archmage::X64V3CryptoToken);
        check_token!(archmage::X64V3GfniCryptoToken);
        check_token!(archmage::X64V3VnniToken);
        check_token!(archmage::X64V3Vnni2Token);
        check_token!(archmage::X64V4Token);
        check_token!(archmage::X64V4xToken);
        check_token!(archmage::Avx512Fp16Token);
    }
    #[cfg(target_arch = "aarch64")]
    {
        check_token!(archmage::NeonToken);
        check_token!(archmage::NeonAesToken);
        check_token!(archmage::NeonSha3Token);
        check_token!(archmage::NeonCrcToken);
        check_token!(archmage::Arm64V2Token);
        check_token!(archmage::Arm64V3Token);
    }
    #[cfg(target_arch = "wasm32")]
    {
        check_token!(archmage::Wasm128Token);
        check_token!(archmage::Wasm128RelaxedToken);
        check_token!(archmage::Wasm128RelaxedFmaToken);
    }
}

#[cfg(target_arch = "x86_64")]
#[test]
fn subtier_tokens_produce_bit_identical_results_to_canonical_base() {
    let inputs = [0.25f32, 0.5, 1.5, 2.75, 4.0, 9.5, 0.125, 3.25];
    if let (Some(v3), Some(sub)) = (
        archmage::X64V3Token::summon(),
        archmage::X64V3CryptoToken::summon(),
    ) {
        let a = try_exp_f32x8_with::<EXP_PORTABLE, _>(v3, f32x8::from_array_t(v3, inputs))
            .unwrap()
            .to_array();
        let b = try_exp_f32x8_with::<EXP_PORTABLE, _>(sub, f32x8::from_array_t(sub, inputs))
            .unwrap()
            .to_array();
        assert_eq!(a.map(f32::to_bits), b.map(f32::to_bits));

        let pa =
            try_powf_f32x8_with::<POW_PORTABLE_HIGHP, _>(v3, f32x8::from_array_t(v3, inputs), 2.4)
                .unwrap()
                .to_array();
        let pb = try_powf_f32x8_with::<POW_PORTABLE_HIGHP, _>(
            sub,
            f32x8::from_array_t(sub, inputs),
            2.4,
        )
        .unwrap()
        .to_array();
        assert_eq!(pa.map(f32::to_bits), pb.map(f32::to_bits));
    }

    if let (Some(sc), Some(v2)) = (
        archmage::ScalarToken::summon(),
        archmage::X64V2Token::summon(),
    ) {
        let a = try_ln_f32x8_with::<LOG_PORTABLE, _>(sc, f32x8::from_array_t(sc, inputs))
            .unwrap()
            .to_array();
        let b = try_ln_f32x8_with::<LOG_PORTABLE, _>(v2, f32x8::from_array_t(v2, inputs))
            .unwrap()
            .to_array();
        assert_eq!(a.map(f32::to_bits), b.map(f32::to_bits));
    }
}
