//! Numerical-study callbacks; no production API, dependencies or export attrs.
#![forbid(unsafe_code)]
use archmage::SimdToken;
use magetypes::powf::{PowfDomain, PowfErrorKind, PowfExecution, PowfOptions};
#[allow(dead_code)]
pub mod rounding {
    include!(concat!(env!("OUT_DIR"), "/rounding_kernels.rs"));
}

#[cfg(test)]
mod rounding_tests {
    use super::*;
    macro_rules! width {
        ($test:ident,$kernel:ident,$vector:ident,$n:literal) => {
            #[archmage::magetypes(v3, neon, wasm128, scalar)]
            fn $test(token: Token) {
                use magetypes::simd::generic::$vector as V;
                macro_rules! modes {
                    ($lowp:literal,$fused:literal) => {{
                        let mut x = [0.75; $n];
                        x[1] = f32::NAN;
                        x[$n - 1] = -0.0;
                        let result = rounding::$kernel::<Token, $lowp, $fused, true, 2>(
                            token,
                            V::from_array_t(token, x),
                            2.4,
                        )
                        .unwrap_err();
                        assert_eq!(result.kind, PowfErrorKind::InvalidBase);
                        assert_eq!(result.lane_mask, (1 << 1) | (1 << ($n - 1)));
                        let result = rounding::$kernel::<Token, $lowp, $fused, true, 2>(
                            token,
                            V::from_array_t(token, x),
                            f32::from_bits(1),
                        )
                        .unwrap_err();
                        assert_eq!(result.kind, PowfErrorKind::InvalidExponent);
                        assert_eq!(result.lane_mask, ((1u32 << $n) - 1) as u16);
                        x[1] = 0.;
                        x[$n - 1] = 65536.;
                        for exponent in [0., -0., 0.5, 2.4, 4.] {
                            let v = V::from_array_t(token, x);
                            let a = rounding::$kernel::<Token, $lowp, $fused, true, 2>(
                                token, v, exponent,
                            )
                            .unwrap()
                            .to_array();
                            let b = rounding::$kernel::<Token, $lowp, $fused, false, 2>(
                                token, v, exponent,
                            )
                            .unwrap()
                            .to_array();
                            assert_eq!(a.map(f32::to_bits), b.map(f32::to_bits));
                            assert_eq!(
                                a[1].to_bits(),
                                if exponent == 0. { 1f32.to_bits() } else { 0 }
                            );
                            assert!(a.iter().all(|v| v.is_normal() || *v == 0.));
                        }
                        x[0] = f32::from_bits(1);
                        let error = rounding::$kernel::<Token, $lowp, $fused, true, 2>(
                            token,
                            V::from_array_t(token, x),
                            0.,
                        )
                        .unwrap_err();
                        assert_eq!(error.kind, PowfErrorKind::InvalidBase);
                        assert_eq!(error.lane_mask, 1);
                    }};
                }
                modes!(false, false);
                modes!(false, true);
                modes!(true, false);
                modes!(true, true);
            }
        };
    }
    width!(check4, kernel4, f32x4, 4);
    width!(check8, kernel8, f32x8, 8);
    width!(check16, kernel16, f32x16, 16);
    #[test]
    fn scalar_mixed_lanes() {
        check4_scalar(archmage::ScalarToken);
        check8_scalar(archmage::ScalarToken);
        check16_scalar(archmage::ScalarToken);
    }
    #[test]
    fn explicit_native_mixed_lanes() {
        #[cfg(target_arch = "x86_64")]
        {
            let token = archmage::X64V3Token::summon().expect("study requires V3");
            check4_v3(token);
            check8_v3(token);
            check16_v3(token);
        }
        #[cfg(target_arch = "aarch64")]
        {
            let token = archmage::NeonToken::summon().expect("study requires NEON");
            check4_neon(token);
            check8_neon(token);
            check16_neon(token);
        }
        #[cfg(target_arch = "wasm32")]
        {
            let token = archmage::Wasm128Token::summon().expect("study requires SIMD128");
            check4_wasm128(token);
            check8_wasm128(token);
            check16_wasm128(token);
        }
    }
}

#[archmage::magetypes(v3, neon, wasm128, scalar)]
fn rounding_value(token: Token, base: f32, n: f32, method: u32, domain: u32) -> u64 {
    use magetypes::simd::generic::f32x4;
    let input = f32x4::splat_t(token, base);
    macro_rules! domain {
        ($lowp:literal, $fused:literal) => {
            match domain {
                0 => rounding::kernel4::<Token, $lowp, $fused, true, 0>(token, input, n),
                1 => rounding::kernel4::<Token, $lowp, $fused, true, 1>(token, input, n),
                2 => rounding::kernel4::<Token, $lowp, $fused, true, 2>(token, input, n),
                _ => panic!("unknown study domain"),
            }
        };
    }
    let result = match method {
        0 => domain!(false, false),
        1 => domain!(false, true),
        2 => domain!(true, false),
        3 => domain!(true, true),
        _ => panic!("unknown controlled rounding method"),
    };
    match result {
        Ok(value) => u64::from(value.to_array()[0].to_bits()),
        Err(error) => {
            (match error.kind {
                PowfErrorKind::InvalidExponent => 1_u64,
                PowfErrorKind::InvalidBase => 2,
                PowfErrorKind::OutsideApproximationRange => 3,
                PowfErrorKind::UnsupportedWidth => 4,
                _ => panic!("unknown rejection stage"),
            }) << 32
        }
    }
}

/// Controlled benchmark-only callback: midp separate/fused, lowp separate/fused.
#[inline(never)]
pub extern "C" fn rounding_evaluate(
    base: f32,
    n: f32,
    method: u32,
    domain: u32,
    native: u32,
) -> u64 {
    assert!(method < 4 && domain < 3 && native < 2);
    if native == 0 {
        return rounding_value_scalar(archmage::ScalarToken, base, n, method, domain);
    }
    #[cfg(target_arch = "x86_64")]
    return rounding_value_v3(
        archmage::X64V3Token::summon().expect("required V3"),
        base,
        n,
        method,
        domain,
    );
    #[cfg(target_arch = "aarch64")]
    return rounding_value_neon(
        archmage::NeonToken::summon().expect("required NEON"),
        base,
        n,
        method,
        domain,
    );
    #[cfg(target_arch = "wasm32")]
    return rounding_value_wasm128(
        archmage::Wasm128Token::summon().expect("required WASM SIMD"),
        base,
        n,
        method,
        domain,
    );
    #[cfg(not(any(
        target_arch = "x86_64",
        target_arch = "aarch64",
        target_arch = "wasm32"
    )))]
    panic!("no native study context on this target");
}
pub const METHOD_NAMES: [&str; 8] = [
    "canonical",
    "pow_lowp",
    "pow_lowp_unchecked",
    "pow_midp",
    "pow_midp_unchecked",
    "pow_midp_precise",
    "std_f32_powf",
    "std_promoted_f64_powf",
];

#[archmage::magetypes(v3, neon, wasm128, scalar)]
fn evaluate(token: Token, base: f32, n: f32, choice: u32, domain: u32) -> u64 {
    use magetypes::simd::generic::f32x4;
    let v = f32x4::splat_t(token, base);
    let value = match choice {
        0 => {
            let result = match domain {
                0 => v.try_powf_with::<{
                    PowfOptions::new()
                        .with_execution(PowfExecution::Caller)
                        .bits()
                }>(n),
                1 => v.try_powf_with::<{
                    PowfOptions::new()
                        .with_domain(PowfDomain::NonNegativeNormal)
                        .with_execution(PowfExecution::Caller)
                        .bits()
                }>(n),
                2 => v.try_powf_with::<{
                    PowfOptions::new()
                        .with_domain(PowfDomain::Gamma)
                        .with_execution(PowfExecution::Caller)
                        .bits()
                }>(n),
                _ => panic!("unknown numerical domain"),
            };
            match result {
                Ok(value) => value.to_array()[0],
                Err(error) => {
                    return (match error.kind {
                        PowfErrorKind::InvalidExponent => 1_u64,
                        PowfErrorKind::InvalidBase => 2,
                        PowfErrorKind::OutsideApproximationRange => 3,
                        PowfErrorKind::UnsupportedWidth => 4,
                        _ => panic!("unknown rejection stage: {:?}", error.kind),
                    }) << 32;
                }
            }
        }
        1 => v.pow_lowp(n).to_array()[0],
        2 => v.pow_lowp_unchecked(n).to_array()[0],
        3 => v.pow_midp(n).to_array()[0],
        4 => v.pow_midp_unchecked(n).to_array()[0],
        5 => v.pow_midp_precise(n).to_array()[0],
        6 => base.powf(n),
        7 => f64::from(base).powf(f64::from(n)) as f32,
        _ => panic!("unknown numerical candidate"),
    };
    u64::from(value.to_bits())
}

/// Upper word is zero for an actual result, or the canonical rejection stage.
/// C declares this compiler-assigned symbol by its exact name after `nm`.
#[inline(never)]
pub extern "C" fn replacement_evaluate(
    base: f32,
    n: f32,
    choice: u32,
    domain: u32,
    native: u32,
) -> u64 {
    assert!(
        domain <= 2 && native <= 1 && choice < 8,
        "invalid study selector"
    );
    if native == 0 {
        return evaluate_scalar(archmage::ScalarToken, base, n, choice, domain);
    }
    evaluate_native(base, n, choice, domain)
}
#[cfg(target_arch = "x86_64")]
fn evaluate_native(base: f32, n: f32, choice: u32, domain: u32) -> u64 {
    evaluate_v3(
        archmage::X64V3Token::summon().expect("required native V3"),
        base,
        n,
        choice,
        domain,
    )
}
#[cfg(target_arch = "aarch64")]
fn evaluate_native(base: f32, n: f32, choice: u32, domain: u32) -> u64 {
    evaluate_neon(
        archmage::NeonToken::summon().expect("required native NEON"),
        base,
        n,
        choice,
        domain,
    )
}
#[cfg(target_arch = "wasm32")]
fn evaluate_native(base: f32, n: f32, choice: u32, domain: u32) -> u64 {
    evaluate_wasm128(
        archmage::Wasm128Token::summon().expect("required native WASM SIMD"),
        base,
        n,
        choice,
        domain,
    )
}
#[cfg(not(any(
    target_arch = "x86_64",
    target_arch = "aarch64",
    target_arch = "wasm32"
)))]
fn evaluate_native(_base: f32, _n: f32, _choice: u32, _domain: u32) -> u64 {
    panic!("no native study backend on this target")
}

#[used]
static RETAIN_CALLBACK: extern "C" fn(f32, f32, u32, u32, u32) -> u64 = replacement_evaluate;
