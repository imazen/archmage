//! Numerical-study callbacks; no production API, dependencies or export attrs.
#![forbid(unsafe_code)]
use archmage::SimdToken;
use magetypes::powf::{PowfDomain, PowfErrorKind, PowfExecution, PowfOptions};
#[allow(dead_code)]
pub mod rounding {
    include!(concat!(env!("OUT_DIR"), "/rounding_kernels.rs"));
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
        Err(error) => (match error.kind {
            PowfErrorKind::InvalidExponent => 1_u64,
            PowfErrorKind::InvalidBase => 2,
            PowfErrorKind::OutsideApproximationRange => 3,
            PowfErrorKind::UnsupportedWidth => 4,
            _ => panic!("unknown rejection stage"),
        }) << 32,
    }
}

/// Controlled benchmark-only callback: midp separate/fused, lowp separate/fused.
#[inline(never)]
pub extern "C" fn rounding_evaluate(
    base: f32, n: f32, method: u32, domain: u32, native: u32,
) -> u64 {
    assert!(method < 4 && domain < 3 && native < 2);
    if native == 0 {
        return rounding_value_scalar(archmage::ScalarToken, base, n, method, domain);
    }
    #[cfg(target_arch = "x86_64")]
    return rounding_value_v3(archmage::X64V3Token::summon().expect("required V3"), base, n, method, domain);
    #[cfg(target_arch = "aarch64")]
    return rounding_value_neon(archmage::NeonToken::summon().expect("required NEON"), base, n, method, domain);
    #[cfg(target_arch = "wasm32")]
    return rounding_value_wasm128(archmage::Wasm128Token::summon().expect("required WASM SIMD"), base, n, method, domain);
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64", target_arch = "wasm32")))]
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
