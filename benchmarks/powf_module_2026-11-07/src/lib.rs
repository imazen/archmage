//! Numerical-study callbacks; no production API, dependencies or export attrs.
#![forbid(unsafe_code)]
use archmage::SimdToken;
use magetypes::powf::{PowfDomain, PowfErrorKind, PowfExecution, PowfOptions};
pub const METHOD_NAMES:[&str;8]=["canonical","pow_lowp","pow_lowp_unchecked","pow_midp","pow_midp_unchecked","pow_midp_precise","std_f32_powf","std_promoted_f64_powf"];

#[archmage::magetypes(v3,neon,wasm128,scalar)]
fn evaluate(token:Token,base:f32,n:f32,choice:u32,domain:u32)->u64 {
    use magetypes::simd::generic::f32x4;
    let v=f32x4::splat_t(token,base);
    let value=match choice {
        0=>{
            let result=match domain {
                0=>v.try_powf_with::<{PowfOptions::new().with_execution(PowfExecution::Caller).bits()}>(n),
                1=>v.try_powf_with::<{PowfOptions::new().with_domain(PowfDomain::NonNegativeNormal).with_execution(PowfExecution::Caller).bits()}>(n),
                2=>v.try_powf_with::<{PowfOptions::new().with_domain(PowfDomain::Gamma).with_execution(PowfExecution::Caller).bits()}>(n),
                _=>panic!("unknown numerical domain"),
            };
            match result {
                Ok(value)=>value.to_array()[0],
                Err(error)=>return (match error.kind{PowfErrorKind::InvalidExponent=>1_u64,PowfErrorKind::InvalidBase=>2,PowfErrorKind::OutsideApproximationRange=>3,PowfErrorKind::UnsupportedWidth=>4})<<32,
            }
        },
        1=>v.pow_lowp(n).to_array()[0],
        2=>v.pow_lowp_unchecked(n).to_array()[0],
        3=>v.pow_midp(n).to_array()[0],
        4=>v.pow_midp_unchecked(n).to_array()[0],
        5=>v.pow_midp_precise(n).to_array()[0],
        6=>base.powf(n),
        7=>f64::from(base).powf(f64::from(n))as f32,
        _=>panic!("unknown numerical candidate"),
    };
    u64::from(value.to_bits())
}

/// Upper word is zero for an actual result, or the canonical rejection stage.
/// C declares this compiler-assigned symbol by its exact name after `nm`.
#[inline(never)]
pub extern "C" fn replacement_evaluate(base:f32,n:f32,choice:u32,domain:u32,native:u32)->u64 {
    assert!(domain<=2 && native<=1 && choice<8,"invalid study selector");
    if native==0 {return evaluate_scalar(archmage::ScalarToken,base,n,choice,domain);}
    #[cfg(target_arch="x86_64")]
    {return evaluate_v3(archmage::X64V3Token::summon().expect("required native V3"),base,n,choice,domain);}
    #[cfg(target_arch="aarch64")]
    {return evaluate_neon(archmage::NeonToken::summon().expect("required native NEON"),base,n,choice,domain);}
    #[cfg(target_arch="wasm32")]
    {return evaluate_wasm128(archmage::Wasm128Token::summon().expect("required native WASM SIMD"),base,n,choice,domain);}
    #[allow(unreachable_code)]
    {panic!("no native study backend on this target")}
}

#[used]
static RETAIN_CALLBACK:extern "C" fn(f32,f32,u32,u32,u32)->u64=replacement_evaluate;
