//! Actual public std HIGHP output and all-width, all-policy contract checks.
#![forbid(unsafe_code)]
use magetypes::powf::*;
use std::io::{self, BufRead};
fn same<const N:usize>(a:Result<[f32;N],PowfError>,b:Result<[f32;N],PowfError>)->Result<[f32;N],PowfError>{
 match(a,b){
  (Ok(a),Ok(b))=>{for i in 0..N{assert!(a[i].to_bits()==b[i].to_bits() || a[i].is_nan()&&b[i].is_nan(),"HIGHP lane {i}");}Ok(a)}
  (Err(a),Err(b))=>{assert_eq!(a,b);Err(a)}
  _=>panic!("HIGHP vector/array or const/runtime admission mismatch"),
 }
}
macro_rules! kernel{
 ($name:ident,$ty:ident,$width:literal)=>{
  #[archmage::magetypes(v3,neon,wasm128,scalar)]
  fn $name<const FLAGS:u32>(token:Token,input:[f32;$width],n:f32,opts:PowfOptions)->Result<[f32;$width],PowfError>{
   let v=magetypes::simd::generic::$ty::from_array_t(token,input);
   same(v.try_powf_with::<FLAGS>(n).map(|v|v.to_array()),v.try_powf_runtime(n,opts).map(|v|v.to_array()))
  }
 };
}
kernel!(highp4,f32x4,4);kernel!(highp8,f32x8,8);kernel!(highp16,f32x16,16);
fn vector<const FLAGS:u32,const N:usize>(context:&str,input:[f32;N],n:f32,opts:PowfOptions)->Result<[f32;N],PowfError>{
 #[cfg(any(target_arch="x86_64",target_arch="aarch64",target_arch="wasm32"))]
 use archmage::SimdToken;
 if N==1{
  let value=magetypes::simd::scalar::f32x1(input[0]).try_powf_with::<FLAGS>(n)?;
  let mut output=[0.;N];output[0]=value.0;return Ok(output);
 }
 macro_rules! go{($width:literal,$scalar:ident,$v3:ident,$neon:ident,$wasm:ident)=>{if N==$width{
  let mut fixed=[0.;$width];fixed.copy_from_slice(&input);
  let output=match context{
   "scalar"=>$scalar::<FLAGS>(archmage::ScalarToken,fixed,n,opts),
   #[cfg(target_arch="x86_64")]"native"=>$v3::<FLAGS>(archmage::X64V3Token::summon().expect("required V3"),fixed,n,opts),
   #[cfg(target_arch="aarch64")]"native"=>$neon::<FLAGS>(archmage::NeonToken::summon().expect("required NEON"),fixed,n,opts),
   #[cfg(target_arch="wasm32")]"native"=>$wasm::<FLAGS>(archmage::Wasm128Token::summon().expect("required SIMD128"),fixed,n,opts),
   _=>panic!("requested native context unsupported"),
  }?;
  let mut result=[0.;N];result.copy_from_slice(&output);return Ok(result);
 }}}
 go!(4,highp4_scalar,highp4_v3,highp4_neon,highp4_wasm128);
 go!(8,highp8_scalar,highp8_v3,highp8_neon,highp8_wasm128);
 go!(16,highp16_scalar,highp16_v3,highp16_neon,highp16_wasm128);
 unreachable!("supported test widths only")
}
fn check<const F: u32, const N: usize>(
    x: f32,
    n: f32,
    opts: PowfOptions,
    context: &str,
) -> Result<[f32; N], PowfError> {
    let input = std::hint::black_box(core::array::from_fn(|i| if i % 2 == 0 { x } else { 1. }));
    let native = vector::<F,N>(context,input,n,opts);
    let a = same(try_powf_array_with::<F, N>(input, n),native);
    let b = try_powf_array_runtime(input, n, opts);
    match (a, b) {
        (Ok(a), Ok(b)) => {
            for i in 0..N {
                assert!(a[i].to_bits() == b[i].to_bits() || a[i].is_nan() && b[i].is_nan());
                let expected = input[i].powf(n);
                assert!(a[i].to_bits() == expected.to_bits() || a[i].is_nan() && expected.is_nan());
            }
            Ok(a)
        }
        (Err(a), Err(b)) => {
            assert_eq!(a, b);
            Err(a)
        }
        _ => panic!("const/runtime HIGHP admission mismatch"),
    }
}
fn main() {
    let context=std::env::args().nth(1).expect("explicit scalar/native context required");
    assert!(matches!(context.as_str(),"scalar"|"native"));
    for line in io::stdin().lock().lines() {
        let line = line.unwrap();
        let fields: Vec<_> = line.split_whitespace().collect();
        assert_eq!(fields.len(), 3);
        let xb = u32::from_str_radix(fields[0], 16).unwrap();
        let nb = u32::from_str_radix(fields[1], 16).unwrap();
        let x = f32::from_bits(xb);
        let n = f32::from_bits(nb);
        {
            let opts = PowfOptions::STD_HIGHP
                .with_domain(PowfDomain::PositiveNormal)
                .with_execution(PowfExecution::Automatic);
            let _ = check::<80, 1>(x, n, opts, &context);
            let _ = check::<80, 4>(x, n, opts, &context);
            let _ = check::<80, 8>(x, n, opts, &context);
            let _ = check::<80, 16>(x, n, opts, &context);
        }
        {
            let opts = PowfOptions::STD_HIGHP
                .with_domain(PowfDomain::NonNegativeNormal)
                .with_execution(PowfExecution::Automatic);
            let _ = check::<81, 1>(x, n, opts, &context);
            let _ = check::<81, 4>(x, n, opts, &context);
            let _ = check::<81, 8>(x, n, opts, &context);
            let _ = check::<81, 16>(x, n, opts, &context);
        }
        {
            let opts = PowfOptions::STD_HIGHP
                .with_domain(PowfDomain::Gamma)
                .with_execution(PowfExecution::Automatic);
            let _ = check::<82, 1>(x, n, opts, &context);
            let _ = check::<82, 4>(x, n, opts, &context);
            let _ = check::<82, 8>(x, n, opts, &context);
            let _ = check::<82, 16>(x, n, opts, &context);
        }
        {
            let opts = PowfOptions::STD_HIGHP
                .with_domain(PowfDomain::Full)
                .with_execution(PowfExecution::Automatic);
            let _ = check::<83, 1>(x, n, opts, &context);
            let _ = check::<83, 4>(x, n, opts, &context);
            let _ = check::<83, 8>(x, n, opts, &context);
            let _ = check::<83, 16>(x, n, opts, &context);
        }
        {
            let opts = PowfOptions::STD_HIGHP
                .with_domain(PowfDomain::PositiveNormal)
                .with_execution(PowfExecution::Caller);
            let _ = check::<84, 1>(x, n, opts, &context);
            let _ = check::<84, 4>(x, n, opts, &context);
            let _ = check::<84, 8>(x, n, opts, &context);
            let _ = check::<84, 16>(x, n, opts, &context);
        }
        {
            let opts = PowfOptions::STD_HIGHP
                .with_domain(PowfDomain::NonNegativeNormal)
                .with_execution(PowfExecution::Caller);
            let _ = check::<85, 1>(x, n, opts, &context);
            let _ = check::<85, 4>(x, n, opts, &context);
            let _ = check::<85, 8>(x, n, opts, &context);
            let _ = check::<85, 16>(x, n, opts, &context);
        }
        {
            let opts = PowfOptions::STD_HIGHP
                .with_domain(PowfDomain::Gamma)
                .with_execution(PowfExecution::Caller);
            let _ = check::<86, 1>(x, n, opts, &context);
            let _ = check::<86, 4>(x, n, opts, &context);
            let _ = check::<86, 8>(x, n, opts, &context);
            let _ = check::<86, 16>(x, n, opts, &context);
        }
        {
            let opts = PowfOptions::STD_HIGHP
                .with_domain(PowfDomain::Full)
                .with_execution(PowfExecution::Caller);
            let _ = check::<87, 1>(x, n, opts, &context);
            let _ = check::<87, 4>(x, n, opts, &context);
            let _ = check::<87, 8>(x, n, opts, &context);
            let _ = check::<87, 16>(x, n, opts, &context);
        }
        {
            let opts = PowfOptions::STD_HIGHP
                .with_domain(PowfDomain::PositiveNormal)
                .with_execution(PowfExecution::Automatic);
            if try_powf_runtime(x, n, opts).is_ok() {
                let opts = opts.with_validation(PowfValidation::AssumeValid);
                let _ = check::<112, 1>(x, n, opts, &context);
                let _ = check::<112, 4>(x, n, opts, &context);
                let _ = check::<112, 8>(x, n, opts, &context);
                let _ = check::<112, 16>(x, n, opts, &context);
            }
        }
        {
            let opts = PowfOptions::STD_HIGHP
                .with_domain(PowfDomain::NonNegativeNormal)
                .with_execution(PowfExecution::Automatic);
            if try_powf_runtime(x, n, opts).is_ok() {
                let opts = opts.with_validation(PowfValidation::AssumeValid);
                let _ = check::<113, 1>(x, n, opts, &context);
                let _ = check::<113, 4>(x, n, opts, &context);
                let _ = check::<113, 8>(x, n, opts, &context);
                let _ = check::<113, 16>(x, n, opts, &context);
            }
        }
        {
            let opts = PowfOptions::STD_HIGHP
                .with_domain(PowfDomain::Gamma)
                .with_execution(PowfExecution::Automatic);
            if try_powf_runtime(x, n, opts).is_ok() {
                let opts = opts.with_validation(PowfValidation::AssumeValid);
                let _ = check::<114, 1>(x, n, opts, &context);
                let _ = check::<114, 4>(x, n, opts, &context);
                let _ = check::<114, 8>(x, n, opts, &context);
                let _ = check::<114, 16>(x, n, opts, &context);
            }
        }
        {
            let opts = PowfOptions::STD_HIGHP
                .with_domain(PowfDomain::Full)
                .with_execution(PowfExecution::Automatic);
            if try_powf_runtime(x, n, opts).is_ok() {
                let opts = opts.with_validation(PowfValidation::AssumeValid);
                let _ = check::<115, 1>(x, n, opts, &context);
                let _ = check::<115, 4>(x, n, opts, &context);
                let _ = check::<115, 8>(x, n, opts, &context);
                let _ = check::<115, 16>(x, n, opts, &context);
            }
        }
        {
            let opts = PowfOptions::STD_HIGHP
                .with_domain(PowfDomain::PositiveNormal)
                .with_execution(PowfExecution::Caller);
            if try_powf_runtime(x, n, opts).is_ok() {
                let opts = opts.with_validation(PowfValidation::AssumeValid);
                let _ = check::<116, 1>(x, n, opts, &context);
                let _ = check::<116, 4>(x, n, opts, &context);
                let _ = check::<116, 8>(x, n, opts, &context);
                let _ = check::<116, 16>(x, n, opts, &context);
            }
        }
        {
            let opts = PowfOptions::STD_HIGHP
                .with_domain(PowfDomain::NonNegativeNormal)
                .with_execution(PowfExecution::Caller);
            if try_powf_runtime(x, n, opts).is_ok() {
                let opts = opts.with_validation(PowfValidation::AssumeValid);
                let _ = check::<117, 1>(x, n, opts, &context);
                let _ = check::<117, 4>(x, n, opts, &context);
                let _ = check::<117, 8>(x, n, opts, &context);
                let _ = check::<117, 16>(x, n, opts, &context);
            }
        }
        {
            let opts = PowfOptions::STD_HIGHP
                .with_domain(PowfDomain::Gamma)
                .with_execution(PowfExecution::Caller);
            if try_powf_runtime(x, n, opts).is_ok() {
                let opts = opts.with_validation(PowfValidation::AssumeValid);
                let _ = check::<118, 1>(x, n, opts, &context);
                let _ = check::<118, 4>(x, n, opts, &context);
                let _ = check::<118, 8>(x, n, opts, &context);
                let _ = check::<118, 16>(x, n, opts, &context);
            }
        }
        {
            let opts = PowfOptions::STD_HIGHP
                .with_domain(PowfDomain::Full)
                .with_execution(PowfExecution::Caller);
            if try_powf_runtime(x, n, opts).is_ok() {
                let opts = opts.with_validation(PowfValidation::AssumeValid);
                let _ = check::<119, 1>(x, n, opts, &context);
                let _ = check::<119, 4>(x, n, opts, &context);
                let _ = check::<119, 8>(x, n, opts, &context);
                let _ = check::<119, 16>(x, n, opts, &context);
            }
        }
        let result = try_powf_with::<{ PowfOptions::STD_HIGHP.bits() }>(x, n).unwrap();
        println!("{xb:08x} {nb:08x} {} {:08x}", fields[2], result.to_bits());
    }
}
