//! Actual public std HIGHP output and all-width, all-policy contract checks.
#![forbid(unsafe_code)]
use magetypes::powf::*;
use std::io::{self, BufRead};
fn check<const F: u32, const N: usize>(
    x: f32,
    n: f32,
    opts: PowfOptions,
) -> Result<[f32; N], PowfError> {
    let input = core::array::from_fn(|i| if i % 2 == 0 { x } else { 1. });
    let a = try_powf_array_with::<F, N>(input, n);
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
            let _ = check::<80, 1>(x, n, opts);
            let _ = check::<80, 4>(x, n, opts);
            let _ = check::<80, 8>(x, n, opts);
            let _ = check::<80, 16>(x, n, opts);
        }
        {
            let opts = PowfOptions::STD_HIGHP
                .with_domain(PowfDomain::NonNegativeNormal)
                .with_execution(PowfExecution::Automatic);
            let _ = check::<81, 1>(x, n, opts);
            let _ = check::<81, 4>(x, n, opts);
            let _ = check::<81, 8>(x, n, opts);
            let _ = check::<81, 16>(x, n, opts);
        }
        {
            let opts = PowfOptions::STD_HIGHP
                .with_domain(PowfDomain::Gamma)
                .with_execution(PowfExecution::Automatic);
            let _ = check::<82, 1>(x, n, opts);
            let _ = check::<82, 4>(x, n, opts);
            let _ = check::<82, 8>(x, n, opts);
            let _ = check::<82, 16>(x, n, opts);
        }
        {
            let opts = PowfOptions::STD_HIGHP
                .with_domain(PowfDomain::Full)
                .with_execution(PowfExecution::Automatic);
            let _ = check::<83, 1>(x, n, opts);
            let _ = check::<83, 4>(x, n, opts);
            let _ = check::<83, 8>(x, n, opts);
            let _ = check::<83, 16>(x, n, opts);
        }
        {
            let opts = PowfOptions::STD_HIGHP
                .with_domain(PowfDomain::PositiveNormal)
                .with_execution(PowfExecution::Caller);
            let _ = check::<84, 1>(x, n, opts);
            let _ = check::<84, 4>(x, n, opts);
            let _ = check::<84, 8>(x, n, opts);
            let _ = check::<84, 16>(x, n, opts);
        }
        {
            let opts = PowfOptions::STD_HIGHP
                .with_domain(PowfDomain::NonNegativeNormal)
                .with_execution(PowfExecution::Caller);
            let _ = check::<85, 1>(x, n, opts);
            let _ = check::<85, 4>(x, n, opts);
            let _ = check::<85, 8>(x, n, opts);
            let _ = check::<85, 16>(x, n, opts);
        }
        {
            let opts = PowfOptions::STD_HIGHP
                .with_domain(PowfDomain::Gamma)
                .with_execution(PowfExecution::Caller);
            let _ = check::<86, 1>(x, n, opts);
            let _ = check::<86, 4>(x, n, opts);
            let _ = check::<86, 8>(x, n, opts);
            let _ = check::<86, 16>(x, n, opts);
        }
        {
            let opts = PowfOptions::STD_HIGHP
                .with_domain(PowfDomain::Full)
                .with_execution(PowfExecution::Caller);
            let _ = check::<87, 1>(x, n, opts);
            let _ = check::<87, 4>(x, n, opts);
            let _ = check::<87, 8>(x, n, opts);
            let _ = check::<87, 16>(x, n, opts);
        }
        {
            let opts = PowfOptions::STD_HIGHP
                .with_domain(PowfDomain::PositiveNormal)
                .with_execution(PowfExecution::Automatic);
            if try_powf_runtime(x, n, opts).is_ok() {
                let opts = opts.with_validation(PowfValidation::AssumeValid);
                let _ = check::<112, 1>(x, n, opts);
                let _ = check::<112, 4>(x, n, opts);
                let _ = check::<112, 8>(x, n, opts);
                let _ = check::<112, 16>(x, n, opts);
            }
        }
        {
            let opts = PowfOptions::STD_HIGHP
                .with_domain(PowfDomain::NonNegativeNormal)
                .with_execution(PowfExecution::Automatic);
            if try_powf_runtime(x, n, opts).is_ok() {
                let opts = opts.with_validation(PowfValidation::AssumeValid);
                let _ = check::<113, 1>(x, n, opts);
                let _ = check::<113, 4>(x, n, opts);
                let _ = check::<113, 8>(x, n, opts);
                let _ = check::<113, 16>(x, n, opts);
            }
        }
        {
            let opts = PowfOptions::STD_HIGHP
                .with_domain(PowfDomain::Gamma)
                .with_execution(PowfExecution::Automatic);
            if try_powf_runtime(x, n, opts).is_ok() {
                let opts = opts.with_validation(PowfValidation::AssumeValid);
                let _ = check::<114, 1>(x, n, opts);
                let _ = check::<114, 4>(x, n, opts);
                let _ = check::<114, 8>(x, n, opts);
                let _ = check::<114, 16>(x, n, opts);
            }
        }
        {
            let opts = PowfOptions::STD_HIGHP
                .with_domain(PowfDomain::Full)
                .with_execution(PowfExecution::Automatic);
            if try_powf_runtime(x, n, opts).is_ok() {
                let opts = opts.with_validation(PowfValidation::AssumeValid);
                let _ = check::<115, 1>(x, n, opts);
                let _ = check::<115, 4>(x, n, opts);
                let _ = check::<115, 8>(x, n, opts);
                let _ = check::<115, 16>(x, n, opts);
            }
        }
        {
            let opts = PowfOptions::STD_HIGHP
                .with_domain(PowfDomain::PositiveNormal)
                .with_execution(PowfExecution::Caller);
            if try_powf_runtime(x, n, opts).is_ok() {
                let opts = opts.with_validation(PowfValidation::AssumeValid);
                let _ = check::<116, 1>(x, n, opts);
                let _ = check::<116, 4>(x, n, opts);
                let _ = check::<116, 8>(x, n, opts);
                let _ = check::<116, 16>(x, n, opts);
            }
        }
        {
            let opts = PowfOptions::STD_HIGHP
                .with_domain(PowfDomain::NonNegativeNormal)
                .with_execution(PowfExecution::Caller);
            if try_powf_runtime(x, n, opts).is_ok() {
                let opts = opts.with_validation(PowfValidation::AssumeValid);
                let _ = check::<117, 1>(x, n, opts);
                let _ = check::<117, 4>(x, n, opts);
                let _ = check::<117, 8>(x, n, opts);
                let _ = check::<117, 16>(x, n, opts);
            }
        }
        {
            let opts = PowfOptions::STD_HIGHP
                .with_domain(PowfDomain::Gamma)
                .with_execution(PowfExecution::Caller);
            if try_powf_runtime(x, n, opts).is_ok() {
                let opts = opts.with_validation(PowfValidation::AssumeValid);
                let _ = check::<118, 1>(x, n, opts);
                let _ = check::<118, 4>(x, n, opts);
                let _ = check::<118, 8>(x, n, opts);
                let _ = check::<118, 16>(x, n, opts);
            }
        }
        {
            let opts = PowfOptions::STD_HIGHP
                .with_domain(PowfDomain::Full)
                .with_execution(PowfExecution::Caller);
            if try_powf_runtime(x, n, opts).is_ok() {
                let opts = opts.with_validation(PowfValidation::AssumeValid);
                let _ = check::<119, 1>(x, n, opts);
                let _ = check::<119, 4>(x, n, opts);
                let _ = check::<119, 8>(x, n, opts);
                let _ = check::<119, 16>(x, n, opts);
            }
        }
        let result = try_powf_with::<{ PowfOptions::STD_HIGHP.bits() }>(x, n).unwrap();
        println!("{xb:08x} {nb:08x} {} {:08x}", fields[2], result.to_bits());
    }
}
