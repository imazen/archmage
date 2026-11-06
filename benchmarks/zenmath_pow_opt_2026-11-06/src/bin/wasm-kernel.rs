//! Strict module-load SIMD128; allocations per host invocation amortized by repeats.
#![forbid(unsafe_code)]
use std::hint::black_box;
use zenmath_pow_opt_study::*;
pub extern "C" fn pow_study_kernel(
    domain: i32,
    distribution: i32,
    uniform: i32,
    case: i32,
    n: i32,
    repeats: i32,
) -> f64 {
    assert!((1..=65536).contains(&n));
    assert!((1..=256).contains(&repeats));
    assert!(domain >= 0 && distribution >= 0 && case >= 0);
    let d = DOMAINS[domain as usize];
    let distribution = distributions(d)[distribution as usize];
    let cases = domain_cases(d, uniform != 0);
    let c = &cases[case as usize];
    let (x, y) = distributed_inputs(d, distribution, n as usize);
    let mut out = vec![0.0; n as usize];
    for _ in 0..repeats {
        (c.run)(black_box(&x), black_box(&y), black_box(&mut out));
        black_box(&out);
    }
    out[0] as f64
}
fn main() {
    verify();
    for (domain, d) in DOMAINS.into_iter().enumerate() {
        for (distribution, dist) in distributions(d).iter().enumerate() {
            for uniform in [false, true] {
                if uniform && !matches!(d, Domain::Decode | Domain::Encode) {
                    continue;
                }
                let group = if *dist == Distribution::Uniform {
                    d.name().to_owned()
                } else {
                    format!("{}@{}", d.name(), dist.name())
                };
                for (case, c) in domain_cases(d, uniform).into_iter().enumerate() {
                    println!(
                        "C\t{domain}\t{distribution}\t{}\t{case}\t{group}\t{}",
                        i32::from(uniform),
                        c.name
                    );
                }
            }
        }
    }
    black_box(pow_study_kernel as extern "C" fn(i32, i32, i32, i32, i32, i32) -> f64);
}
