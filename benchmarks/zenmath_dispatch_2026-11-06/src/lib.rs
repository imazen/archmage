#![forbid(unsafe_code)]
use archmage::prelude::*;
pub type Kernel<T> = Box<dyn Fn(&[T], &[T], &[T], &mut [T]) + Send + Sync>;
pub struct Case<T> {
    pub name: &'static str,
    pub run: Kernel<T>,
}
pub mod generated;
pub use generated::{cases32, cases64};
pub fn inventory() {
    let tiers = [
        ("scalar", ScalarToken::summon().is_some()),
        ("v1", X64V1Token::summon().is_some()),
        ("v2", X64V2Token::summon().is_some()),
        ("v3", X64V3Token::summon().is_some()),
        ("neon", NeonToken::summon().is_some()),
        ("wasm128", Wasm128Token::summon().is_some()),
    ];
    for (n, ok) in tiers {
        println!("TIER {n} supported={ok}");
    }
    #[cfg(feature = "avx512")]
    println!("TIER v4x supported={}", X64V4xToken::summon().is_some());
    #[cfg(not(feature = "avx512"))]
    println!("TIER v4x compiled=false supported=false");
    if let Ok(required) = std::env::var("ZENMATH_REQUIRE_TIERS") {
        for name in required.split(',') {
            if name == "v4x" {
                #[cfg(feature = "avx512")]
                assert!(X64V4xToken::summon().is_some());
                #[cfg(not(feature = "avx512"))]
                panic!("v4x feature not compiled");
            } else {
                assert!(
                    tiers.iter().any(|(n, ok)| *n == name && *ok),
                    "required tier unavailable: {name}"
                );
            }
        }
    }
}
pub fn inputs32(op: &str, n: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let x = (0..n)
        .map(|i| {
            let v = ((i * 719) % 1024) as f32 / 1024.;
            match op {
                "ln" | "log2" | "pow" => 0.1234567 + 15.654321 * v,
                "pow_rescue" => [
                    0.,
                    -0.,
                    f32::INFINITY,
                    f32::NEG_INFINITY,
                    f32::NAN,
                    f32::from_bits(1),
                    -2.,
                    -3.,
                ][i % 8],
                "sqrt" => 0.125 + 16. * v,
                _ => -7.812345 + 15.937654 * v,
            }
        })
        .collect();
    let y = (0..n)
        .map(|i| {
            if op.starts_with("pow") {
                2.4
            } else {
                f32::from_bits(0x3f000001 | ((i as u32).wrapping_mul(0x9e3779b9) & 0x007fffff))
            }
        })
        .collect();
    let z = (0..n)
        .map(|i| f32::from_bits(0x3e000001 | ((i as u32).wrapping_mul(0x85ebca6b) & 0x007fffff)))
        .collect();
    (x, y, z)
}
pub fn inputs64(op: &str, n: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let x = (0..n)
        .map(|i| {
            let v = ((i * 719) % 1024) as f64 / 1024.;
            if op == "sqrt" {
                0.123456789012345 + 16.0987654321 * v
            } else {
                -7.81234567890123 + 15.9376543210987 * v
            }
        })
        .collect();
    let y = (0..n)
        .map(|i| {
            f64::from_bits(
                0x3fe0000000000001
                    | ((i as u64).wrapping_mul(0x9e3779b97f4a7c15) & 0x000fffffffffffff),
            )
        })
        .collect();
    let z = (0..n)
        .map(|i| {
            f64::from_bits(
                0x3fc0000000000001
                    | ((i as u64).wrapping_mul(0x85ebca6b9e3779b9) & 0x000fffffffffffff),
            )
        })
        .collect();
    (x, y, z)
}
pub fn verify() {
    inventory();
    for op in OPS32.iter().copied() {
        for n in [1, 7, 16, 1024] {
            let (x, y, z) = inputs32(op, n);
            let mut want = vec![0.; n];
            let mut cases = cases32(op, false);
            (cases.remove(0).run)(&x, &y, &z, &mut want);
            for c in cases {
                if c.name.starts_with("std") {
                    continue;
                }
                let mut got = vec![0.; n];
                (c.run)(&x, &y, &z, &mut got);
                for i in 0..n {
                    assert!(
                        got[i].is_nan() && want[i].is_nan()
                            || got[i].to_bits() == want[i].to_bits(),
                        "{op}/{} n={n} i={i} input={:08x} got={:08x} want={:08x}",
                        c.name,
                        x[i].to_bits(),
                        got[i].to_bits(),
                        want[i].to_bits()
                    );
                }
            }
        }
    }
    for op in OPS64.iter().copied() {
        let (x, y, z) = inputs64(op, 1024);
        let mut want = vec![0.; 1024];
        let mut cases = cases64(op, false);
        (cases.remove(0).run)(&x, &y, &z, &mut want);
        for c in cases {
            if c.name.starts_with("std") {
                continue;
            }
            let mut got = vec![0.; 1024];
            (c.run)(&x, &y, &z, &mut got);
            for i in 0..1024 {
                assert!(
                    got[i].is_nan() && want[i].is_nan() || got[i].to_bits() == want[i].to_bits(),
                    "{op}/{} input={:016x}",
                    c.name,
                    x[i].to_bits()
                );
            }
        }
    }
    for op in OPS32.iter().copied().filter(|op| *op != "pow_rescue") {
        let (x, y, z) = inputs32(op, 1024);
        let mut want = vec![0.; 1024];
        let mut cases = cases32(op, true);
        (cases.remove(0).run)(&x, &y, &z, &mut want);
        for c in cases {
            if c.name.starts_with("std") {
                continue;
            }
            let mut got = vec![0.; 1024];
            (c.run)(&x, &y, &z, &mut got);
            assert_eq!(
                got[0].to_bits(),
                want[0].to_bits(),
                "feedback {op}/{}",
                c.name
            );
        }
    }
    for op in OPS64.iter().copied() {
        let (x, y, z) = inputs64(op, 1024);
        let mut want = vec![0.; 1024];
        let mut cases = cases64(op, true);
        (cases.remove(0).run)(&x, &y, &z, &mut want);
        for c in cases {
            if c.name.starts_with("std") {
                continue;
            }
            let mut got = vec![0.; 1024];
            (c.run)(&x, &y, &z, &mut got);
            assert_eq!(
                got[0].to_bits(),
                want[0].to_bits(),
                "feedback {op}/{}",
                c.name
            );
        }
    }
    println!(
        "PASS exact stream/feedback benchmark adapter checks; std is timing control, not oracle"
    );
}
#[test]
fn adapters() {
    verify();
}

pub const OPS32: &[&str] = &[
    "sqrt",
    "floor",
    "ceil",
    "trunc",
    "roundeven",
    "round",
    "fma",
    "exp",
    "ln",
    "log2",
    "pow",
    "pow_rescue",
];
pub const OPS64: &[&str] = &[
    "sqrt",
    "floor",
    "ceil",
    "trunc",
    "roundeven",
    "round",
    "fma",
];
