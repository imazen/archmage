//! Strict WASM module-load selection, no runtime engine feature detection.
#![forbid(unsafe_code)]
use std::hint::black_box;
use zenmath_dispatch_study::*;
pub extern "C" fn dispatch_kernel(
    group: i32,
    feedback: i32,
    case: i32,
    n: i32,
    repeats: i32,
) -> f64 {
    assert!((1..=65536).contains(&n));
    assert!((1..=256).contains(&repeats));
    assert!(case >= 0 && group >= 0);
    let feedback = feedback != 0;
    if (group as usize) < OPS32.len() {
        let op = OPS32[group as usize];
        let cs = cases32(op, feedback);
        let c = &cs[case as usize];
        let (x, y, z) = inputs32(op, n as usize);
        let mut out = vec![0.; n as usize];
        for _ in 0..repeats {
            (c.run)(
                black_box(&x),
                black_box(&y),
                black_box(&z),
                black_box(&mut out),
            );
            black_box(&out);
        }
        out[0] as f64
    } else {
        let op = OPS64[group as usize - OPS32.len()];
        let cs = cases64(op, feedback);
        let c = &cs[case as usize];
        let (x, y, z) = inputs64(op, n as usize);
        let mut out = vec![0.; n as usize];
        for _ in 0..repeats {
            (c.run)(
                black_box(&x),
                black_box(&y),
                black_box(&z),
                black_box(&mut out),
            );
            black_box(&out);
        }
        out[0]
    }
}
fn main() {
    verify();
    for (group, op) in OPS32.iter().chain(OPS64.iter()).enumerate() {
        for feedback in [false, true] {
            if *op == "pow_rescue" && feedback {
                continue;
            }
            let names: Vec<_> = if group < OPS32.len() {
                cases32(op, feedback).iter().map(|c| c.name).collect()
            } else {
                cases64(op, feedback).iter().map(|c| c.name).collect()
            };
            for (case, name) in names.into_iter().enumerate() {
                println!(
                    "C\t{group}\t{}\t{case}\t{}\t{op}\t{name}",
                    i32::from(feedback),
                    if group < OPS32.len() { "f32" } else { "f64" }
                );
            }
        }
    }
    black_box(dispatch_kernel as extern "C" fn(i32, i32, i32, i32, i32) -> f64);
}
