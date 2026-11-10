//! Numerical graph witness only, copied from clean linear-srgb 8900ffc.
//! No timing or claim that this binary is the linear-srgb crate.
#![forbid(unsafe_code)]
#![allow(clippy::excessive_precision)] // Preserve the original f32 coefficient encodings.
const P: [f32; 5] = [
    1.672_814_6e-2,
    8.089_536_4e-1,
    1.288_439_66e1,
    6.854_374_7e1,
    8.224_625_4e1,
];
const Q: [f32; 5] = [
    2.003_807_83e1,
    9.681_468_2e1,
    5.377_303_3e1,
    -7.125_638,
    1.0,
];
fn eval32(x: f32) -> f32 {
    let mut p = P[4];
    let mut q = Q[4];
    for i in (0..4).rev() {
        p = p.mul_add(x, P[i]);
        q = q.mul_add(x, Q[i]);
    }
    p / q
}
fn eval64(x: f32) -> f32 {
    let x = x as f64;
    let mut p = P[4] as f64;
    let mut q = Q[4] as f64;
    for i in (0..4).rev() {
        p = p.mul_add(x, P[i] as f64);
        q = q.mul_add(x, Q[i] as f64);
    }
    (p / q) as f32
}
fn main() {
    let mut mismatches = 0;
    for x in [0.0625_f32, 0.1, 0.125, 0.2, 0.25, 0.3, 0.5, 0.75, 0.9] {
        let a = eval32(x);
        let b = eval64(x);
        if a.to_bits() != b.to_bits() {
            mismatches += 1;
            println!(
                "input={:08x} f32-fused-graph={:08x} f64-scalar-graph={:08x}",
                x.to_bits(),
                a.to_bits(),
                b.to_bits()
            );
        }
    }
    println!("mismatching graph samples {mismatches}/9; this is not an exhaustive sweep");
}
