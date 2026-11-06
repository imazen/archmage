//! Kept/exported using wasm-ld export-all: no unsafe export attributes.
#![forbid(unsafe_code)]
use math_matrix_study::*;
use std::hint::black_box;
pub extern "C" fn matrix_kernel(case: i32, band: i32, repeats: i32) -> f64 {
    let cs = cases32();
    let band = ["ordinary", "edge", "special"][band as usize];
    let id = case as usize;
    if id < cs.len() {
        let c = &cs[id];
        let (x, y) = inputs(c.op, band, 1024);
        let mut o = vec![0.; 1024];
        for _ in 0..repeats {
            (c.run)(black_box(&x), black_box(&y), black_box(&mut o));
            black_box(&o);
        }
        o[17] as f64
    } else {
        let cs = cases64();
        let c = &cs[id - cases32().len()];
        let (x, y) = inputs64(band, 1024);
        let mut o = vec![0.; 1024];
        for _ in 0..repeats {
            (c.run)(black_box(&x), black_box(&y), black_box(&mut o));
            black_box(&o);
        }
        o[17]
    }
}
pub extern "C" fn algebraic_kernel(case: i32, repeats: i32) -> f64 {
    let cs = algebraic::cases_f32();
    let id = case as usize;
    if id < cs.len() {
        let c = &cs[id];
        let x: Vec<_> = (0..1024).map(|i| i as f32 / 1024.).collect();
        let y = x.clone();
        let mut o = vec![0.; 1024];
        for _ in 0..repeats {
            (c.run)(black_box(&x), black_box(&y), black_box(&mut o));
            black_box(&o);
        }
        o[0] as f64
    } else {
        let cs = algebraic::cases_f64();
        let c = &cs[id - algebraic::cases_f32().len()];
        let x: Vec<_> = (0..1024).map(|i| i as f64 / 1024.).collect();
        let y = x.clone();
        let mut o = vec![0.; 1024];
        for _ in 0..repeats {
            (c.run)(black_box(&x), black_box(&y), black_box(&mut o));
            black_box(&o);
        }
        o[0]
    }
}
fn main() {
    tier_inventory();
    for (i, c) in cases32()
        .iter()
        .map(|c| (c.op, c.label))
        .chain(cases64().iter().map(|c| (c.op, c.label)))
        .enumerate()
    {
        println!("M\t{i}\t{}\t{}", c.0, c.1);
    }
    for (i, c) in algebraic::cases_f32()
        .iter()
        .map(|c| (c.op, c.label, "f32"))
        .chain(
            algebraic::cases_f64()
                .iter()
                .map(|c| (c.op, c.label, "f64")),
        )
        .enumerate()
    {
        println!(
            "A\t{i}\t{}/{}\t{}",
            c.0.split('_').next().unwrap(),
            c.2,
            c.1
        );
    }
    black_box(matrix_kernel as extern "C" fn(i32, i32, i32) -> f64);
    black_box(algebraic_kernel as extern "C" fn(i32, i32) -> f64);
}
