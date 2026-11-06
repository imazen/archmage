//! Exported using link-dead-code/export-all, with no unsafe export attributes.
#![forbid(unsafe_code)]
use algebraic_ext_consumer::*;
use std::hint::black_box;
pub extern "C" fn ext_kernel(id: i32, repeats: i32) -> f64 {
    let cs = cases_f32();
    let id = id as usize;
    if id < cs.len() {
        let c = &cs[id];
        let (x, y) = inputs_f32(c.group);
        let mut o = vec![0.; 1024];
        for _ in 0..repeats {
            (c.run)(black_box(&x), black_box(&y), black_box(&mut o));
            black_box(&o);
        }
        o[0] as f64
    } else {
        let cc = cases_f64();
        let c = &cc[id - cs.len()];
        let (x, y) = inputs_f64(c.group);
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
    println!("exact fixtures {}", fixtures::check_all());
    for (i, (g, n)) in cases_f32()
        .iter()
        .map(|c| (c.group, c.name))
        .chain(cases_f64().iter().map(|c| (c.group, c.name)))
        .enumerate()
    {
        println!("C\t{i}\t{g}\t{n}");
    }
    black_box(ext_kernel as extern "C" fn(i32, i32) -> f64);
}
