#![forbid(unsafe_code)]
use std::{fs::File, io::Write};
use zenmath_pow_opt_study::*;
fn main() {
    let a: Vec<_> = std::env::args().collect();
    let d = Domain::parse(&a[1]);
    let (x, y) = inputs(d, 65536);
    let mut f = File::create(&a[2]).unwrap();
    f.write_all(b"POW32LE1").unwrap();
    f.write_all(&(x.len() as u64).to_le_bytes()).unwrap();
    for (x, y) in x.into_iter().zip(y) {
        f.write_all(&x.to_bits().to_le_bytes()).unwrap();
        f.write_all(&y.to_bits().to_le_bytes()).unwrap();
    }
}
