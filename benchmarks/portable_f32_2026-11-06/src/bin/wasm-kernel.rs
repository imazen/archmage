//! Exported by wasm-ld --export-all; no unsafe export attributes required.
#![forbid(unsafe_code)]
use portable_f32_study::{CANDIDATES, eval};
use std::hint::black_box;
pub extern "C" fn portable_bench_kernel(candidate: i32, operation: i32, repeats: i32) -> f64 {
    let candidate = CANDIDATES[usize::try_from(candidate).unwrap()];
    let op = ["exp", "ln", "log2", "pow", "pow"][usize::try_from(operation).unwrap()];
    let input: [f32; 1024] = core::array::from_fn(|i| {
        let t = ((i * 719) % 1024) as f32 / 1024.0;
        if op == "exp" {
            -8.0 + 16.0 * t
        } else {
            0.01 + 4.0 * t
        }
    });
    let y: [f32; 1024] = core::array::from_fn(|i| {
        if operation == 4 {
            -4.0 + 8.0 * ((i * 313) % 1024) as f32 / 1024.0
        } else {
            2.4
        }
    });
    let mut out = [0.0; 1024];
    for _ in 0..repeats {
        out.copy_from_slice(black_box(&input));
        eval(candidate, op, black_box(&mut out), black_box(&y));
        black_box(&out);
    }
    black_box(out[17]) as f64
}
fn main() {
    black_box(portable_bench_kernel as extern "C" fn(i32, i32, i32) -> f64);
}
