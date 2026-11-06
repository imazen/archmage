//! `benches/portable_cost.rs` for wasm32, where zenbench does not build.
//!
//! Prints the best of several passes per form, in ns per element. Compare
//! strict and relaxed SIMD, under an engine that fuses relaxed madd or not:
//!
//!   RUSTFLAGS="-Ctarget-feature=+simd128" cargo build -p magemath \
//!     --example wasm_cost --release --target wasm32-wasip1
//!   wasmtime run target/wasm32-wasip1/release/examples/wasm_cost.wasm

#[path = "../benches/support/forms.rs"]
#[allow(dead_code, unused_imports)]
mod forms;

use archmage::prelude::*;
use forms::*;
use std::hint::black_box;
use std::time::Instant;

const N: usize = 1024;
const PASSES: usize = 200;

fn time(name: &str, input: &[[f32; 8]], run: impl Fn(&[[f32; 8]], &mut [[f32; 8]])) {
    let mut out = vec![[0.0f32; 8]; N];
    let mut best = f64::MAX;
    for _ in 0..PASSES {
        let start = Instant::now();
        run(black_box(input), &mut out);
        black_box(&out);
        best = best.min(start.elapsed().as_secs_f64());
    }
    println!("{name:28} {:8.3} ns/element", best * 1e9 / (N * 8) as f64);
}

macro_rules! timed {
    ($input:expr, $op:ident) => {
        timed!(@forms $input, $op, fused, fused_unchecked, unfused, unfused_unchecked, relaxed, relaxed_unchecked, method);
    };
    (@forms $input:expr, $op:ident, $($form:ident),+) => {
        $(time(concat!(stringify!($op), " ", stringify!($form)), $input, |i, o| {
            attuned!($op::$form(i, o), [_v4x(cfg(avx512)), _v3, _neon, _wasm128, _scalar])
        });)+
    };
}

fn main() {
    let exp_input = inputs(-20.0, 20.0, N);
    let log_input = inputs(0.001, 1000.0, N);
    let pow_input = inputs(0.001, 1.0, N);
    timed!(&exp_input, exp2_midp);
    timed!(&log_input, log2_midp);
    timed!(&pow_input, pow_midp);
    timed!(&exp_input, exp2_lowp);
    timed!(&log_input, log2_lowp);
    for (name, fused) in [("feedback fused", true), ("feedback unfused", false)] {
        time(name, &log_input, |i, o| {
            if fused {
                attuned!(feedback_fused(i, o), [_wasm128, _scalar]);
            } else {
                attuned!(feedback_unfused(i, o), [_wasm128, _scalar]);
            }
        });
    }
}
