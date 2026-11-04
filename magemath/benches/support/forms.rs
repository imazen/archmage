//! Kernels under test, each one `#[attune]` family streaming `[f32; 8]` values,
//! so a form's loop is one feature region. Shared by the `portable_cost` bench
//! and the `wasm_cost` example.

use archmage::prelude::*;
use magemath::portable::kernels;

macro_rules! form {
    ($name:ident, |$token:ident, $v:ident| $body:expr) => {
        #[attune(_*_t, +v4x(cfg(avx512)), define(f32x8))]
        pub fn $name($token: Token, input: &[[f32; 8]], out: &mut [[f32; 8]]) {
            for (o, i) in out.iter_mut().zip(input) {
                let $v = f32x8::load_t($token, i);
                ($body).store(o);
            }
        }
    };
}

form!(exp2_midp_portable, |t, v| kernels::exp2_midp(t, v));
form!(exp2_midp_unchecked, |t, v| kernels::exp2_midp_unchecked(
    t, v
));
form!(exp2_midp_relaxed, |t, v| v.exp2_midp());

form!(log2_midp_portable, |t, v| kernels::log2_midp(t, v));
form!(log2_midp_unchecked, |t, v| kernels::log2_midp_unchecked(
    t, v
));
form!(log2_midp_relaxed, |t, v| v.log2_midp());

form!(pow_midp_portable, |t, v| kernels::pow_midp(t, v, 2.4));
form!(pow_midp_unchecked, |t, v| kernels::pow_midp_unchecked(
    t, v, 2.4
));
form!(pow_midp_relaxed, |t, v| v.pow_midp(2.4));

form!(exp2_lowp_portable, |t, v| kernels::exp2_lowp(t, v));
form!(exp2_lowp_unchecked, |t, v| kernels::exp2_lowp_unchecked(
    t, v
));
form!(exp2_lowp_relaxed, |t, v| v.exp2_lowp());

form!(log2_lowp_portable, |t, v| kernels::log2_lowp(t, v));
form!(log2_lowp_unchecked, |t, v| kernels::log2_lowp_unchecked(
    t, v
));
form!(log2_lowp_relaxed, |t, v| v.log2_lowp());

/// `n` vectors of values spaced evenly across `lo..hi`.
pub fn inputs(lo: f32, hi: f32, n: usize) -> Vec<[f32; 8]> {
    let lanes = (n * 8) as f32;
    (0..n)
        .map(|i| {
            core::array::from_fn(|j| {
                let k = (i * 8 + j) as f32;
                lo + (hi - lo) * k / lanes
            })
        })
        .collect()
}
