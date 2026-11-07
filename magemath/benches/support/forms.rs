//! Kernels under test, each one `#[attune]` family streaming `[f32; 8]` values,
//! so a form's loop is one feature region. Shared by the `portable_cost` bench
//! and the `wasm_cost` example.
//!
//! - `fused`, `fused_unchecked`: `magemath::portable`, one rounding per step.
//! - `unfused`, `unfused_unchecked`: `magemath::unfused`, multiply then add.
//! - `relaxed`, `relaxed_unchecked`: `magemath::relaxed`, backend-dependent rounding.
//! - `method`: the matching magetypes `f32x8` method.

use archmage::prelude::*;

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

macro_rules! op_forms {
    ($op:ident, $unchecked:ident $(, $arg:expr)?) => {
        pub mod $op {
            use archmage::prelude::*;
            use magemath as mm;

            form!(fused, |t, v| mm::portable::kernels::$op(t, v $(, $arg)?));
            form!(fused_unchecked, |t, v| mm::portable::kernels::$unchecked(t, v $(, $arg)?));
            form!(unfused, |t, v| mm::unfused::kernels::$op(t, v $(, $arg)?));
            form!(unfused_unchecked, |t, v| mm::unfused::kernels::$unchecked(t, v $(, $arg)?));
            form!(relaxed, |t, v| mm::relaxed::kernels::$op(t, v $(, $arg)?));
            form!(relaxed_unchecked, |t, v| mm::relaxed::kernels::$unchecked(t, v $(, $arg)?));
            form!(method, |t, v| {
                let _ = t;
                // Keep the original inherent method as the benchmark control.
                #[allow(deprecated)]
                { v.$op($($arg)?) }
            });
        }
    };
}

op_forms!(exp2_midp, exp2_midp_unchecked);
op_forms!(log2_midp, log2_midp_unchecked);
op_forms!(pow_midp, pow_midp_unchecked, 2.4);
op_forms!(exp2_lowp, exp2_lowp_unchecked);
op_forms!(log2_lowp, log2_lowp_unchecked);

/// Each result feeds the next call, so the kernel's latency sets the time rather
/// than the throughput the streaming forms measure. `2^-x` converges quickly and
/// stays in range. This is the worst shape for a separate multiply and add.
macro_rules! feedback_form {
    ($name:ident, $surface:ident) => {
        #[attune(_*_t, +v4x(cfg(avx512)), define(f32x8))]
        pub fn $name(token: Token, input: &[[f32; 8]], out: &mut [[f32; 8]]) {
            let mut v = f32x8::load_t(token, &input[0]);
            for o in out.iter_mut() {
                v = magemath::$surface::kernels::exp2_midp(token, -v);
                v.store(o);
            }
        }
    };
}

feedback_form!(feedback_fused, portable);
feedback_form!(feedback_unfused, unfused);

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
