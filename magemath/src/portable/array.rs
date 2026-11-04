//! Fixed-array operations: `[f32; 8]` in, `[f32; 8]` out.
//!
//! Each op is an `#[attune]` family: the public name dispatches at runtime,
//! `_v3_t`-style proof wrappers take a token, and `_v3`-style direct variants
//! are for callers already in that feature context. Each call is one load, one
//! kernel and one store; to keep a pipeline in registers across several
//! operations, call [`kernels`] from one `#[attune]` body instead.

use super::kernels;
use archmage::prelude::*;

macro_rules! array_op {
    ($public:ident, $kernel:ident $(, $arg:ident)?) => {
        #[doc = concat!(
            "[`kernels::", stringify!($kernel), "`] over `[f32; 8]`. See the module ",
            "docs for the generated tier variants."
        )]
        #[attune(_*, _*_t, dispatch(inline(hint)), +v4x(cfg(avx512)), define(f32x8))]
        pub fn $public(a: [f32; 8] $(, $arg: f32)?) -> [f32; 8] {
            let token = Token::from_context();
            kernels::$kernel(token, f32x8::from_array_t(token, a) $(, $arg)?).to_array()
        }
    };
}

array_op!(sqrt_f32x8, sqrt);
array_op!(floor_f32x8, floor);
array_op!(ceil_f32x8, ceil);
array_op!(roundeven_f32x8, roundeven);

array_op!(log2_lowp_f32x8, log2_lowp);
array_op!(exp2_lowp_f32x8, exp2_lowp);
array_op!(ln_lowp_f32x8, ln_lowp);
array_op!(exp_lowp_f32x8, exp_lowp);
array_op!(log10_lowp_f32x8, log10_lowp);
array_op!(pow_lowp_f32x8, pow_lowp, n);
array_op!(cbrt_lowp_f32x8, cbrt_lowp);

array_op!(log2_midp_f32x8, log2_midp);
array_op!(exp2_midp_f32x8, exp2_midp);
array_op!(ln_midp_f32x8, ln_midp);
array_op!(exp_midp_f32x8, exp_midp);
array_op!(log10_midp_f32x8, log10_midp);
array_op!(pow_midp_f32x8, pow_midp, n);
array_op!(cbrt_midp_f32x8, cbrt_midp);
array_op!(cbrt_midp_precise_f32x8, cbrt_midp_precise);
array_op!(sigmoid_midp_f32x8, sigmoid_midp);
array_op!(silu_midp_f32x8, silu_midp);
