//! Bulk operations on `&mut [f32]`, dispatched through archmage tiers.
//!
//! Each op is an `#[attune]` family: the public name dispatches at runtime,
//! `_v3_t`-style proof wrappers take a token, and `_v3`-style direct variants
//! are for callers already in that feature context. Lanes left over after the
//! last whole `f32x8` go through the same kernel in a padded vector, so every
//! element of a slice, whatever its position, takes one code path.
//!
//! Dispatch happens once per call: pass a whole buffer, not one element at a
//! time. To chain several operations per element, write one `#[attune]` body
//! over [`kernels`] instead of calling these in sequence.

use super::kernels;
use archmage::prelude::*;

macro_rules! slice_op {
    ($public:ident, $kernel:ident $(, $arg:ident)?) => {
        #[doc = concat!(
            "[`kernels::", stringify!($kernel), "`] over a slice. See the module ",
            "docs for the generated tier variants."
        )]
        #[attune(_*, _*_t, dispatch(inline(hint)), +v4x(cfg(avx512)), define(f32x8))]
        pub fn $public(data: &mut [f32] $(, $arg: f32)?) {
            let token = Token::from_context();
            let (chunks, tail) = f32x8::partition_slice_mut_t(token, data);
            let mut pad = [0.0f32; 8];
            let tail_vectors = usize::from(!tail.is_empty());
            if tail_vectors != 0 {
                pad[..tail.len()].copy_from_slice(tail);
            }
            // One loop with one inlined kernel, not a loop and then a tail: on the
            // scalar tier a kernel is thousands of instructions.
            for i in 0..chunks.len() + tail_vectors {
                let chunk = if i < chunks.len() { &mut chunks[i] } else { &mut pad };
                kernels::$kernel(token, f32x8::load_t(token, chunk) $(, $arg)?).store(chunk);
            }
            if tail_vectors != 0 {
                tail.copy_from_slice(&pad[..tail.len()]);
            }
        }
    };
}

slice_op!(sqrt_f32, sqrt);
slice_op!(floor_f32, floor);
slice_op!(ceil_f32, ceil);
slice_op!(roundeven_f32, roundeven);

slice_op!(log2_lowp_f32, log2_lowp);
slice_op!(exp2_lowp_f32, exp2_lowp);
slice_op!(ln_lowp_f32, ln_lowp);
slice_op!(exp_lowp_f32, exp_lowp);
slice_op!(log10_lowp_f32, log10_lowp);
slice_op!(pow_lowp_f32, pow_lowp, n);
slice_op!(cbrt_lowp_f32, cbrt_lowp);

slice_op!(log2_midp_f32, log2_midp);
slice_op!(exp2_midp_f32, exp2_midp);
slice_op!(ln_midp_f32, ln_midp);
slice_op!(exp_midp_f32, exp_midp);
slice_op!(log10_midp_f32, log10_midp);
slice_op!(pow_midp_f32, pow_midp, n);
slice_op!(cbrt_midp_f32, cbrt_midp);
slice_op!(cbrt_midp_precise_f32, cbrt_midp_precise);
slice_op!(sigmoid_midp_f32, sigmoid_midp);
slice_op!(silu_midp_f32, silu_midp);
