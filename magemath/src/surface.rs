//! What each public module (`portable`, `relaxed`, ...) is made of: the same ops
//! under the same names, so the import path alone states the contract.

macro_rules! kernel {
    ($step:ty, $name:ident, $doc:literal) => {
        #[doc = $doc]
        #[inline(always)]
        pub fn $name<T: F32x8Convert>(token: T, x: f32x8<T>) -> f32x8<T> {
            $crate::bodies::$name::<$step, T>(token, x)
        }
    };
    ($step:ty, $name:ident, $doc:literal, n) => {
        #[doc = $doc]
        #[inline(always)]
        pub fn $name<T: F32x8Convert>(token: T, x: f32x8<T>, n: f32) -> f32x8<T> {
            $crate::bodies::$name::<$step, T>(token, x, n)
        }
    };
}

/// Token-generic kernels: every op, plus the `_unchecked` forms.
macro_rules! kernels_surface {
    ($step:ty) => {
        use magetypes::simd::backends::F32x8Convert;
        use magetypes::simd::generic::f32x8;

        $crate::surface::kernel!($step, sqrt, "Square root, correctly rounded.");
        $crate::surface::kernel!($step, floor, "Round toward negative infinity.");
        $crate::surface::kernel!($step, ceil, "Round toward positive infinity.");
        $crate::surface::kernel!(
            $step,
            roundeven,
            "Round to nearest, ties to even. Not `f32::round`, which ties away from zero."
        );

        #[cfg(feature = "configurable")]
        $crate::surface::kernel!($step, log2_lowp, "`log2`, lowp, per lane.");
        #[cfg(feature = "configurable")]
        $crate::surface::kernel!(
            $step,
            log2_lowp_unchecked,
            "[`log2_lowp`] for positive finite `x`."
        );
        #[cfg(feature = "configurable")]
        $crate::surface::kernel!($step, exp2_lowp, "`2^x`, lowp, per lane.");
        #[cfg(feature = "configurable")]
        $crate::surface::kernel!(
            $step,
            exp2_lowp_unchecked,
            "[`exp2_lowp`] for `x` in [-126, 127.99]."
        );
        #[cfg(feature = "configurable")]
        $crate::surface::kernel!($step, ln_lowp, "`ln`, lowp, per lane.");
        #[cfg(feature = "configurable")]
        $crate::surface::kernel!(
            $step,
            ln_lowp_unchecked,
            "[`ln_lowp`] for positive finite `x`."
        );
        #[cfg(feature = "configurable")]
        $crate::surface::kernel!($step, exp_lowp, "`e^x`, lowp, per lane.");
        #[cfg(feature = "configurable")]
        $crate::surface::kernel!(
            $step,
            exp_lowp_unchecked,
            "[`exp_lowp`] for `x · log2(e)` in [-126, 127.99]."
        );
        #[cfg(feature = "configurable")]
        $crate::surface::kernel!($step, log10_lowp, "`log10`, lowp, per lane.");
        #[cfg(feature = "configurable")]
        $crate::surface::kernel!(
            $step,
            log10_lowp_unchecked,
            "[`log10_lowp`] for positive finite `x`."
        );
        #[cfg(feature = "configurable")]
        $crate::surface::kernel!(
            $step,
            pow_lowp,
            "`x^n`, lowp, per lane, with one exponent for all lanes.",
            n
        );
        #[cfg(feature = "configurable")]
        $crate::surface::kernel!(
            $step,
            pow_lowp_unchecked,
            "[`pow_lowp`] for positive finite `x` with `log2(x) · n` in [-126, 127.99].",
            n
        );
        #[cfg(feature = "configurable")]
        $crate::surface::kernel!($step, cbrt_lowp, "Cube root, lowp, per lane.");

        #[cfg(feature = "configurable")]
        $crate::surface::kernel!($step, log2_midp, "`log2`, midp, per lane.");
        #[cfg(feature = "configurable")]
        $crate::surface::kernel!(
            $step,
            log2_midp_unchecked,
            "[`log2_midp`] for positive finite `x`."
        );
        #[cfg(feature = "configurable")]
        $crate::surface::kernel!($step, exp2_midp, "`2^x`, midp, per lane.");
        #[cfg(feature = "configurable")]
        $crate::surface::kernel!(
            $step,
            exp2_midp_unchecked,
            "[`exp2_midp`] for `x` in [-126, 128)."
        );
        #[cfg(feature = "configurable")]
        $crate::surface::kernel!($step, ln_midp, "`ln`, midp, per lane.");
        #[cfg(feature = "configurable")]
        $crate::surface::kernel!(
            $step,
            ln_midp_unchecked,
            "[`ln_midp`] for positive finite `x`."
        );
        #[cfg(feature = "configurable")]
        $crate::surface::kernel!($step, exp_midp, "`e^x`, midp, per lane.");
        #[cfg(feature = "configurable")]
        $crate::surface::kernel!(
            $step,
            exp_midp_unchecked,
            "[`exp_midp`] for `x · log2(e)` in [-126, 128)."
        );
        #[cfg(feature = "configurable")]
        $crate::surface::kernel!($step, log10_midp, "`log10`, midp, per lane.");
        #[cfg(feature = "configurable")]
        $crate::surface::kernel!(
            $step,
            log10_midp_unchecked,
            "[`log10_midp`] for positive finite `x`."
        );
        #[cfg(feature = "configurable")]
        $crate::surface::kernel!(
            $step,
            pow_midp,
            "`x^n`, midp, per lane, with one exponent for all lanes.",
            n
        );
        #[cfg(feature = "configurable")]
        $crate::surface::kernel!(
            $step,
            pow_midp_unchecked,
            "[`pow_midp`] for positive finite `x` with `log2(x) · n` in [-126, 128).",
            n
        );
        #[cfg(feature = "configurable")]
        $crate::surface::kernel!($step, cbrt_midp, "Cube root, midp, per lane.");
        #[cfg(feature = "configurable")]
        $crate::surface::kernel!(
            $step,
            cbrt_midp_precise,
            "Cube root, midp, over the whole f32 range, per lane."
        );
        #[cfg(feature = "configurable")]
        $crate::surface::kernel!(
            $step,
            sigmoid_midp,
            "Logistic sigmoid `1 / (1 + e^-x)`, midp, per lane."
        );
        #[cfg(feature = "configurable")]
        $crate::surface::kernel!($step, silu_midp, "SiLU `x · sigmoid(x)`, midp, per lane.");
    };
}

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

/// `&mut [f32]` families over the `kernels` module in scope at the call site.
macro_rules! slice_surface {
    () => {
        use super::kernels;
        use archmage::prelude::*;

        $crate::surface::slice_op!(sqrt_f32, sqrt);
        $crate::surface::slice_op!(floor_f32, floor);
        $crate::surface::slice_op!(ceil_f32, ceil);
        $crate::surface::slice_op!(roundeven_f32, roundeven);

        #[cfg(feature = "configurable")]
        $crate::surface::slice_op!(log2_lowp_f32, log2_lowp);
        #[cfg(feature = "configurable")]
        $crate::surface::slice_op!(exp2_lowp_f32, exp2_lowp);
        #[cfg(feature = "configurable")]
        $crate::surface::slice_op!(ln_lowp_f32, ln_lowp);
        #[cfg(feature = "configurable")]
        $crate::surface::slice_op!(exp_lowp_f32, exp_lowp);
        #[cfg(feature = "configurable")]
        $crate::surface::slice_op!(log10_lowp_f32, log10_lowp);
        #[cfg(feature = "configurable")]
        $crate::surface::slice_op!(pow_lowp_f32, pow_lowp, n);
        #[cfg(feature = "configurable")]
        $crate::surface::slice_op!(cbrt_lowp_f32, cbrt_lowp);

        #[cfg(feature = "configurable")]
        $crate::surface::slice_op!(log2_midp_f32, log2_midp);
        #[cfg(feature = "configurable")]
        $crate::surface::slice_op!(exp2_midp_f32, exp2_midp);
        #[cfg(feature = "configurable")]
        $crate::surface::slice_op!(ln_midp_f32, ln_midp);
        #[cfg(feature = "configurable")]
        $crate::surface::slice_op!(exp_midp_f32, exp_midp);
        #[cfg(feature = "configurable")]
        $crate::surface::slice_op!(log10_midp_f32, log10_midp);
        #[cfg(feature = "configurable")]
        $crate::surface::slice_op!(pow_midp_f32, pow_midp, n);
        #[cfg(feature = "configurable")]
        $crate::surface::slice_op!(cbrt_midp_f32, cbrt_midp);
        #[cfg(feature = "configurable")]
        $crate::surface::slice_op!(cbrt_midp_precise_f32, cbrt_midp_precise);
        #[cfg(feature = "configurable")]
        $crate::surface::slice_op!(sigmoid_midp_f32, sigmoid_midp);
        #[cfg(feature = "configurable")]
        $crate::surface::slice_op!(silu_midp_f32, silu_midp);
    };
}

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

/// `[f32; 8]` families over the `kernels` module in scope at the call site.
macro_rules! array_surface {
    () => {
        use super::kernels;
        use archmage::prelude::*;

        $crate::surface::array_op!(sqrt_f32x8, sqrt);
        $crate::surface::array_op!(floor_f32x8, floor);
        $crate::surface::array_op!(ceil_f32x8, ceil);
        $crate::surface::array_op!(roundeven_f32x8, roundeven);

        #[cfg(feature = "configurable")]
        $crate::surface::array_op!(log2_lowp_f32x8, log2_lowp);
        #[cfg(feature = "configurable")]
        $crate::surface::array_op!(exp2_lowp_f32x8, exp2_lowp);
        #[cfg(feature = "configurable")]
        $crate::surface::array_op!(ln_lowp_f32x8, ln_lowp);
        #[cfg(feature = "configurable")]
        $crate::surface::array_op!(exp_lowp_f32x8, exp_lowp);
        #[cfg(feature = "configurable")]
        $crate::surface::array_op!(log10_lowp_f32x8, log10_lowp);
        #[cfg(feature = "configurable")]
        $crate::surface::array_op!(pow_lowp_f32x8, pow_lowp, n);
        #[cfg(feature = "configurable")]
        $crate::surface::array_op!(cbrt_lowp_f32x8, cbrt_lowp);

        #[cfg(feature = "configurable")]
        $crate::surface::array_op!(log2_midp_f32x8, log2_midp);
        #[cfg(feature = "configurable")]
        $crate::surface::array_op!(exp2_midp_f32x8, exp2_midp);
        #[cfg(feature = "configurable")]
        $crate::surface::array_op!(ln_midp_f32x8, ln_midp);
        #[cfg(feature = "configurable")]
        $crate::surface::array_op!(exp_midp_f32x8, exp_midp);
        #[cfg(feature = "configurable")]
        $crate::surface::array_op!(log10_midp_f32x8, log10_midp);
        #[cfg(feature = "configurable")]
        $crate::surface::array_op!(pow_midp_f32x8, pow_midp, n);
        #[cfg(feature = "configurable")]
        $crate::surface::array_op!(cbrt_midp_f32x8, cbrt_midp);
        #[cfg(feature = "configurable")]
        $crate::surface::array_op!(cbrt_midp_precise_f32x8, cbrt_midp_precise);
        #[cfg(feature = "configurable")]
        $crate::surface::array_op!(sigmoid_midp_f32x8, sigmoid_midp);
        #[cfg(feature = "configurable")]
        $crate::surface::array_op!(silu_midp_f32x8, silu_midp);
    };
}

pub(crate) use {array_op, array_surface, kernel, kernels_surface, slice_op, slice_surface};
