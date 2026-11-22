//! Const-specialized `Logf<CFG>` and runtime `LogfRuntime` entry points.
use super::{LogfError, LogfErrorKind, LogfOptions, dispatch, graph};
use crate::{Execution, SimdVector, SliceError};

pub(crate) struct Cfg<const CFG: u128>;

impl<const CFG: u128> graph::Kernel for Cfg<CFG> {
    const RANGED: bool = graph::KernelCfg::from_options(LogfOptions::decode_const(CFG)).ranged;
    const ZERO: bool = graph::KernelCfg::from_options(LogfOptions::decode_const(CFG)).zero;
    const SUBNORM: bool = graph::KernelCfg::from_options(LogfOptions::decode_const(CFG)).subnorm;
    const FLUSH_SUB: bool =
        graph::KernelCfg::from_options(LogfOptions::decode_const(CFG)).flush_sub;
    const SPECIALS: bool = graph::KernelCfg::from_options(LogfOptions::decode_const(CFG)).specials;
    const CLAMP_OOB: bool =
        graph::KernelCfg::from_options(LogfOptions::decode_const(CFG)).clamp_oob;
    const PRESERVE_CLAMPED_NAN: bool = LogfOptions::decode_const(CFG).preserve_clamped_nan();
    const LOWP: bool = graph::KernelCfg::from_options(LogfOptions::decode_const(CFG)).lowp;
    const ASSUME: bool = graph::KernelCfg::from_options(LogfOptions::decode_const(CFG)).assume;
    const ELIDE_RED: bool =
        graph::KernelCfg::from_options(LogfOptions::decode_const(CFG)).elide_red;
    const ELIDE_RED_LOG1P: bool = false;
    const IN_LO: f32 = graph::KernelCfg::from_options(LogfOptions::decode_const(CFG)).in_lo;
    const IN_HI: f32 = graph::KernelCfg::from_options(LogfOptions::decode_const(CFG)).in_hi;
}

macro_rules! vector {
    ($run:ident, $profile:ident, $dispatch:ident, $float:ident, $bound:ident, $n:literal) => {
        #[inline(always)]
        pub(crate) fn $run<const CFG: u128, const OP: u8, T: magetypes::simd::backends::$bound>(
            token: T,
            input: magetypes::simd::generic::$float<T>,
        ) -> Result<magetypes::simd::generic::$float<T>, LogfError> {
            let opts = const {
                let o = LogfOptions::decode_const(CFG);
                o.validate();
                o
            };
            let backend = opts.is_backend();
            let caller = dispatch::select(token, Execution::Caller, backend, false);
            let widen_v3 = $n == 16;
            if matches!(caller, dispatch::Plan::Software) || (widen_v3 && !opts.is_caller()) {
                if opts.is_caller() {
                    return if backend {
                        graph::$profile::<T, Cfg<CFG>, OP, { graph::NATIVE }>(token, input)
                    } else {
                        graph::$profile::<T, Cfg<CFG>, OP, { graph::SOFTWARE }>(token, input)
                    };
                }
                let plan = dispatch::select(token, Execution::Upgrade, backend, widen_v3);
                if !plan.same_tier(caller) || matches!(caller, dispatch::Plan::Software) {
                    let arr = input.to_array();
                    let result = if backend {
                        dispatch::$dispatch::<Cfg<CFG>, OP, true>(plan, arr)
                    } else {
                        dispatch::$dispatch::<Cfg<CFG>, OP, false>(plan, arr)
                    }?;
                    return Ok(magetypes::simd::generic::$float::from_array_t(
                        token, result,
                    ));
                }
            }
            #[cfg(target_arch = "wasm32")]
            if matches!(caller, dispatch::Plan::Wasm(_, false)) {
                return graph::$profile::<T, Cfg<CFG>, OP, { graph::WASM_SOFTWARE }>(token, input);
            }
            graph::$profile::<T, Cfg<CFG>, OP, { graph::NATIVE }>(token, input)
        }
    };
}
vector!(run4, profile4, run4_const, f32x4, F32x4Convert, 4);
vector!(run8, profile8, run8_const, f32x8, F32x8Convert, 8);
#[cfg(feature = "w512")]
vector!(run16, profile16, run16_const, f32x16, F32x16Convert, 16);

#[inline(always)]
fn scalar_with_token<const CFG: u128, const OP: u8, T: magetypes::simd::backends::F32x4Convert>(
    token: T,
    x: f32,
) -> Result<f32, LogfError> {
    let opts = const {
        let opts = LogfOptions::decode_const(CFG);
        opts.validate();
        opts
    };
    let backend = opts.is_backend();
    if token.as_scalar().is_some() {
        let plan = if opts.is_caller() {
            dispatch::Plan::Software
        } else {
            dispatch::select(token, Execution::Upgrade, backend, false)
        };
        return if backend {
            dispatch::run1_const::<Cfg<CFG>, OP, true>(plan, x)
        } else {
            dispatch::run1_const::<Cfg<CFG>, OP, false>(plan, x)
        };
    }
    run4::<CFG, OP, T>(token, magetypes::simd::generic::f32x4::splat_t(token, x))
        .map(|v| v.to_array()[0])
        .map_err(|error| LogfError {
            lane_mask: 1,
            ..error
        })
}

#[inline(always)]
fn array_const<const CFG: u128, const OP: u8, const N: usize>(
    inputs: [f32; N],
) -> Result<[f32; N], LogfError> {
    use archmage::ScalarToken;
    use magetypes::simd::generic::{f32x4, f32x8};
    const {
        LogfOptions::decode_const(CFG).validate();
    }
    if !matches!(N, 1 | 4 | 8) && !(cfg!(feature = "w512") && N == 16) {
        return Err(LogfError {
            kind: LogfErrorKind::UnsupportedWidth,
            lane_mask: 0,
        });
    }
    let mut output = [0.0; N];
    if N == 1 {
        output[0] = scalar_with_token::<CFG, OP, _>(ScalarToken, inputs[0])?;
    } else if N == 4 {
        let mut input = [0.0; 4];
        input.copy_from_slice(&inputs);
        output.copy_from_slice(
            &run4::<CFG, OP, _>(ScalarToken, f32x4::from_array_t(ScalarToken, input))?.to_array(),
        );
    } else if N == 8 {
        let mut input = [0.0; 8];
        input.copy_from_slice(&inputs);
        output.copy_from_slice(
            &run8::<CFG, OP, _>(ScalarToken, f32x8::from_array_t(ScalarToken, input))?.to_array(),
        );
    } else {
        #[cfg(feature = "w512")]
        if N == 16 {
            use magetypes::simd::generic::f32x16;
            let mut input = [0.0; 16];
            input.copy_from_slice(&inputs);
            output.copy_from_slice(
                &run16::<CFG, OP, _>(ScalarToken, f32x16::from_array_t(ScalarToken, input))?
                    .to_array(),
            );
            return Ok(output);
        }
        #[allow(unreachable_code)]
        return Err(LogfError {
            kind: LogfErrorKind::UnsupportedWidth,
            lane_mask: 0,
        });
    }
    Ok(output)
}

#[cfg_attr(debug_assertions, inline(never))]
#[cfg_attr(not(debug_assertions), inline(always))]
pub(crate) fn runtime_token<T: archmage::IntoConcreteToken, const OP: u8, const N: usize>(
    token: T,
    inputs: [f32; N],
    options: LogfOptions,
) -> Result<[f32; N], LogfError> {
    if !options.is_valid() {
        return Err(LogfError {
            kind: LogfErrorKind::UnsupportedOptions,
            lane_mask: 0,
        });
    }
    if !matches!(N, 1 | 4 | 8) && !(cfg!(feature = "w512") && N == 16) {
        return Err(LogfError {
            kind: LogfErrorKind::UnsupportedWidth,
            lane_mask: 0,
        });
    }
    let exec = if options.is_caller() {
        Execution::Caller
    } else {
        Execution::Upgrade
    };
    let backend = options.is_backend();
    let plan = dispatch::select(token, exec, backend, N == 16);
    let cfg = graph::KernelCfg::from_options(options);
    let mut output = [0.0; N];
    if N == 1 {
        output[0] = dispatch::run1_cfg(plan, inputs[0], OP, backend, cfg)?;
    } else if N == 4 {
        let mut input = [0.0; 4];
        input.copy_from_slice(&inputs);
        output.copy_from_slice(&dispatch::run4_cfg(plan, input, OP, backend, cfg)?);
    } else if N == 8 {
        let mut input = [0.0; 8];
        input.copy_from_slice(&inputs);
        output.copy_from_slice(&dispatch::run8_cfg(plan, input, OP, backend, cfg)?);
    } else {
        #[cfg(feature = "w512")]
        {
            let mut input = [0.0; 16];
            input.copy_from_slice(&inputs);
            output.copy_from_slice(&dispatch::run16_cfg(plan, input, OP, backend, cfg)?);
        }
    }
    Ok(output)
}

#[inline(always)]
fn slice_const<const CFG: u128, const OP: u8, T: archmage::IntoConcreteToken>(
    token: T,
    src: &[f32],
    dst: &mut [f32],
) -> Result<(), SliceError<LogfErrorKind>> {
    let opts = const {
        let o = LogfOptions::decode_const(CFG);
        o.validate();
        o
    };
    assert_eq!(src.len(), dst.len(), "logf slice lengths differ");
    let exec = if opts.is_caller() {
        Execution::Caller
    } else {
        Execution::Upgrade
    };
    let backend = opts.is_backend();
    let plan = dispatch::select(token, exec, backend, true);
    let res = if backend {
        dispatch::run_slice_const::<Cfg<CFG>, OP, true>(plan, src, dst)
    } else {
        dispatch::run_slice_const::<Cfg<CFG>, OP, false>(plan, src, dst)
    };
    res.map_err(|(base, e)| SliceError {
        kind: e.kind,
        index: base + e.lane_mask.trailing_zeros().min(15) as usize,
    })
}

#[cfg_attr(debug_assertions, inline(never))]
#[cfg_attr(not(debug_assertions), inline(always))]
fn slice_runtime<T: archmage::IntoConcreteToken, const OP: u8>(
    token: T,
    src: &[f32],
    dst: &mut [f32],
    options: LogfOptions,
) -> Result<(), SliceError<LogfErrorKind>> {
    assert_eq!(src.len(), dst.len(), "logf slice lengths differ");
    if !options.is_valid() {
        return Err(SliceError {
            kind: LogfErrorKind::UnsupportedOptions,
            index: 0,
        });
    }
    let exec = if options.is_caller() {
        Execution::Caller
    } else {
        Execution::Upgrade
    };
    let backend = options.is_backend();
    let plan = dispatch::select(token, exec, backend, true);
    let cfg = graph::KernelCfg::from_options(options);
    dispatch::run_slice_cfg(plan, src, dst, OP, backend, cfg).map_err(|(base, e)| SliceError {
        kind: e.kind,
        index: base + e.lane_mask.trailing_zeros().min(15) as usize,
    })
}

/// Compile-time specialized logarithm (`log2`, `ln`, `log10`) operator
/// configured by `const CFG: u128` (produced via [`LogfOptions::bits()`]).
///
/// ```
/// use magemath::logf::{Logf, LogfDomain, LogfOptions};
/// assert_eq!(<Logf>::log2(8.0).unwrap(), 3.0);
///
/// const NEAR_ONE: u128 = LogfOptions::new().domain(LogfDomain::SingleOctave).bits();
/// assert_eq!(Logf::<NEAR_ONE>::ln(1.0).unwrap(), 0.0);
/// ```
#[derive(Clone, Copy, Debug, Default)]
pub struct Logf<const CFG: u128 = { LogfOptions::new().bits() }>;

macro_rules! logf_const_methods {
    (
        $scalar:ident,
        $infallible:ident,
        $with_token:ident,
        $array:ident,
        $slice:ident,
        $slice_with_token:ident,
        $simd:ident,
        $op:literal,
        $label:literal
    ) => {
        #[doc = concat!("Checked scalar `", $label, "(x)`.")]
        #[inline(always)]
        pub fn $scalar(x: f32) -> Result<f32, LogfError> {
            array_const::<CFG, $op, 1>([x]).map(|r| r[0])
        }

        #[doc = concat!("Infallible scalar `", $label, "(x)` (requires `.clamp_out_of_bounds()`, `.assume_in_bounds()` / `.unchecked()`, or `.admit_all_floats()`).")]
        #[inline(always)]
        pub fn $infallible(x: f32) -> f32 {
            const {
                let opts = LogfOptions::decode_const(CFG);
                opts.validate();
                assert!(
                    opts.is_infallible(),
                    concat!(
                        "Logf::<CFG>::",
                        stringify!($infallible),
                        " requires `.clamp_out_of_bounds()`, `.assume_in_bounds()` / `.unchecked()`, or `.admit_all_floats()`"
                    )
                );
            }
            match Self::$scalar(x) {
                Ok(v) => v,
                Err(_) => unreachable!("infallible configuration returned Err"),
            }
        }

        #[doc = concat!("Scalar `", $label, "(x)` retaining a vector caller's token.")]
        #[inline(always)]
        pub fn $with_token<T: magetypes::simd::backends::F32x4Convert>(
            token: T,
            x: f32,
        ) -> Result<f32, LogfError> {
            scalar_with_token::<CFG, $op, T>(token, x)
        }

        #[doc = concat!("Array `", $label, "(x)` for widths `1`, `4`, `8` and (with `w512`) `16`.")]
        #[inline(always)]
        pub fn $array<const N: usize>(inputs: [f32; N]) -> Result<[f32; N], LogfError> {
            array_const::<CFG, $op, N>(inputs)
        }

        #[doc = concat!("Whole-slice `", $label, "(src[i]) -> dst[i]`.\n\n# Panics\n\nPanics if `src.len() != dst.len()`.")]
        #[inline(always)]
        pub fn $slice(src: &[f32], dst: &mut [f32]) -> Result<(), SliceError<LogfErrorKind>> {
            slice_const::<CFG, $op, _>(archmage::ScalarToken, src, dst)
        }

        #[doc = concat!("Whole-slice `", $label, "(src[i]) -> dst[i]` starting from `token`.\n\n# Panics\n\nPanics if `src.len() != dst.len()`.")]
        #[inline(always)]
        pub fn $slice_with_token(
            token: impl archmage::IntoConcreteToken,
            src: &[f32],
            dst: &mut [f32],
        ) -> Result<(), SliceError<LogfErrorKind>> {
            slice_const::<CFG, $op, _>(token, src, dst)
        }

        #[doc = concat!("Vector `", $label, "(input)` reusing `input.token()`.")]
        #[inline(always)]
        pub fn $simd<V: SimdVector>(input: V) -> Result<V, LogfError> {
            input.eval_logf_const::<CFG, $op>()
        }
    };
}

impl<const CFG: u128> Logf<CFG> {
    logf_const_methods!(
        log2,
        log2_infallible,
        log2_with_token,
        log2_array,
        log2_slice,
        log2_slice_with_token,
        log2_simd,
        0,
        "log2"
    );
    logf_const_methods!(
        ln,
        ln_infallible,
        ln_with_token,
        ln_array,
        ln_slice,
        ln_slice_with_token,
        ln_simd,
        1,
        "ln"
    );
    logf_const_methods!(
        log10,
        log10_infallible,
        log10_with_token,
        log10_array,
        log10_slice,
        log10_slice_with_token,
        log10_simd,
        2,
        "log10"
    );
}

/// Dynamic runtime logarithm evaluator created via [`LogfOptions::runtime()`].
#[derive(Clone, Copy, Debug)]
pub struct LogfRuntime {
    options: LogfOptions,
}

macro_rules! logf_runtime_methods {
    (
        $scalar:ident,
        $with_token:ident,
        $array:ident,
        $slice:ident,
        $slice_with_token:ident,
        $simd:ident,
        $op:literal,
        $label:literal
    ) => {
        #[doc = concat!("Checked scalar `", $label, "(x)` with runtime options.")]
        #[inline(always)]
        pub fn $scalar(self, x: f32) -> Result<f32, LogfError> {
            self.$array::<1>([x]).map(|r| r[0])
        }

        #[doc = concat!("Scalar `", $label, "(x)` retaining `token` with runtime options.")]
        #[inline(always)]
        pub fn $with_token<T: magetypes::simd::backends::F32x4Convert>(
            self,
            token: T,
            x: f32,
        ) -> Result<f32, LogfError> {
            runtime_token::<T, $op, 1>(token, [x], self.options).map(|r| r[0])
        }

        #[doc = concat!("Array `", $label, "(x)` with runtime options.")]
        #[inline(always)]
        pub fn $array<const N: usize>(self, inputs: [f32; N]) -> Result<[f32; N], LogfError> {
            runtime_token::<_, $op, N>(archmage::ScalarToken, inputs, self.options)
        }

        #[doc = concat!("Whole-slice `", $label, "(src[i]) -> dst[i]` with runtime options.\n\n# Panics\n\nPanics if `src.len() != dst.len()`.")]
        #[inline]
        pub fn $slice(
            self,
            src: &[f32],
            dst: &mut [f32],
        ) -> Result<(), SliceError<LogfErrorKind>> {
            slice_runtime::<_, $op>(archmage::ScalarToken, src, dst, self.options)
        }

        #[doc = concat!("Whole-slice `", $label, "(src[i]) -> dst[i]` starting from `token` with runtime options.\n\n# Panics\n\nPanics if `src.len() != dst.len()`.")]
        #[inline]
        pub fn $slice_with_token(
            self,
            token: impl archmage::IntoConcreteToken,
            src: &[f32],
            dst: &mut [f32],
        ) -> Result<(), SliceError<LogfErrorKind>> {
            slice_runtime::<_, $op>(token, src, dst, self.options)
        }

        #[doc = concat!("Vector `", $label, "(input)` with runtime options.")]
        #[inline(always)]
        pub fn $simd<V: SimdVector>(self, input: V) -> Result<V, LogfError> {
            input.eval_logf_runtime::<$op>(self.options)
        }
    };
}

impl LogfRuntime {
    #[inline(always)]
    pub(crate) const fn new(options: LogfOptions) -> Self {
        Self { options }
    }

    /// Return the underlying [`LogfOptions`].
    #[inline(always)]
    pub const fn options(self) -> LogfOptions {
        self.options
    }

    logf_runtime_methods!(
        log2,
        log2_with_token,
        log2_array,
        log2_slice,
        log2_slice_with_token,
        log2_simd,
        0,
        "log2"
    );
    logf_runtime_methods!(
        ln,
        ln_with_token,
        ln_array,
        ln_slice,
        ln_slice_with_token,
        ln_simd,
        1,
        "ln"
    );
    logf_runtime_methods!(
        log10,
        log10_with_token,
        log10_array,
        log10_slice,
        log10_slice_with_token,
        log10_simd,
        2,
        "log10"
    );
}
