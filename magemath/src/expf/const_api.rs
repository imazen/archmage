//! Const-specialized `Expf<CFG>` and runtime `ExpfRuntime` entry points.
use super::{ExpfError, ExpfErrorKind, ExpfOptions, dispatch, graph};
use crate::{Execution, SimdVector, SliceError};

pub(crate) struct Cfg<const CFG: u128>;

impl<const CFG: u128> graph::Kernel for Cfg<CFG> {
    const RANGED: bool = graph::KernelCfg::from_options(ExpfOptions::decode_const(CFG)).ranged;
    const FULL: bool = graph::KernelCfg::from_options(ExpfOptions::decode_const(CFG)).full;
    const CLAMP_OOB: bool =
        graph::KernelCfg::from_options(ExpfOptions::decode_const(CFG)).clamp_oob;
    const PRESERVE_CLAMPED_NAN: bool = ExpfOptions::decode_const(CFG).preserve_clamped_nan();
    const LOWP: bool = graph::KernelCfg::from_options(ExpfOptions::decode_const(CFG)).lowp;
    const ASSUME: bool = graph::KernelCfg::from_options(ExpfOptions::decode_const(CFG)).assume;
    const ELIDE_EXP2: bool =
        graph::KernelCfg::from_options(ExpfOptions::decode_const(CFG)).elide_exp2;
    const ELIDE_EXP: bool =
        graph::KernelCfg::from_options(ExpfOptions::decode_const(CFG)).elide_exp;
    const ELIDE_OVERFLOW_EXP2: bool =
        graph::KernelCfg::from_options(ExpfOptions::decode_const(CFG)).elide_overflow_exp2;
    const ELIDE_OVERFLOW_EXP: bool =
        graph::KernelCfg::from_options(ExpfOptions::decode_const(CFG)).elide_overflow_exp;
    const ELIDE_RED_EXP2: bool =
        graph::KernelCfg::from_options(ExpfOptions::decode_const(CFG)).elide_red_exp2;
    const ELIDE_RED_EXP: bool =
        graph::KernelCfg::from_options(ExpfOptions::decode_const(CFG)).elide_red_exp;
    const IN_LO_EXP2: f32 =
        graph::KernelCfg::from_options(ExpfOptions::decode_const(CFG)).in_lo_exp2;
    const IN_HI_EXP2: f32 =
        graph::KernelCfg::from_options(ExpfOptions::decode_const(CFG)).in_hi_exp2;
    const IN_LO_EXP: f32 = graph::KernelCfg::from_options(ExpfOptions::decode_const(CFG)).in_lo_exp;
    const IN_HI_EXP: f32 = graph::KernelCfg::from_options(ExpfOptions::decode_const(CFG)).in_hi_exp;
}

macro_rules! vector {
    ($run:ident, $profile:ident, $dispatch:ident, $float:ident, $bound:ident, $n:literal) => {
        #[inline(always)]
        pub(crate) fn $run<const CFG: u128, const OP: u8, T: magetypes::simd::backends::$bound>(
            token: T,
            input: magetypes::simd::generic::$float<T>,
        ) -> Result<magetypes::simd::generic::$float<T>, ExpfError> {
            let opts = const {
                let o = ExpfOptions::decode_const(CFG);
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
) -> Result<f32, ExpfError> {
    let opts = const {
        let opts = ExpfOptions::decode_const(CFG);
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
        .map_err(|error| ExpfError {
            lane_mask: 1,
            ..error
        })
}

#[inline(always)]
fn array_const<const CFG: u128, const OP: u8, const N: usize>(
    inputs: [f32; N],
) -> Result<[f32; N], ExpfError> {
    use archmage::ScalarToken;
    use magetypes::simd::generic::{f32x4, f32x8};
    const {
        ExpfOptions::decode_const(CFG).validate();
    }
    if !matches!(N, 1 | 4 | 8) && !(cfg!(feature = "w512") && N == 16) {
        return Err(ExpfError {
            kind: ExpfErrorKind::UnsupportedWidth,
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
        return Err(ExpfError {
            kind: ExpfErrorKind::UnsupportedWidth,
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
    options: ExpfOptions,
) -> Result<[f32; N], ExpfError> {
    if !options.is_valid() {
        return Err(ExpfError {
            kind: ExpfErrorKind::UnsupportedOptions,
            lane_mask: 0,
        });
    }
    if !matches!(N, 1 | 4 | 8) && !(cfg!(feature = "w512") && N == 16) {
        return Err(ExpfError {
            kind: ExpfErrorKind::UnsupportedWidth,
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
) -> Result<(), SliceError<ExpfErrorKind>> {
    let opts = const {
        let o = ExpfOptions::decode_const(CFG);
        o.validate();
        o
    };
    assert_eq!(src.len(), dst.len(), "expf slice lengths differ");
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
    options: ExpfOptions,
) -> Result<(), SliceError<ExpfErrorKind>> {
    assert_eq!(src.len(), dst.len(), "expf slice lengths differ");
    if !options.is_valid() {
        return Err(SliceError {
            kind: ExpfErrorKind::UnsupportedOptions,
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

/// Compile-time specialized exponential (`exp2`, `exp`) operator configured by
/// `const CFG: u128` (produced via [`ExpfOptions::bits()`]).
///
/// Use `<Expf>::exp2(x)` / `<Expf>::exp(x)` for default [`ExpfOptions::new()`]
/// options or `Expf::<CFG>::exp2(x)` for a custom `const CFG: u128`:
///
/// ```
/// use magemath::expf::{Expf, ExpfOptions};
/// assert_eq!(<Expf>::exp2(3.0).unwrap(), 8.0);
///
/// const SMALL: u128 = ExpfOptions::new().small_exp2_range().bits();
/// assert_eq!(Expf::<SMALL>::exp2(0.0).unwrap(), 1.0);
/// ```
#[derive(Clone, Copy, Debug, Default)]
pub struct Expf<const CFG: u128 = { ExpfOptions::new().bits() }>;

impl<const CFG: u128> Expf<CFG> {
    /// Checked scalar `2^x`.
    #[inline(always)]
    pub fn exp2(x: f32) -> Result<f32, ExpfError> {
        array_const::<CFG, 0, 1>([x]).map(|r| r[0])
    }

    /// Infallible scalar `2^x` (requires `.clamp_out_of_bounds()`,
    /// `.assume_in_bounds()` / `.unchecked()`, or `.admit_all_floats()`).
    #[inline(always)]
    pub fn exp2_infallible(x: f32) -> f32 {
        const {
            let opts = ExpfOptions::decode_const(CFG);
            opts.validate();
            assert!(
                opts.is_infallible(),
                "Expf::<CFG>::exp2_infallible requires `.clamp_out_of_bounds()`, `.assume_in_bounds()` / `.unchecked()`, or `.admit_all_floats()`"
            );
        }
        match Self::exp2(x) {
            Ok(v) => v,
            Err(_) => unreachable!("infallible configuration returned Err"),
        }
    }

    /// Scalar `2^x` retaining a vector caller's token.
    #[inline(always)]
    pub fn exp2_with_token<T: magetypes::simd::backends::F32x4Convert>(
        token: T,
        x: f32,
    ) -> Result<f32, ExpfError> {
        scalar_with_token::<CFG, 0, T>(token, x)
    }

    /// Array `2^x` for widths `1`, `4`, `8` and (with `w512`) `16`.
    #[inline(always)]
    pub fn exp2_array<const N: usize>(inputs: [f32; N]) -> Result<[f32; N], ExpfError> {
        array_const::<CFG, 0, N>(inputs)
    }

    /// Whole-slice `2^src[i] -> dst[i]`.
    ///
    /// # Panics
    ///
    /// Panics if `src.len() != dst.len()`.
    #[inline(always)]
    pub fn exp2_slice(src: &[f32], dst: &mut [f32]) -> Result<(), SliceError<ExpfErrorKind>> {
        slice_const::<CFG, 0, _>(archmage::ScalarToken, src, dst)
    }

    /// Whole-slice `2^src[i] -> dst[i]` starting from `token`.
    ///
    /// # Panics
    ///
    /// Panics if `src.len() != dst.len()`.
    #[inline(always)]
    pub fn exp2_slice_with_token(
        token: impl archmage::IntoConcreteToken,
        src: &[f32],
        dst: &mut [f32],
    ) -> Result<(), SliceError<ExpfErrorKind>> {
        slice_const::<CFG, 0, _>(token, src, dst)
    }

    /// Vector `2^input` reusing `input.token()`.
    #[inline(always)]
    pub fn exp2_simd<V: SimdVector>(input: V) -> Result<V, ExpfError> {
        input.eval_expf_const::<CFG, 0>()
    }

    /// Checked scalar `e^x`.
    #[inline(always)]
    pub fn exp(x: f32) -> Result<f32, ExpfError> {
        array_const::<CFG, 1, 1>([x]).map(|r| r[0])
    }

    /// Infallible scalar `e^x` (requires `.clamp_out_of_bounds()`,
    /// `.assume_in_bounds()` / `.unchecked()`, or `.admit_all_floats()`).
    #[inline(always)]
    pub fn exp_infallible(x: f32) -> f32 {
        const {
            let opts = ExpfOptions::decode_const(CFG);
            opts.validate();
            assert!(
                opts.is_infallible(),
                "Expf::<CFG>::exp_infallible requires `.clamp_out_of_bounds()`, `.assume_in_bounds()` / `.unchecked()`, or `.admit_all_floats()`"
            );
        }
        match Self::exp(x) {
            Ok(v) => v,
            Err(_) => unreachable!("infallible configuration returned Err"),
        }
    }

    /// Scalar `e^x` retaining a vector caller's token.
    #[inline(always)]
    pub fn exp_with_token<T: magetypes::simd::backends::F32x4Convert>(
        token: T,
        x: f32,
    ) -> Result<f32, ExpfError> {
        scalar_with_token::<CFG, 1, T>(token, x)
    }

    /// Array `e^x` for widths `1`, `4`, `8` and (with `w512`) `16`.
    #[inline(always)]
    pub fn exp_array<const N: usize>(inputs: [f32; N]) -> Result<[f32; N], ExpfError> {
        array_const::<CFG, 1, N>(inputs)
    }

    /// Whole-slice `e^src[i] -> dst[i]`.
    ///
    /// # Panics
    ///
    /// Panics if `src.len() != dst.len()`.
    #[inline(always)]
    pub fn exp_slice(src: &[f32], dst: &mut [f32]) -> Result<(), SliceError<ExpfErrorKind>> {
        slice_const::<CFG, 1, _>(archmage::ScalarToken, src, dst)
    }

    /// Whole-slice `e^src[i] -> dst[i]` starting from `token`.
    ///
    /// # Panics
    ///
    /// Panics if `src.len() != dst.len()`.
    #[inline(always)]
    pub fn exp_slice_with_token(
        token: impl archmage::IntoConcreteToken,
        src: &[f32],
        dst: &mut [f32],
    ) -> Result<(), SliceError<ExpfErrorKind>> {
        slice_const::<CFG, 1, _>(token, src, dst)
    }

    /// Vector `e^input` reusing `input.token()`.
    #[inline(always)]
    pub fn exp_simd<V: SimdVector>(input: V) -> Result<V, ExpfError> {
        input.eval_expf_const::<CFG, 1>()
    }
}

/// Dynamic runtime exponential evaluator created via [`ExpfOptions::runtime()`].
#[derive(Clone, Copy, Debug)]
pub struct ExpfRuntime {
    options: ExpfOptions,
}

impl ExpfRuntime {
    #[inline(always)]
    pub(crate) const fn new(options: ExpfOptions) -> Self {
        Self { options }
    }

    /// Return the underlying [`ExpfOptions`].
    #[inline(always)]
    pub const fn options(self) -> ExpfOptions {
        self.options
    }

    /// Checked scalar `2^x` with runtime options.
    #[inline(always)]
    pub fn exp2(self, x: f32) -> Result<f32, ExpfError> {
        self.exp2_array::<1>([x]).map(|r| r[0])
    }

    /// Scalar `2^x` retaining `token` with runtime options.
    #[inline(always)]
    pub fn exp2_with_token<T: magetypes::simd::backends::F32x4Convert>(
        self,
        token: T,
        x: f32,
    ) -> Result<f32, ExpfError> {
        runtime_token::<T, 0, 1>(token, [x], self.options).map(|r| r[0])
    }

    /// Array `2^x` with runtime options.
    #[inline(always)]
    pub fn exp2_array<const N: usize>(self, inputs: [f32; N]) -> Result<[f32; N], ExpfError> {
        runtime_token::<_, 0, N>(archmage::ScalarToken, inputs, self.options)
    }

    /// Whole-slice `2^src[i] -> dst[i]` with runtime options.
    ///
    /// # Panics
    ///
    /// Panics if `src.len() != dst.len()`.
    #[inline]
    pub fn exp2_slice(self, src: &[f32], dst: &mut [f32]) -> Result<(), SliceError<ExpfErrorKind>> {
        slice_runtime::<_, 0>(archmage::ScalarToken, src, dst, self.options)
    }

    /// Whole-slice `2^src[i] -> dst[i]` starting from `token` with runtime options.
    ///
    /// # Panics
    ///
    /// Panics if `src.len() != dst.len()`.
    #[inline]
    pub fn exp2_slice_with_token(
        self,
        token: impl archmage::IntoConcreteToken,
        src: &[f32],
        dst: &mut [f32],
    ) -> Result<(), SliceError<ExpfErrorKind>> {
        slice_runtime::<_, 0>(token, src, dst, self.options)
    }

    /// Vector `2^input` with runtime options.
    #[inline(always)]
    pub fn exp2_simd<V: SimdVector>(self, input: V) -> Result<V, ExpfError> {
        input.eval_expf_runtime::<0>(self.options)
    }

    /// Checked scalar `e^x` with runtime options.
    #[inline(always)]
    pub fn exp(self, x: f32) -> Result<f32, ExpfError> {
        self.exp_array::<1>([x]).map(|r| r[0])
    }

    /// Scalar `e^x` retaining `token` with runtime options.
    #[inline(always)]
    pub fn exp_with_token<T: magetypes::simd::backends::F32x4Convert>(
        self,
        token: T,
        x: f32,
    ) -> Result<f32, ExpfError> {
        runtime_token::<T, 1, 1>(token, [x], self.options).map(|r| r[0])
    }

    /// Array `e^x` with runtime options.
    #[inline(always)]
    pub fn exp_array<const N: usize>(self, inputs: [f32; N]) -> Result<[f32; N], ExpfError> {
        runtime_token::<_, 1, N>(archmage::ScalarToken, inputs, self.options)
    }

    /// Whole-slice `e^src[i] -> dst[i]` with runtime options.
    ///
    /// # Panics
    ///
    /// Panics if `src.len() != dst.len()`.
    #[inline]
    pub fn exp_slice(self, src: &[f32], dst: &mut [f32]) -> Result<(), SliceError<ExpfErrorKind>> {
        slice_runtime::<_, 1>(archmage::ScalarToken, src, dst, self.options)
    }

    /// Whole-slice `e^src[i] -> dst[i]` starting from `token` with runtime options.
    ///
    /// # Panics
    ///
    /// Panics if `src.len() != dst.len()`.
    #[inline]
    pub fn exp_slice_with_token(
        self,
        token: impl archmage::IntoConcreteToken,
        src: &[f32],
        dst: &mut [f32],
    ) -> Result<(), SliceError<ExpfErrorKind>> {
        slice_runtime::<_, 1>(token, src, dst, self.options)
    }

    /// Vector `e^input` with runtime options.
    #[inline(always)]
    pub fn exp_simd<V: SimdVector>(self, input: V) -> Result<V, ExpfError> {
        input.eval_expf_runtime::<1>(self.options)
    }
}
