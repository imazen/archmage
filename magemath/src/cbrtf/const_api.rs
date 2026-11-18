//! Const-specialized `Cbrtf<CFG>` and runtime `CbrtfRuntime` entry points.
use super::{CbrtfError, CbrtfErrorKind, CbrtfOptions, dispatch, graph};
use crate::{Execution, SimdVector, SliceError};

pub(crate) struct Cfg<const CFG: u128>;

impl<const CFG: u128> graph::Kernel for Cfg<CFG> {
    const RANGED: bool = graph::KernelCfg::from_options(CbrtfOptions::decode_const(CFG)).ranged;
    const DENORM_POSSIBLE: bool =
        graph::KernelCfg::from_options(CbrtfOptions::decode_const(CFG)).denorm_possible;
    const SCALE_SUBNORM: bool =
        graph::KernelCfg::from_options(CbrtfOptions::decode_const(CFG)).scale_subnorm;
    const FULL: bool = graph::KernelCfg::from_options(CbrtfOptions::decode_const(CFG)).full;
    const CLAMP_OOB: bool =
        graph::KernelCfg::from_options(CbrtfOptions::decode_const(CFG)).clamp_oob;
    const LOWP: bool = graph::KernelCfg::from_options(CbrtfOptions::decode_const(CFG)).lowp;
    const ASSUME: bool = graph::KernelCfg::from_options(CbrtfOptions::decode_const(CFG)).assume;
    const IN_LO: f32 = graph::KernelCfg::from_options(CbrtfOptions::decode_const(CFG)).in_lo;
    const IN_HI: f32 = graph::KernelCfg::from_options(CbrtfOptions::decode_const(CFG)).in_hi;
}

macro_rules! vector {
    ($run:ident, $profile:ident, $dispatch:ident, $float:ident, $bound:ident, $n:literal) => {
        #[inline(always)]
        pub(crate) fn $run<const CFG: u128, T: magetypes::simd::backends::$bound>(
            token: T,
            input: magetypes::simd::generic::$float<T>,
        ) -> Result<magetypes::simd::generic::$float<T>, CbrtfError> {
            let opts = const {
                let o = CbrtfOptions::decode_const(CFG);
                o.validate();
                o
            };
            let backend = opts.is_backend();
            let caller = dispatch::select(token, Execution::Caller, false);
            let widen_v3 = $n == 16;
            if matches!(caller, dispatch::Plan::Software) || (widen_v3 && !opts.is_caller()) {
                if opts.is_caller() {
                    return if backend {
                        graph::$profile::<T, Cfg<CFG>, true>(token, input)
                    } else {
                        graph::$profile::<T, Cfg<CFG>, false>(token, input)
                    };
                }
                let plan = dispatch::select(token, Execution::Upgrade, widen_v3);
                if !plan.same_tier(caller) || matches!(caller, dispatch::Plan::Software) {
                    let arr = input.to_array();
                    let result = if backend {
                        dispatch::$dispatch::<Cfg<CFG>, true>(plan, arr)
                    } else {
                        dispatch::$dispatch::<Cfg<CFG>, false>(plan, arr)
                    }?;
                    return Ok(magetypes::simd::generic::$float::from_array_t(
                        token, result,
                    ));
                }
            }
            if backend {
                graph::$profile::<T, Cfg<CFG>, true>(token, input)
            } else {
                graph::$profile::<T, Cfg<CFG>, false>(token, input)
            }
        }
    };
}
vector!(run4, profile4, run4_const, f32x4, F32x4Convert, 4);
vector!(run8, profile8, run8_const, f32x8, F32x8Convert, 8);
#[cfg(feature = "w512")]
vector!(run16, profile16, run16_const, f32x16, F32x16Convert, 16);

#[inline(always)]
fn scalar_with_token<const CFG: u128, T: magetypes::simd::backends::F32x4Convert>(
    token: T,
    x: f32,
) -> Result<f32, CbrtfError> {
    run4::<CFG, T>(token, magetypes::simd::generic::f32x4::splat_t(token, x))
        .map(|v| v.to_array()[0])
        .map_err(|error| CbrtfError {
            lane_mask: 1,
            ..error
        })
}

#[inline(always)]
fn array_const<const CFG: u128, const N: usize>(inputs: [f32; N]) -> Result<[f32; N], CbrtfError> {
    use archmage::ScalarToken;
    use magetypes::simd::generic::{f32x4, f32x8};
    const {
        CbrtfOptions::decode_const(CFG).validate();
    }
    if !matches!(N, 1 | 4 | 8) && !(cfg!(feature = "w512") && N == 16) {
        return Err(CbrtfError {
            kind: CbrtfErrorKind::UnsupportedWidth,
            lane_mask: 0,
        });
    }
    let mut output = [0.0; N];
    if N == 1 {
        let value = run4::<CFG, _>(ScalarToken, f32x4::splat_t(ScalarToken, inputs[0])).map_err(
            |error| CbrtfError {
                lane_mask: 1,
                ..error
            },
        )?;
        output[0] = value.to_array()[0];
    } else if N == 4 {
        let mut input = [0.0; 4];
        input.copy_from_slice(&inputs);
        output.copy_from_slice(
            &run4::<CFG, _>(ScalarToken, f32x4::from_array_t(ScalarToken, input))?.to_array(),
        );
    } else if N == 8 {
        let mut input = [0.0; 8];
        input.copy_from_slice(&inputs);
        output.copy_from_slice(
            &run8::<CFG, _>(ScalarToken, f32x8::from_array_t(ScalarToken, input))?.to_array(),
        );
    } else {
        #[cfg(feature = "w512")]
        if N == 16 {
            use magetypes::simd::generic::f32x16;
            let mut input = [0.0; 16];
            input.copy_from_slice(&inputs);
            output.copy_from_slice(
                &run16::<CFG, _>(ScalarToken, f32x16::from_array_t(ScalarToken, input))?.to_array(),
            );
            return Ok(output);
        }
        #[allow(unreachable_code)]
        return Err(CbrtfError {
            kind: CbrtfErrorKind::UnsupportedWidth,
            lane_mask: 0,
        });
    }
    Ok(output)
}

#[cfg_attr(debug_assertions, inline(never))]
#[cfg_attr(not(debug_assertions), inline(always))]
pub(crate) fn runtime_token<T: archmage::IntoConcreteToken, const N: usize>(
    token: T,
    inputs: [f32; N],
    options: CbrtfOptions,
) -> Result<[f32; N], CbrtfError> {
    if !options.is_valid() {
        return Err(CbrtfError {
            kind: CbrtfErrorKind::UnsupportedOptions,
            lane_mask: 0,
        });
    }
    if !matches!(N, 1 | 4 | 8) && !(cfg!(feature = "w512") && N == 16) {
        return Err(CbrtfError {
            kind: CbrtfErrorKind::UnsupportedWidth,
            lane_mask: 0,
        });
    }
    let exec = if options.is_caller() {
        Execution::Caller
    } else {
        Execution::Upgrade
    };
    let plan = dispatch::select(token, exec, N == 16);
    let cfg = graph::KernelCfg::from_options(options);
    let backend = options.is_backend();
    let mut output = [0.0; N];
    if N == 1 {
        let value =
            dispatch::run4_cfg(plan, [inputs[0]; 4], backend, cfg).map_err(|error| CbrtfError {
                lane_mask: 1,
                ..error
            })?;
        output[0] = value[0];
    } else if N == 4 {
        let mut input = [0.0; 4];
        input.copy_from_slice(&inputs);
        output.copy_from_slice(&dispatch::run4_cfg(plan, input, backend, cfg)?);
    } else if N == 8 {
        let mut input = [0.0; 8];
        input.copy_from_slice(&inputs);
        output.copy_from_slice(&dispatch::run8_cfg(plan, input, backend, cfg)?);
    } else {
        #[cfg(feature = "w512")]
        {
            let mut input = [0.0; 16];
            input.copy_from_slice(&inputs);
            output.copy_from_slice(&dispatch::run16_cfg(plan, input, backend, cfg)?);
        }
    }
    Ok(output)
}

#[inline(always)]
fn slice_const<const CFG: u128, T: archmage::IntoConcreteToken>(
    token: T,
    src: &[f32],
    dst: &mut [f32],
) -> Result<(), SliceError<CbrtfErrorKind>> {
    let opts = const {
        let o = CbrtfOptions::decode_const(CFG);
        o.validate();
        o
    };
    assert_eq!(src.len(), dst.len(), "cbrtf slice lengths differ");
    let exec = if opts.is_caller() {
        Execution::Caller
    } else {
        Execution::Upgrade
    };
    let plan = dispatch::select(token, exec, true);
    let res = if opts.is_backend() {
        dispatch::run_slice_const::<Cfg<CFG>, true>(plan, src, dst)
    } else {
        dispatch::run_slice_const::<Cfg<CFG>, false>(plan, src, dst)
    };
    res.map_err(|(base, e)| SliceError {
        kind: e.kind,
        index: base + e.lane_mask.trailing_zeros().min(15) as usize,
    })
}

#[cfg_attr(debug_assertions, inline(never))]
#[cfg_attr(not(debug_assertions), inline(always))]
fn slice_runtime<T: archmage::IntoConcreteToken>(
    token: T,
    src: &[f32],
    dst: &mut [f32],
    options: CbrtfOptions,
) -> Result<(), SliceError<CbrtfErrorKind>> {
    assert_eq!(src.len(), dst.len(), "cbrtf slice lengths differ");
    if !options.is_valid() {
        return Err(SliceError {
            kind: CbrtfErrorKind::UnsupportedOptions,
            index: 0,
        });
    }
    let exec = if options.is_caller() {
        Execution::Caller
    } else {
        Execution::Upgrade
    };
    let plan = dispatch::select(token, exec, true);
    let cfg = graph::KernelCfg::from_options(options);
    dispatch::run_slice_cfg(plan, src, dst, options.is_backend(), cfg).map_err(|(base, e)| {
        SliceError {
            kind: e.kind,
            index: base + e.lane_mask.trailing_zeros().min(15) as usize,
        }
    })
}

/// Compile-time specialized cube-root (`cbrt`) operator configured by
/// `const CFG: u128` (produced via [`CbrtfOptions::bits()`]).
///
/// Use `<Cbrtf>::scalar(x)` for default [`CbrtfOptions::new()`] options or
/// `Cbrtf::<CFG>::scalar(x)` for a custom `const CFG: u128`:
///
/// ```
/// use magemath::cbrtf::{Cbrtf, CbrtfOptions};
/// assert!((<Cbrtf>::scalar(8.0).unwrap() - 2.0).abs() < 1e-6);
///
/// const UNIT: u128 = CbrtfOptions::new().unit_range().bits();
/// assert!((Cbrtf::<UNIT>::scalar(-0.125).unwrap() + 0.5).abs() < 1e-6);
/// ```
#[derive(Clone, Copy, Debug, Default)]
pub struct Cbrtf<const CFG: u128 = { CbrtfOptions::new().bits() }>;

impl<const CFG: u128> Cbrtf<CFG> {
    /// Checked scalar `cbrt(x)`.
    #[inline(always)]
    pub fn scalar(x: f32) -> Result<f32, CbrtfError> {
        array_const::<CFG, 1>([x]).map(|r| r[0])
    }

    /// Infallible scalar `cbrt(x)` (requires `.clamp_out_of_bounds()`,
    /// `.assume_in_bounds()` / `.unchecked()`, or `.admit_all_floats()`).
    #[inline(always)]
    pub fn scalar_infallible(x: f32) -> f32 {
        const {
            let opts = CbrtfOptions::decode_const(CFG);
            opts.validate();
            assert!(
                opts.is_infallible(),
                "Cbrtf::<CFG>::scalar_infallible requires `.clamp_out_of_bounds()`, `.assume_in_bounds()` / `.unchecked()`, or `.admit_all_floats()`"
            );
        }
        match Self::scalar(x) {
            Ok(v) => v,
            Err(_) => unreachable!("infallible configuration returned Err"),
        }
    }

    /// Scalar `cbrt(x)` retaining a vector caller's capability token.
    #[inline(always)]
    pub fn scalar_with_token<T: magetypes::simd::backends::F32x4Convert>(
        token: T,
        x: f32,
    ) -> Result<f32, CbrtfError> {
        scalar_with_token::<CFG, T>(token, x)
    }

    /// Array `cbrt(x)` for widths `1`, `4`, `8` and (with `w512`) `16`.
    #[inline(always)]
    pub fn array<const N: usize>(inputs: [f32; N]) -> Result<[f32; N], CbrtfError> {
        array_const::<CFG, N>(inputs)
    }

    /// Whole-slice `cbrt(src[i]) -> dst[i]`.
    ///
    /// # Panics
    ///
    /// Panics if `src.len() != dst.len()`.
    #[inline]
    pub fn slice(src: &[f32], dst: &mut [f32]) -> Result<(), SliceError<CbrtfErrorKind>> {
        slice_const::<CFG, _>(archmage::ScalarToken, src, dst)
    }

    /// Whole-slice `cbrt(src[i]) -> dst[i]` starting from `token`.
    ///
    /// # Panics
    ///
    /// Panics if `src.len() != dst.len()`.
    #[inline(always)]
    pub fn slice_with_token(
        token: impl archmage::IntoConcreteToken,
        src: &[f32],
        dst: &mut [f32],
    ) -> Result<(), SliceError<CbrtfErrorKind>> {
        slice_const::<CFG, _>(token, src, dst)
    }

    /// Vector `cbrt(input)` reusing `input.token()`.
    #[inline(always)]
    pub fn simd<V: SimdVector>(input: V) -> Result<V, CbrtfError> {
        input.eval_cbrtf_const::<CFG>()
    }
}

/// Dynamic runtime cube-root evaluator created via [`CbrtfOptions::runtime()`].
#[derive(Clone, Copy, Debug)]
pub struct CbrtfRuntime {
    options: CbrtfOptions,
}

impl CbrtfRuntime {
    #[inline(always)]
    pub(crate) const fn new(options: CbrtfOptions) -> Self {
        Self { options }
    }

    /// Return the underlying [`CbrtfOptions`].
    #[inline(always)]
    pub const fn options(self) -> CbrtfOptions {
        self.options
    }

    /// Checked scalar `cbrt(x)` with runtime options.
    #[inline(always)]
    pub fn scalar(self, x: f32) -> Result<f32, CbrtfError> {
        self.array::<1>([x]).map(|r| r[0])
    }

    /// Scalar `cbrt(x)` retaining `token` with runtime options.
    #[inline(always)]
    pub fn scalar_with_token<T: magetypes::simd::backends::F32x4Convert>(
        self,
        token: T,
        x: f32,
    ) -> Result<f32, CbrtfError> {
        runtime_token::<T, 1>(token, [x], self.options).map(|r| r[0])
    }

    /// Array `cbrt(x)` with runtime options.
    #[inline(always)]
    pub fn array<const N: usize>(self, inputs: [f32; N]) -> Result<[f32; N], CbrtfError> {
        runtime_token::<_, N>(archmage::ScalarToken, inputs, self.options)
    }

    /// Whole-slice `cbrt(src[i]) -> dst[i]` with runtime options.
    ///
    /// # Panics
    ///
    /// Panics if `src.len() != dst.len()`.
    #[inline]
    pub fn slice(self, src: &[f32], dst: &mut [f32]) -> Result<(), SliceError<CbrtfErrorKind>> {
        slice_runtime(archmage::ScalarToken, src, dst, self.options)
    }

    /// Whole-slice `cbrt(src[i]) -> dst[i]` starting from `token` with runtime options.
    ///
    /// # Panics
    ///
    /// Panics if `src.len() != dst.len()`.
    #[inline]
    pub fn slice_with_token(
        self,
        token: impl archmage::IntoConcreteToken,
        src: &[f32],
        dst: &mut [f32],
    ) -> Result<(), SliceError<CbrtfErrorKind>> {
        slice_runtime(token, src, dst, self.options)
    }

    /// Vector `cbrt(input)` with runtime options.
    #[inline(always)]
    pub fn simd<V: SimdVector>(self, input: V) -> Result<V, CbrtfError> {
        input.eval_cbrtf_runtime(self.options)
    }
}
