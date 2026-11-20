//! Const-specialized `Log1pf<CFG>` and runtime `Log1pfRuntime` entry points,
//! reusing the shared `OP = 3` SIMD dispatch in [`crate::logf`].

use super::{Log1pfError, Log1pfErrorKind, Log1pfOptions};
use crate::{
    Execution, SimdVector, SliceError,
    logf::{LogfError, dispatch, graph},
};

#[inline(always)]
const fn map_err(err: LogfError) -> Log1pfError {
    Log1pfError {
        kind: Log1pfErrorKind::InvalidInput,
        lane_mask: err.lane_mask,
    }
}

pub(crate) struct Cfg<const CFG: u128>;

impl<const CFG: u128> graph::Kernel for Cfg<CFG> {
    const RANGED: bool = Log1pfOptions::decode_const(CFG).to_kernel_cfg().ranged;
    const ZERO: bool = Log1pfOptions::decode_const(CFG).to_kernel_cfg().zero;
    const SUBNORM: bool = Log1pfOptions::decode_const(CFG).to_kernel_cfg().subnorm;
    const FLUSH_SUB: bool = Log1pfOptions::decode_const(CFG).to_kernel_cfg().flush_sub;
    const SPECIALS: bool = Log1pfOptions::decode_const(CFG).to_kernel_cfg().specials;
    const CLAMP_OOB: bool = Log1pfOptions::decode_const(CFG).to_kernel_cfg().clamp_oob;
    const PRESERVE_CLAMPED_NAN: bool = Log1pfOptions::decode_const(CFG).preserve_clamped_nan();
    const LOWP: bool = Log1pfOptions::decode_const(CFG).to_kernel_cfg().lowp;
    const ASSUME: bool = Log1pfOptions::decode_const(CFG).to_kernel_cfg().assume;
    const ELIDE_RED: bool = false;
    const ELIDE_RED_LOG1P: bool = Log1pfOptions::decode_const(CFG)
        .to_kernel_cfg()
        .elide_red_log1p;
    const IN_LO: f32 = Log1pfOptions::decode_const(CFG).to_kernel_cfg().in_lo;
    const IN_HI: f32 = Log1pfOptions::decode_const(CFG).to_kernel_cfg().in_hi;
}

macro_rules! vector {
    ($run:ident, $profile:ident, $dispatch:ident, $float:ident, $bound:ident, $n:literal) => {
        #[inline(always)]
        pub(crate) fn $run<const CFG: u128, T: magetypes::simd::backends::$bound>(
            token: T,
            input: magetypes::simd::generic::$float<T>,
        ) -> Result<magetypes::simd::generic::$float<T>, Log1pfError> {
            let opts = const {
                let o = Log1pfOptions::decode_const(CFG);
                o.validate();
                o
            };
            let backend = opts.is_backend();
            let caller = dispatch::select(token, Execution::Caller, backend, false);
            let widen_v3 = $n == 16;
            if matches!(caller, dispatch::Plan::Software) || (widen_v3 && !opts.is_caller()) {
                if opts.is_caller() {
                    return if backend {
                        graph::$profile::<T, Cfg<CFG>, 3, { graph::NATIVE }>(token, input)
                            .map_err(map_err)
                    } else {
                        graph::$profile::<T, Cfg<CFG>, 3, { graph::SOFTWARE }>(token, input)
                            .map_err(map_err)
                    };
                }
                let plan = dispatch::select(token, Execution::Upgrade, backend, widen_v3);
                if !plan.same_tier(caller) || matches!(caller, dispatch::Plan::Software) {
                    let arr = input.to_array();
                    let result = if backend {
                        dispatch::$dispatch::<Cfg<CFG>, 3, true>(plan, arr)
                    } else {
                        dispatch::$dispatch::<Cfg<CFG>, 3, false>(plan, arr)
                    }
                    .map_err(map_err)?;
                    return Ok(magetypes::simd::generic::$float::from_array_t(
                        token, result,
                    ));
                }
            }
            #[cfg(target_arch = "wasm32")]
            if matches!(caller, dispatch::Plan::Wasm(_, false)) {
                return graph::$profile::<T, Cfg<CFG>, 3, { graph::WASM_SOFTWARE }>(token, input)
                    .map_err(map_err);
            }
            graph::$profile::<T, Cfg<CFG>, 3, { graph::NATIVE }>(token, input).map_err(map_err)
        }
    };
}
vector!(run4, profile4, run4_const, f32x4, F32x4Convert, 4);
vector!(run8, profile8, run8_const, f32x8, F32x8Convert, 8);
#[cfg(feature = "w512")]
vector!(run16, profile16, run16_const, f32x16, F32x16Convert, 16);

#[cfg_attr(debug_assertions, inline(never))]
#[cfg_attr(not(debug_assertions), inline(always))]
pub(crate) fn runtime_token<T: archmage::IntoConcreteToken, const N: usize>(
    token: T,
    inputs: [f32; N],
    options: Log1pfOptions,
) -> Result<[f32; N], Log1pfError> {
    if !options.is_valid() {
        return Err(Log1pfError {
            kind: Log1pfErrorKind::UnsupportedOptions,
            lane_mask: 0,
        });
    }
    if !matches!(N, 1 | 4 | 8) && !(cfg!(feature = "w512") && N == 16) {
        return Err(Log1pfError {
            kind: Log1pfErrorKind::UnsupportedWidth,
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
    let cfg = options.to_kernel_cfg();
    let mut output = [0.0; N];
    if N == 1 {
        let value = dispatch::run4_cfg(plan, [inputs[0]; 4], 3, backend, cfg).map_err(|error| {
            Log1pfError {
                kind: Log1pfErrorKind::InvalidInput,
                lane_mask: 1,
            }
            .with_mask_1(error)
        })?;
        output[0] = value[0];
    } else if N == 4 {
        let mut input = [0.0; 4];
        input.copy_from_slice(&inputs);
        output.copy_from_slice(&dispatch::run4_cfg(plan, input, 3, backend, cfg).map_err(map_err)?);
    } else if N == 8 {
        let mut input = [0.0; 8];
        input.copy_from_slice(&inputs);
        output.copy_from_slice(&dispatch::run8_cfg(plan, input, 3, backend, cfg).map_err(map_err)?);
    } else {
        #[cfg(feature = "w512")]
        {
            let mut input = [0.0; 16];
            input.copy_from_slice(&inputs);
            output.copy_from_slice(
                &dispatch::run16_cfg(plan, input, 3, backend, cfg).map_err(map_err)?,
            );
        }
    }
    Ok(output)
}

impl Log1pfError {
    #[inline(always)]
    const fn with_mask_1(self, _orig: LogfError) -> Self {
        self
    }
}

#[cfg_attr(debug_assertions, inline(never))]
#[cfg_attr(not(debug_assertions), inline(always))]
fn slice_runtime<T: archmage::IntoConcreteToken>(
    token: T,
    src: &[f32],
    dst: &mut [f32],
    options: Log1pfOptions,
) -> Result<(), SliceError<Log1pfErrorKind>> {
    assert_eq!(src.len(), dst.len(), "log1pf slice lengths differ");
    if !options.is_valid() {
        return Err(SliceError {
            kind: Log1pfErrorKind::UnsupportedOptions,
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
    let cfg = options.to_kernel_cfg();
    dispatch::run_slice_cfg(plan, src, dst, 3, backend, cfg).map_err(|(base, e)| SliceError {
        kind: Log1pfErrorKind::InvalidInput,
        index: base + e.lane_mask.trailing_zeros().min(15) as usize,
    })
}

/// Compile-time specialized `ln(1 + x)` (`log1p`) operator configured by
/// `const CFG: u128` (produced via [`Log1pfOptions::bits()`]).
///
/// ```
/// use magemath::log1pf::{Log1pf, Log1pfDomain, Log1pfOptions};
/// assert_eq!(<Log1pf>::scalar(0.0).unwrap(), 0.0);
///
/// const AROUND_ZERO: u128 = Log1pfOptions::new().domain(Log1pfDomain::AroundZero).bits();
/// assert_eq!(Log1pf::<AROUND_ZERO>::scalar(0.0).unwrap(), 0.0);
/// ```
#[derive(Clone, Copy, Debug, Default)]
pub struct Log1pf<const CFG: u128 = { Log1pfOptions::new().bits() }>;

impl<const CFG: u128> Log1pf<CFG> {
    /// Checked scalar `ln(1 + x)`.
    #[inline(always)]
    pub fn scalar(x: f32) -> Result<f32, Log1pfError> {
        Self::array::<1>([x]).map(|r| r[0])
    }

    /// Infallible scalar `ln(1 + x)` (requires `.clamp_out_of_bounds()`,
    /// `.assume_in_bounds()` / `.unchecked()`, or `.admit_all_floats()`).
    #[inline(always)]
    pub fn scalar_infallible(x: f32) -> f32 {
        const {
            let opts = Log1pfOptions::decode_const(CFG);
            opts.validate();
            assert!(
                opts.is_infallible(),
                "Log1pf::<CFG>::scalar_infallible requires `.clamp_out_of_bounds()`, `.assume_in_bounds()` / `.unchecked()`, or `.admit_all_floats()`"
            );
        }
        match Self::scalar(x) {
            Ok(v) => v,
            Err(_) => unreachable!("infallible configuration returned Err"),
        }
    }

    /// Scalar `ln(1 + x)` retaining a vector caller's token.
    #[inline(always)]
    pub fn scalar_with_token<T: magetypes::simd::backends::F32x4Convert>(
        token: T,
        x: f32,
    ) -> Result<f32, Log1pfError> {
        run4::<CFG, T>(token, magetypes::simd::generic::f32x4::splat_t(token, x))
            .map(|v| v.to_array()[0])
            .map_err(|error| Log1pfError {
                lane_mask: 1,
                ..error
            })
    }

    /// Array `ln(1 + x)` for widths `1`, `4`, `8` and (with `w512`) `16`.
    #[inline(always)]
    pub fn array<const N: usize>(inputs: [f32; N]) -> Result<[f32; N], Log1pfError> {
        use archmage::ScalarToken;
        use magetypes::simd::generic::{f32x4, f32x8};
        const {
            Log1pfOptions::decode_const(CFG).validate();
        }
        if !matches!(N, 1 | 4 | 8) && !(cfg!(feature = "w512") && N == 16) {
            return Err(Log1pfError {
                kind: Log1pfErrorKind::UnsupportedWidth,
                lane_mask: 0,
            });
        }
        let mut output = [0.0; N];
        if N == 1 {
            let value = run4::<CFG, _>(ScalarToken, f32x4::splat_t(ScalarToken, inputs[0]))
                .map_err(|error| Log1pfError {
                    lane_mask: 1,
                    ..error
                })?;
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
                    &run16::<CFG, _>(ScalarToken, f32x16::from_array_t(ScalarToken, input))?
                        .to_array(),
                );
                return Ok(output);
            }
            #[allow(unreachable_code)]
            return Err(Log1pfError {
                kind: Log1pfErrorKind::UnsupportedWidth,
                lane_mask: 0,
            });
        }
        Ok(output)
    }

    /// Whole-slice `ln(1 + src[i]) -> dst[i]`.
    ///
    /// # Panics
    ///
    /// Panics if `src.len() != dst.len()`.
    #[inline]
    pub fn slice(src: &[f32], dst: &mut [f32]) -> Result<(), SliceError<Log1pfErrorKind>> {
        Self::slice_with_token(archmage::ScalarToken, src, dst)
    }

    /// Whole-slice `ln(1 + src[i]) -> dst[i]` starting from `token`.
    ///
    /// # Panics
    ///
    /// Panics if `src.len() != dst.len()`.
    #[inline(always)]
    pub fn slice_with_token(
        token: impl archmage::IntoConcreteToken,
        src: &[f32],
        dst: &mut [f32],
    ) -> Result<(), SliceError<Log1pfErrorKind>> {
        let opts = const {
            let o = Log1pfOptions::decode_const(CFG);
            o.validate();
            o
        };
        assert_eq!(src.len(), dst.len(), "log1pf slice lengths differ");
        let exec = if opts.is_caller() {
            Execution::Caller
        } else {
            Execution::Upgrade
        };
        let backend = opts.is_backend();
        let plan = dispatch::select(token, exec, backend, true);
        let res = if backend {
            dispatch::run_slice_const::<Cfg<CFG>, 3, true>(plan, src, dst)
        } else {
            dispatch::run_slice_const::<Cfg<CFG>, 3, false>(plan, src, dst)
        };
        res.map_err(|(base, e)| SliceError {
            kind: Log1pfErrorKind::InvalidInput,
            index: base + e.lane_mask.trailing_zeros().min(15) as usize,
        })
    }

    /// Vector `ln(1 + input)` reusing `input.token()`.
    #[inline(always)]
    pub fn simd<V: SimdVector>(input: V) -> Result<V, Log1pfError> {
        input.eval_log1pf_const::<CFG>()
    }
}

/// Dynamic runtime `log1p` evaluator created via [`Log1pfOptions::runtime()`].
#[derive(Clone, Copy, Debug)]
pub struct Log1pfRuntime {
    options: Log1pfOptions,
}

impl Log1pfRuntime {
    #[inline(always)]
    pub(crate) const fn new(options: Log1pfOptions) -> Self {
        Self { options }
    }

    /// Return the underlying [`Log1pfOptions`].
    #[inline(always)]
    pub const fn options(self) -> Log1pfOptions {
        self.options
    }

    /// Checked scalar `ln(1 + x)` with runtime options.
    #[inline(always)]
    pub fn scalar(self, x: f32) -> Result<f32, Log1pfError> {
        self.array::<1>([x]).map(|r| r[0])
    }

    /// Scalar `ln(1 + x)` retaining `token` with runtime options.
    #[inline(always)]
    pub fn scalar_with_token<T: magetypes::simd::backends::F32x4Convert>(
        self,
        token: T,
        x: f32,
    ) -> Result<f32, Log1pfError> {
        runtime_token::<T, 1>(token, [x], self.options).map(|r| r[0])
    }

    /// Array `ln(1 + x)` with runtime options.
    #[inline(always)]
    pub fn array<const N: usize>(self, inputs: [f32; N]) -> Result<[f32; N], Log1pfError> {
        runtime_token::<_, N>(archmage::ScalarToken, inputs, self.options)
    }

    /// Whole-slice `ln(1 + src[i]) -> dst[i]` with runtime options.
    ///
    /// # Panics
    ///
    /// Panics if `src.len() != dst.len()`.
    #[inline]
    pub fn slice(self, src: &[f32], dst: &mut [f32]) -> Result<(), SliceError<Log1pfErrorKind>> {
        slice_runtime(archmage::ScalarToken, src, dst, self.options)
    }

    /// Whole-slice `ln(1 + src[i]) -> dst[i]` starting from `token` with runtime options.
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
    ) -> Result<(), SliceError<Log1pfErrorKind>> {
        slice_runtime(token, src, dst, self.options)
    }

    /// Vector `ln(1 + input)` with runtime options.
    #[inline(always)]
    pub fn simd<V: SimdVector>(self, input: V) -> Result<V, Log1pfError> {
        input.eval_log1pf_runtime(self.options)
    }
}
