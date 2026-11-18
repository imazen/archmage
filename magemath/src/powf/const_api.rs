//! Const-specialized `Powf<CFG>` and runtime `PowfRuntime` entry points.
use super::{PowfError, PowfErrorKind, PowfOptions, dispatch, graph};
use crate::{Execution, SimdVector, SliceError};

pub(crate) struct Cfg<const CFG: u128>;

impl<const CFG: u128> graph::Kernel for Cfg<CFG> {
    const ZERO: bool = graph::KernelCfg::from_options(PowfOptions::decode_const(CFG)).zero;
    const SUBNORM: bool = graph::KernelCfg::from_options(PowfOptions::decode_const(CFG)).subnorm;
    const RANGED: bool = graph::KernelCfg::from_options(PowfOptions::decode_const(CFG)).ranged;
    const ELIDE: bool = graph::KernelCfg::from_options(PowfOptions::decode_const(CFG)).elide;
    const CLAMP: bool = graph::KernelCfg::from_options(PowfOptions::decode_const(CFG)).clamp;
    const CLAMP_OOB: bool =
        graph::KernelCfg::from_options(PowfOptions::decode_const(CFG)).clamp_oob;
    const LOWP: bool = graph::KernelCfg::from_options(PowfOptions::decode_const(CFG)).lowp;
    const ASSUME: bool = graph::KernelCfg::from_options(PowfOptions::decode_const(CFG)).assume;
    const BASE_LO: f32 = graph::KernelCfg::from_options(PowfOptions::decode_const(CFG)).base_lo;
    const BASE_HI: f32 = graph::KernelCfg::from_options(PowfOptions::decode_const(CFG)).base_hi;
}

#[inline(always)]
fn prepare_exponent<const N: usize>(opts: PowfOptions, n: f32) -> Result<Option<f32>, PowfError> {
    if opts.assume_valid() || opts.is_full() {
        return Ok(Some(n));
    }
    if opts.clamp_oob() {
        if n.is_nan() {
            return Ok(None);
        }
        let (lo, hi) = opts.exponent_bounds().unwrap_or((-f32::MAX, f32::MAX));
        let mut c = if n < lo {
            lo
        } else if n > hi {
            hi
        } else {
            n
        };
        if !opts.admits_subnormals() {
            let abs_b = c.to_bits() & 0x7fff_ffff;
            if abs_b != 0 && abs_b < 0x0080_0000 {
                c = 0.0;
            }
        }
        return Ok(Some(c));
    }
    let bits = n.to_bits() & 0x7fff_ffff;
    let in_range = match opts.exponent_bounds() {
        Some((lo, hi)) => n >= lo && n <= hi,
        None => true,
    };
    let subnorm_bad = !opts.admits_subnormals() && bits != 0 && bits < 0x0080_0000;
    if bits >= 0x7f80_0000 || subnorm_bad || !in_range {
        return Err(PowfError {
            kind: PowfErrorKind::InvalidExponent,
            lane_mask: ((1_u32 << N.min(16)) - 1) as u16,
        });
    }
    Ok(Some(n))
}

#[inline(always)]
fn select_plan<T: archmage::IntoConcreteToken>(
    opts: PowfOptions,
    token: T,
    execution: Execution,
    widen_v3: bool,
) -> dispatch::Plan {
    if opts.is_highp() {
        if opts.is_backend() {
            dispatch::Plan::Software
        } else {
            dispatch::select_highp(token, execution, widen_v3)
        }
    } else if opts.is_backend() {
        dispatch::select_relaxed(token, execution, widen_v3)
    } else {
        dispatch::select(token, execution, widen_v3)
    }
}

#[cold]
#[inline(never)]
fn refine_slice_error(
    plan: dispatch::Plan,
    backend: bool,
    cfg: graph::KernelCfg,
    src: &[f32],
    base: usize,
    fallback: PowfError,
    n: f32,
) -> SliceError<PowfErrorKind> {
    let end = (base + 16).min(src.len());
    for (offset, &x) in src[base..end].iter().enumerate() {
        if let Err(err) = dispatch::run4_cfg(plan, backend, [x; 4], n, cfg) {
            return SliceError {
                kind: err.kind,
                index: base + offset,
            };
        }
    }
    SliceError {
        kind: fallback.kind,
        index: base + fallback.lane_mask.trailing_zeros().min(15) as usize,
    }
}

#[cold]
#[inline(never)]
fn refine_highp_slice_error<T: archmage::IntoConcreteToken>(
    token: T,
    opts: PowfOptions,
    src: &[f32],
    base: usize,
    fallback: PowfError,
    n: f32,
) -> SliceError<PowfErrorKind> {
    let end = (base + 8).min(src.len());
    for (offset, &x) in src[base..end].iter().enumerate() {
        if let Err(err) = super::highp_dispatch::for_token_cfg::<T, 1>(token, opts, [x], n) {
            return SliceError {
                kind: err.kind,
                index: base + offset,
            };
        }
    }
    SliceError {
        kind: fallback.kind,
        index: base + fallback.lane_mask.trailing_zeros().min(7) as usize,
    }
}

macro_rules! vector {
    ($run:ident, $profile:ident, $dispatch:ident, $float:ident, $bound:ident, $n:literal) => {
        #[inline(always)]
        pub(crate) fn $run<const CFG: u128, T: magetypes::simd::backends::$bound>(
            token: T,
            input: magetypes::simd::generic::$float<T>,
            n: f32,
        ) -> Result<magetypes::simd::generic::$float<T>, PowfError> {
            let opts = const {
                let o = PowfOptions::decode_const(CFG);
                o.validate();
                o
            };
            let Some(n) = prepare_exponent::<$n>(opts, n)? else {
                return Ok(magetypes::simd::generic::$float::splat_t(token, f32::NAN));
            };
            if opts.is_highp() {
                let result = super::highp_dispatch::for_token_cfg::<T, $n>(
                    token,
                    opts,
                    input.to_array(),
                    n,
                )?;
                return Ok(magetypes::simd::generic::$float::from_array_t(
                    token, result,
                ));
            }
            let caller = select_plan(opts, token, Execution::Caller, false);
            let widen_v3 = $n == 16;
            if matches!(caller, dispatch::Plan::Software) || (widen_v3 && !opts.is_caller()) {
                if opts.is_caller() {
                    if opts.is_backend() {
                        return graph::$profile::<T, Cfg<CFG>, { graph::RELAXED }>(token, input, n);
                    }
                    return graph::$profile::<T, Cfg<CFG>, { graph::SOFTWARE }>(token, input, n);
                }
                let plan = select_plan(opts, token, Execution::Upgrade, widen_v3);
                if !plan.same_tier(caller) || matches!(caller, dispatch::Plan::Software) {
                    let arr = input.to_array();
                    let result = if opts.is_backend() {
                        dispatch::$dispatch::<Cfg<CFG>, true>(plan, arr, n)
                    } else {
                        dispatch::$dispatch::<Cfg<CFG>, false>(plan, arr, n)
                    }?;
                    return Ok(magetypes::simd::generic::$float::from_array_t(
                        token, result,
                    ));
                }
            }
            #[cfg(target_arch = "wasm32")]
            if matches!(caller, dispatch::Plan::Wasm(_, false)) {
                return graph::$profile::<T, Cfg<CFG>, { graph::WASM_SOFTWARE }>(token, input, n);
            }
            graph::$profile::<T, Cfg<CFG>, { graph::NATIVE }>(token, input, n)
        }
    };
}
vector!(run4, profile4, run4_const, f32x4, F32x4Convert, 4);
vector!(run8, profile8, run8_const, f32x8, F32x8Convert, 8);
#[cfg(feature = "w512")]
vector!(run16, profile16, run16_const, f32x16, F32x16Convert, 16);

#[cfg_attr(debug_assertions, inline(never))]
#[cfg_attr(not(debug_assertions), inline(always))]
pub(crate) fn runtime<T: archmage::IntoConcreteToken, const N: usize>(
    token: T,
    bases: [f32; N],
    n: f32,
    options: PowfOptions,
) -> Result<[f32; N], PowfError> {
    if !options.is_valid() {
        return Err(PowfError {
            kind: PowfErrorKind::UnsupportedOptions,
            lane_mask: 0,
        });
    }
    if !matches!(N, 1 | 4 | 8) && !(cfg!(feature = "w512") && N == 16) {
        return Err(PowfError {
            kind: PowfErrorKind::UnsupportedWidth,
            lane_mask: 0,
        });
    }
    let Some(n) = prepare_exponent::<N>(options, n)? else {
        return Ok([f32::NAN; N]);
    };
    if options.is_highp() {
        return super::highp_dispatch::for_token_cfg::<T, N>(token, options, bases, n);
    }
    let exec = if options.is_caller() {
        Execution::Caller
    } else {
        Execution::Upgrade
    };
    let backend = options.is_backend();
    let widen_v3 = N == 16;
    let plan = if backend {
        dispatch::select_relaxed(token, exec, widen_v3)
    } else {
        dispatch::select(token, exec, widen_v3)
    };
    let cfg = graph::KernelCfg::from_options(options);
    let mut output = [0.; N];
    if N == 1 {
        let arr = dispatch::run4_cfg(plan, backend, [bases[0]; 4], n, cfg).map_err(|error| {
            PowfError {
                lane_mask: 1,
                ..error
            }
        })?;
        output[0] = arr[0];
    } else if N == 4 {
        let mut input = [0.; 4];
        input.copy_from_slice(&bases);
        let res = dispatch::run4_cfg(plan, backend, input, n, cfg)?;
        output.copy_from_slice(&res);
    } else if N == 8 {
        let mut input = [0.; 8];
        input.copy_from_slice(&bases);
        let res = dispatch::run8_cfg(plan, backend, input, n, cfg)?;
        output.copy_from_slice(&res);
    } else {
        #[cfg(feature = "w512")]
        {
            let mut input = [0.; 16];
            input.copy_from_slice(&bases);
            let res = dispatch::run16_cfg(plan, backend, input, n, cfg)?;
            output.copy_from_slice(&res);
        }
    }
    Ok(output)
}

#[cfg_attr(debug_assertions, inline(never))]
#[cfg_attr(not(debug_assertions), inline(always))]
fn slice_runtime<T: archmage::IntoConcreteToken>(
    token: T,
    src: &[f32],
    dst: &mut [f32],
    n: f32,
    options: PowfOptions,
) -> Result<(), SliceError<PowfErrorKind>> {
    assert_eq!(src.len(), dst.len(), "powf slice lengths differ");
    if !options.is_valid() {
        return Err(SliceError {
            kind: PowfErrorKind::UnsupportedOptions,
            index: 0,
        });
    }
    let Some(n) = prepare_exponent::<1>(options, n).map_err(|e| SliceError {
        kind: e.kind,
        index: 0,
    })?
    else {
        dst.fill(f32::NAN);
        return Ok(());
    };
    if options.is_highp() {
        return super::highp_dispatch::for_token_slice_cfg::<T>(token, options, src, dst, n)
            .map_err(|(base, e)| refine_highp_slice_error(token, options, src, base, e, n));
    }
    let exec = if options.is_caller() {
        Execution::Caller
    } else {
        Execution::Upgrade
    };
    let backend = options.is_backend();
    let plan = select_plan(options, token, exec, true);
    let cfg = graph::KernelCfg::from_options(options);
    dispatch::run_slice_cfg(plan, backend, src, dst, n, cfg)
        .map_err(|(base, e)| refine_slice_error(plan, backend, cfg, src, base, e, n))
}

/// Compile-time specialized `powf` operator configured by `const CFG: u128`
/// (produced via [`PowfOptions::bits()`]).
///
/// Use `<Powf>::scalar(base, n)` for default [`PowfOptions::new()`] options or
/// `Powf::<CFG>::scalar(base, n)` for a custom `const CFG: u128`:
///
/// ```
/// use magemath::powf::{Powf, PowfOptions};
/// assert_eq!(<Powf>::scalar(4.0, 0.5).unwrap(), 2.0);
///
/// const GAMMA: u128 = PowfOptions::new().gamma_domain().caller().bits();
/// assert_eq!(Powf::<GAMMA>::scalar(4.0, 0.5).unwrap(), 2.0);
/// ```
///
/// Unencoded integers, another family's `bits()`, or invalid option combinations
/// fail at compile time:
///
/// ```compile_fail
/// let _ = magemath::powf::Powf::<0>::scalar(1.0, 2.4);
/// ```
/// ```compile_fail
/// const EXP: u128 = magemath::expf::ExpfOptions::new().bits();
/// let _ = magemath::powf::Powf::<EXP>::scalar(1.0, 2.4);
/// ```
/// ```compile_fail
/// const BAD: u128 = magemath::powf::PowfOptions::new().full_domain().bits();
/// let _ = magemath::powf::Powf::<BAD>::scalar(1.0, 2.4);
/// ```
#[derive(Clone, Copy, Debug, Default)]
pub struct Powf<const CFG: u128 = { PowfOptions::new().bits() }>;

impl<const CFG: u128> Powf<CFG> {
    /// Checked scalar `base^n`.
    #[inline(always)]
    pub fn scalar(base: f32, n: f32) -> Result<f32, PowfError> {
        Self::array::<1>([base], n).map(|result| result[0])
    }

    /// Infallible scalar `base^n`.
    ///
    /// Fails to compile unless `CFG` sets `.clamp_out_of_bounds()`,
    /// `.assume_in_bounds()` / `.unchecked()`, or `.admit_all_floats().highp()`.
    #[inline(always)]
    pub fn scalar_infallible(base: f32, n: f32) -> f32 {
        const {
            let opts = PowfOptions::decode_const(CFG);
            opts.validate();
            assert!(
                opts.is_infallible(),
                "Powf::<CFG>::scalar_infallible requires `.clamp_out_of_bounds()`, `.assume_in_bounds()` / `.unchecked()`, or `.admit_all_floats().highp()`; use `Powf::<CFG>::scalar` for checked options"
            );
        }
        match Self::scalar(base, n) {
            Ok(v) => v,
            Err(_) => unreachable!("infallible configuration returned Err"),
        }
    }

    /// Scalar `base^n` retaining a vector caller's capability token (for scalar tails).
    #[inline(always)]
    pub fn scalar_with_token<T: magetypes::simd::backends::F32x4Convert>(
        token: T,
        base: f32,
        n: f32,
    ) -> Result<f32, PowfError> {
        let opts = const {
            let o = PowfOptions::decode_const(CFG);
            o.validate();
            o
        };
        if opts.is_highp() {
            let Some(n) = prepare_exponent::<1>(opts, n)? else {
                return Ok(f32::NAN);
            };
            return super::highp_dispatch::for_token_cfg::<T, 1>(token, opts, [base], n)
                .map(|v| v[0]);
        }
        run4::<CFG, T>(
            token,
            magetypes::simd::generic::f32x4::splat_t(token, base),
            n,
        )
        .map(|v| v.to_array()[0])
        .map_err(|error| PowfError {
            lane_mask: 1,
            ..error
        })
    }

    /// Version-independent array boundary for widths `1`, `4`, `8` and (with `w512`) `16`.
    #[inline(always)]
    pub fn array<const N: usize>(bases: [f32; N], n: f32) -> Result<[f32; N], PowfError> {
        use archmage::ScalarToken;
        use magetypes::simd::generic::{f32x4, f32x8};
        let opts = const {
            let o = PowfOptions::decode_const(CFG);
            o.validate();
            o
        };
        if !matches!(N, 1 | 4 | 8) && !(cfg!(feature = "w512") && N == 16) {
            return Err(PowfError {
                kind: PowfErrorKind::UnsupportedWidth,
                lane_mask: 0,
            });
        }
        if opts.is_highp() {
            let Some(n) = prepare_exponent::<N>(opts, n)? else {
                return Ok([f32::NAN; N]);
            };
            return super::highp_dispatch::for_token_cfg::<_, N>(ScalarToken, opts, bases, n);
        }
        let mut output = [0.0; N];
        if N == 1 {
            let value = run4::<CFG, _>(ScalarToken, f32x4::splat_t(ScalarToken, bases[0]), n)
                .map_err(|error| PowfError {
                    lane_mask: 1,
                    ..error
                })?;
            output[0] = value.to_array()[0];
        } else if N == 4 {
            let mut input = [0.0; 4];
            input.copy_from_slice(&bases);
            output.copy_from_slice(
                &run4::<CFG, _>(ScalarToken, f32x4::from_array_t(ScalarToken, input), n)?
                    .to_array(),
            );
        } else if N == 8 {
            let mut input = [0.0; 8];
            input.copy_from_slice(&bases);
            output.copy_from_slice(
                &run8::<CFG, _>(ScalarToken, f32x8::from_array_t(ScalarToken, input), n)?
                    .to_array(),
            );
        } else {
            #[cfg(feature = "w512")]
            if N == 16 {
                use magetypes::simd::generic::f32x16;
                let mut input = [0.0; 16];
                input.copy_from_slice(&bases);
                output.copy_from_slice(
                    &run16::<CFG, _>(ScalarToken, f32x16::from_array_t(ScalarToken, input), n)?
                        .to_array(),
                );
                return Ok(output);
            }
            #[allow(unreachable_code)]
            return Err(PowfError {
                kind: PowfErrorKind::UnsupportedWidth,
                lane_mask: 0,
            });
        }
        Ok(output)
    }

    /// Whole-slice `src[i]^n -> dst[i]`, selecting the best SIMD backend once.
    ///
    /// # Panics
    ///
    /// Panics if `src.len() != dst.len()`.
    #[inline]
    pub fn slice(src: &[f32], dst: &mut [f32], n: f32) -> Result<(), SliceError<PowfErrorKind>> {
        Self::slice_with_token(archmage::ScalarToken, src, dst, n)
    }

    /// Whole-slice `src[i]^n -> dst[i]` starting from `token` (reuses `token`
    /// under `Execution::Caller`, or upgrades to `V4`/`V4x` when beneficial
    /// under `Execution::Upgrade`).
    ///
    /// # Panics
    ///
    /// Panics if `src.len() != dst.len()`.
    #[inline(always)]
    pub fn slice_with_token(
        token: impl archmage::IntoConcreteToken,
        src: &[f32],
        dst: &mut [f32],
        n: f32,
    ) -> Result<(), SliceError<PowfErrorKind>> {
        let opts = const {
            let o = PowfOptions::decode_const(CFG);
            o.validate();
            o
        };
        assert_eq!(src.len(), dst.len(), "powf slice lengths differ");
        let Some(n) = prepare_exponent::<1>(opts, n).map_err(|e| SliceError {
            kind: e.kind,
            index: 0,
        })?
        else {
            dst.fill(f32::NAN);
            return Ok(());
        };
        if opts.is_highp() {
            return super::highp_dispatch::for_token_slice_cfg(token, opts, src, dst, n)
                .map_err(|(base, e)| refine_highp_slice_error(token, opts, src, base, e, n));
        }
        let exec = if opts.is_caller() {
            Execution::Caller
        } else {
            Execution::Upgrade
        };
        let backend = opts.is_backend();
        let plan = select_plan(opts, token, exec, true);
        let res = if backend {
            dispatch::run_slice::<Cfg<CFG>, true>(plan, src, dst, n)
        } else {
            dispatch::run_slice::<Cfg<CFG>, false>(plan, src, dst, n)
        };
        res.map_err(|(base, e)| {
            let cfg = graph::KernelCfg::from_options(opts);
            refine_slice_error(plan, backend, cfg, src, base, e, n)
        })
    }

    /// Evaluate `input^n` on any Magetypes `f32` vector (`f32x4`, `f32x8`, or
    /// `f32x16`), reusing `input.token()`.
    #[inline(always)]
    pub fn simd<V: SimdVector>(input: V, n: f32) -> Result<V, PowfError> {
        input.eval_powf_const::<CFG>(n)
    }
}

/// Dynamic runtime `powf` evaluator created via [`PowfOptions::runtime()`].
#[derive(Clone, Copy, Debug)]
pub struct PowfRuntime {
    options: PowfOptions,
}

impl PowfRuntime {
    #[inline(always)]
    pub(crate) const fn new(options: PowfOptions) -> Self {
        Self { options }
    }

    /// Return the underlying [`PowfOptions`].
    #[inline(always)]
    pub const fn options(self) -> PowfOptions {
        self.options
    }

    /// Checked scalar `base^n` with runtime options.
    #[inline(always)]
    pub fn scalar(self, base: f32, n: f32) -> Result<f32, PowfError> {
        self.array::<1>([base], n).map(|r| r[0])
    }

    /// Scalar `base^n` retaining a vector caller's token with runtime options.
    #[inline(always)]
    pub fn scalar_with_token<T: magetypes::simd::backends::F32x4Convert>(
        self,
        token: T,
        base: f32,
        n: f32,
    ) -> Result<f32, PowfError> {
        runtime::<T, 1>(token, [base], n, self.options).map(|r| r[0])
    }

    /// Array `bases^n` for widths `1`, `4`, `8` and (with `w512`) `16`.
    #[inline(always)]
    pub fn array<const N: usize>(self, bases: [f32; N], n: f32) -> Result<[f32; N], PowfError> {
        runtime::<_, N>(archmage::ScalarToken, bases, n, self.options)
    }

    /// Whole-slice `src[i]^n -> dst[i]` with runtime options.
    ///
    /// # Panics
    ///
    /// Panics if `src.len() != dst.len()`.
    #[inline]
    pub fn slice(
        self,
        src: &[f32],
        dst: &mut [f32],
        n: f32,
    ) -> Result<(), SliceError<PowfErrorKind>> {
        slice_runtime(archmage::ScalarToken, src, dst, n, self.options)
    }

    /// Whole-slice `src[i]^n -> dst[i]` retaining or upgrading from `token`.
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
        n: f32,
    ) -> Result<(), SliceError<PowfErrorKind>> {
        slice_runtime(token, src, dst, n, self.options)
    }

    /// Evaluate `input^n` on any Magetypes `f32` vector (`f32x4`, `f32x8`, or `f32x16`).
    #[inline(always)]
    pub fn simd<V: SimdVector>(self, input: V, n: f32) -> Result<V, PowfError> {
        input.eval_powf_runtime(n, self.options)
    }
}
