//! Const-specialized `Powf<CFG>` and runtime `PowfRuntime` entry points.
use super::{PowfError, PowfErrorKind, PowfOptions, dispatch, graph};
use crate::{Execution, SimdVector, SliceError};

pub(crate) struct Cfg<const CFG: u128>;

impl<const CFG: u128> graph::Kernel for Cfg<CFG> {
    const ZERO: bool = graph::KernelCfg::from_options(PowfOptions::decode_const(CFG)).zero;
    const SUBNORM: bool = graph::KernelCfg::from_options(PowfOptions::decode_const(CFG)).subnorm;
    const FLUSH_SUB: bool =
        graph::KernelCfg::from_options(PowfOptions::decode_const(CFG)).flush_sub;
    const RANGED: bool = graph::KernelCfg::from_options(PowfOptions::decode_const(CFG)).ranged;
    const ELIDE: bool = graph::KernelCfg::from_options(PowfOptions::decode_const(CFG)).elide;
    const CLAMP: bool = graph::KernelCfg::from_options(PowfOptions::decode_const(CFG)).clamp;
    const CLAMP_OOB: bool =
        graph::KernelCfg::from_options(PowfOptions::decode_const(CFG)).clamp_oob;
    const LOWP: bool = graph::KernelCfg::from_options(PowfOptions::decode_const(CFG)).lowp;
    const ASSUME: bool = graph::KernelCfg::from_options(PowfOptions::decode_const(CFG)).assume;
    const BASE_LO: f32 = graph::KernelCfg::from_options(PowfOptions::decode_const(CFG)).base_lo;
    const BASE_HI: f32 = graph::KernelCfg::from_options(PowfOptions::decode_const(CFG)).base_hi;
    const ALG_KIND: u8 = graph::KernelCfg::from_options(PowfOptions::decode_const(CFG)).alg_kind;
    const HAS_CONST_EXP: bool =
        graph::KernelCfg::from_options(PowfOptions::decode_const(CFG)).has_const_exp;
    const CONST_EXP: f32 = graph::KernelCfg::from_options(PowfOptions::decode_const(CFG)).const_exp;
}

#[inline(always)]
pub(super) fn prepare_exponent<const N: usize>(
    opts: PowfOptions,
    n: f32,
) -> Result<Option<f32>, PowfError> {
    if opts.assume_valid() || opts.is_full() {
        return Ok(Some(n));
    }
    if opts.clamp_oob() {
        if n.is_nan() {
            return Ok(None);
        }
        let (raw_lo, raw_hi) = opts.exponent_bounds().unwrap_or((-f32::MAX, f32::MAX));
        let (eff_lo, eff_hi) = if !opts.admits_subnormals() {
            let eff_lo = if raw_lo > 0.0 && raw_lo < f32::MIN_POSITIVE {
                f32::MIN_POSITIVE
            } else {
                raw_lo
            };
            let eff_hi = if raw_hi < 0.0 && raw_hi > -f32::MIN_POSITIVE {
                -f32::MIN_POSITIVE
            } else {
                raw_hi
            };
            (eff_lo, eff_hi)
        } else {
            (raw_lo, raw_hi)
        };
        let mut c = if n < eff_lo {
            eff_lo
        } else if n > eff_hi {
            eff_hi
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
    if bits >= 0x7f80_0000 {
        return Err(PowfError {
            kind: PowfErrorKind::InvalidExponent,
            lane_mask: ((1_u32 << N.min(16)) - 1) as u16,
        });
    }
    let mut eff_n = n;
    let is_subnorm = bits != 0 && bits < 0x0080_0000;
    if is_subnorm {
        if opts.flushes_subnormals() {
            match opts.exponent_bounds() {
                None => eff_n = 0.0,
                Some((lo, hi)) => {
                    if lo == 0.0 && hi == 0.0 {
                        eff_n = 0.0;
                    } else if lo <= 0.0 && hi >= 0.0 {
                        if !((lo == 0.0 && n < 0.0) || (hi == 0.0 && n > 0.0)) {
                            eff_n = 0.0;
                        }
                    } else if lo > 0.0 && n > 0.0 {
                        eff_n = lo.max(f32::MIN_POSITIVE);
                    } else if hi < 0.0 && n < 0.0 {
                        eff_n = hi.min(-f32::MIN_POSITIVE);
                    }
                }
            }
        } else if !opts.admits_subnormals() {
            return Err(PowfError {
                kind: PowfErrorKind::InvalidExponent,
                lane_mask: ((1_u32 << N.min(16)) - 1) as u16,
            });
        }
    }
    let in_range = match opts.exponent_bounds() {
        Some((lo, hi)) => eff_n >= lo && eff_n <= hi,
        None => true,
    };
    if !in_range {
        return Err(PowfError {
            kind: PowfErrorKind::InvalidExponent,
            lane_mask: ((1_u32 << N.min(16)) - 1) as u16,
        });
    }
    Ok(Some(eff_n))
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
    ($run:ident, $run_const:ident, $inner:ident, $profile:ident, $dispatch:ident, $float:ident, $bound:ident, $n:literal) => {
        #[inline(always)]
        fn $inner<const CFG: u128, T: magetypes::simd::backends::$bound>(
            opts: PowfOptions,
            token: T,
            input: magetypes::simd::generic::$float<T>,
            n: f32,
        ) -> Result<magetypes::simd::generic::$float<T>, PowfError> {
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

        #[inline(always)]
        pub(crate) fn $run<const CFG: u128, T: magetypes::simd::backends::$bound>(
            token: T,
            input: magetypes::simd::generic::$float<T>,
            n: f32,
        ) -> Result<magetypes::simd::generic::$float<T>, PowfError> {
            let opts = const {
                let o = PowfOptions::decode_const(CFG);
                o.validate();
                assert!(
                    !o.has_const_exponent(),
                    "CFG sets `.const_exponent(...)`; use `*_const` methods so the constant exponent is not ignored"
                );
                o
            };
            let Some(n) = prepare_exponent::<$n>(opts, n)? else {
                return Ok(magetypes::simd::generic::$float::splat_t(token, f32::NAN));
            };
            $inner::<CFG, T>(opts, token, input, n)
        }

        #[inline(always)]
        pub(crate) fn $run_const<const CFG: u128, T: magetypes::simd::backends::$bound>(
            token: T,
            input: magetypes::simd::generic::$float<T>,
        ) -> Result<magetypes::simd::generic::$float<T>, PowfError> {
            let (opts, const_exp) = const {
                let o = PowfOptions::decode_const(CFG);
                o.validate();
                let Some(e) = o.const_exponent_value() else {
                    panic!("`*_const` methods require `.const_exponent(n)` on PowfOptions");
                };
                (o, e)
            };
            $inner::<CFG, T>(opts, token, input, const_exp)
        }
    };
}
vector!(
    run4,
    run4_const,
    run4_inner,
    profile4,
    run4_const,
    f32x4,
    F32x4Convert,
    4
);
vector!(
    run8,
    run8_const,
    run8_inner,
    profile8,
    run8_const,
    f32x8,
    F32x8Convert,
    8
);
#[cfg(feature = "w512")]
vector!(
    run16,
    run16_const,
    run16_inner,
    profile16,
    run16_const,
    f32x16,
    F32x16Convert,
    16
);

#[cfg_attr(debug_assertions, inline(never))]
#[cfg_attr(not(debug_assertions), inline)]
fn runtime_eval<T: archmage::IntoConcreteToken, const N: usize>(
    token: T,
    bases: [f32; N],
    n: f32,
    options: PowfOptions,
) -> Result<[f32; N], PowfError> {
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
#[cfg_attr(not(debug_assertions), inline)]
pub(crate) fn runtime<T: archmage::IntoConcreteToken, const N: usize>(
    token: T,
    bases: [f32; N],
    n: f32,
    options: PowfOptions,
) -> Result<[f32; N], PowfError> {
    if !options.is_valid() || options.has_const_exponent() {
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
    runtime_eval::<T, N>(token, bases, n, options)
}

#[cfg_attr(debug_assertions, inline(never))]
#[cfg_attr(not(debug_assertions), inline)]
pub(crate) fn runtime_const_exp<T: archmage::IntoConcreteToken, const N: usize>(
    token: T,
    bases: [f32; N],
    options: PowfOptions,
) -> Result<[f32; N], PowfError> {
    if !options.is_valid() {
        return Err(PowfError {
            kind: PowfErrorKind::UnsupportedOptions,
            lane_mask: 0,
        });
    }
    let Some(n) = options.const_exponent_value() else {
        return Err(PowfError {
            kind: PowfErrorKind::UnsupportedOptions,
            lane_mask: 0,
        });
    };
    if !matches!(N, 1 | 4 | 8) && !(cfg!(feature = "w512") && N == 16) {
        return Err(PowfError {
            kind: PowfErrorKind::UnsupportedWidth,
            lane_mask: 0,
        });
    }
    runtime_eval::<T, N>(token, bases, n, options)
}

#[cfg_attr(debug_assertions, inline(never))]
#[cfg_attr(not(debug_assertions), inline)]
fn slice_runtime_eval<T: archmage::IntoConcreteToken>(
    token: T,
    src: &[f32],
    dst: &mut [f32],
    n: f32,
    options: PowfOptions,
) -> Result<(), SliceError<PowfErrorKind>> {
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

#[cfg_attr(debug_assertions, inline(never))]
#[cfg_attr(not(debug_assertions), inline)]
fn slice_runtime<T: archmage::IntoConcreteToken>(
    token: T,
    src: &[f32],
    dst: &mut [f32],
    n: f32,
    options: PowfOptions,
) -> Result<(), SliceError<PowfErrorKind>> {
    assert_eq!(src.len(), dst.len(), "powf slice lengths differ");
    if !options.is_valid() || options.has_const_exponent() {
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
    slice_runtime_eval(token, src, dst, n, options)
}

#[cfg_attr(debug_assertions, inline(never))]
#[cfg_attr(not(debug_assertions), inline)]
fn slice_runtime_const_exp<T: archmage::IntoConcreteToken>(
    token: T,
    src: &[f32],
    dst: &mut [f32],
    options: PowfOptions,
) -> Result<(), SliceError<PowfErrorKind>> {
    assert_eq!(src.len(), dst.len(), "powf slice lengths differ");
    if !options.is_valid() {
        return Err(SliceError {
            kind: PowfErrorKind::UnsupportedOptions,
            index: 0,
        });
    }
    let Some(n) = options.const_exponent_value() else {
        return Err(SliceError {
            kind: PowfErrorKind::UnsupportedOptions,
            index: 0,
        });
    };
    slice_runtime_eval(token, src, dst, n, options)
}

#[inline]
fn slice_const_eval<const CFG: u128>(
    opts: PowfOptions,
    token: impl archmage::IntoConcreteToken,
    src: &[f32],
    dst: &mut [f32],
    n: f32,
) -> Result<(), SliceError<PowfErrorKind>> {
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

/// Compile-time specialized `powf` operator configured by `const CFG: u128`
/// (produced via [`PowfOptions::bits()`]).
///
/// Use `<Powf>::scalar(base, n)` for default [`PowfOptions::new()`] options,
/// `Powf::<CFG>::scalar(base, n)` for a custom `const CFG: u128`, or
/// `Powf::<CFG>::scalar_const(base)` / `slice_const` when `CFG` sets
/// [`PowfOptions::const_exponent`]:
///
/// ```
/// use magemath::powf::{Powf, PowfDomain, PowfOptions};
/// assert_eq!(<Powf>::scalar(4.0, 0.5).unwrap(), 2.0);
///
/// const GAMMA: u128 = PowfOptions::new().gamma_domain().caller().bits();
/// assert_eq!(Powf::<GAMMA>::scalar(4.0, 0.5).unwrap(), 2.0);
///
/// const SQRT: u128 = PowfOptions::new().domain(PowfDomain::Unit).const_exponent(0.5).bits();
/// assert_eq!(Powf::<SQRT>::scalar_const(0.25).unwrap(), 0.5);
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

    /// Checked scalar `base^N` where `N` is configured via [`PowfOptions::const_exponent`].
    #[inline(always)]
    pub fn scalar_const(base: f32) -> Result<f32, PowfError> {
        Self::array_const::<1>([base]).map(|result| result[0])
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
                !opts.has_const_exponent(),
                "CFG sets `.const_exponent(...)`; call `Powf::<CFG>::scalar_const_infallible` instead"
            );
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

    /// Infallible scalar `base^N` where `N` is configured via [`PowfOptions::const_exponent`].
    #[inline(always)]
    pub fn scalar_const_infallible(base: f32) -> f32 {
        const {
            let opts = PowfOptions::decode_const(CFG);
            opts.validate();
            assert!(
                opts.has_const_exponent(),
                "Powf::<CFG>::scalar_const_infallible requires `.const_exponent(n)` on PowfOptions"
            );
            assert!(
                opts.is_infallible(),
                "Powf::<CFG>::scalar_const_infallible requires `.clamp_out_of_bounds()`, `.assume_in_bounds()` / `.unchecked()`, or `.admit_all_floats().highp()`; use `Powf::<CFG>::scalar_const` for checked options"
            );
        }
        match Self::scalar_const(base) {
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
            assert!(
                !o.has_const_exponent(),
                "CFG sets `.const_exponent(...)`; call `Powf::<CFG>::scalar_const_with_token` instead"
            );
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

    /// Scalar `base^N` retaining a vector caller's capability token where `N` is
    /// configured via [`PowfOptions::const_exponent`].
    #[inline(always)]
    pub fn scalar_const_with_token<T: magetypes::simd::backends::F32x4Convert>(
        token: T,
        base: f32,
    ) -> Result<f32, PowfError> {
        let (opts, const_exp) = const {
            let o = PowfOptions::decode_const(CFG);
            o.validate();
            let Some(e) = o.const_exponent_value() else {
                panic!("Powf::<CFG>::scalar_const_with_token requires `.const_exponent(n)`");
            };
            (o, e)
        };
        if opts.is_highp() {
            return super::highp_dispatch::for_token_cfg::<T, 1>(token, opts, [base], const_exp)
                .map(|v| v[0]);
        }
        run4_const::<CFG, T>(token, magetypes::simd::generic::f32x4::splat_t(token, base))
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
            assert!(
                !o.has_const_exponent(),
                "CFG sets `.const_exponent(...)`; call `Powf::<CFG>::array_const` instead"
            );
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

    /// Version-independent array boundary for widths `1`, `4`, `8` and (with `w512`) `16`
    /// where the exponent is configured via [`PowfOptions::const_exponent`].
    #[inline(always)]
    pub fn array_const<const N: usize>(bases: [f32; N]) -> Result<[f32; N], PowfError> {
        use archmage::ScalarToken;
        use magetypes::simd::generic::{f32x4, f32x8};
        let (opts, const_exp) = const {
            let o = PowfOptions::decode_const(CFG);
            o.validate();
            let Some(e) = o.const_exponent_value() else {
                panic!("Powf::<CFG>::array_const requires `.const_exponent(n)`");
            };
            (o, e)
        };
        if !matches!(N, 1 | 4 | 8) && !(cfg!(feature = "w512") && N == 16) {
            return Err(PowfError {
                kind: PowfErrorKind::UnsupportedWidth,
                lane_mask: 0,
            });
        }
        if opts.is_highp() {
            return super::highp_dispatch::for_token_cfg::<_, N>(
                ScalarToken,
                opts,
                bases,
                const_exp,
            );
        }
        let mut output = [0.0; N];
        if N == 1 {
            let value = run4_const::<CFG, _>(ScalarToken, f32x4::splat_t(ScalarToken, bases[0]))
                .map_err(|error| PowfError {
                    lane_mask: 1,
                    ..error
                })?;
            output[0] = value.to_array()[0];
        } else if N == 4 {
            let mut input = [0.0; 4];
            input.copy_from_slice(&bases);
            output.copy_from_slice(
                &run4_const::<CFG, _>(ScalarToken, f32x4::from_array_t(ScalarToken, input))?
                    .to_array(),
            );
        } else if N == 8 {
            let mut input = [0.0; 8];
            input.copy_from_slice(&bases);
            output.copy_from_slice(
                &run8_const::<CFG, _>(ScalarToken, f32x8::from_array_t(ScalarToken, input))?
                    .to_array(),
            );
        } else {
            #[cfg(feature = "w512")]
            if N == 16 {
                use magetypes::simd::generic::f32x16;
                let mut input = [0.0; 16];
                input.copy_from_slice(&bases);
                output.copy_from_slice(
                    &run16_const::<CFG, _>(ScalarToken, f32x16::from_array_t(ScalarToken, input))?
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

    /// Whole-slice `src[i]^N -> dst[i]` where `N` is configured via [`PowfOptions::const_exponent`].
    ///
    /// # Panics
    ///
    /// Panics if `src.len() != dst.len()`.
    #[inline]
    pub fn slice_const(src: &[f32], dst: &mut [f32]) -> Result<(), SliceError<PowfErrorKind>> {
        Self::slice_const_with_token(archmage::ScalarToken, src, dst)
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
            assert!(
                !o.has_const_exponent(),
                "CFG sets `.const_exponent(...)`; call `Powf::<CFG>::slice_const_with_token` instead"
            );
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
        slice_const_eval::<CFG>(opts, token, src, dst, n)
    }

    /// Whole-slice `src[i]^N -> dst[i]` starting from `token` where `N` is
    /// configured via [`PowfOptions::const_exponent`].
    ///
    /// # Panics
    ///
    /// Panics if `src.len() != dst.len()`.
    #[inline(always)]
    pub fn slice_const_with_token(
        token: impl archmage::IntoConcreteToken,
        src: &[f32],
        dst: &mut [f32],
    ) -> Result<(), SliceError<PowfErrorKind>> {
        let (opts, const_exp) = const {
            let o = PowfOptions::decode_const(CFG);
            o.validate();
            let Some(e) = o.const_exponent_value() else {
                panic!("Powf::<CFG>::slice_const_with_token requires `.const_exponent(n)`");
            };
            (o, e)
        };
        assert_eq!(src.len(), dst.len(), "powf slice lengths differ");
        slice_const_eval::<CFG>(opts, token, src, dst, const_exp)
    }

    /// Evaluate `input^n` on any Magetypes `f32` vector (`f32x4`, `f32x8`, or
    /// `f32x16`), reusing `input.token()`.
    #[inline(always)]
    pub fn simd<V: SimdVector>(input: V, n: f32) -> Result<V, PowfError> {
        input.eval_powf_const::<CFG>(n)
    }

    /// Evaluate `input^N` on any Magetypes `f32` vector where `N` is configured
    /// via [`PowfOptions::const_exponent`].
    #[inline(always)]
    pub fn simd_const<V: SimdVector>(input: V) -> Result<V, PowfError> {
        input.eval_powf_const_exp::<CFG>()
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

    /// Checked scalar `base^N` with runtime options that set [`PowfOptions::const_exponent`].
    #[inline(always)]
    pub fn scalar_const(self, base: f32) -> Result<f32, PowfError> {
        self.array_const::<1>([base]).map(|r| r[0])
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

    /// Scalar `base^N` retaining a vector caller's token with runtime options
    /// that set [`PowfOptions::const_exponent`].
    #[inline(always)]
    pub fn scalar_const_with_token<T: magetypes::simd::backends::F32x4Convert>(
        self,
        token: T,
        base: f32,
    ) -> Result<f32, PowfError> {
        runtime_const_exp::<T, 1>(token, [base], self.options).map(|r| r[0])
    }

    /// Array `bases^n` for widths `1`, `4`, `8` and (with `w512`) `16`.
    #[inline(always)]
    pub fn array<const N: usize>(self, bases: [f32; N], n: f32) -> Result<[f32; N], PowfError> {
        runtime::<_, N>(archmage::ScalarToken, bases, n, self.options)
    }

    /// Array `bases^N` for widths `1`, `4`, `8` and (with `w512`) `16` with
    /// runtime options that set [`PowfOptions::const_exponent`].
    #[inline(always)]
    pub fn array_const<const N: usize>(self, bases: [f32; N]) -> Result<[f32; N], PowfError> {
        runtime_const_exp::<_, N>(archmage::ScalarToken, bases, self.options)
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

    /// Whole-slice `src[i]^N -> dst[i]` with runtime options that set
    /// [`PowfOptions::const_exponent`].
    ///
    /// # Panics
    ///
    /// Panics if `src.len() != dst.len()`.
    #[inline]
    pub fn slice_const(
        self,
        src: &[f32],
        dst: &mut [f32],
    ) -> Result<(), SliceError<PowfErrorKind>> {
        slice_runtime_const_exp(archmage::ScalarToken, src, dst, self.options)
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

    /// Whole-slice `src[i]^N -> dst[i]` retaining or upgrading from `token`
    /// where `N` is configured via [`PowfOptions::const_exponent`].
    ///
    /// # Panics
    ///
    /// Panics if `src.len() != dst.len()`.
    #[inline]
    pub fn slice_const_with_token(
        self,
        token: impl archmage::IntoConcreteToken,
        src: &[f32],
        dst: &mut [f32],
    ) -> Result<(), SliceError<PowfErrorKind>> {
        slice_runtime_const_exp(token, src, dst, self.options)
    }

    /// Evaluate `input^n` on any Magetypes `f32` vector (`f32x4`, `f32x8`, or `f32x16`).
    #[inline(always)]
    pub fn simd<V: SimdVector>(self, input: V, n: f32) -> Result<V, PowfError> {
        input.eval_powf_runtime(n, self.options)
    }

    /// Evaluate `input^N` on any Magetypes `f32` vector where `N` is configured
    /// via [`PowfOptions::const_exponent`].
    #[inline(always)]
    pub fn simd_const<V: SimdVector>(self, input: V) -> Result<V, PowfError> {
        input.eval_powf_runtime_const_exp(self.options)
    }
}
