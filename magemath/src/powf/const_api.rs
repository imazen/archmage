//! Typed profile entry points and runtime adapter for `powf`.
use super::{PowfError, PowfErrorKind, PowfExecution, PowfOptions, PowfProfile, dispatch, graph};

struct Of<P>(core::marker::PhantomData<P>);
impl<P: PowfProfile> graph::Kernel for Of<P> {
    const ZERO: bool = P::SPEC.admits_zero();
    const RANGED: bool = P::SPEC.base_bounds().is_some();
    const BASE_LO: f32 = match P::SPEC.base_bounds() {
        Some((lo, _)) => lo,
        None => f32::MIN_POSITIVE,
    };
    const BASE_HI: f32 = match P::SPEC.base_bounds() {
        Some((_, hi)) => hi,
        None => f32::MAX,
    };
    const ELIDE: bool = P::SPEC.elides_range_check();
    const CLAMP: bool = P::SPEC.should_clamp();
    const LOWP: bool = P::SPEC.is_lowp();
    const ASSUME: bool = P::SPEC.assume_valid();
}

pub(super) fn exponent_spec<const N: usize>(
    spec: super::PowfSpec,
    n: f32,
) -> Result<(), PowfError> {
    if spec.assume_valid() || spec.is_full() {
        return Ok(());
    }
    let bits = n.to_bits() & 0x7fff_ffff;
    let in_range = match spec.exponent_bounds() {
        Some((lo, hi)) => n >= lo && n <= hi,
        None => true,
    };
    if bits >= 0x7f80_0000 || (bits != 0 && bits < 0x0080_0000) || !in_range {
        return Err(PowfError {
            kind: PowfErrorKind::InvalidExponent,
            lane_mask: ((1_u32 << N.min(16)) - 1) as u16,
        });
    }
    Ok(())
}

#[inline(always)]
pub(super) fn exponent<P: PowfProfile, const N: usize>(n: f32) -> Result<(), PowfError> {
    const {
        P::SPEC.validate();
    }
    exponent_spec::<N>(P::SPEC, n)
}

#[inline(always)]
fn select<P: PowfProfile, T: archmage::IntoConcreteToken>(
    token: T,
    execution: PowfExecution,
) -> dispatch::Plan {
    if P::SPEC.is_highp() {
        if P::SPEC.is_backend() {
            dispatch::Plan::Software
        } else {
            dispatch::select_highp(token, execution)
        }
    } else if P::SPEC.is_backend() {
        dispatch::select_relaxed(token, execution)
    } else {
        dispatch::select(token, execution)
    }
}

/// Configurable scalar power under profile `P`.
///
/// ```
/// use magemath::{Caller, powf::{PowfOptions, PowfSpec, PowfProfile, presets::GammaMidp, try_powf_with}};
/// assert_eq!(try_powf_with::<Caller<GammaMidp>>(4.0, 0.5).unwrap(), 2.0);
/// ```
/// Invalid option encodings or inconsistent specs are compile errors:
/// ```compile_fail
/// let _ = magemath::powf::try_powf_with::<magemath::powf::Options<3>>(1.0, 2.4);
/// ```
/// ```compile_fail
/// let _ = magemath::powf::try_powf_with::<magemath::powf::Options<128>>(1.0, 2.4);
/// ```
/// ```compile_fail
/// struct Bad;
/// impl magemath::powf::PowfProfile for Bad {
///     const SPEC: magemath::powf::PowfSpec = magemath::powf::PowfSpec::new().full_domain();
/// }
/// let _ = magemath::powf::try_powf_with::<Bad>(1.0, 2.4);
/// ```
#[inline(always)]
pub fn try_powf_with<P: PowfProfile>(base: f32, n: f32) -> Result<f32, PowfError> {
    try_powf_array_with::<P, 1>([base], n).map(|result| result[0])
}

/// Scalar power retaining the vector caller's feature proof under profile `P`.
///
/// Use this for single values and slice tails inside a Magetypes kernel. Even
/// `Backend<Caller<P>>` evaluates the same graph as a vector with this token.
/// An error's lane mask is always one.
///
/// ```
/// use magemath::{Backend, Caller, powf::{presets::PositiveGammaMidp, try_powf_with_token}};
/// assert_eq!(
///     try_powf_with_token::<Backend<Caller<PositiveGammaMidp>>, _>(archmage::ScalarToken, 4.0, 0.5).unwrap(),
///     2.0
/// );
/// ```
#[inline(always)]
pub fn try_powf_with_token<P: PowfProfile, T: magetypes::simd::backends::F32x4Convert>(
    token: T,
    base: f32,
    n: f32,
) -> Result<f32, PowfError> {
    const {
        P::SPEC.validate();
    }
    if P::SPEC.is_highp() {
        exponent::<P, 1>(n)?;
        return super::highp_dispatch::for_token::<P, _, 1>(token, [base], n).map(|v| v[0]);
    }
    run4::<P, T>(
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

/// Version-independent array boundary under profile `P`.
///
/// Supported widths are 1, 4, 8 and, with `w512`, 16. Every lane must pass;
/// unsupported widths return `UnsupportedWidth` with mask zero. This shares
/// the numerical contract with [`super::try_powf_array_runtime`].
#[inline(always)]
pub fn try_powf_array_with<P: PowfProfile, const N: usize>(
    bases: [f32; N],
    n: f32,
) -> Result<[f32; N], PowfError> {
    use archmage::ScalarToken;
    use magetypes::simd::generic::{f32x4, f32x8};
    const {
        P::SPEC.validate();
    }
    if !matches!(N, 1 | 4 | 8) && !(cfg!(feature = "w512") && N == 16) {
        return Err(PowfError {
            kind: PowfErrorKind::UnsupportedWidth,
            lane_mask: 0,
        });
    }
    if P::SPEC.is_highp() {
        exponent::<P, N>(n)?;
        return super::highp_dispatch::for_token::<P, _, N>(ScalarToken, bases, n);
    }
    let mut output = [0.0; N];
    if N == 1 {
        let value = run4::<P, _>(ScalarToken, f32x4::splat_t(ScalarToken, bases[0]), n).map_err(
            |error| PowfError {
                lane_mask: 1,
                ..error
            },
        )?;
        output[0] = value.to_array()[0];
    } else if N == 4 {
        let mut input = [0.0; 4];
        input.copy_from_slice(&bases);
        output.copy_from_slice(
            &run4::<P, _>(ScalarToken, f32x4::from_array_t(ScalarToken, input), n)?.to_array(),
        );
    } else if N == 8 {
        let mut input = [0.0; 8];
        input.copy_from_slice(&bases);
        output.copy_from_slice(
            &run8::<P, _>(ScalarToken, f32x8::from_array_t(ScalarToken, input), n)?.to_array(),
        );
    } else {
        #[cfg(feature = "w512")]
        if N == 16 {
            use magetypes::simd::generic::f32x16;
            let mut input = [0.0; 16];
            input.copy_from_slice(&bases);
            output.copy_from_slice(
                &run16::<P, _>(ScalarToken, f32x16::from_array_t(ScalarToken, input), n)?
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

macro_rules! vector {
    ($run:ident, $profile:ident, $dispatch:ident, $float:ident, $bound:ident, $n:literal) => {
        #[inline(always)]
        pub(crate) fn $run<P: PowfProfile, T: magetypes::simd::backends::$bound>(
            token: T,
            input: magetypes::simd::generic::$float<T>,
            n: f32,
        ) -> Result<magetypes::simd::generic::$float<T>, PowfError> {
            exponent::<P, $n>(n)?;
            if P::SPEC.is_highp() {
                let result =
                    super::highp_dispatch::for_token::<P, _, $n>(token, input.to_array(), n)?;
                return Ok(magetypes::simd::generic::$float::from_array_t(
                    token, result,
                ));
            }
            let caller = select::<P, _>(token, PowfExecution::Caller);
            if matches!(caller, dispatch::Plan::Software) {
                if P::SPEC.is_caller() {
                    if P::SPEC.is_backend() {
                        return graph::$profile::<T, Of<P>, { graph::RELAXED }>(token, input, n);
                    }
                    return graph::$profile::<T, Of<P>, { graph::SOFTWARE }>(token, input, n);
                }
                let auto = select::<P, _>(token, PowfExecution::Automatic);
                let arr = input.to_array();
                let result = if P::SPEC.is_backend() {
                    dispatch::$dispatch::<Of<P>, true>(auto, arr, n)
                } else {
                    dispatch::$dispatch::<Of<P>, false>(auto, arr, n)
                }?;
                return Ok(magetypes::simd::generic::$float::from_array_t(
                    token, result,
                ));
            }
            #[cfg(target_arch = "wasm32")]
            if matches!(caller, dispatch::Plan::Wasm(_, false)) {
                return graph::$profile::<T, Of<P>, { graph::WASM_SOFTWARE }>(token, input, n);
            }
            graph::$profile::<T, Of<P>, { graph::NATIVE }>(token, input, n)
        }
    };
}
vector!(run4, profile4, run4_const, f32x4, F32x4Convert, 4);
vector!(run8, profile8, run8_const, f32x8, F32x8Convert, 8);
#[cfg(feature = "w512")]
vector!(run16, profile16, run16_const, f32x16, F32x16Convert, 16);

#[cfg_attr(debug_assertions, inline(never))]
#[cfg_attr(not(debug_assertions), inline(always))]
pub(super) fn runtime<T: archmage::IntoConcreteToken, const N: usize>(
    token: T,
    bases: [f32; N],
    n: f32,
    options: PowfOptions,
) -> Result<[f32; N], PowfError> {
    let Some(options) = PowfOptions::from_bits(options.bits()) else {
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
    let spec = super::PowfSpec::from_options(options);
    exponent_spec::<N>(spec, n)?;
    if spec.is_highp() {
        return super::highp_dispatch::for_token_cfg::<T, N>(token, spec, bases, n);
    }
    let exec = if spec.is_caller() {
        PowfExecution::Caller
    } else {
        PowfExecution::Automatic
    };
    let backend = spec.is_backend();
    let plan = if backend {
        dispatch::select_relaxed(token, exec)
    } else {
        dispatch::select(token, exec)
    };
    let cfg = graph::KernelCfg::from_spec(spec);
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

/// Power of every element of `src` into `dst` under profile `P`, selecting a
/// backend once.
///
/// Results are the same bits the array and vector entries produce for each
/// element. Checked profiles stop at the first rejected vector and report the
/// index of its first rejected element; `dst` is then unspecified. An invalid
/// exponent reports index 0.
///
/// ```
/// use magemath::powf::{presets::GammaMidp, try_powf_slice_with};
/// let src = [0.0, 0.25, 0.5, 1.0, 2.0];
/// let mut dst = [0.0; 5];
/// try_powf_slice_with::<GammaMidp>(&src, &mut dst, 2.0).unwrap();
/// assert_eq!(dst[0], 0.0);
/// assert_eq!(dst[3], 1.0);
/// ```
///
/// # Panics
///
/// Panics if `src.len() != dst.len()`.
#[inline]
pub fn try_powf_slice_with<P: PowfProfile>(
    src: &[f32],
    dst: &mut [f32],
    n: f32,
) -> Result<(), crate::SliceError<PowfErrorKind>> {
    try_powf_slice_with_token::<P, _>(archmage::ScalarToken, src, dst, n)
}

/// [`try_powf_slice_with`] retaining the caller's feature proof: with
/// `Execution::Caller` the loop runs on `token`'s tier.
///
/// # Panics
///
/// Panics if `src.len() != dst.len()`.
#[inline(always)]
pub fn try_powf_slice_with_token<P: PowfProfile, T: archmage::IntoConcreteToken>(
    token: T,
    src: &[f32],
    dst: &mut [f32],
    n: f32,
) -> Result<(), crate::SliceError<PowfErrorKind>> {
    const {
        P::SPEC.validate();
    }
    assert_eq!(src.len(), dst.len(), "powf slice lengths differ");
    let first = |base: usize, e: PowfError| crate::SliceError {
        kind: e.kind,
        index: base + e.lane_mask.trailing_zeros().min(7) as usize,
    };
    exponent::<P, 1>(n).map_err(|e| crate::SliceError {
        kind: e.kind,
        index: 0,
    })?;
    if P::SPEC.is_highp() {
        return super::highp_dispatch::for_token_slice::<P, T>(token, src, dst, n)
            .map_err(|(base, e)| first(base, e));
    }
    let plan = select::<P, _>(
        token,
        if P::SPEC.is_caller() {
            PowfExecution::Caller
        } else {
            PowfExecution::Automatic
        },
    );
    let res = if P::SPEC.is_backend() {
        dispatch::run_slice::<Of<P>, true>(plan, src, dst, n)
    } else {
        dispatch::run_slice::<Of<P>, false>(plan, src, dst, n)
    };
    res.map_err(|(base, e)| first(base, e))
}
