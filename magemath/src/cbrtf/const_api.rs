//! Typed profile entry points and runtime adapter for `cbrtf`.
use super::{
    CbrtfError, CbrtfErrorKind, CbrtfExecution, CbrtfOptions, CbrtfProfile, dispatch, graph,
};

struct Of<P>(core::marker::PhantomData<P>);
impl<P: CbrtfProfile> graph::Kernel for Of<P> {
    const RANGED: bool = P::SPEC.input_bounds().is_some();
    const IN_LO: f32 = match P::SPEC.input_bounds() {
        Some((lo, _)) => lo,
        None => 0.0,
    };
    const IN_HI: f32 = match P::SPEC.input_bounds() {
        Some((_, hi)) => hi,
        None => 0.0,
    };
    const DENORM_POSSIBLE: bool = P::SPEC.denorm_possible();
    const FULL: bool = P::SPEC.is_full();
    const LOWP: bool = P::SPEC.is_lowp();
    const ASSUME: bool = P::SPEC.assume_valid();
}

macro_rules! vector {
    ($run:ident, $profile:ident, $dispatch:ident, $float:ident, $bound:ident) => {
        #[inline(always)]
        pub(crate) fn $run<P: CbrtfProfile, T: magetypes::simd::backends::$bound>(
            token: T,
            input: magetypes::simd::generic::$float<T>,
        ) -> Result<magetypes::simd::generic::$float<T>, CbrtfError> {
            const {
                P::SPEC.validate();
            }
            let caller = dispatch::select(token, CbrtfExecution::Caller);
            if matches!(caller, dispatch::Plan::Software) {
                if P::SPEC.is_caller() {
                    return if P::SPEC.is_backend() {
                        graph::$profile::<T, Of<P>, true>(token, input)
                    } else {
                        graph::$profile::<T, Of<P>, false>(token, input)
                    };
                }
                let plan = dispatch::select(token, CbrtfExecution::Automatic);
                let arr = input.to_array();
                let result = if P::SPEC.is_backend() {
                    dispatch::$dispatch::<Of<P>, true>(plan, arr)
                } else {
                    dispatch::$dispatch::<Of<P>, false>(plan, arr)
                }?;
                return Ok(magetypes::simd::generic::$float::from_array_t(
                    token, result,
                ));
            }
            if P::SPEC.is_backend() {
                graph::$profile::<T, Of<P>, true>(token, input)
            } else {
                graph::$profile::<T, Of<P>, false>(token, input)
            }
        }
    };
}

vector!(run4, profile4, run4_const, f32x4, F32x4Convert);
vector!(run8, profile8, run8_const, f32x8, F32x8Convert);
#[cfg(feature = "w512")]
vector!(run16, profile16, run16_const, f32x16, F32x16Convert);

/// Configurable scalar `cbrt` under profile `P`.
#[inline(always)]
pub fn try_cbrt_with<P: CbrtfProfile>(x: f32) -> Result<f32, CbrtfError> {
    try_cbrt_array_with::<P, 1>([x]).map(|result| result[0])
}

/// Scalar `cbrt` retaining the vector caller's feature proof under profile `P`.
#[inline(always)]
pub fn try_cbrt_with_token<P: CbrtfProfile, T: magetypes::simd::backends::F32x4Convert>(
    token: T,
    x: f32,
) -> Result<f32, CbrtfError> {
    run4::<P, T>(token, magetypes::simd::generic::f32x4::splat_t(token, x))
        .map(|v| v.to_array()[0])
        .map_err(|error| CbrtfError {
            lane_mask: 1,
            ..error
        })
}

/// Version-independent array boundary under profile `P`.
///
/// Supported widths are 1, 4, 8 and, with `w512`, 16. Every lane must pass
/// its domain; unsupported widths return `UnsupportedWidth` with mask zero.
#[inline(always)]
pub fn try_cbrt_array_with<P: CbrtfProfile, const N: usize>(
    inputs: [f32; N],
) -> Result<[f32; N], CbrtfError> {
    use archmage::ScalarToken;
    use magetypes::simd::generic::{f32x4, f32x8};
    const {
        P::SPEC.validate();
    }
    if !matches!(N, 1 | 4 | 8) && !(cfg!(feature = "w512") && N == 16) {
        return Err(CbrtfError {
            kind: CbrtfErrorKind::UnsupportedWidth,
            lane_mask: 0,
        });
    }
    let mut output = [0.0; N];
    if N == 1 {
        let value =
            run4::<P, _>(ScalarToken, f32x4::splat_t(ScalarToken, inputs[0])).map_err(|error| {
                CbrtfError {
                    lane_mask: 1,
                    ..error
                }
            })?;
        output[0] = value.to_array()[0];
    } else if N == 4 {
        let mut input = [0.0; 4];
        input.copy_from_slice(&inputs);
        output.copy_from_slice(
            &run4::<P, _>(ScalarToken, f32x4::from_array_t(ScalarToken, input))?.to_array(),
        );
    } else if N == 8 {
        let mut input = [0.0; 8];
        input.copy_from_slice(&inputs);
        output.copy_from_slice(
            &run8::<P, _>(ScalarToken, f32x8::from_array_t(ScalarToken, input))?.to_array(),
        );
    } else {
        #[cfg(feature = "w512")]
        if N == 16 {
            use magetypes::simd::generic::f32x16;
            let mut input = [0.0; 16];
            input.copy_from_slice(&inputs);
            output.copy_from_slice(
                &run16::<P, _>(ScalarToken, f32x16::from_array_t(ScalarToken, input))?.to_array(),
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

/// Runtime-options scalar `cbrt`; same numerical contract as the profile API.
#[inline(always)]
pub fn try_cbrt_runtime(x: f32, options: CbrtfOptions) -> Result<f32, CbrtfError> {
    try_cbrt_array_runtime([x], options).map(|result| result[0])
}

/// Runtime-options array `cbrt`.
#[cfg_attr(debug_assertions, inline(never))]
#[cfg_attr(not(debug_assertions), inline(always))]
pub fn try_cbrt_array_runtime<const N: usize>(
    inputs: [f32; N],
    options: CbrtfOptions,
) -> Result<[f32; N], CbrtfError> {
    let Some(options) = CbrtfOptions::from_bits(options.bits()) else {
        return Err(CbrtfError {
            kind: CbrtfErrorKind::UnsupportedOptions,
            lane_mask: 0,
        });
    };
    if !matches!(N, 1 | 4 | 8) && !(cfg!(feature = "w512") && N == 16) {
        return Err(CbrtfError {
            kind: CbrtfErrorKind::UnsupportedWidth,
            lane_mask: 0,
        });
    }
    let spec = super::CbrtfSpec::from_options(options);
    let plan = dispatch::select(archmage::ScalarToken, options.execution);
    let cfg = graph::KernelCfg::from_spec(spec);
    let backend = spec.is_backend();
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
fn slice<P: CbrtfProfile, T: archmage::IntoConcreteToken>(
    token: T,
    src: &[f32],
    dst: &mut [f32],
) -> Result<(), crate::SliceError<CbrtfErrorKind>> {
    const {
        P::SPEC.validate();
    }
    assert_eq!(src.len(), dst.len(), "cbrtf slice lengths differ");
    let plan = dispatch::select(
        token,
        if P::SPEC.is_caller() {
            CbrtfExecution::Caller
        } else {
            CbrtfExecution::Automatic
        },
    );
    let res = if P::SPEC.is_backend() {
        dispatch::run_slice_const::<Of<P>, true>(plan, src, dst)
    } else {
        dispatch::run_slice_const::<Of<P>, false>(plan, src, dst)
    };
    res.map_err(|(base, e)| crate::SliceError {
        kind: e.kind,
        index: base + e.lane_mask.trailing_zeros().min(7) as usize,
    })
}

/// Cube root (`cbrt`) of every element of `src` into `dst` under profile `P`,
/// selecting a backend once.
///
/// # Panics
///
/// Panics if `src.len() != dst.len()`.
#[inline]
pub fn try_cbrt_slice_with<P: CbrtfProfile>(
    src: &[f32],
    dst: &mut [f32],
) -> Result<(), crate::SliceError<CbrtfErrorKind>> {
    slice::<P, _>(archmage::ScalarToken, src, dst)
}

/// [`try_cbrt_slice_with`] retaining the caller's feature proof.
///
/// # Panics
///
/// Panics if `src.len() != dst.len()`.
#[inline(always)]
pub fn try_cbrt_slice_with_token<P: CbrtfProfile, T: archmage::IntoConcreteToken>(
    token: T,
    src: &[f32],
    dst: &mut [f32],
) -> Result<(), crate::SliceError<CbrtfErrorKind>> {
    slice::<P, T>(token, src, dst)
}
