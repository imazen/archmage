//! Typed profile entry points and runtime adapter for `expf`.
use super::{ExpfError, ExpfErrorKind, ExpfExecution, ExpfOptions, ExpfProfile, dispatch, graph};

struct Of<P>(core::marker::PhantomData<P>);
impl<P: ExpfProfile> graph::Kernel for Of<P> {
    const RANGED: bool = P::SPEC.input_bounds().is_some();
    const IN_LO: f32 = match P::SPEC.input_bounds() {
        Some((lo, _)) => lo,
        None => 0.0,
    };
    const IN_HI: f32 = match P::SPEC.input_bounds() {
        Some((_, hi)) => hi,
        None => 0.0,
    };
    const FULL: bool = P::SPEC.is_full();
    const LOWP: bool = P::SPEC.is_lowp();
    const ASSUME: bool = P::SPEC.assume_valid();
    const ELIDE_EXP2: bool = P::SPEC.elides_clamp();
    const ELIDE_EXP: bool = P::SPEC.elides_exp_clamp();
}

macro_rules! vector {
    ($run:ident, $profile:ident, $dispatch:ident, $float:ident, $bound:ident, $n:literal) => {
        #[inline(always)]
        pub(crate) fn $run<P: ExpfProfile, const OP: u8, T: magetypes::simd::backends::$bound>(
            token: T,
            input: magetypes::simd::generic::$float<T>,
        ) -> Result<magetypes::simd::generic::$float<T>, ExpfError> {
            const {
                P::SPEC.validate();
            }
            let caller = dispatch::select(token, ExpfExecution::Caller);
            if matches!(caller, dispatch::Plan::Software) {
                if P::SPEC.is_caller() {
                    return if P::SPEC.is_backend() {
                        graph::$profile::<T, Of<P>, OP, true>(token, input)
                    } else {
                        graph::$profile::<T, Of<P>, OP, false>(token, input)
                    };
                }
                let plan = dispatch::select(token, ExpfExecution::Automatic);
                let arr = input.to_array();
                let result = if P::SPEC.is_backend() {
                    dispatch::$dispatch::<Of<P>, OP, true>(plan, arr)
                } else {
                    dispatch::$dispatch::<Of<P>, OP, false>(plan, arr)
                }?;
                return Ok(magetypes::simd::generic::$float::from_array_t(
                    token, result,
                ));
            }
            if P::SPEC.is_backend() {
                graph::$profile::<T, Of<P>, OP, true>(token, input)
            } else {
                graph::$profile::<T, Of<P>, OP, false>(token, input)
            }
        }
    };
}
vector!(run4, profile4, run4_const, f32x4, F32x4Convert, 4);
vector!(run8, profile8, run8_const, f32x8, F32x8Convert, 8);
#[cfg(feature = "w512")]
vector!(run16, profile16, run16_const, f32x16, F32x16Convert, 16);

#[inline(always)]
fn scalar_with_token<P: ExpfProfile, const OP: u8, T: magetypes::simd::backends::F32x4Convert>(
    token: T,
    x: f32,
) -> Result<f32, ExpfError> {
    run4::<P, OP, T>(token, magetypes::simd::generic::f32x4::splat_t(token, x))
        .map(|v| v.to_array()[0])
        .map_err(|error| ExpfError {
            lane_mask: 1,
            ..error
        })
}

#[inline(always)]
fn array_with<P: ExpfProfile, const OP: u8, const N: usize>(
    inputs: [f32; N],
) -> Result<[f32; N], ExpfError> {
    use archmage::ScalarToken;
    use magetypes::simd::generic::{f32x4, f32x8};
    const {
        P::SPEC.validate();
    }
    if !matches!(N, 1 | 4 | 8) && !(cfg!(feature = "w512") && N == 16) {
        return Err(ExpfError {
            kind: ExpfErrorKind::UnsupportedWidth,
            lane_mask: 0,
        });
    }
    let mut output = [0.0; N];
    if N == 1 {
        let value = run4::<P, OP, _>(ScalarToken, f32x4::splat_t(ScalarToken, inputs[0])).map_err(
            |error| ExpfError {
                lane_mask: 1,
                ..error
            },
        )?;
        output[0] = value.to_array()[0];
    } else if N == 4 {
        let mut input = [0.0; 4];
        input.copy_from_slice(&inputs);
        output.copy_from_slice(
            &run4::<P, OP, _>(ScalarToken, f32x4::from_array_t(ScalarToken, input))?.to_array(),
        );
    } else if N == 8 {
        let mut input = [0.0; 8];
        input.copy_from_slice(&inputs);
        output.copy_from_slice(
            &run8::<P, OP, _>(ScalarToken, f32x8::from_array_t(ScalarToken, input))?.to_array(),
        );
    } else {
        #[cfg(feature = "w512")]
        if N == 16 {
            use magetypes::simd::generic::f32x16;
            let mut input = [0.0; 16];
            input.copy_from_slice(&inputs);
            output.copy_from_slice(
                &run16::<P, OP, _>(ScalarToken, f32x16::from_array_t(ScalarToken, input))?
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

/// Configurable scalar `exp2` under profile `P`.
#[inline(always)]
pub fn try_exp2_with<P: ExpfProfile>(x: f32) -> Result<f32, ExpfError> {
    try_exp2_array_with::<P, 1>([x]).map(|result| result[0])
}

/// Scalar `exp` (base e) under profile `P`.
#[inline(always)]
pub fn try_exp_with<P: ExpfProfile>(x: f32) -> Result<f32, ExpfError> {
    try_exp_array_with::<P, 1>([x]).map(|result| result[0])
}

/// Scalar `exp2` retaining the vector caller's feature proof under profile `P`.
#[inline(always)]
pub fn try_exp2_with_token<P: ExpfProfile, T: magetypes::simd::backends::F32x4Convert>(
    token: T,
    x: f32,
) -> Result<f32, ExpfError> {
    scalar_with_token::<P, 0, T>(token, x)
}

/// Scalar `exp` retaining the vector caller's feature proof under profile `P`.
#[inline(always)]
pub fn try_exp_with_token<P: ExpfProfile, T: magetypes::simd::backends::F32x4Convert>(
    token: T,
    x: f32,
) -> Result<f32, ExpfError> {
    scalar_with_token::<P, 1, T>(token, x)
}

/// Version-independent array boundary for `exp2` under profile `P`.
///
/// Supported widths are 1, 4, 8 and, with `w512`, 16.
#[inline(always)]
pub fn try_exp2_array_with<P: ExpfProfile, const N: usize>(
    inputs: [f32; N],
) -> Result<[f32; N], ExpfError> {
    array_with::<P, 0, N>(inputs)
}

/// Array `exp` (base e) under profile `P`.
#[inline(always)]
pub fn try_exp_array_with<P: ExpfProfile, const N: usize>(
    inputs: [f32; N],
) -> Result<[f32; N], ExpfError> {
    array_with::<P, 1, N>(inputs)
}

/// Runtime-options scalar `exp2`; same numerical contract as the profile API.
#[inline(always)]
pub fn try_exp2_runtime(x: f32, options: ExpfOptions) -> Result<f32, ExpfError> {
    try_exp2_array_runtime([x], options).map(|result| result[0])
}

/// Runtime-options scalar `exp`.
#[inline(always)]
pub fn try_exp_runtime(x: f32, options: ExpfOptions) -> Result<f32, ExpfError> {
    try_exp_array_runtime([x], options).map(|result| result[0])
}

#[cfg_attr(debug_assertions, inline(never))]
#[cfg_attr(not(debug_assertions), inline(always))]
fn array_runtime<const OP: u8, const N: usize>(
    inputs: [f32; N],
    options: ExpfOptions,
) -> Result<[f32; N], ExpfError> {
    let Some(options) = ExpfOptions::from_bits(options.bits()) else {
        return Err(ExpfError {
            kind: ExpfErrorKind::UnsupportedOptions,
            lane_mask: 0,
        });
    };
    if !matches!(N, 1 | 4 | 8) && !(cfg!(feature = "w512") && N == 16) {
        return Err(ExpfError {
            kind: ExpfErrorKind::UnsupportedWidth,
            lane_mask: 0,
        });
    }
    let spec = super::ExpfSpec::from_options(options);
    let plan = dispatch::select(archmage::ScalarToken, options.execution);
    let cfg = graph::KernelCfg::from_spec(spec);
    let backend = spec.is_backend();
    let mut output = [0.0; N];
    if N == 1 {
        let value =
            dispatch::run4_cfg(plan, [inputs[0]; 4], OP, backend, cfg).map_err(|error| {
                ExpfError {
                    lane_mask: 1,
                    ..error
                }
            })?;
        output[0] = value[0];
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

/// Runtime-options array `exp`.
#[inline(always)]
pub fn try_exp_array_runtime<const N: usize>(
    inputs: [f32; N],
    options: ExpfOptions,
) -> Result<[f32; N], ExpfError> {
    array_runtime::<1, N>(inputs, options)
}

/// Runtime-options array `exp2`.
#[inline(always)]
pub fn try_exp2_array_runtime<const N: usize>(
    inputs: [f32; N],
    options: ExpfOptions,
) -> Result<[f32; N], ExpfError> {
    array_runtime::<0, N>(inputs, options)
}

#[inline(always)]
fn slice<P: ExpfProfile, const OP: u8, T: archmage::IntoConcreteToken>(
    token: T,
    src: &[f32],
    dst: &mut [f32],
) -> Result<(), crate::SliceError<ExpfErrorKind>> {
    const {
        P::SPEC.validate();
    }
    assert_eq!(src.len(), dst.len(), "expf slice lengths differ");
    let plan = dispatch::select(
        token,
        if P::SPEC.is_caller() {
            ExpfExecution::Caller
        } else {
            ExpfExecution::Automatic
        },
    );
    let res = if P::SPEC.is_backend() {
        dispatch::run_slice_const::<Of<P>, OP, true>(plan, src, dst)
    } else {
        dispatch::run_slice_const::<Of<P>, OP, false>(plan, src, dst)
    };
    res.map_err(|(base, e)| crate::SliceError {
        kind: e.kind,
        index: base + e.lane_mask.trailing_zeros().min(7) as usize,
    })
}

/// Base-2 exponential (`exp2`) of every element of `src` into `dst` under profile `P`.
///
/// # Panics
///
/// Panics if `src.len() != dst.len()`.
#[inline]
pub fn try_exp2_slice_with<P: ExpfProfile>(
    src: &[f32],
    dst: &mut [f32],
) -> Result<(), crate::SliceError<ExpfErrorKind>> {
    slice::<P, 0, _>(archmage::ScalarToken, src, dst)
}

/// [`try_exp2_slice_with`] retaining the caller's feature proof.
///
/// # Panics
///
/// Panics if `src.len() != dst.len()`.
#[inline(always)]
pub fn try_exp2_slice_with_token<P: ExpfProfile, T: archmage::IntoConcreteToken>(
    token: T,
    src: &[f32],
    dst: &mut [f32],
) -> Result<(), crate::SliceError<ExpfErrorKind>> {
    slice::<P, 0, T>(token, src, dst)
}

/// Natural exponential (`exp`) of every element of `src` into `dst` under profile `P`.
///
/// # Panics
///
/// Panics if `src.len() != dst.len()`.
#[inline]
pub fn try_exp_slice_with<P: ExpfProfile>(
    src: &[f32],
    dst: &mut [f32],
) -> Result<(), crate::SliceError<ExpfErrorKind>> {
    slice::<P, 1, _>(archmage::ScalarToken, src, dst)
}

/// [`try_exp_slice_with`] retaining the caller's feature proof.
///
/// # Panics
///
/// Panics if `src.len() != dst.len()`.
#[inline(always)]
pub fn try_exp_slice_with_token<P: ExpfProfile, T: archmage::IntoConcreteToken>(
    token: T,
    src: &[f32],
    dst: &mut [f32],
) -> Result<(), crate::SliceError<ExpfErrorKind>> {
    slice::<P, 1, T>(token, src, dst)
}
