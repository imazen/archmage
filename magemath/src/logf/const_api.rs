//! Typed profile entry points and runtime adapter for `logf`.
use super::{LogfError, LogfErrorKind, LogfExecution, LogfOptions, LogfProfile, dispatch, graph};

struct Of<P>(core::marker::PhantomData<P>);
impl<P: LogfProfile> graph::Kernel for Of<P> {
    const RANGED: bool = P::SPEC.input_bounds().is_some();
    const IN_LO: f32 = match P::SPEC.input_bounds() {
        Some((lo, _)) => lo,
        None => 0.0,
    };
    const IN_HI: f32 = match P::SPEC.input_bounds() {
        Some((_, hi)) => hi,
        None => 0.0,
    };
    const SPECIALS: bool = P::SPEC.is_specials();
    const LOWP: bool = P::SPEC.is_lowp();
    const ASSUME: bool = P::SPEC.assume_valid();
}

macro_rules! vector {
    ($run:ident, $profile:ident, $dispatch:ident, $float:ident, $bound:ident, $n:literal) => {
        #[inline(always)]
        pub(crate) fn $run<P: LogfProfile, const OP: u8, T: magetypes::simd::backends::$bound>(
            token: T,
            input: magetypes::simd::generic::$float<T>,
        ) -> Result<magetypes::simd::generic::$float<T>, LogfError> {
            const {
                P::SPEC.validate_op(OP);
            }
            let caller = dispatch::select(token, LogfExecution::Caller);
            if matches!(caller, dispatch::Plan::Software) {
                if P::SPEC.is_caller() {
                    return if P::SPEC.is_backend() {
                        graph::$profile::<T, Of<P>, OP, true>(token, input)
                    } else {
                        graph::$profile::<T, Of<P>, OP, false>(token, input)
                    };
                }
                let plan = dispatch::select(token, LogfExecution::Automatic);
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
fn scalar_with_token<P: LogfProfile, const OP: u8, T: magetypes::simd::backends::F32x4Convert>(
    token: T,
    x: f32,
) -> Result<f32, LogfError> {
    run4::<P, OP, T>(token, magetypes::simd::generic::f32x4::splat_t(token, x))
        .map(|v| v.to_array()[0])
        .map_err(|error| LogfError {
            lane_mask: 1,
            ..error
        })
}

#[inline(always)]
fn array_with<P: LogfProfile, const OP: u8, const N: usize>(
    inputs: [f32; N],
) -> Result<[f32; N], LogfError> {
    use archmage::ScalarToken;
    use magetypes::simd::generic::{f32x4, f32x8};
    const {
        P::SPEC.validate_op(OP);
    }
    if !matches!(N, 1 | 4 | 8) && !(cfg!(feature = "w512") && N == 16) {
        return Err(LogfError {
            kind: LogfErrorKind::UnsupportedWidth,
            lane_mask: 0,
        });
    }
    let mut output = [0.0; N];
    if N == 1 {
        let value = run4::<P, OP, _>(ScalarToken, f32x4::splat_t(ScalarToken, inputs[0])).map_err(
            |error| LogfError {
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
        return Err(LogfError {
            kind: LogfErrorKind::UnsupportedWidth,
            lane_mask: 0,
        });
    }
    Ok(output)
}

/// Configurable scalar `log2` under profile `P`.
#[inline(always)]
pub fn try_log2_with<P: LogfProfile>(x: f32) -> Result<f32, LogfError> {
    try_log2_array_with::<P, 1>([x]).map(|result| result[0])
}
/// Scalar `ln`: `log2(x) * ln(2)` under profile `P`.
#[inline(always)]
pub fn try_ln_with<P: LogfProfile>(x: f32) -> Result<f32, LogfError> {
    try_ln_array_with::<P, 1>([x]).map(|result| result[0])
}
/// Scalar `log10`: `log2(x) * log10(2)` under profile `P`.
#[inline(always)]
pub fn try_log10_with<P: LogfProfile>(x: f32) -> Result<f32, LogfError> {
    try_log10_array_with::<P, 1>([x]).map(|result| result[0])
}
/// Scalar `log1p`: `ln(1+x)`, with a correction for rounding `1+x`, under profile `P`.
#[inline(always)]
pub fn try_log1p_with<P: LogfProfile>(x: f32) -> Result<f32, LogfError> {
    try_log1p_array_with::<P, 1>([x]).map(|result| result[0])
}
/// Scalar `log2` retaining the vector caller's feature proof under profile `P`.
#[inline(always)]
pub fn try_log2_with_token<P: LogfProfile, T: magetypes::simd::backends::F32x4Convert>(
    token: T,
    x: f32,
) -> Result<f32, LogfError> {
    scalar_with_token::<P, 0, T>(token, x)
}
/// Scalar `ln` retaining the vector caller's feature proof under profile `P`.
#[inline(always)]
pub fn try_ln_with_token<P: LogfProfile, T: magetypes::simd::backends::F32x4Convert>(
    token: T,
    x: f32,
) -> Result<f32, LogfError> {
    scalar_with_token::<P, 1, T>(token, x)
}
/// Scalar `log10` retaining the vector caller's feature proof under profile `P`.
#[inline(always)]
pub fn try_log10_with_token<P: LogfProfile, T: magetypes::simd::backends::F32x4Convert>(
    token: T,
    x: f32,
) -> Result<f32, LogfError> {
    scalar_with_token::<P, 2, T>(token, x)
}
/// Scalar `log1p` retaining the vector caller's feature proof under profile `P`.
#[inline(always)]
pub fn try_log1p_with_token<P: LogfProfile, T: magetypes::simd::backends::F32x4Convert>(
    token: T,
    x: f32,
) -> Result<f32, LogfError> {
    scalar_with_token::<P, 3, T>(token, x)
}

/// Version-independent array boundary for `log2` under profile `P`.
///
/// Supported widths are 1, 4, 8 and, with `w512`, 16.
#[inline(always)]
pub fn try_log2_array_with<P: LogfProfile, const N: usize>(
    inputs: [f32; N],
) -> Result<[f32; N], LogfError> {
    array_with::<P, 0, N>(inputs)
}
/// Array `ln` under profile `P`.
#[inline(always)]
pub fn try_ln_array_with<P: LogfProfile, const N: usize>(
    inputs: [f32; N],
) -> Result<[f32; N], LogfError> {
    array_with::<P, 1, N>(inputs)
}
/// Array `log10` under profile `P`.
#[inline(always)]
pub fn try_log10_array_with<P: LogfProfile, const N: usize>(
    inputs: [f32; N],
) -> Result<[f32; N], LogfError> {
    array_with::<P, 2, N>(inputs)
}
/// Array `log1p` under profile `P`.
#[inline(always)]
pub fn try_log1p_array_with<P: LogfProfile, const N: usize>(
    inputs: [f32; N],
) -> Result<[f32; N], LogfError> {
    array_with::<P, 3, N>(inputs)
}

/// Runtime-options scalar `log2`.
#[inline(always)]
pub fn try_log2_runtime(x: f32, options: LogfOptions) -> Result<f32, LogfError> {
    try_log2_array_runtime([x], options).map(|result| result[0])
}
/// Runtime-options scalar `ln`.
#[inline(always)]
pub fn try_ln_runtime(x: f32, options: LogfOptions) -> Result<f32, LogfError> {
    try_ln_array_runtime([x], options).map(|result| result[0])
}
/// Runtime-options scalar `log1p`.
#[inline(always)]
pub fn try_log1p_runtime(x: f32, options: LogfOptions) -> Result<f32, LogfError> {
    try_log1p_array_runtime::<1>([x], options).map(|r| r[0])
}
/// Runtime-options scalar `log10`.
#[inline(always)]
pub fn try_log10_runtime(x: f32, options: LogfOptions) -> Result<f32, LogfError> {
    try_log10_array_runtime([x], options).map(|result| result[0])
}

#[cfg_attr(debug_assertions, inline(never))]
#[cfg_attr(not(debug_assertions), inline(always))]
fn array_runtime<const OP: u8, const N: usize>(
    inputs: [f32; N],
    options: LogfOptions,
) -> Result<[f32; N], LogfError> {
    let Some(options) = LogfOptions::from_bits(options.bits()) else {
        return Err(LogfError {
            kind: LogfErrorKind::UnsupportedOptions,
            lane_mask: 0,
        });
    };
    if !matches!(N, 1 | 4 | 8) && !(cfg!(feature = "w512") && N == 16) {
        return Err(LogfError {
            kind: LogfErrorKind::UnsupportedWidth,
            lane_mask: 0,
        });
    }
    let spec = super::LogfSpec::from_options(options);
    let plan = dispatch::select(archmage::ScalarToken, options.execution);
    let cfg = graph::KernelCfg::from_spec(spec);
    let backend = spec.is_backend();
    let mut output = [0.0; N];
    if N == 1 {
        let value =
            dispatch::run4_cfg(plan, [inputs[0]; 4], OP, backend, cfg).map_err(|error| {
                LogfError {
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

/// Runtime-options array `log2`.
#[inline(always)]
pub fn try_log2_array_runtime<const N: usize>(
    inputs: [f32; N],
    options: LogfOptions,
) -> Result<[f32; N], LogfError> {
    array_runtime::<0, N>(inputs, options)
}
/// Runtime-options array `ln`.
#[inline(always)]
pub fn try_ln_array_runtime<const N: usize>(
    inputs: [f32; N],
    options: LogfOptions,
) -> Result<[f32; N], LogfError> {
    array_runtime::<1, N>(inputs, options)
}
/// Runtime-options array `log10`.
#[inline(always)]
pub fn try_log10_array_runtime<const N: usize>(
    inputs: [f32; N],
    options: LogfOptions,
) -> Result<[f32; N], LogfError> {
    array_runtime::<2, N>(inputs, options)
}
/// Runtime-options array `log1p`.
#[inline(always)]
pub fn try_log1p_array_runtime<const N: usize>(
    inputs: [f32; N],
    options: LogfOptions,
) -> Result<[f32; N], LogfError> {
    array_runtime::<3, N>(inputs, options)
}

#[inline(always)]
fn slice<P: LogfProfile, const OP: u8, T: archmage::IntoConcreteToken>(
    token: T,
    src: &[f32],
    dst: &mut [f32],
) -> Result<(), crate::SliceError<LogfErrorKind>> {
    const {
        P::SPEC.validate_op(OP);
    }
    assert_eq!(src.len(), dst.len(), "logf slice lengths differ");
    let plan = dispatch::select(
        token,
        if P::SPEC.is_caller() {
            LogfExecution::Caller
        } else {
            LogfExecution::Automatic
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

/// Base-2 logarithm (`log2`) of every element of `src` into `dst` under profile `P`.
///
/// # Panics
///
/// Panics if `src.len() != dst.len()`.
#[inline]
pub fn try_log2_slice_with<P: LogfProfile>(
    src: &[f32],
    dst: &mut [f32],
) -> Result<(), crate::SliceError<LogfErrorKind>> {
    slice::<P, 0, _>(archmage::ScalarToken, src, dst)
}

/// [`try_log2_slice_with`] retaining the caller's feature proof.
///
/// # Panics
///
/// Panics if `src.len() != dst.len()`.
#[inline(always)]
pub fn try_log2_slice_with_token<P: LogfProfile, T: archmage::IntoConcreteToken>(
    token: T,
    src: &[f32],
    dst: &mut [f32],
) -> Result<(), crate::SliceError<LogfErrorKind>> {
    slice::<P, 0, T>(token, src, dst)
}

/// Natural logarithm (`ln`) of every element of `src` into `dst` under profile `P`.
///
/// # Panics
///
/// Panics if `src.len() != dst.len()`.
#[inline]
pub fn try_ln_slice_with<P: LogfProfile>(
    src: &[f32],
    dst: &mut [f32],
) -> Result<(), crate::SliceError<LogfErrorKind>> {
    slice::<P, 1, _>(archmage::ScalarToken, src, dst)
}

/// [`try_ln_slice_with`] retaining the caller's feature proof.
///
/// # Panics
///
/// Panics if `src.len() != dst.len()`.
#[inline(always)]
pub fn try_ln_slice_with_token<P: LogfProfile, T: archmage::IntoConcreteToken>(
    token: T,
    src: &[f32],
    dst: &mut [f32],
) -> Result<(), crate::SliceError<LogfErrorKind>> {
    slice::<P, 1, T>(token, src, dst)
}

/// Base-10 logarithm (`log10`) of every element of `src` into `dst` under profile `P`.
///
/// # Panics
///
/// Panics if `src.len() != dst.len()`.
#[inline]
pub fn try_log10_slice_with<P: LogfProfile>(
    src: &[f32],
    dst: &mut [f32],
) -> Result<(), crate::SliceError<LogfErrorKind>> {
    slice::<P, 2, _>(archmage::ScalarToken, src, dst)
}

/// [`try_log10_slice_with`] retaining the caller's feature proof.
///
/// # Panics
///
/// Panics if `src.len() != dst.len()`.
#[inline(always)]
pub fn try_log10_slice_with_token<P: LogfProfile, T: archmage::IntoConcreteToken>(
    token: T,
    src: &[f32],
    dst: &mut [f32],
) -> Result<(), crate::SliceError<LogfErrorKind>> {
    slice::<P, 2, T>(token, src, dst)
}

/// Shifted natural logarithm (`ln(1 + x)`) of every element of `src` into
/// `dst` under profile `P`.
///
/// # Panics
///
/// Panics if `src.len() != dst.len()`.
#[inline]
pub fn try_log1p_slice_with<P: LogfProfile>(
    src: &[f32],
    dst: &mut [f32],
) -> Result<(), crate::SliceError<LogfErrorKind>> {
    slice::<P, 3, _>(archmage::ScalarToken, src, dst)
}

/// [`try_log1p_slice_with`] retaining the caller's feature proof.
///
/// # Panics
///
/// Panics if `src.len() != dst.len()`.
#[inline(always)]
pub fn try_log1p_slice_with_token<P: LogfProfile, T: archmage::IntoConcreteToken>(
    token: T,
    src: &[f32],
    dst: &mut [f32],
) -> Result<(), crate::SliceError<LogfErrorKind>> {
    slice::<P, 3, T>(token, src, dst)
}
