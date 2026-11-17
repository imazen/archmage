//! Caller-visible const flags, kept through the feature boundary and graph.
use super::{LogfError, LogfErrorKind, LogfExecution, LogfOptions, dispatch, graph};

macro_rules! vector {
    ($run:ident, $profile:ident, $dispatch:ident, $float:ident, $bound:ident, $n:literal) => {
        #[inline(always)]
        pub(crate) fn $run<
            T: magetypes::simd::backends::$bound,
            const OP: u8,
            const OPTIONS: u32,
        >(
            token: T,
            input: magetypes::simd::generic::$float<T>,
        ) -> Result<magetypes::simd::generic::$float<T>, LogfError> {
            const {
                assert!(LogfOptions::valid(OPTIONS), "invalid logf const options");
            }
            let caller = dispatch::select(token, LogfExecution::Caller);
            if matches!(caller, dispatch::Plan::Software) {
                if OPTIONS & LogfOptions::CALLER != 0 {
                    return graph::$profile::<T, OP, OPTIONS>(token, input);
                }
                return dispatch::$dispatch::<OP, OPTIONS>(
                    dispatch::select(token, LogfExecution::Automatic),
                    input.to_array(),
                )
                .map(|result| magetypes::simd::generic::$float::from_array_t(token, result));
            }
            graph::$profile::<T, OP, OPTIONS>(token, input)
        }
    };
}
vector!(run4, profile4, run4_const, f32x4, F32x4Convert, 4);
vector!(run8, profile8, run8_const, f32x8, F32x8Convert, 8);
#[cfg(feature = "w512")]
vector!(run16, profile16, run16_const, f32x16, F32x16Convert, 16);

#[inline(always)]
fn scalar_with_token<
    const OP: u8,
    const OPTIONS: u32,
    T: magetypes::simd::backends::F32x4Convert,
>(
    token: T,
    x: f32,
) -> Result<f32, LogfError> {
    run4::<T, OP, OPTIONS>(token, magetypes::simd::generic::f32x4::splat_t(token, x))
        .map(|v| v.to_array()[0])
        .map_err(|error| LogfError {
            lane_mask: 1,
            ..error
        })
}

#[inline(always)]
fn array_with<const OP: u8, const OPTIONS: u32, const N: usize>(
    inputs: [f32; N],
) -> Result<[f32; N], LogfError> {
    use archmage::ScalarToken;
    use magetypes::simd::generic::{f32x4, f32x8};
    const {
        assert!(LogfOptions::valid(OPTIONS), "invalid logf const options");
    }
    if !matches!(N, 1 | 4 | 8) && !(cfg!(feature = "w512") && N == 16) {
        return Err(LogfError {
            kind: LogfErrorKind::UnsupportedWidth,
            lane_mask: 0,
        });
    }
    let mut output = [0.0; N];
    if N == 1 {
        let value = run4::<_, OP, OPTIONS>(ScalarToken, f32x4::splat_t(ScalarToken, inputs[0]))
            .map_err(|error| LogfError {
                lane_mask: 1,
                ..error
            })?;
        output[0] = value.to_array()[0];
    } else if N == 4 {
        let mut input = [0.0; 4];
        input.copy_from_slice(&inputs);
        output.copy_from_slice(
            &run4::<_, OP, OPTIONS>(ScalarToken, f32x4::from_array_t(ScalarToken, input))?
                .to_array(),
        );
    } else if N == 8 {
        let mut input = [0.0; 8];
        input.copy_from_slice(&inputs);
        output.copy_from_slice(
            &run8::<_, OP, OPTIONS>(ScalarToken, f32x8::from_array_t(ScalarToken, input))?
                .to_array(),
        );
    } else {
        #[cfg(feature = "w512")]
        if N == 16 {
            use magetypes::simd::generic::f32x16;
            let mut input = [0.0; 16];
            input.copy_from_slice(&inputs);
            output.copy_from_slice(
                &run16::<_, OP, OPTIONS>(ScalarToken, f32x16::from_array_t(ScalarToken, input))?
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

/// Configurable scalar log2 with integer const options.
#[inline(always)]
pub fn try_log2_with<const OPTIONS: u32>(x: f32) -> Result<f32, LogfError> {
    try_log2_array_with::<OPTIONS, 1>([x]).map(|result| result[0])
}
/// Scalar ln: `log2(x) * ln(2)`.
#[inline(always)]
pub fn try_ln_with<const OPTIONS: u32>(x: f32) -> Result<f32, LogfError> {
    try_ln_array_with::<OPTIONS, 1>([x]).map(|result| result[0])
}
/// Scalar log10: `log2(x) * log10(2)`.
#[inline(always)]
pub fn try_log10_with<const OPTIONS: u32>(x: f32) -> Result<f32, LogfError> {
    try_log10_array_with::<OPTIONS, 1>([x]).map(|result| result[0])
}
/// Scalar log1p: `ln(1+x)`, with a correction for rounding `1+x`.
/// MIDP handles tiny inputs; LOWP retains the log2 fit bias near zero and
/// can return a negative approximation for a tiny positive input.
/// Domain checks apply to `u = 1+x` — the Positive domain rejects x <= -1.
#[inline(always)]
pub fn try_log1p_with<const OPTIONS: u32>(x: f32) -> Result<f32, LogfError> {
    try_log1p_array_with::<OPTIONS, 1>([x]).map(|result| result[0])
}
/// Scalar log2 retaining the vector caller's feature proof.
#[inline(always)]
pub fn try_log2_with_token<const OPTIONS: u32, T: magetypes::simd::backends::F32x4Convert>(
    token: T,
    x: f32,
) -> Result<f32, LogfError> {
    scalar_with_token::<0, OPTIONS, T>(token, x)
}
/// Scalar ln retaining the vector caller's feature proof.
#[inline(always)]
pub fn try_ln_with_token<const OPTIONS: u32, T: magetypes::simd::backends::F32x4Convert>(
    token: T,
    x: f32,
) -> Result<f32, LogfError> {
    scalar_with_token::<1, OPTIONS, T>(token, x)
}
/// Scalar log10 retaining the vector caller's feature proof.
#[inline(always)]
pub fn try_log10_with_token<const OPTIONS: u32, T: magetypes::simd::backends::F32x4Convert>(
    token: T,
    x: f32,
) -> Result<f32, LogfError> {
    scalar_with_token::<2, OPTIONS, T>(token, x)
}
/// Scalar log1p retaining the vector caller's feature proof.
#[inline(always)]
pub fn try_log1p_with_token<const OPTIONS: u32, T: magetypes::simd::backends::F32x4Convert>(
    token: T,
    x: f32,
) -> Result<f32, LogfError> {
    scalar_with_token::<3, OPTIONS, T>(token, x)
}

/// Version-independent array boundary with caller-visible integer const flags.
///
/// Supported widths are 1, 4, 8 and, with `w512`, 16.
#[inline(always)]
pub fn try_log2_array_with<const OPTIONS: u32, const N: usize>(
    inputs: [f32; N],
) -> Result<[f32; N], LogfError> {
    array_with::<0, OPTIONS, N>(inputs)
}
/// Array ln.
#[inline(always)]
pub fn try_ln_array_with<const OPTIONS: u32, const N: usize>(
    inputs: [f32; N],
) -> Result<[f32; N], LogfError> {
    array_with::<1, OPTIONS, N>(inputs)
}
/// Array log10.
#[inline(always)]
pub fn try_log10_array_with<const OPTIONS: u32, const N: usize>(
    inputs: [f32; N],
) -> Result<[f32; N], LogfError> {
    array_with::<2, OPTIONS, N>(inputs)
}
/// Array log1p.
#[inline(always)]
pub fn try_log1p_array_with<const OPTIONS: u32, const N: usize>(
    inputs: [f32; N],
) -> Result<[f32; N], LogfError> {
    array_with::<3, OPTIONS, N>(inputs)
}

/// Runtime-options scalar log2.
#[inline(always)]
pub fn try_log2_runtime(x: f32, options: LogfOptions) -> Result<f32, LogfError> {
    try_log2_array_runtime([x], options).map(|result| result[0])
}
/// Runtime-options scalar ln.
#[inline(always)]
pub fn try_ln_runtime(x: f32, options: LogfOptions) -> Result<f32, LogfError> {
    try_ln_array_runtime([x], options).map(|result| result[0])
}
/// Runtime-options scalar log1p.
#[inline(always)]
pub fn try_log1p_runtime(x: f32, options: LogfOptions) -> Result<f32, LogfError> {
    try_log1p_array_runtime::<1>([x], options).map(|r| r[0])
}
/// Runtime-options scalar log10.
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
    token_array::<_, OP, N>(archmage::ScalarToken, inputs, options.bits())
}

/// Runtime-options array log2.
#[inline(always)]
pub fn try_log2_array_runtime<const N: usize>(
    inputs: [f32; N],
    options: LogfOptions,
) -> Result<[f32; N], LogfError> {
    array_runtime::<0, N>(inputs, options)
}
/// Runtime-options array ln.
#[inline(always)]
pub fn try_ln_array_runtime<const N: usize>(
    inputs: [f32; N],
    options: LogfOptions,
) -> Result<[f32; N], LogfError> {
    array_runtime::<1, N>(inputs, options)
}
/// Runtime-options array log10.
#[inline(always)]
pub fn try_log10_array_runtime<const N: usize>(
    inputs: [f32; N],
    options: LogfOptions,
) -> Result<[f32; N], LogfError> {
    array_runtime::<2, N>(inputs, options)
}
/// Runtime-options array log1p.
#[inline(always)]
pub fn try_log1p_array_runtime<const N: usize>(
    inputs: [f32; N],
    options: LogfOptions,
) -> Result<[f32; N], LogfError> {
    array_runtime::<3, N>(inputs, options)
}

#[cfg_attr(debug_assertions, inline(never))]
#[cfg_attr(not(debug_assertions), inline(always))]
fn token_array<T: archmage::IntoConcreteToken, const OP: u8, const N: usize>(
    token: T,
    inputs: [f32; N],
    options: u32,
) -> Result<[f32; N], LogfError> {
    macro_rules! dispatch {
        ($o:literal) => {
            array_with::<OP, $o, N>(inputs)
        };
    }
    match options {
        0 => dispatch!(0),
        1 => dispatch!(1),
        4 => dispatch!(4),
        5 => dispatch!(5),
        8 => dispatch!(8),
        9 => dispatch!(9),
        12 => dispatch!(12),
        13 => dispatch!(13),
        16 => dispatch!(16),
        17 => dispatch!(17),
        20 => dispatch!(20),
        21 => dispatch!(21),
        24 => dispatch!(24),
        25 => dispatch!(25),
        28 => dispatch!(28),
        29 => dispatch!(29),
        32 => dispatch!(32),
        33 => dispatch!(33),
        36 => dispatch!(36),
        37 => dispatch!(37),
        40 => dispatch!(40),
        41 => dispatch!(41),
        44 => dispatch!(44),
        45 => dispatch!(45),
        48 => dispatch!(48),
        49 => dispatch!(49),
        52 => dispatch!(52),
        53 => dispatch!(53),
        56 => dispatch!(56),
        57 => dispatch!(57),
        60 => dispatch!(60),
        61 => dispatch!(61),
        _ => {
            let _ = token;
            Err(LogfError {
                kind: LogfErrorKind::UnsupportedOptions,
                lane_mask: 0,
            })
        }
    }
}

#[inline(always)]
fn slice<const OP: u8, const OPTIONS: u32, T: archmage::IntoConcreteToken>(
    token: T,
    src: &[f32],
    dst: &mut [f32],
) -> Result<(), crate::SliceError<LogfErrorKind>> {
    const {
        assert!(LogfOptions::valid(OPTIONS), "invalid logf const options");
    }
    assert_eq!(src.len(), dst.len(), "logf slice lengths differ");
    let plan = dispatch::select(
        token,
        if OPTIONS & LogfOptions::CALLER != 0 {
            LogfExecution::Caller
        } else {
            LogfExecution::Automatic
        },
    );
    dispatch::run_slice_const::<OP, OPTIONS>(plan, src, dst).map_err(|(base, e)| {
        crate::SliceError {
            kind: e.kind,
            index: base + e.lane_mask.trailing_zeros().min(7) as usize,
        }
    })
}

/// Base-2 logarithm (`log2`) of every element of `src` into `dst`, selecting a
/// backend once.
///
/// # Panics
///
/// Panics if `src.len() != dst.len()`.
#[inline]
pub fn try_log2_slice_with<const OPTIONS: u32>(
    src: &[f32],
    dst: &mut [f32],
) -> Result<(), crate::SliceError<LogfErrorKind>> {
    slice::<0, OPTIONS, _>(archmage::ScalarToken, src, dst)
}

/// [`try_log2_slice_with`] retaining the caller's feature proof.
///
/// # Panics
///
/// Panics if `src.len() != dst.len()`.
#[inline(always)]
pub fn try_log2_slice_with_token<const OPTIONS: u32, T: archmage::IntoConcreteToken>(
    token: T,
    src: &[f32],
    dst: &mut [f32],
) -> Result<(), crate::SliceError<LogfErrorKind>> {
    slice::<0, OPTIONS, T>(token, src, dst)
}

/// Natural logarithm (`ln`) of every element of `src` into `dst`, selecting a
/// backend once.
///
/// # Panics
///
/// Panics if `src.len() != dst.len()`.
#[inline]
pub fn try_ln_slice_with<const OPTIONS: u32>(
    src: &[f32],
    dst: &mut [f32],
) -> Result<(), crate::SliceError<LogfErrorKind>> {
    slice::<1, OPTIONS, _>(archmage::ScalarToken, src, dst)
}

/// [`try_ln_slice_with`] retaining the caller's feature proof.
///
/// # Panics
///
/// Panics if `src.len() != dst.len()`.
#[inline(always)]
pub fn try_ln_slice_with_token<const OPTIONS: u32, T: archmage::IntoConcreteToken>(
    token: T,
    src: &[f32],
    dst: &mut [f32],
) -> Result<(), crate::SliceError<LogfErrorKind>> {
    slice::<1, OPTIONS, T>(token, src, dst)
}

/// Base-10 logarithm (`log10`) of every element of `src` into `dst`, selecting
/// a backend once.
///
/// # Panics
///
/// Panics if `src.len() != dst.len()`.
#[inline]
pub fn try_log10_slice_with<const OPTIONS: u32>(
    src: &[f32],
    dst: &mut [f32],
) -> Result<(), crate::SliceError<LogfErrorKind>> {
    slice::<2, OPTIONS, _>(archmage::ScalarToken, src, dst)
}

/// [`try_log10_slice_with`] retaining the caller's feature proof.
///
/// # Panics
///
/// Panics if `src.len() != dst.len()`.
#[inline(always)]
pub fn try_log10_slice_with_token<const OPTIONS: u32, T: archmage::IntoConcreteToken>(
    token: T,
    src: &[f32],
    dst: &mut [f32],
) -> Result<(), crate::SliceError<LogfErrorKind>> {
    slice::<2, OPTIONS, T>(token, src, dst)
}

/// Shifted natural logarithm (`ln(1 + x)`) of every element of `src` into
/// `dst`, selecting a backend once.
///
/// # Panics
///
/// Panics if `src.len() != dst.len()`.
#[inline]
pub fn try_log1p_slice_with<const OPTIONS: u32>(
    src: &[f32],
    dst: &mut [f32],
) -> Result<(), crate::SliceError<LogfErrorKind>> {
    slice::<3, OPTIONS, _>(archmage::ScalarToken, src, dst)
}

/// [`try_log1p_slice_with`] retaining the caller's feature proof.
///
/// # Panics
///
/// Panics if `src.len() != dst.len()`.
#[inline(always)]
pub fn try_log1p_slice_with_token<const OPTIONS: u32, T: archmage::IntoConcreteToken>(
    token: T,
    src: &[f32],
    dst: &mut [f32],
) -> Result<(), crate::SliceError<LogfErrorKind>> {
    slice::<3, OPTIONS, T>(token, src, dst)
}
