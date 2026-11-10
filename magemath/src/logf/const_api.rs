//! Caller-visible const flags, kept through the feature boundary and graph.
use super::{LogfError, LogfErrorKind, LogfExecution, LogfOptions, dispatch, graph};

macro_rules! vector {
    ($run:ident, $profile:ident, $dispatch:ident, $float:ident, $bound:ident, $n:literal) => {
        #[inline(always)]
        pub(crate) fn $run<T: magetypes::simd::backends::$bound, const OPTIONS: u32>(
            token: T,
            input: magetypes::simd::generic::$float<T>,
        ) -> Result<magetypes::simd::generic::$float<T>, LogfError> {
            const {
                assert!(LogfOptions::valid(OPTIONS), "invalid logf const options");
            }
            let caller = dispatch::select(token, LogfExecution::Caller);
            if matches!(caller, dispatch::Plan::Software) {
                if OPTIONS & LogfOptions::CALLER != 0 {
                    return graph::$profile::<T, OPTIONS>(token, input);
                }
                return dispatch::$dispatch::<OPTIONS>(
                    dispatch::select(token, LogfExecution::Automatic),
                    input.to_array(),
                )
                .map(|result| magetypes::simd::generic::$float::from_array_t(token, result));
            }
            graph::$profile::<T, OPTIONS>(token, input)
        }
    };
}
vector!(run4, profile4, run4_const, f32x4, F32x4Convert, 4);
vector!(run8, profile8, run8_const, f32x8, F32x8Convert, 8);
#[cfg(feature = "w512")]
vector!(run16, profile16, run16_const, f32x16, F32x16Convert, 16);

/// Configurable scalar log2 with integer const options.
#[inline(always)]
pub fn try_log2_with<const OPTIONS: u32>(x: f32) -> Result<f32, LogfError> {
    try_log2_array_with::<OPTIONS, 1>([x]).map(|result| result[0])
}
/// Scalar ln: `log2(x) * ln(2)`.
#[inline(always)]
pub fn try_ln_with<const OPTIONS: u32>(x: f32) -> Result<f32, LogfError> {
    try_log2_with::<OPTIONS>(x).map(|v| v * core::f32::consts::LN_2)
}
/// Scalar log10: `log2(x) * log10(2)`.
#[inline(always)]
pub fn try_log10_with<const OPTIONS: u32>(x: f32) -> Result<f32, LogfError> {
    try_log2_with::<OPTIONS>(x).map(|v| v * core::f32::consts::LOG10_2)
}
/// Per-input `u = 1+x` and fdlibm correction: `c = (x-(u-1))/u` for
/// `u < 2`, else `(1-(u-x))/u`. Zero for non-finite or non-positive `u` —
/// the kernel's domain handling owns those cases (reject or NaN/-inf/+inf).
#[inline(always)]
pub(super) fn log1p_parts(x: f32) -> (f32, f32) {
    let u = x + 1.0;
    let c = if !(u > 0.0 && u.is_finite()) {
        0.0
    } else if u >= 2.0 {
        (1.0 - (u - x)) / u
    } else {
        (x - (u - 1.0)) / u
    };
    (u, c)
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
    run4::<T, OPTIONS>(token, magetypes::simd::generic::f32x4::splat_t(token, x))
        .map(|v| v.to_array()[0])
        .map_err(|error| LogfError {
            lane_mask: 1,
            ..error
        })
}
/// Scalar ln retaining the vector caller's feature proof.
#[inline(always)]
pub fn try_ln_with_token<const OPTIONS: u32, T: magetypes::simd::backends::F32x4Convert>(
    token: T,
    x: f32,
) -> Result<f32, LogfError> {
    try_log2_with_token::<OPTIONS, T>(token, x).map(|v| v * core::f32::consts::LN_2)
}
/// Scalar log1p retaining the vector caller's feature proof.
#[inline(always)]
pub fn try_log1p_with_token<const OPTIONS: u32, T: magetypes::simd::backends::F32x4Convert>(
    token: T,
    x: f32,
) -> Result<f32, LogfError> {
    let (u, c) = log1p_parts(x);
    run4::<T, OPTIONS>(token, magetypes::simd::generic::f32x4::splat_t(token, u))
        .map(|v| log1p_finish(v.to_array()[0], c, x))
        .map_err(|error| LogfError {
            lane_mask: 1,
            ..error
        })
}
/// Scalar log10 retaining the vector caller's feature proof.
#[inline(always)]
pub fn try_log10_with_token<const OPTIONS: u32, T: magetypes::simd::backends::F32x4Convert>(
    token: T,
    x: f32,
) -> Result<f32, LogfError> {
    try_log2_with_token::<OPTIONS, T>(token, x).map(|v| v * core::f32::consts::LOG10_2)
}

/// Version-independent array boundary with caller-visible integer const flags.
///
/// Supported widths are 1, 4, 8 and, with `w512`, 16.
#[inline(always)]
pub fn try_log2_array_with<const OPTIONS: u32, const N: usize>(
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
        let value = run4::<_, OPTIONS>(ScalarToken, f32x4::splat_t(ScalarToken, inputs[0]))
            .map_err(|error| LogfError {
                lane_mask: 1,
                ..error
            })?;
        output[0] = value.to_array()[0];
    } else if N == 4 {
        let mut input = [0.0; 4];
        input.copy_from_slice(&inputs);
        output.copy_from_slice(
            &run4::<_, OPTIONS>(ScalarToken, f32x4::from_array_t(ScalarToken, input))?.to_array(),
        );
    } else if N == 8 {
        let mut input = [0.0; 8];
        input.copy_from_slice(&inputs);
        output.copy_from_slice(
            &run8::<_, OPTIONS>(ScalarToken, f32x8::from_array_t(ScalarToken, input))?.to_array(),
        );
    } else {
        #[cfg(feature = "w512")]
        if N == 16 {
            use magetypes::simd::generic::f32x16;
            let mut input = [0.0; 16];
            input.copy_from_slice(&inputs);
            output.copy_from_slice(
                &run16::<_, OPTIONS>(ScalarToken, f32x16::from_array_t(ScalarToken, input))?
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
/// Array ln.
#[inline(always)]
pub fn try_ln_array_with<const OPTIONS: u32, const N: usize>(
    inputs: [f32; N],
) -> Result<[f32; N], LogfError> {
    try_log2_array_with::<OPTIONS, N>(inputs).map(|o| o.map(|v| v * core::f32::consts::LN_2))
}
/// Array log10.
#[inline(always)]
pub fn try_log10_array_with<const OPTIONS: u32, const N: usize>(
    inputs: [f32; N],
) -> Result<[f32; N], LogfError> {
    try_log2_array_with::<OPTIONS, N>(inputs).map(|o| o.map(|v| v * core::f32::consts::LOG10_2))
}
/// Array log1p.
#[inline(always)]
pub fn try_log1p_array_with<const OPTIONS: u32, const N: usize>(
    inputs: [f32; N],
) -> Result<[f32; N], LogfError> {
    let mut u = [0.0f32; N];
    let mut c = [0.0f32; N];
    for i in 0..N {
        let (ui, ci) = log1p_parts(inputs[i]);
        u[i] = ui;
        c[i] = ci;
    }
    try_log2_array_with::<OPTIONS, N>(u).map(|o| {
        let mut out = o;
        for i in 0..N {
            out[i] = log1p_finish(o[i], c[i], inputs[i]);
        }
        out
    })
}
/// Runtime-options scalar log2.
#[inline(always)]
pub fn try_log2_runtime(x: f32, options: LogfOptions) -> Result<f32, LogfError> {
    try_log2_array_runtime([x], options).map(|result| result[0])
}
/// Runtime-options scalar ln.
#[inline(always)]
pub fn try_ln_runtime(x: f32, options: LogfOptions) -> Result<f32, LogfError> {
    try_log2_runtime(x, options).map(|v| v * core::f32::consts::LN_2)
}
/// Runtime-options scalar log1p.
#[inline(always)]
pub fn try_log1p_runtime(x: f32, options: LogfOptions) -> Result<f32, LogfError> {
    try_log1p_array_runtime::<1>([x], options).map(|r| r[0])
}
/// Runtime-options array log1p.
#[cfg_attr(debug_assertions, inline(never))]
#[cfg_attr(not(debug_assertions), inline(always))]
pub fn try_log1p_array_runtime<const N: usize>(
    inputs: [f32; N],
    options: LogfOptions,
) -> Result<[f32; N], LogfError> {
    let mut u = [0.0f32; N];
    let mut c = [0.0f32; N];
    for i in 0..N {
        let (ui, ci) = log1p_parts(inputs[i]);
        u[i] = ui;
        c[i] = ci;
    }
    try_log2_array_runtime::<N>(u, options).map(|o| {
        let mut out = o;
        for i in 0..N {
            out[i] = log1p_finish(o[i], c[i], inputs[i]);
        }
        out
    })
}
/// Runtime-options scalar log10.
#[inline(always)]
pub fn try_log10_runtime(x: f32, options: LogfOptions) -> Result<f32, LogfError> {
    try_log2_runtime(x, options).map(|v| v * core::f32::consts::LOG10_2)
}
/// Runtime-options array log2.
#[cfg_attr(debug_assertions, inline(never))]
#[cfg_attr(not(debug_assertions), inline(always))]
pub fn try_log2_array_runtime<const N: usize>(
    inputs: [f32; N],
    options: LogfOptions,
) -> Result<[f32; N], LogfError> {
    let Some(options) = LogfOptions::from_bits(options.bits()) else {
        return Err(LogfError {
            kind: LogfErrorKind::UnsupportedOptions,
            lane_mask: 0,
        });
    };
    let _ = options;
    token_array(archmage::ScalarToken, inputs, options.bits())
}

#[cfg_attr(debug_assertions, inline(never))]
#[cfg_attr(not(debug_assertions), inline(always))]
fn token_array<T: archmage::SimdToken, const N: usize>(
    token: T,
    inputs: [f32; N],
    options: u32,
) -> Result<[f32; N], LogfError> {
    macro_rules! dispatch {
        ($o:literal) => {
            try_log2_array_with::<$o, N>(inputs)
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
fn log1p_finish(log2: f32, correction: f32, input: f32) -> f32 {
    if input == 0.0 {
        return input;
    }
    #[cfg(feature = "std")]
    {
        log2.mul_add(core::f32::consts::LN_2, correction)
    }
    #[cfg(not(feature = "std"))]
    {
        crate::soft_fma::fmaf_soft_lanes([log2], [core::f32::consts::LN_2], [correction])[0]
    }
}
