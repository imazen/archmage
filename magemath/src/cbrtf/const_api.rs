//! Caller-visible const flags for `cbrtf`, kept through the feature boundary and graph.
use super::{CbrtfError, CbrtfErrorKind, CbrtfExecution, CbrtfOptions, dispatch, graph};

macro_rules! vector {
    ($run:ident, $profile:ident, $dispatch:ident, $float:ident, $bound:ident) => {
        #[inline(always)]
        pub(crate) fn $run<T: magetypes::simd::backends::$bound, const OPTIONS: u32>(
            token: T,
            input: magetypes::simd::generic::$float<T>,
        ) -> Result<magetypes::simd::generic::$float<T>, CbrtfError> {
            const {
                assert!(CbrtfOptions::valid(OPTIONS), "invalid cbrtf const options");
            }
            let caller = dispatch::select(token, CbrtfExecution::Caller);
            if matches!(caller, dispatch::Plan::Software) {
                if OPTIONS & CbrtfOptions::CALLER != 0 {
                    return graph::$profile::<T, OPTIONS>(token, input);
                }
                return dispatch::$dispatch::<OPTIONS>(
                    dispatch::select(token, CbrtfExecution::Automatic),
                    input.to_array(),
                )
                .map(|result| magetypes::simd::generic::$float::from_array_t(token, result));
            }
            graph::$profile::<T, OPTIONS>(token, input)
        }
    };
}

vector!(run4, profile4, run4_const, f32x4, F32x4Convert);
vector!(run8, profile8, run8_const, f32x8, F32x8Convert);
#[cfg(feature = "w512")]
vector!(run16, profile16, run16_const, f32x16, F32x16Convert);

/// Configurable scalar `cbrt` with integer const options.
#[inline(always)]
pub fn try_cbrt_with<const OPTIONS: u32>(x: f32) -> Result<f32, CbrtfError> {
    try_cbrt_array_with::<OPTIONS, 1>([x]).map(|result| result[0])
}

/// Scalar `cbrt` retaining the vector caller's feature proof.
#[inline(always)]
pub fn try_cbrt_with_token<const OPTIONS: u32, T: magetypes::simd::backends::F32x4Convert>(
    token: T,
    x: f32,
) -> Result<f32, CbrtfError> {
    run4::<T, OPTIONS>(token, magetypes::simd::generic::f32x4::splat_t(token, x))
        .map(|v| v.to_array()[0])
        .map_err(|error| CbrtfError {
            lane_mask: 1,
            ..error
        })
}

/// Version-independent array boundary with caller-visible integer const flags.
///
/// Supported widths are 1, 4, 8 and, with `w512`, 16. Every lane must pass
/// its domain; unsupported widths return `UnsupportedWidth` with mask zero.
#[inline(always)]
pub fn try_cbrt_array_with<const OPTIONS: u32, const N: usize>(
    inputs: [f32; N],
) -> Result<[f32; N], CbrtfError> {
    use archmage::ScalarToken;
    use magetypes::simd::generic::{f32x4, f32x8};
    const {
        assert!(CbrtfOptions::valid(OPTIONS), "invalid cbrtf const options");
    }
    if !matches!(N, 1 | 4 | 8) && !(cfg!(feature = "w512") && N == 16) {
        return Err(CbrtfError {
            kind: CbrtfErrorKind::UnsupportedWidth,
            lane_mask: 0,
        });
    }
    let mut output = [0.0; N];
    if N == 1 {
        let value = run4::<_, OPTIONS>(ScalarToken, f32x4::splat_t(ScalarToken, inputs[0]))
            .map_err(|error| CbrtfError {
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
        return Err(CbrtfError {
            kind: CbrtfErrorKind::UnsupportedWidth,
            lane_mask: 0,
        });
    }
    Ok(output)
}

/// Runtime-options scalar `cbrt`; same numerical contract as the const API.
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
    token_array(archmage::ScalarToken, inputs, options.bits())
}

#[cfg_attr(debug_assertions, inline(never))]
#[cfg_attr(not(debug_assertions), inline(always))]
fn token_array<T: archmage::SimdToken, const N: usize>(
    token: T,
    inputs: [f32; N],
    options: u32,
) -> Result<[f32; N], CbrtfError> {
    macro_rules! dispatch {
        ($o:literal) => {
            try_cbrt_array_with::<$o, N>(inputs)
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
            Err(CbrtfError {
                kind: CbrtfErrorKind::UnsupportedOptions,
                lane_mask: 0,
            })
        }
    }
}
