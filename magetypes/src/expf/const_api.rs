//! Caller-visible const flags, kept through the feature boundary and graph.
use super::{ExpfError, ExpfErrorKind, ExpfExecution, ExpfOptions, dispatch, graph};

macro_rules! vector {
    ($run:ident, $profile:ident, $dispatch:ident, $float:ident, $bound:ident, $n:literal) => {
        #[inline(always)]
        pub(crate) fn $run<T: crate::simd::backends::$bound, const OPTIONS: u32>(
            token: T,
            input: crate::simd::generic::$float<T>,
        ) -> Result<crate::simd::generic::$float<T>, ExpfError> {
            const {
                assert!(ExpfOptions::valid(OPTIONS), "invalid expf const options");
            }
            let caller = dispatch::select(token, ExpfExecution::Caller);
            if matches!(caller, dispatch::Plan::Software) {
                if OPTIONS & ExpfOptions::CALLER != 0 {
                    return graph::$profile::<T, OPTIONS>(token, input);
                }
                return dispatch::$dispatch::<OPTIONS>(
                    dispatch::select(token, ExpfExecution::Automatic),
                    input.to_array(),
                )
                .map(|result| crate::simd::generic::$float::from_array_t(token, result));
            }
            graph::$profile::<T, OPTIONS>(token, input)
        }
    };
}
vector!(run4, profile4, run4_const, f32x4, F32x4Convert, 4);
vector!(run8, profile8, run8_const, f32x8, F32x8Convert, 8);
#[cfg(feature = "w512")]
vector!(run16, profile16, run16_const, f32x16, F32x16Convert, 16);

/// Configurable scalar exp2 with integer const options.
#[inline(always)]
pub fn try_exp2_with<const OPTIONS: u32>(x: f32) -> Result<f32, ExpfError> {
    try_exp2_array_with::<OPTIONS, 1>([x]).map(|result| result[0])
}

/// Scalar exp (base e) with integer const options: `exp2(x * log2(e))`.
#[inline(always)]
pub fn try_exp_with<const OPTIONS: u32>(x: f32) -> Result<f32, ExpfError> {
    try_exp2_with::<OPTIONS>(x * core::f32::consts::LOG2_E)
}

/// Scalar exp2 retaining the vector caller's feature proof.
#[inline(always)]
pub fn try_exp2_with_token<const OPTIONS: u32, T: crate::simd::backends::F32x4Convert>(
    token: T,
    x: f32,
) -> Result<f32, ExpfError> {
    run4::<T, OPTIONS>(token, crate::simd::generic::f32x4::splat_t(token, x))
        .map(|v| v.to_array()[0])
        .map_err(|error| ExpfError {
            lane_mask: 1,
            ..error
        })
}

/// Scalar exp retaining the vector caller's feature proof.
#[inline(always)]
pub fn try_exp_with_token<const OPTIONS: u32, T: crate::simd::backends::F32x4Convert>(
    token: T,
    x: f32,
) -> Result<f32, ExpfError> {
    try_exp2_with_token::<OPTIONS, T>(token, x * core::f32::consts::LOG2_E)
}

/// Version-independent array boundary with caller-visible integer const flags.
///
/// Supported widths are 1, 4, 8 and, with `w512`, 16. Every lane must pass
/// its domain; unsupported widths return `UnsupportedWidth` with mask zero.
#[inline(always)]
pub fn try_exp2_array_with<const OPTIONS: u32, const N: usize>(
    inputs: [f32; N],
) -> Result<[f32; N], ExpfError> {
    use crate::simd::generic::{f32x4, f32x8};
    use archmage::ScalarToken;
    const {
        assert!(ExpfOptions::valid(OPTIONS), "invalid expf const options");
    }
    if !matches!(N, 1 | 4 | 8) && !(cfg!(feature = "w512") && N == 16) {
        return Err(ExpfError {
            kind: ExpfErrorKind::UnsupportedWidth,
            lane_mask: 0,
        });
    }
    let mut output = [0.0; N];
    if N == 1 {
        let value = run4::<_, OPTIONS>(ScalarToken, f32x4::splat_t(ScalarToken, inputs[0]))
            .map_err(|error| ExpfError {
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
            use crate::simd::generic::f32x16;
            let mut input = [0.0; 16];
            input.copy_from_slice(&inputs);
            output.copy_from_slice(
                &run16::<_, OPTIONS>(ScalarToken, f32x16::from_array_t(ScalarToken, input))?
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

/// Array exp (base e): per-lane `exp2(x * log2(e))`.
#[inline(always)]
pub fn try_exp_array_with<const OPTIONS: u32, const N: usize>(
    inputs: [f32; N],
) -> Result<[f32; N], ExpfError> {
    try_exp2_array_with::<OPTIONS, N>(inputs.map(|x| x * core::f32::consts::LOG2_E))
}

/// Runtime-options scalar exp2; same numerical contract as the const API.
#[inline(always)]
pub fn try_exp2_runtime(x: f32, options: ExpfOptions) -> Result<f32, ExpfError> {
    try_exp2_array_runtime([x], options).map(|result| result[0])
}

/// Runtime-options scalar exp.
#[inline(always)]
pub fn try_exp_runtime(x: f32, options: ExpfOptions) -> Result<f32, ExpfError> {
    try_exp2_runtime(x * core::f32::consts::LOG2_E, options)
}

/// Runtime-options array exp2.
#[cfg_attr(debug_assertions, inline(never))]
#[cfg_attr(not(debug_assertions), inline(always))]
pub fn try_exp2_array_runtime<const N: usize>(
    inputs: [f32; N],
    options: ExpfOptions,
) -> Result<[f32; N], ExpfError> {
    let Some(options) = ExpfOptions::from_bits(options.bits()) else {
        return Err(ExpfError {
            kind: ExpfErrorKind::UnsupportedOptions,
            lane_mask: 0,
        });
    };
    let _ = options;
    token_array(archmage::ScalarToken, inputs, options.bits())
}

// Keep the runtime decoder and each selected graph in separate frames
// in unoptimized builds, mirroring the powf adapter.
#[cfg_attr(debug_assertions, inline(never))]
#[cfg_attr(not(debug_assertions), inline(always))]
fn token_array<T: archmage::SimdToken, const N: usize>(
    token: T,
    inputs: [f32; N],
    options: u32,
) -> Result<[f32; N], ExpfError> {
    macro_rules! dispatch {
        ($o:literal) => {
            try_exp2_array_with::<$o, N>(inputs)
        };
    }
    match options {
        o if o == ExpfOptions::DEFAULT => dispatch!(0),
        o if o == (ExpfOptions::DEFAULT | ExpfOptions::FULL) => dispatch!(1),
        o if o == (ExpfOptions::DEFAULT | ExpfOptions::LOWP) => dispatch!(8),
        o if o == (ExpfOptions::DEFAULT | ExpfOptions::LOWP | ExpfOptions::FULL) => dispatch!(9),
        o if o == (ExpfOptions::DEFAULT | ExpfOptions::BACKEND) => dispatch!(16),
        o if o == (ExpfOptions::DEFAULT | ExpfOptions::BACKEND | ExpfOptions::LOWP) => {
            dispatch!(24)
        }
        o if o == (ExpfOptions::DEFAULT | ExpfOptions::ASSUME) => dispatch!(32),
        o if o == (ExpfOptions::DEFAULT | ExpfOptions::ASSUME | ExpfOptions::LOWP) => {
            dispatch!(40)
        }
        o if o == (ExpfOptions::DEFAULT | ExpfOptions::ASSUME | ExpfOptions::BACKEND) => {
            dispatch!(48)
        }
        o if o
            == (ExpfOptions::DEFAULT
                | ExpfOptions::ASSUME
                | ExpfOptions::BACKEND
                | ExpfOptions::LOWP) =>
        {
            dispatch!(56)
        }
        _ => {
            let _ = token;
            Err(ExpfError {
                kind: ExpfErrorKind::UnsupportedOptions,
                lane_mask: 0,
            })
        }
    }
}

// Existing infallible methods use the same polynomial engine. Their clamp and
// edge policy is a private const configuration; no runtime decoder, validation
// Result, or portable-FMA cost enters this path.
macro_rules! approximate {
    ($name:ident, $profile:ident, $v:ident, $bound:ident) => {
        #[inline(always)]
        pub(crate) fn $name<
            T: crate::simd::backends::$bound,
            const LOWP: bool,
            const CHECKED: bool,
        >(
            token: T,
            input: crate::simd::generic::$v<T>,
        ) -> crate::simd::generic::$v<T> {
            let result = if LOWP {
                graph::$profile::<T, { ExpfOptions::ASSUME | ExpfOptions::LOWP }>(token, input)
            } else if CHECKED {
                graph::$profile::<
                    T,
                    { ExpfOptions::ASSUME | ExpfOptions::BACKEND | ExpfOptions::LEGACY },
                >(token, input)
            } else {
                graph::$profile::<T, { ExpfOptions::ASSUME | graph::CLAMP }>(token, input)
            };
            match result {
                Ok(value) => value,
                Err(_) => unreachable!("private assumed configuration cannot reject"),
            }
        }
    };
}
approximate!(approximate4, profile4, f32x4, F32x4Convert);
approximate!(approximate8, profile8, f32x8, F32x8Convert);
#[cfg(feature = "w512")]
approximate!(approximate16, profile16, f32x16, F32x16Convert);

// Expand the scalar composition in the caller's frame; the type predicate
// folds away and performs no runtime CPU detection.
macro_rules! approximate_exp2 {
    ($float:ident, $t:ty, $input:expr, $kernel:ident, $lowp:literal, $checked:literal, $scalar:expr) => {{
        if core::any::TypeId::of::<$t>() == core::any::TypeId::of::<archmage::ScalarToken>() {
            $scalar
        } else {
            crate::expf::const_api::$kernel::<$t, $lowp, $checked>($input.1, $input)
        }
    }};
}
pub(crate) use approximate_exp2;
