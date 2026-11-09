//! Caller-visible const flags, kept through the feature boundary and graph.
use super::{PowfError, PowfErrorKind, PowfExecution, PowfOptions, dispatch, graph};

#[inline(always)]
pub(super) fn exponent<const OPTIONS: u32, const N: usize>(n: f32) -> Result<(), PowfError> {
    const {
        assert!(PowfOptions::valid(OPTIONS), "invalid powf const options");
    }
    if OPTIONS & PowfOptions::ASSUME != 0 || OPTIONS & 3 == 3 {
        return Ok(());
    }
    let bits = n.to_bits() & 0x7fff_ffff;
    if bits >= 0x7f80_0000
        || (bits != 0 && bits < 0x0080_0000)
        || (OPTIONS & 3 == PowfOptions::GAMMA && !(0.0..=4.0).contains(&n))
    {
        return Err(PowfError {
            kind: PowfErrorKind::InvalidExponent,
            lane_mask: ((1_u32 << N.min(16)) - 1) as u16,
        });
    }
    Ok(())
}

/// Configurable scalar power with integer const options.
///
/// Build the configuration in const evaluation, then supply its integer bits:
/// ```
/// use magetypes::powf::{PowfOptions, PowfDomain, PowfExecution, try_powf_with};
/// const OPTIONS: u32 = PowfOptions::new()
///     .with_domain(PowfDomain::Gamma)
///     .with_execution(PowfExecution::Caller)
///     .bits();
/// assert_eq!(try_powf_with::<OPTIONS>(4.0, 0.5).unwrap(), 2.0);
/// ```
/// Invalid encodings are compile errors:
/// ```compile_fail
/// let _ = magetypes::powf::try_powf_with::<3>(1.0, 2.4);
/// ```
/// ```compile_fail
/// let _ = magetypes::powf::try_powf_with::<128>(1.0, 2.4);
/// ```
/// Options monomorphize through the caller, backend selection and arithmetic
/// graph. Runtime input checks retain the whole-result rejection contract.
#[inline(always)]
pub fn try_powf_with<const OPTIONS: u32>(base: f32, n: f32) -> Result<f32, PowfError> {
    try_powf_array_with::<OPTIONS, 1>([base], n).map(|result| result[0])
}

/// Version-independent array boundary with caller-visible integer const flags.
///
/// Supported widths are 1, 4, 8 and, with `w512`, 16. Every lane must pass;
/// unsupported widths return `UnsupportedWidth` with mask zero. This shares
/// the numerical contract with [`super::try_powf_array_runtime`].
#[inline(always)]
pub fn try_powf_array_with<const OPTIONS: u32, const N: usize>(
    bases: [f32; N],
    n: f32,
) -> Result<[f32; N], PowfError> {
    use crate::simd::generic::{f32x4, f32x8};
    const {
        assert!(PowfOptions::valid(OPTIONS), "invalid powf const options");
    }
    #[cfg(feature = "std")]
    if OPTIONS & PowfOptions::HIGHP != 0 {
        if !matches!(N, 1 | 4 | 8) && !(cfg!(feature="w512") && N==16) {
            return Err(PowfError {kind:PowfErrorKind::UnsupportedWidth,lane_mask:0});
        }
        exponent::<OPTIONS,N>(n)?;
        return super::highp::run::<OPTIONS,N>(bases,n);
    }
    use archmage::ScalarToken;
    let mut output = [0.0; N];
    if N == 1 {
        let value = run4::<_, OPTIONS>(ScalarToken, f32x4::splat_t(ScalarToken, bases[0]), n)
            .map_err(|error| PowfError {
                lane_mask: 1,
                ..error
            })?;
        output[0] = value.to_array()[0];
    } else if N == 4 {
        let mut input = [0.0; 4];
        input.copy_from_slice(&bases);
        output.copy_from_slice(
            &run4::<_, OPTIONS>(ScalarToken, f32x4::from_array_t(ScalarToken, input), n)?
                .to_array(),
        );
    } else if N == 8 {
        let mut input = [0.0; 8];
        input.copy_from_slice(&bases);
        output.copy_from_slice(
            &run8::<_, OPTIONS>(ScalarToken, f32x8::from_array_t(ScalarToken, input), n)?
                .to_array(),
        );
    } else {
        #[cfg(feature = "w512")]
        if N == 16 {
            use crate::simd::generic::f32x16;
            let mut input = [0.0; 16];
            input.copy_from_slice(&bases);
            output.copy_from_slice(
                &run16::<_, OPTIONS>(ScalarToken, f32x16::from_array_t(ScalarToken, input), n)?
                    .to_array(),
            );
            return Ok(output);
        }
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
        pub(crate) fn $run<T: crate::simd::backends::$bound, const OPTIONS: u32>(
            token: T,
            input: crate::simd::generic::$float<T>,
            n: f32,
        ) -> Result<crate::simd::generic::$float<T>, PowfError> {
            exponent::<OPTIONS, $n>(n)?;
            #[cfg(feature = "std")]
            if OPTIONS & PowfOptions::HIGHP != 0 {
                let result = super::highp::run::<OPTIONS, $n>(input.to_array(), n)?;
                return Ok(crate::simd::generic::$float::from_array_t(token, result));
            }
            let caller = select::<T, OPTIONS>(token, PowfExecution::Caller);
            if matches!(caller, dispatch::Plan::Software) {
                if OPTIONS & PowfOptions::CALLER != 0 {
                    if OPTIONS & PowfOptions::BACKEND != 0 {
                        return graph::$profile::<T, { graph::RELAXED }, OPTIONS>(token, input, n);
                    }
                    return graph::$profile::<T, { graph::SOFTWARE }, OPTIONS>(token, input, n);
                }
                return dispatch::$dispatch::<OPTIONS>(
                    select::<T, OPTIONS>(token, PowfExecution::Automatic),
                    input.to_array(),
                    n,
                )
                .map(|result| crate::simd::generic::$float::from_array_t(token, result));
            }
            #[cfg(target_arch = "wasm32")]
            if matches!(caller, dispatch::Plan::Wasm(_, false)) {
                return graph::$profile::<T, { graph::WASM_SOFTWARE }, OPTIONS>(token, input, n);
            }
            // BitExact reaches this point only with a native FMA proof or
            // a screened relaxed-WASM provider. BackendDependent also admits
            // strict/nonfusing WASM. Keep vectors in the caller's feature region.
            graph::$profile::<T, { graph::NATIVE }, OPTIONS>(token, input, n)
        }
    };
}
vector!(run4, profile4, run4_const, f32x4, F32x4Convert, 4);
vector!(run8, profile8, run8_const, f32x8, F32x8Convert, 8);
#[cfg(feature = "w512")]
vector!(run16, profile16, run16_const, f32x16, F32x16Convert, 16);

#[inline(always)]
fn select<T: archmage::SimdToken, const OPTIONS: u32>(
    token: T,
    execution: PowfExecution,
) -> dispatch::Plan {
    if OPTIONS & PowfOptions::BACKEND != 0 {
        dispatch::select_relaxed(token, execution)
    } else {
        dispatch::select(token, execution)
    }
}

// The runtime adapter shares literal const instantiations with the hot API.
// Every policy arm is outside the polynomial; no dynamic flags enter Horner steps.
#[inline(always)]
fn token_array<T: archmage::SimdToken, const OPTIONS: u32, const N: usize>(
    token: T,
    bases: [f32; N],
    n: f32,
) -> Result<[f32; N], PowfError> {
    if !matches!(N, 1 | 4 | 8) && !(cfg!(feature = "w512") && N == 16) {
        return Err(PowfError {
            kind: PowfErrorKind::UnsupportedWidth,
            lane_mask: 0,
        });
    }
    exponent::<OPTIONS, N>(n)?;
    #[cfg(feature = "std")]
    if OPTIONS & PowfOptions::HIGHP != 0 {
        return super::highp::run::<OPTIONS, N>(bases, n);
    }
    let plan = select::<T, OPTIONS>(
        token,
        if OPTIONS & PowfOptions::CALLER != 0 {
            PowfExecution::Caller
        } else {
            PowfExecution::Automatic
        },
    );
    let mut output = [0.; N];
    if N == 1 {
        let value =
            dispatch::run4_const::<OPTIONS>(plan, [bases[0]; 4], n).map_err(|error| PowfError {
                lane_mask: 1,
                ..error
            })?;
        output[0] = value[0];
    } else if N == 4 {
        let mut input = [0.; 4];
        input.copy_from_slice(&bases);
        output.copy_from_slice(&dispatch::run4_const::<OPTIONS>(plan, input, n)?);
    } else if N == 8 {
        let mut input = [0.; 8];
        input.copy_from_slice(&bases);
        output.copy_from_slice(&dispatch::run8_const::<OPTIONS>(plan, input, n)?);
    } else {
        #[cfg(feature = "w512")]
        {
            let mut input = [0.; 16];
            input.copy_from_slice(&bases);
            output.copy_from_slice(&dispatch::run16_const::<OPTIONS>(plan, input, n)?);
        }
    }
    Ok(output)
}
#[inline(always)]
pub(super) fn runtime<T: archmage::SimdToken, const N: usize>(
    token: T,
    bases: [f32; N],
    n: f32,
    options: PowfOptions,
) -> Result<[f32; N], PowfError> {
    match options.bits() {
        0 => token_array::<T, 0, N>(token, bases, n),
        1 => token_array::<T, 1, N>(token, bases, n),
        2 => token_array::<T, 2, N>(token, bases, n),
        4 => token_array::<T, 4, N>(token, bases, n),
        5 => token_array::<T, 5, N>(token, bases, n),
        6 => token_array::<T, 6, N>(token, bases, n),
        8 => token_array::<T, 8, N>(token, bases, n),
        9 => token_array::<T, 9, N>(token, bases, n),
        10 => token_array::<T, 10, N>(token, bases, n),
        12 => token_array::<T, 12, N>(token, bases, n),
        13 => token_array::<T, 13, N>(token, bases, n),
        14 => token_array::<T, 14, N>(token, bases, n),
        16 => token_array::<T, 16, N>(token, bases, n),
        17 => token_array::<T, 17, N>(token, bases, n),
        18 => token_array::<T, 18, N>(token, bases, n),
        20 => token_array::<T, 20, N>(token, bases, n),
        21 => token_array::<T, 21, N>(token, bases, n),
        22 => token_array::<T, 22, N>(token, bases, n),
        24 => token_array::<T, 24, N>(token, bases, n),
        25 => token_array::<T, 25, N>(token, bases, n),
        26 => token_array::<T, 26, N>(token, bases, n),
        28 => token_array::<T, 28, N>(token, bases, n),
        29 => token_array::<T, 29, N>(token, bases, n),
        30 => token_array::<T, 30, N>(token, bases, n),
        32 => token_array::<T, 32, N>(token, bases, n),
        33 => token_array::<T, 33, N>(token, bases, n),
        34 => token_array::<T, 34, N>(token, bases, n),
        36 => token_array::<T, 36, N>(token, bases, n),
        37 => token_array::<T, 37, N>(token, bases, n),
        38 => token_array::<T, 38, N>(token, bases, n),
        40 => token_array::<T, 40, N>(token, bases, n),
        41 => token_array::<T, 41, N>(token, bases, n),
        42 => token_array::<T, 42, N>(token, bases, n),
        44 => token_array::<T, 44, N>(token, bases, n),
        45 => token_array::<T, 45, N>(token, bases, n),
        46 => token_array::<T, 46, N>(token, bases, n),
        48 => token_array::<T, 48, N>(token, bases, n),
        49 => token_array::<T, 49, N>(token, bases, n),
        50 => token_array::<T, 50, N>(token, bases, n),
        52 => token_array::<T, 52, N>(token, bases, n),
        53 => token_array::<T, 53, N>(token, bases, n),
        54 => token_array::<T, 54, N>(token, bases, n),
        56 => token_array::<T, 56, N>(token, bases, n),
        57 => token_array::<T, 57, N>(token, bases, n),
        58 => token_array::<T, 58, N>(token, bases, n),
        60 => token_array::<T, 60, N>(token, bases, n),
        61 => token_array::<T, 61, N>(token, bases, n),
        62 => token_array::<T, 62, N>(token, bases, n),
        #[cfg(feature = "std")]
        80 => token_array::<T, 80, N>(token, bases, n),
        #[cfg(feature = "std")]
        81 => token_array::<T, 81, N>(token, bases, n),
        #[cfg(feature = "std")]
        82 => token_array::<T, 82, N>(token, bases, n),
        #[cfg(feature = "std")]
        83 => token_array::<T, 83, N>(token, bases, n),
        #[cfg(feature = "std")]
        84 => token_array::<T, 84, N>(token, bases, n),
        #[cfg(feature = "std")]
        85 => token_array::<T, 85, N>(token, bases, n),
        #[cfg(feature = "std")]
        86 => token_array::<T, 86, N>(token, bases, n),
        #[cfg(feature = "std")]
        87 => token_array::<T, 87, N>(token, bases, n),
        #[cfg(feature = "std")]
        112 => token_array::<T, 112, N>(token, bases, n),
        #[cfg(feature = "std")]
        113 => token_array::<T, 113, N>(token, bases, n),
        #[cfg(feature = "std")]
        114 => token_array::<T, 114, N>(token, bases, n),
        #[cfg(feature = "std")]
        115 => token_array::<T, 115, N>(token, bases, n),
        #[cfg(feature = "std")]
        116 => token_array::<T, 116, N>(token, bases, n),
        #[cfg(feature = "std")]
        117 => token_array::<T, 117, N>(token, bases, n),
        #[cfg(feature = "std")]
        118 => token_array::<T, 118, N>(token, bases, n),
        #[cfg(feature = "std")]
        119 => token_array::<T, 119, N>(token, bases, n),
        _ => Err(PowfError {
            kind: PowfErrorKind::UnsupportedOptions,
            lane_mask: 0,
        }),
    }
}
