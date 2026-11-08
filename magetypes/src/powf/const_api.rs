//! Caller-visible const flags, kept through the feature boundary and graph.
use super::{PowfError, PowfErrorKind, PowfExecution, PowfOptions, dispatch, graph};

#[inline(always)]
fn exponent<const OPTIONS: u32, const N: usize>(n: f32) -> Result<(), PowfError> {
    const {
        assert!(
            OPTIONS & !7 == 0 && OPTIONS & 3 < 3,
            "invalid powf const options"
        );
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

/// Bounded scalar power with integer const options.
///
/// Combine one domain and an execution flag, for example
/// `try_powf_with::<{ PowfOptions::GAMMA | PowfOptions::CALLER }>(x, n)`.
/// Invalid encodings are compile errors:
/// ```compile_fail
/// let _ = magetypes::powf::try_powf_with::<3>(1.0, 2.4);
/// ```
/// ```compile_fail
/// let _ = magetypes::powf::try_powf_with::<8>(1.0, 2.4);
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
    const { assert!(OPTIONS & !7 == 0 && OPTIONS & 3 < 3, "invalid powf const options"); }
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
            let caller = dispatch::select(token, PowfExecution::Caller);
            if matches!(caller, dispatch::Plan::Software) {
                if OPTIONS & PowfOptions::CALLER != 0 {
                    return graph::$profile::<T, { graph::SOFTWARE }, OPTIONS>(token, input, n);
                }
                return dispatch::$dispatch::<OPTIONS>(
                    dispatch::select(token, PowfExecution::Automatic),
                    input.to_array(),
                    n,
                )
                .map(|result| crate::simd::generic::$float::from_array_t(token, result));
            }
            #[cfg(target_arch = "wasm32")]
            if matches!(caller, dispatch::Plan::Wasm(_, false)) {
                return graph::$profile::<T, { graph::WASM_SOFTWARE }, OPTIONS>(token, input, n);
            }
            // Caller selection admits only a native FMA proof here (or a
            // relaxed-WASM proof after its cached fusion screen). Keep the
            // vector representation in the caller's optimization region.
            graph::$profile::<T, { graph::NATIVE }, OPTIONS>(token, input, n)
        }
    };
}
vector!(run4, profile4, run4_const, f32x4, F32x4Convert, 4);
vector!(run8, profile8, run8_const, f32x8, F32x8Convert, 8);
#[cfg(feature = "w512")]
vector!(run16, profile16, run16_const, f32x16, F32x16Convert, 16);
