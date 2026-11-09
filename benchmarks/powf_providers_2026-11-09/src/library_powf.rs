//! Actual public vector delegates, with const options retained into each kernel.
use magetypes::powf::{PowfExecution, PowfOptions, PowfValidation};
use magetypes::simd::backends::{F32x4Convert, F32x8Convert, F32x16Convert};
pub trait LibraryToken: F32x4Convert + F32x8Convert + F32x16Convert {}
impl<T: F32x4Convert + F32x8Convert + F32x16Convert> LibraryToken for T {}
const FULL: u32 = PowfOptions::PORTABLE_HIGHP
    .with_execution(PowfExecution::Caller)
    .bits();
const GAMMA: u32 = PowfOptions::PORTABLE_GAMMA_HIGHP
    .with_execution(PowfExecution::Caller)
    .bits();
const ASSUMED: u32 = PowfOptions::PORTABLE_GAMMA_HIGHP
    .with_execution(PowfExecution::Caller)
    .with_validation(PowfValidation::AssumeValid)
    .bits();
#[inline(always)]
fn uniform<T: LibraryToken, const N: usize, const M: u8>(t: T, x: [f32; N], n: f32) -> [f32; N] {
    let mut out = [0.; N];
    macro_rules! vector {
        ($v:ident, $width:literal) => {{
            let mut a = [0.; $width];
            if N == 1 {
                a.fill(x[0]);
            } else {
                a.copy_from_slice(&x);
            }
            let v = magetypes::simd::generic::$v::from_array_t(t, a);
            let result = match M {
                12 => v.try_powf_with::<FULL>(n).unwrap(),
                13 => v.try_powf_with::<GAMMA>(n).unwrap(),
                14 => v.try_powf_with::<ASSUMED>(n).unwrap(),
                15 => v.pow_midp(n),
                16 => v.pow_lowp(n),
                _ => panic!("unknown library method"),
            }
            .to_array();
            if N == 1 {
                out[0] = result[0];
            } else {
                out.copy_from_slice(&result);
            }
        }};
    }
    if N == 1 {
        // A scalar boundary must perform one power, not four broadcasts.
        out[0] = match M {
            12 => magetypes::powf::try_powf_with::<{ PowfOptions::PORTABLE_HIGHP.bits() }>(x[0], n)
                .unwrap(),
            13 => magetypes::powf::try_powf_with::<{ PowfOptions::PORTABLE_GAMMA_HIGHP.bits() }>(
                x[0], n,
            )
            .unwrap(),
            14 => magetypes::powf::try_powf_with::<
                {
                    PowfOptions::PORTABLE_GAMMA_HIGHP
                        .with_validation(PowfValidation::AssumeValid)
                        .bits()
                },
            >(x[0], n)
            .unwrap(),
            // Legacy generic API has no f32x1 equivalent of these vector kernels.
            _ => {
                vector!(f32x4, 4);
                out[0]
            }
        };
    } else if N == 4 {
        vector!(f32x4, 4);
    } else if N == 8 {
        vector!(f32x8, 8);
    } else {
        assert_eq!(N, 16);
        vector!(f32x16, 16);
    }
    out
}
#[inline(always)]
pub fn calculate<T: LibraryToken, const N: usize, const M: u8>(
    t: T,
    x: [f32; N],
    n: [f32; N],
) -> [f32; N] {
    if n.iter().all(|v| v.to_bits() == n[0].to_bits()) {
        return uniform::<T, N, M>(t, x, n[0]);
    }
    // The API exponent is uniform. Mixed-exponent numerical fixtures evaluate
    // the full mixed-base vector once for each requested lane exponent.
    core::array::from_fn(|i| uniform::<T, N, M>(t, x, n[i])[i])
}
