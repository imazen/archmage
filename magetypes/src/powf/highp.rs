//! Shared whole-vector HIGHP validation and the explicitly std-backed policy.
use super::{PowfError, PowfErrorKind, PowfOptions};

#[inline(always)]
pub(super) fn check_bases<const OPTIONS: u32, const N: usize>(
    bases: &[f32; N],
    exponent: f32,
) -> Result<(), PowfError> {
    let domain = OPTIONS & 3;
    if OPTIONS & PowfOptions::ASSUME == 0 && domain != 3 {
        let mut mask = 0;
        for (i, &x) in bases.iter().enumerate() {
            let valid = if domain == 2 {
                (OPTIONS & PowfOptions::POSITIVE == 0 && x.to_bits() == 0)
                    || (1.0 / 65_536.0..=65_536.0).contains(&x)
            } else {
                (x.is_normal() && x > 0.) || (domain == 1 && x.to_bits() == 0 && exponent >= 0.)
            };
            mask |= u16::from(!valid) << i;
        }
        if mask != 0 {
            return Err(PowfError {
                kind: PowfErrorKind::InvalidBase,
                lane_mask: mask,
            });
        }
    }
    Ok(())
}
#[inline(always)]
pub(super) fn check_results<const OPTIONS: u32, const N: usize>(
    bases: &[f32; N],
    result: &[f32; N],
    exponent: f32,
) -> Result<(), PowfError> {
    let domain = OPTIONS & 3;
    if OPTIONS & PowfOptions::ASSUME == 0 && domain < 2 {
        let mut mask = 0;
        for (i, &x) in result.iter().enumerate() {
            let valid = x.is_normal()
                || (domain == 1 && bases[i].to_bits() == 0 && exponent > 0. && x.to_bits() == 0);
            mask |= u16::from(!valid) << i;
        }
        if mask != 0 {
            return Err(PowfError {
                kind: PowfErrorKind::OutsideApproximationRange,
                lane_mask: mask,
            });
        }
    }
    Ok(())
}
#[cfg(feature = "std")]
#[inline(always)]
pub(super) fn run<const OPTIONS: u32, const N: usize>(
    bases: [f32; N],
    exponent: f32,
) -> Result<[f32; N], PowfError> {
    check_bases::<OPTIONS, N>(&bases, exponent)?;
    let result = bases.map(|x| x.powf(exponent));
    check_results::<OPTIONS, N>(&bases, &result, exponent)?;
    Ok(result)
}
