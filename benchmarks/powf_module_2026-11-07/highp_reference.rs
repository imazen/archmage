// Original HIGHP Result<Array> provider from c3abfbcd; benchmark reference only.
// Scalar std backend. No cross-platform bit or ULP guarantee is inferred.
use magetypes::powf::{PowfError, PowfErrorKind};

#[inline(always)]
pub(super) fn run<const OPTIONS: u32, const N: usize>(
    bases: [f32; N],
    exponent: f32,
) -> Result<[f32; N], PowfError> {
    let domain = OPTIONS & 3;
    if OPTIONS & 32 == 0 && domain != 3 {
        let mut mask = 0;
        for (i, &x) in bases.iter().enumerate() {
            let valid = if domain == 2 {
                x.to_bits() == 0 || (1.0 / 65_536.0..=65_536.0).contains(&x)
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
    let result = bases.map(|x| x.powf(exponent));
    if OPTIONS & 32 == 0 && domain < 2 {
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
    Ok(result)
}
