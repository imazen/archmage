//! Shared whole-vector HIGHP validation and the explicitly std-backed policy.
use super::{PowfError, PowfErrorKind, PowfOptions};

#[derive(Clone, Copy)]
pub(super) struct HighpCfg {
    pub(super) zero: bool,
    pub(super) full: bool,
    pub(super) assume: bool,
    pub(super) elide: bool,
    pub(super) base: Option<(f32, f32)>,
}

impl HighpCfg {
    pub(super) const fn from_options(opts: PowfOptions) -> Self {
        Self {
            zero: opts.admits_zero(),
            full: opts.is_full(),
            assume: opts.assume_valid(),
            elide: opts.elides_range_check(),
            base: opts.base_bounds(),
        }
    }
}

#[cfg_attr(debug_assertions, inline(never))]
#[cfg_attr(not(debug_assertions), inline(always))]
pub(super) fn check_bases_cfg(
    cfg: HighpCfg,
    bases: &[f32],
    exponent: f32,
) -> Result<(), PowfError> {
    if cfg.assume || cfg.full {
        return Ok(());
    }
    let mut mask = 0;
    for (i, &x) in bases.iter().enumerate() {
        let valid = match cfg.base {
            Some((lo, hi)) => {
                (cfg.zero && exponent >= 0. && x.to_bits() == 0) || (lo..=hi).contains(&x)
            }
            None => (x.is_normal() && x > 0.) || (cfg.zero && x.to_bits() == 0 && exponent >= 0.),
        };
        mask |= u16::from(!valid) << i;
    }
    if mask != 0 {
        return Err(PowfError {
            kind: PowfErrorKind::InvalidBase,
            lane_mask: mask,
        });
    }
    Ok(())
}

#[cfg_attr(debug_assertions, inline(never))]
#[cfg_attr(not(debug_assertions), inline(always))]
pub(super) fn check_results_cfg(
    cfg: HighpCfg,
    bases: &[f32],
    result: &[f32],
    exponent: f32,
) -> Result<(), PowfError> {
    if cfg.assume || cfg.full || cfg.elide {
        return Ok(());
    }
    let mut mask = 0;
    for (i, &x) in result.iter().enumerate() {
        let valid = x.is_normal()
            || (cfg.zero && bases[i].to_bits() == 0 && exponent > 0. && x.to_bits() == 0);
        mask |= u16::from(!valid) << i;
    }
    if mask != 0 {
        return Err(PowfError {
            kind: PowfErrorKind::OutsideApproximationRange,
            lane_mask: mask,
        });
    }
    Ok(())
}

#[cfg(feature = "std")]
#[cfg_attr(debug_assertions, inline(never))]
#[cfg_attr(not(debug_assertions), inline(always))]
pub(super) fn run_cfg<const N: usize>(
    cfg: HighpCfg,
    bases: [f32; N],
    exponent: f32,
) -> Result<[f32; N], PowfError> {
    check_bases_cfg(cfg, &bases, exponent)?;
    // Prevent LLVM's `@llvm.pow.f32(2^k, y) -> exp2f(k * y)` rewrite on
    // constant power-of-two bases, which rounds `k * y` in binary32 (~6 ULP)
    // and breaks parity between constant-folded scalar calls and vector packs.
    let exp = core::hint::black_box(exponent);
    let result = bases.map(|x| core::hint::black_box(x).powf(exp));
    check_results_cfg(cfg, &bases, &result, exponent)?;
    Ok(result)
}
