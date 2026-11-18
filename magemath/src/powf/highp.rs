//! Shared whole-vector HIGHP validation and the explicitly std-backed policy.
use super::{PowfError, PowfErrorKind, PowfOptions};

#[derive(Clone, Copy)]
pub(super) struct HighpCfg {
    pub(super) zero: bool,
    pub(super) subnormals: bool,
    pub(super) full: bool,
    pub(super) clamp_oob: bool,
    pub(super) assume: bool,
    pub(super) elide: bool,
    pub(super) base: Option<(f32, f32)>,
}

impl HighpCfg {
    pub(super) const fn from_options(opts: PowfOptions) -> Self {
        Self {
            zero: opts.admits_zero(),
            subnormals: opts.admits_subnormals(),
            full: opts.is_full(),
            clamp_oob: opts.clamp_oob(),
            assume: opts.assume_valid(),
            elide: opts.elides_range_check(),
            base: opts.base_bounds(),
        }
    }
}

#[inline(always)]
pub(super) fn sanitize_bases_cfg<const N: usize>(
    cfg: HighpCfg,
    bases: [f32; N],
    exponent: f32,
) -> ([f32; N], u16) {
    if !cfg.clamp_oob || cfg.full {
        return (bases, 0);
    }
    let (lo_bound, hi_bound) = match cfg.base {
        Some((lo, hi)) => (lo, hi),
        None => (
            if cfg.subnormals {
                f32::from_bits(1)
            } else {
                f32::MIN_POSITIVE
            },
            f32::MAX,
        ),
    };
    let mut out = bases;
    let mut nan_mask = 0_u16;
    let mut i = 0;
    while i < N {
        let x = bases[i];
        if x.is_nan() {
            nan_mask |= 1_u16 << i;
            out[i] = 1.0;
        } else if cfg.zero && exponent >= 0.0 && x <= 0.0 {
            out[i] = 0.0;
        } else if x < lo_bound {
            out[i] = lo_bound;
        } else if x > hi_bound {
            out[i] = hi_bound;
        }
        i += 1;
    }
    (out, nan_mask)
}

#[inline(always)]
pub(super) fn restore_nans<const N: usize>(out: &mut [f32; N], nan_mask: u16) {
    if nan_mask == 0 {
        return;
    }
    let mut i = 0;
    while i < N {
        if (nan_mask & (1_u16 << i)) != 0 {
            out[i] = f32::NAN;
        }
        i += 1;
    }
}

#[cfg_attr(debug_assertions, inline(never))]
#[cfg_attr(not(debug_assertions), inline(always))]
pub(super) fn check_bases_cfg(
    cfg: HighpCfg,
    bases: &[f32],
    exponent: f32,
) -> Result<(), PowfError> {
    if cfg.assume || cfg.full || cfg.clamp_oob {
        return Ok(());
    }
    let mut mask = 0;
    for (i, &x) in bases.iter().enumerate() {
        let valid = match cfg.base {
            Some((lo, hi)) => {
                if cfg.subnormals {
                    (cfg.zero && exponent >= 0. && x.to_bits() == 0) || (x > 0. && x <= hi)
                } else {
                    (cfg.zero && exponent >= 0. && x.to_bits() == 0) || (lo..=hi).contains(&x)
                }
            }
            None => {
                let pos_ok = if cfg.subnormals {
                    x > 0. && x.is_finite()
                } else {
                    x.is_normal() && x > 0.
                };
                pos_ok || (cfg.zero && x.to_bits() == 0 && exponent >= 0.)
            }
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
    if cfg.assume || cfg.full || cfg.elide || cfg.clamp_oob {
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
    let (bases, nan_mask) = sanitize_bases_cfg::<N>(cfg, bases, exponent);
    check_bases_cfg(cfg, &bases, exponent)?;
    // Prevent LLVM's `@llvm.pow.f32(2^k, y) -> exp2f(k * y)` rewrite on
    // constant power-of-two bases, which rounds `k * y` in binary32 (~6 ULP)
    // and breaks parity between constant-folded scalar calls and vector packs.
    let exp = core::hint::black_box(exponent);
    let mut result = bases.map(|x| core::hint::black_box(x).powf(exp));
    check_results_cfg(cfg, &bases, &result, exponent)?;
    restore_nans(&mut result, nan_mask);
    Ok(result)
}
