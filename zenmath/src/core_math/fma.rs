//! Reused from magetypes soft_fma.rs at f40f5d38, including its Rust libm 0.2.16
//! integer fallback (MIT, musl lineage; see LICENSE-libm.txt).
//! Nearest-even with gradual underflow.
#[inline]
pub fn fma(x: f64, y: f64, z: f64) -> f64 {
    #[cfg(all(
        feature = "std",
        any(
            target_feature = "fma",
            all(target_arch = "aarch64", target_feature = "neon")
        )
    ))]
    {
        x.mul_add(y, z)
    }
    #[cfg(not(all(
        feature = "std",
        any(
            target_feature = "fma",
            all(target_arch = "aarch64", target_feature = "neon")
        )
    )))]
    {
        let (v, exact) = soft::fast_lane(x, y, z);
        if exact { v } else { soft::fma_exact(x, y, z) }
    }
}
#[cfg(not(all(
    feature = "std",
    any(
        target_feature = "fma",
        all(target_arch = "aarch64", target_feature = "neon")
    )
)))]
mod soft {
    #[inline]
    fn two_sum(a: f64, b: f64) -> (f64, f64) {
        let s = a + b;
        let v = s - a;
        (s, (a - (s - v)) + (b - v))
    }
    /// 2^27 + 1: Veltkamp's constant, splitting an f64 into two 26-bit halves.
    pub(super) const SPLITTER: f64 = 134_217_729.0;
    /// Inputs at or below 2^995 cannot overflow `SPLITTER * x`.
    pub(super) const SPLIT_MAX: f64 = f64::from_bits((1023 + 995) << 52);
    /// Products at or above 2^-960 keep every partial product of the split
    /// exact, and sums at or above it keep TwoSum's error terms normal.
    pub(super) const TINY: f64 = f64::from_bits((1023 - 960) << 52);
    /// Products and addends at or below 2^1020 cannot overflow any partial sum.
    pub(super) const BIG: f64 = f64::from_bits((1023 + 1020) << 52);

    /// One lane of the fast path: the result, and whether it is the correctly
    /// rounded one. Branch-free.
    #[inline(always)]
    pub(super) fn fast_lane(x: f64, y: f64, z: f64) -> (f64, bool) {
        let ph = x * y;
        let (xh, xl) = split(x);
        let (yh, yl) = split(y);
        let pl = ((xh * yh - ph) + xh * yl + xl * yh) + xl * yl;
        let (th, tl) = two_sum(z, ph);
        let (s, se) = two_sum(tl, pl);
        let (fused, residual) = two_sum(th, s);
        // `residual` is exactly half the gap to the neighbor on its side: adding
        // twice it reaches that neighbor exactly.
        let twice = residual + residual;
        let halfway = (residual != 0.0) & ((fused + twice) - fused == twice);

        let zero_factor = (x == 0.0) | (y == 0.0);
        let in_range = (x.abs() <= SPLIT_MAX)
            & (y.abs() <= SPLIT_MAX)
            & (ph.abs() >= TINY)
            & (ph.abs() <= BIG)
            & (z.abs() <= BIG)
            & ((th == 0.0) | (th.abs() >= TINY));
        // A select on SIMD targets; base RISC-V (no conditional move) branches
        // per lane here whether it is written as a select or as bit masks.
        let value = if zero_factor { ph + z } else { fused };
        (value, (in_range & !(halfway & (se != 0.0))) | zero_factor)
    }

    /// Veltkamp's split: `hi + lo == x` exactly, each with at most 26 bits.
    #[inline(always)]
    fn split(x: f64) -> (f64, f64) {
        let t = SPLITTER * x;
        let hi = t - (t - x);
        (hi, x - hi)
    }

    // The exact integer implementation below is ported from the Rust libm crate
    // 0.2.16 (`src/math/generic/fma.rs` and `scalbn.rs`, MIT; origin musl
    // `src/math/fma.c`), specialized to f64.

    /// Exact fused multiply-add on integer significands; correct for every input.
    /// Out of line and cold, like the f32 exact fallback.
    #[cold]
    #[inline(never)]
    pub(crate) fn fma_exact(x: f64, y: f64, z: f64) -> f64 {
        let nx = Norm::from_float(x);
        let ny = Norm::from_float(y);
        let nz = Norm::from_float(z);

        if nx.is_zero_nan_inf() || ny.is_zero_nan_inf() {
            // A zero, infinite or NaN factor: the plain operations are exact.
            return x * y + z;
        }
        if nz.is_zero_nan_inf() {
            if nz.is_zero() {
                return x * y;
            }
            return z;
        }

        // multiply: r = x * y
        let zhi: u64;
        let zlo: u64;
        let wide = u128::from(nx.m) * u128::from(ny.m);
        let (mut rlo, mut rhi) = (wide as u64, (wide >> 64) as u64);

        let mut e: i32 = nx.e + ny.e;
        let mut d: i32 = nz.e - e;
        const SBITS: i32 = 64;

        // Scale `z`: shift `z <<= kz`, `r >>= kr`, so `kz + kr == d`, `e = e + kr`.
        if d > 0 {
            if d < SBITS {
                zlo = nz.m << d;
                zhi = nz.m >> (SBITS - d);
            } else {
                zlo = 0;
                zhi = nz.m;
                d -= SBITS;
                e = nz.e - SBITS;
                if d == 0 {
                } else if d < SBITS {
                    rlo = (rhi << (SBITS - d)) | (rlo >> d);
                    rlo |= u64::from((rlo << (SBITS - d)) != 0);
                    rhi >>= d;
                } else {
                    rlo = 1;
                    rhi = 0;
                }
            }
        } else {
            zhi = 0;
            d = -d;
            if d == 0 {
                zlo = nz.m;
            } else if d < SBITS {
                let sticky = u64::from((nz.m << (SBITS - d)) != 0);
                zlo = (nz.m >> d) | sticky;
            } else {
                zlo = 1;
            }
        }

        // addition
        let mut neg = nx.neg ^ ny.neg;
        let samesign = !neg ^ nz.neg;
        let mut rhi_nonzero = true;

        if samesign {
            rlo = rlo.wrapping_add(zlo);
            rhi = rhi.wrapping_add(zhi.wrapping_add(u64::from(rlo < zlo)));
        } else {
            let (res, borrow) = rlo.overflowing_sub(zlo);
            rlo = res;
            rhi = rhi.wrapping_sub(zhi.wrapping_add(u64::from(borrow)));
            if (rhi >> 63) != 0 {
                rlo = (rlo as i64).wrapping_neg() as u64;
                rhi = ((rhi as i64).wrapping_neg() as u64).wrapping_sub(u64::from(rlo != 0));
                neg = !neg;
            }
            rhi_nonzero = rhi != 0;
        }

        // Shift the result into `rhi`, left-aligned, with a sticky last bit.
        if rhi_nonzero {
            e += SBITS;
            d = rhi.leading_zeros() as i32 - 1;
            rhi = (rhi << d) | (rlo >> (SBITS - d));
            rhi |= u64::from((rlo << d) != 0);
        } else if rlo != 0 {
            d = rlo.leading_zeros() as i32 - 1;
            if d < 0 {
                rhi = (rlo >> 1) | (rlo & 1);
            } else {
                rhi = rlo << d;
            }
        } else {
            // exact +/- 0.0
            return x * y + z;
        }

        e -= d;

        // `i` is in [2^62, 2^63 - 1].
        let mut i = rhi as i64;
        if neg {
            i = -i;
        }
        let mut r = i as f64;

        // Unbiased exponent for the maximum value of `r`.
        const MAX_POW: i32 = 63 + 1023;

        if e < -(MAX_POW - 2) {
            // Subnormal before rounding.
            if e == -(MAX_POW - 1) {
                let c = if neg { -TWO_63 } else { TWO_63 };
                if r == c {
                    // Minimum normal after rounding.
                    return f64::MIN_POSITIVE.copysign(r);
                }
                if (rhi << 53) != 0 {
                    // One bit is lost in `scalbn`; add a top bit to avoid double
                    // rounding if inexact.
                    let iu = (rhi >> 1) | (rhi & 1) | (1 << 62);
                    i = iu as i64;
                    if neg {
                        i = -i;
                    }
                    r = i as f64;
                    r = 2.0 * r - c;
                }
            } else {
                // Round only once when scaled.
                d = 10;
                let sticky = u64::from(rhi << (64 - d) != 0);
                i = (((rhi >> d) | sticky) << d) as i64;
                if neg {
                    i = -i;
                }
                r = i as f64;
            }
        }

        scalbn(r, e)
    }

    const TWO_63: f64 = 9_223_372_036_854_775_808.0;

    /// f64 split into sign, guard-bit-shifted significand and exponent.
    #[derive(Clone, Copy)]
    struct Norm {
        /// Normalized significand with one guard bit: 1.0 is `1 << 53`.
        m: u64,
        /// Exponent such that `m * 2^e == |x|`.
        e: i32,
        neg: bool,
    }

    impl Norm {
        /// Bias plus significand bits plus the guard bit.
        const EXP_UNBIAS: i32 = 1023 + 52 + 1;
        /// Exponents at or above this were infinite, NaN or (via the sentinel) zero.
        const ZERO_INF_NAN: i32 = 2047 - Self::EXP_UNBIAS;

        fn from_float(x: f64) -> Self {
            let mut ix = x.to_bits();
            let mut e = ((ix >> 52) & 0x7ff) as i32;
            let neg = x.is_sign_negative();
            if e == 0 {
                // Normalize subnormals by multiplying by 2^63.
                let scaled = x * TWO_63;
                ix = scaled.to_bits();
                e = ((ix >> 52) & 0x7ff) as i32;
                e = if e == 0 {
                    // Zero: push the exponent past ZERO_INF_NAN.
                    1 << 11
                } else {
                    e - 63
                };
            }
            e -= Self::EXP_UNBIAS;
            ix &= (1 << 52) - 1;
            ix |= 1 << 52;
            ix <<= 1;
            Self { m: ix, e, neg }
        }

        fn is_zero_nan_inf(self) -> bool {
            self.e >= Self::ZERO_INF_NAN
        }

        fn is_zero(self) -> bool {
            self.e > Self::ZERO_INF_NAN
        }
    }

    /// `x * 2^n`, rounding once.
    fn scalbn(mut x: f64, mut n: i32) -> f64 {
        const EXP_MAX: i32 = 1023;
        const EXP_MIN: i32 = -1022;
        let f_exp_max = f64::from_bits(2046 << 52); // 2^1023
        // 2^-1022 * 2^53: prescale toward subnormal without double rounding.
        let mul = f64::from_bits(1 << 52) * f64::from_bits((53 + 1023) << 52);
        let add = -EXP_MIN - 53;
        if n > EXP_MAX {
            x *= f_exp_max;
            n -= EXP_MAX;
            if n > EXP_MAX {
                x *= f_exp_max;
                n -= EXP_MAX;
                if n > EXP_MAX {
                    n = EXP_MAX;
                }
            }
        } else if n < EXP_MIN {
            x *= mul;
            n += add;
            if n < EXP_MIN {
                x *= mul;
                n += add;
                if n < EXP_MIN {
                    n = EXP_MIN;
                }
            }
        }
        x * f64::from_bits(((1023 + n) as u64) << 52)
    }
}
