//! Correctly rounded fused multiply-add over lane arrays.
//!
//! Hardware selection, outside Miri:
//! - std builds whose target features guarantee FMA (x86 `fma`, AArch64
//!   `neon`) use Rust's `mul_add`;
//! - other x86_64 std builds check archmage's cached V3 token once per call
//!   for f64 and send each lane through the FMA instruction when the CPU has
//!   it (the one runtime check in vector arithmetic; see `fma_lanes`);
//! - no_std builds use a token only when `summon` is a compile-time constant
//!   (V3 or NEON target features);
//! - everything else, and f32 without a compile-time guarantee, uses software.
//!
//! The software forms process a whole array with no data-dependent branch per
//! lane, so LLVM can vectorize them wherever the target has SIMD registers.
//! Their common path rounds to nearest; the rare lanes where that could differ
//! from one rounding (a halfway intermediate, or a value outside the fast
//! path's safe range) take one branch per array to an exact per-lane
//! implementation. Hardware and software results are bit-identical; NaN
//! payloads and signs are unspecified; round-to-nearest and gradual underflow
//! are assumed, as in the native vector implementations.
#![forbid(unsafe_code)]

/// The build guarantees an FMA instruction and std provides `mul_add`.
macro_rules! std_hardware_fma {
    ($($item:item)*) => {$(
        #[cfg(all(
            not(miri),
            feature = "std",
            any(
                all(target_arch = "aarch64", target_feature = "neon"),
                all(any(target_arch = "x86", target_arch = "x86_64"), target_feature = "fma")
            )
        ))]
        $item
    )*};
}

/// The complement of [`std_hardware_fma`]: token-proven hardware where a token
/// is available, software otherwise.
macro_rules! token_or_software_fma {
    ($($item:item)*) => {$(
        #[cfg(not(all(
            not(miri),
            feature = "std",
            any(
                all(target_arch = "aarch64", target_feature = "neon"),
                all(any(target_arch = "x86", target_arch = "x86_64"), target_feature = "fma")
            )
        )))]
        $item
    )*};
}

std_hardware_fma! {
    /// `a[i] * b[i] + c[i]` rounded once, per lane.
    #[inline(always)]
    pub(crate) fn fmaf_lanes<const N: usize>(a: [f32; N], b: [f32; N], c: [f32; N]) -> [f32; N] {
        let mut out = [0.0; N];
        for i in 0..N {
            out[i] = a[i].mul_add(b[i], c[i]);
        }
        out
    }

    /// `a[i] * b[i] + c[i]` rounded once, per lane.
    #[inline(always)]
    pub(crate) fn fma_lanes<const N: usize>(a: [f64; N], b: [f64; N], c: [f64; N]) -> [f64; N] {
        let mut out = [0.0; N];
        for i in 0..N {
            out[i] = a[i].mul_add(b[i], c[i]);
        }
        out
    }
}

token_or_software_fma! {
    /// `a[i] * b[i] + c[i]` rounded once, per lane. Hardware only when a
    /// token is proven at compile time (no_std V3 or NEON builds: `summon`
    /// folds to a constant); otherwise software. f32 takes no runtime check:
    /// its vectorized software path measured as fast as or faster than a
    /// per-call hardware path in baseline builds.
    #[inline(always)]
    pub(crate) fn fmaf_lanes<const N: usize>(a: [f32; N], b: [f32; N], c: [f32; N]) -> [f32; N] {
        #[cfg(all(
            not(miri),
            not(feature = "std"),
            any(
                all(target_arch = "x86_64", target_feature = "fma"),
                all(target_arch = "aarch64", target_feature = "neon")
            )
        ))]
        if let Some(token) = hw::token() {
            return core::array::from_fn(|i| hw::fmaf(token, a[i], b[i], c[i]));
        }
        fmaf_soft_lanes(a, b, c)
    }

    /// `a[i] * b[i] + c[i]` rounded once, per lane. On x86_64 with std this is
    /// a cached runtime check (archmage's `summon`, once per call) that sends
    /// every lane through the FMA instruction: on CPUs with FMA that measured
    /// 0.09-0.43x the software path's time, and about 1.0x without. Lanes pass
    /// in float registers, two per call; an array or SIMD-vector argument would
    /// go through memory. Without std, and on AArch64, `summon` is a
    /// compile-time constant. Results equal the software path's bit for bit.
    #[inline(always)]
    pub(crate) fn fma_lanes<const N: usize>(a: [f64; N], b: [f64; N], c: [f64; N]) -> [f64; N] {
        #[cfg(all(
            not(miri),
            any(target_arch = "x86_64", all(target_arch = "aarch64", target_feature = "neon"))
        ))]
        if let Some(token) = hw::token() {
            // Two lanes per call: an f64 pair travels in xmm registers both
            // ways, and fewer calls means fewer spills (every call clobbers
            // the XMM registers). f64x4 streams took 0.60-0.67x the time of
            // one call per lane; chains and single lanes were unchanged.
            let mut out = [0.0; N];
            let mut i = 0;
            while i + 1 < N {
                let (x, y) = hw::fma2(token, a[i], a[i + 1], b[i], b[i + 1], c[i], c[i + 1]);
                out[i] = x;
                out[i + 1] = y;
                i += 2;
            }
            if i < N {
                out[i] = hw::fma(token, a[i], b[i], c[i]);
            }
            return out;
        }
        f64_soft::fma_soft_lanes(a, b, c)
    }
}

/// Token-proven FMA, one lane per call so the operands stay in float
/// registers. Inside a caller with the same features the call inlines.
#[cfg(all(
    not(miri),
    any(
        target_arch = "x86_64",
        all(target_arch = "aarch64", target_feature = "neon")
    ),
    not(all(
        feature = "std",
        any(
            all(target_arch = "aarch64", target_feature = "neon"),
            all(target_arch = "x86_64", target_feature = "fma")
        )
    ))
))]
mod hw {
    use archmage::SimdToken;

    #[cfg(target_arch = "x86_64")]
    pub(super) type Token = archmage::X64V3Token;
    #[cfg(target_arch = "aarch64")]
    pub(super) type Token = archmage::NeonToken;

    #[inline(always)]
    pub(super) fn token() -> Option<Token> {
        Token::summon()
    }

    #[cfg(target_arch = "x86_64")]
    #[archmage::arcane]
    pub(super) fn fma2(
        _token: archmage::X64V3Token,
        a0: f64,
        a1: f64,
        b0: f64,
        b1: f64,
        c0: f64,
        c1: f64,
    ) -> (f64, f64) {
        // Two scalar FMAs, not one packed: inlined into a V3 caller, LLVM
        // merges scalar lanes into one wide FMA, while packing and unpacking
        // left shuffles on the critical path (f64x4 chains 2.5x slower).
        use core::arch::x86_64::*;
        let fma1 =
            |a, b, c| _mm_cvtsd_f64(_mm_fmadd_sd(_mm_set_sd(a), _mm_set_sd(b), _mm_set_sd(c)));
        (fma1(a0, b0, c0), fma1(a1, b1, c1))
    }

    #[cfg(target_arch = "x86_64")]
    #[archmage::arcane]
    pub(super) fn fma(_token: archmage::X64V3Token, a: f64, b: f64, c: f64) -> f64 {
        use core::arch::x86_64::*;
        _mm_cvtsd_f64(_mm_fmadd_sd(_mm_set_sd(a), _mm_set_sd(b), _mm_set_sd(c)))
    }

    #[cfg(all(
        target_arch = "x86_64",
        all(
            not(feature = "std"),
            any(target_arch = "aarch64", target_feature = "fma")
        )
    ))]
    #[archmage::arcane]
    pub(super) fn fmaf(_token: archmage::X64V3Token, a: f32, b: f32, c: f32) -> f32 {
        use core::arch::x86_64::*;
        _mm_cvtss_f32(_mm_fmadd_ss(_mm_set_ss(a), _mm_set_ss(b), _mm_set_ss(c)))
    }

    #[cfg(target_arch = "aarch64")]
    #[archmage::arcane]
    pub(super) fn fma2(
        _token: archmage::NeonToken,
        a0: f64,
        a1: f64,
        b0: f64,
        b1: f64,
        c0: f64,
        c1: f64,
    ) -> (f64, f64) {
        use core::arch::aarch64::*;
        let fma1 =
            |a, b, c| vget_lane_f64::<0>(vfma_f64(vdup_n_f64(c), vdup_n_f64(a), vdup_n_f64(b)));
        (fma1(a0, b0, c0), fma1(a1, b1, c1))
    }

    #[cfg(target_arch = "aarch64")]
    #[archmage::arcane]
    pub(super) fn fma(_token: archmage::NeonToken, a: f64, b: f64, c: f64) -> f64 {
        use core::arch::aarch64::*;
        // vfma_f64(acc, x, y) = acc + x * y, fused.
        vget_lane_f64::<0>(vfma_f64(vdup_n_f64(c), vdup_n_f64(a), vdup_n_f64(b)))
    }

    #[cfg(all(
        target_arch = "aarch64",
        all(
            not(feature = "std"),
            any(target_arch = "aarch64", target_feature = "fma")
        )
    ))]
    #[archmage::arcane]
    pub(super) fn fmaf(_token: archmage::NeonToken, a: f32, b: f32, c: f32) -> f32 {
        use core::arch::aarch64::*;
        vget_lane_f32::<0>(vfma_f32(vdup_n_f32(c), vdup_n_f32(a), vdup_n_f32(b)))
    }
}

/// [`token_or_software_fma`] plus test builds, which exercise the software
/// path everywhere.
macro_rules! software_fma_or_test {
    ($($item:item)*) => {$(
        #[cfg(any(test, not(all(
            not(miri),
            feature = "std",
            any(
                all(target_arch = "aarch64", target_feature = "neon"),
                all(any(target_arch = "x86", target_arch = "x86_64"), target_feature = "fma")
            )
        ))))]
        $item
    )*};
}

/// The 29 low f64 significand bits that narrowing to f32 drops.
const F32_DROPPED: u32 = (1 << 29) - 1;
/// Those bits when the f64 value is halfway between two f32 values.
const F32_HALFWAY: u32 = 1 << 28;
/// The high word of the smallest normal f32's bits as an f64.
const F32_MIN_NORMAL_HIGH: u32 = ((1023 - 126) << 20) as u32;

/// Software f32 FMA. The product is exact in f64 (48 significant bits, exponent
/// range fits), so the f64 sum is the exact result rounded once. Narrowing it
/// rounds a second time, which can differ from one rounding only when the f64
/// sum is itself halfway between two f32 values: every f32 midpoint is an f64
/// value, so none can lie strictly between the exact result and its nearest
/// f64. Those lanes, and lanes in f32's subnormal range (where the midpoints
/// sit higher), take [`fmaf_exact`]. The common path is a multiply, an add and
/// a narrowing, with the check off the critical path.
#[inline(always)]
pub(crate) fn fmaf_soft_lanes<const N: usize>(a: [f32; N], b: [f32; N], c: [f32; N]) -> [f32; N] {
    let mut out = [0.0; N];
    let mut slow = [false; N];
    let mut any_slow = 0u8;
    for i in 0..N {
        let sum = f64::from(a[i]) * f64::from(b[i]) + f64::from(c[i]);
        out[i] = sum as f32;
        let bits = sum.to_bits();
        // 32-bit compares: the dropped bits all sit in the low word.
        let halfway = (bits as u32) & F32_DROPPED == F32_HALFWAY;
        // Exact results are multiples of 2^-298, so the high word is zero only
        // for a zero sum; one unsigned compare finds f32-subnormal magnitudes.
        let high = (bits >> 32) as u32 & 0x7fff_ffff;
        let subnormal = high.wrapping_sub(1) < F32_MIN_NORMAL_HIGH - 1;
        slow[i] = halfway | subnormal;
        any_slow |= u8::from(slow[i]);
    }
    if any_slow != 0 {
        for i in 0..N {
            if slow[i] {
                out[i] = fmaf_exact(a[i], b[i], c[i]);
            }
        }
    }
    out
}

/// Exact f32 FMA for every input: TwoSum recovers the f64 sum's error, and
/// rounding the sum to odd before narrowing avoids double rounding, including
/// at underflow. All finite f32 products and residuals are normal in f64, so
/// TwoSum cannot overflow or underflow. Infinite and NaN sums pass through.
/// Out of line and cold: the fast paths call it per lane, so the result comes
/// back in a float register and the hot loop's lanes stay in registers.
#[cold]
#[inline(never)]
pub(crate) fn fmaf_exact(a: f32, b: f32, c: f32) -> f32 {
    let product = f64::from(a) * f64::from(b);
    let (sum, error) = two_sum(product, f64::from(c));
    round_to_odd(sum, error, sum.is_finite()) as f32
}

/// Knuth's TwoSum: `s + e == a + b` exactly, `s` rounded to nearest.
#[inline(always)]
fn two_sum(a: f64, b: f64) -> (f64, f64) {
    let s = a + b;
    let virtual_b = s - a;
    let e = (a - (s - virtual_b)) + (b - virtual_b);
    (s, e)
}

/// Rounds `sum + error` (an exact TwoSum pair) to odd, branch-free: truncate
/// toward zero (step the bits down when the sum overshot the exact value in
/// magnitude), then set the lowest bit when inexact. The masks come from
/// float compares, so no 64-bit integer compare is needed (SSE2 has none).
/// `apply` masks lanes whose sum is not finite.
#[inline(always)]
fn round_to_odd(sum: f64, error: f64, apply: bool) -> f64 {
    let inexact = apply & (error != 0.0);
    let overshot = inexact & ((error > 0.0) != (sum > 0.0));
    let bits = sum.to_bits() - u64::from(overshot);
    f64::from_bits(bits | u64::from(inexact))
}

software_fma_or_test! {
    /// The f64 software path: unused where the build guarantees hardware FMA.
    mod f64_soft {
        use super::two_sum;

        /// 2^27 + 1: Veltkamp's constant, splitting an f64 into two 26-bit halves.
        pub(super) const SPLITTER: f64 = 134_217_729.0;
        /// Inputs at or below 2^995 cannot overflow `SPLITTER * x`.
        pub(super) const SPLIT_MAX: f64 = f64::from_bits((1023 + 995) << 52);
        /// Products at or above 2^-960 keep every partial product of the split
        /// exact, and sums at or above it keep TwoSum's error terms normal.
        pub(super) const TINY: f64 = f64::from_bits((1023 - 960) << 52);
        /// Products and addends at or below 2^1020 cannot overflow any partial sum.
        pub(super) const BIG: f64 = f64::from_bits((1023 + 1020) << 52);

        /// Software f64 FMA.
        ///
        /// Fast path: Dekker's product gives `ph + pl == a * b` exactly and TwoSum
        /// gives `th + tl == c + ph` exactly, so the result is `th + (tl + pl)`. The
        /// inner sum `s` is tiny next to `th` (or exact, when `c` cancels `ph`), so
        /// `th + s` rounds the same as the exact value unless `s` was inexact and
        /// `th + s` lands exactly halfway between two f64 values. Those lanes, a
        /// rounding-to-odd case (Boldo and Melquiond, IEEE TC 57(4), 2008), take
        /// [`fma_exact`], as do lanes outside the safe range (non-finite values,
        /// overflow or underflow risk). A zero factor gives the ordinary `a * b + c`,
        /// which is exact. The error terms feed only the checks, off the critical
        /// path.
        #[inline(always)]
        pub(crate) fn fma_soft_lanes<const N: usize>(a: [f64; N], b: [f64; N], c: [f64; N]) -> [f64; N] {
            let mut out = [0.0; N];
            let mut slow = [false; N];
            // An integer OR, not `||`: one test per array, no branch per lane.
            let mut any_slow = 0u8;
            for i in 0..N {
                let (value, exact) = fast_lane(a[i], b[i], c[i]);
                out[i] = value;
                slow[i] = !exact;
                any_slow |= u8::from(!exact);
            }
            if any_slow != 0 {
                for i in 0..N {
                    if slow[i] {
                        out[i] = fma_exact(a[i], b[i], c[i]);
                    }
                }
            }
            out
        }

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
        /// Out of line and cold, like [`super::fmaf_exact`].
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
}

#[cfg(test)]
mod tests {
    use super::f64_soft::*;
    use super::*;

    /// xorshift64*, deterministic.
    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 >> 12;
            self.0 ^= self.0 << 25;
            self.0 ^= self.0 >> 27;
            self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
        }
        /// Random sign and significand; exponent uniform in `-span..=span`.
        fn f64_in(&mut self, span: i32) -> f64 {
            let r = self.next();
            let exp = (r % (2 * span as u64 + 1)) as i32 - span;
            let bits = ((1023 + exp) as u64) << 52 | (self.next() >> 12) | (r & 1 << 63);
            f64::from_bits(bits)
        }
        /// Any bit pattern, weighted toward specials.
        fn f64_any(&mut self) -> f64 {
            match self.next() % 16 {
                0 => [
                    0.0,
                    -0.0,
                    f64::INFINITY,
                    f64::NEG_INFINITY,
                    f64::NAN,
                    f64::MIN_POSITIVE,
                    f64::MAX,
                ][(self.next() % 7) as usize],
                1 => f64::from_bits(self.next() & 0x800F_FFFF_FFFF_FFFF), // subnormal
                _ => f64::from_bits(self.next()),
            }
        }
    }

    fn same(a: f64, b: f64) -> bool {
        a.to_bits() == b.to_bits() || (a.is_nan() && b.is_nan())
    }

    /// Checks the array result and, where the fast path claims the lane, its
    /// own value. Returns whether the fast path claimed it.
    fn check(x: f64, y: f64, z: f64) -> bool {
        let want = fma_exact(x, y, z);
        let got = fma_soft_lanes([x], [y], [z])[0];
        assert!(
            same(got, want),
            "fma({x:e}, {y:e}, {z:e}): got {got:e}, exact {want:e}"
        );
        let (fast, claimed) = fast_lane(x, y, z);
        assert!(
            !claimed || same(fast, want),
            "fast fma({x:e}, {y:e}, {z:e}): got {fast:e}, exact {want:e}"
        );
        claimed
    }

    fn cases() -> usize {
        // Safe code only: under Miri, enough cases to run every path.
        if cfg!(miri) { 100 } else { 2_000_000 }
    }

    /// The fast path agrees with the exact implementation on normal operands,
    /// including cancellations built from the rounded product.
    #[test]
    fn fast_path_matches_exact() {
        let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
        let mut claimed = 0;
        for _ in 0..cases() {
            let (x, y) = (rng.f64_in(60), rng.f64_in(60));
            let mut claim = check(x, y, rng.f64_in(130));
            let p = x * y;
            claim &= check(x, y, -p);
            let nudge = (rng.next() % 64 + 1) as f64;
            claim &= check(x, y, -p * (1.0 + nudge * f64::EPSILON));
            claim &= check(x, y, -p + p * f64::EPSILON * rng.f64_in(3));
            // Ties and near-ties: the addend's ULP sits at or above the
            // product's bits.
            claim &= check(
                x,
                y,
                f64::from_bits(((1023 + 60 + (rng.next() % 60) as i32) as u64) << 52),
            );
            claimed += usize::from(claim);
        }
        // Normal operands far from the range limits must stay on the fast path.
        assert_eq!(claimed, cases(), "fast path declined normal operands");
    }

    /// Boundaries of the safe range and every special class.
    #[test]
    fn edges_match_exact() {
        let mut rng = Rng(0x0123_4567_89AB_CDEF);
        for _ in 0..cases() {
            check(rng.f64_any(), rng.f64_any(), rng.f64_any());
            check(rng.f64_in(1023), rng.f64_in(1023), rng.f64_in(1023));
            let lo = rng.f64_in(10) * TINY;
            check(lo, rng.f64_in(10), rng.f64_in(10) * TINY);
            check(
                rng.f64_in(10) * SPLIT_MAX,
                rng.f64_in(4),
                rng.f64_in(10) * BIG,
            );
            // Exact cancellation of the rounded product: the result is the
            // product's error term, at every scale.
            let (x, y) = (rng.f64_in(500), rng.f64_in(500));
            check(x, y, -(x * y));
        }
    }

    /// The textbook double-rounding case: `c` has an even significand and the
    /// product is half an ULP of `c` plus 2^-78 of that, so `th` ties to `c`,
    /// `tl + pl` rounds back to the half ULP, and `th + s` ties the wrong way.
    /// Odd significands and negated operands cover the other directions.
    #[test]
    fn halfway_after_inner_rounding() {
        let mut rng = Rng(0x5DEE_CE66_D1CE_4E5B);
        let x0 = 1.0 + f64::from_bits((1023 - 26) << 52);
        let y = 1.0 - f64::from_bits((1023 - 26) << 52) + f64::from_bits((1023 - 52) << 52);
        assert_eq!(x0 * y, 1.0, "the product's excess is below half an ULP");
        let mut hits = 0;
        for _ in 0..cases() / 10 {
            let m = rng.f64_in(500).abs();
            let half_ulp = f64::from_bits((m.to_bits() & (0x7ff << 52)) - (53 << 52));
            let x = x0 * half_ulp;
            for (xs, ys, zs) in [(x, y, m), (-x, y, -m), (x, -y, -m)] {
                let claimed = check(xs, ys, zs);
                hits += usize::from(!claimed);
                check(xs, ys, f64::from_bits(m.to_bits() ^ 1).copysign(zs));
            }
        }
        // Even significands always hit the halfway case; the check must catch them.
        assert!(hits > 0, "construction never reached the halfway check");
    }

    /// The f32 fast path against the exact f32 path, on every input class.
    #[test]
    fn f32_fast_matches_exact() {
        let mut rng = Rng(42);
        for _ in 0..cases() {
            let r = rng.next();
            let [x, y, z] = [r as u32, (r >> 32) as u32, rng.next() as u32].map(f32::from_bits);
            let got = fmaf_soft_lanes([x], [y], [z])[0];
            let want = fmaf_exact(x, y, z);
            assert!(
                got.to_bits() == want.to_bits() || (got.is_nan() && want.is_nan()),
                "fmaf({x:e}, {y:e}, {z:e})"
            );
            // Midpoint traps: c's ULP sits just above the product.
            let c =
                f32::from_bits(((rng.next() % 200 + 20) as u32) << 23 | (rng.next() as u32 & 1));
            let a = (f64::from(c) * (1.0 / 16_777_216.0)) as f32;
            for (x, y, z) in [
                (a, f32::from_bits(0x3f7f_fffe), c),
                (-a, f32::from_bits(0x3f7f_fffe), -c),
            ] {
                let got = fmaf_soft_lanes([x], [y], [z])[0];
                assert_eq!(
                    got.to_bits(),
                    fmaf_exact(x, y, z).to_bits(),
                    "fmaf({x:e}, {y:e}, {z:e})"
                );
            }
        }
    }

    /// Results in f32's subnormal range, where the midpoints sit above the 29
    /// dropped bits. Davidoff's counterexample (exact result just below the
    /// midpoint 65537.5 minimum subnormals) stays just below a midpoint when `c`
    /// moves by an even number of subnormal ULPs.
    #[test]
    fn f32_subnormal_halfway_family() {
        let [a, b, c] = [0x9700_0800u32, 0x1cff_f001, 0x0001_0002].map(f32::from_bits);
        for m in 0..if cfg!(miri) { 20 } else { 30_000 } {
            let c = f32::from_bits(c.to_bits() + 2 * m);
            for (x, z) in [(a, c), (-a, -c)] {
                let got = fmaf_soft_lanes([x], [b], [z])[0];
                assert_eq!(
                    got.to_bits(),
                    fmaf_exact(x, b, z).to_bits(),
                    "fmaf({x:e}, {b:e}, {z:e})"
                );
            }
        }
        assert_eq!(fmaf_soft_lanes([a], [b], [c])[0].to_bits(), 0x0001_0001);
    }

    /// Lanes in one array take different paths without disturbing each other.
    #[test]
    fn mixed_lanes() {
        let a = [1.5, 0.0, f64::NAN, 3.0, 1e300, -2.0, 1e-200, 7.0];
        let b = [2.5, 5.0, 1.0, f64::INFINITY, 1e300, 0.1, 1e-200, -0.0];
        let c = [0.25, -0.0, 2.0, 1.0, -1.0, 0.2, 1e-310, -0.0];
        let got = fma_soft_lanes(a, b, c);
        for i in 0..8 {
            assert!(same(got[i], fma_exact(a[i], b[i], c[i])), "lane {i}");
        }
    }
}
