//! Whole-array software FMA fast path, reused from Magetypes soft_fma.rs
//! at b1316a4e (MIT/Apache-2.0). No runtime probing; exact scalar rescue is cold.
#[inline(always)]
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

/// Software f64 FMA.
///
/// Fast path: Dekker's product gives `ph + pl == a * b` exactly and TwoSum
/// gives `th + tl == c + ph` exactly, so the result is `th + (tl + pl)`. The
/// inner sum `s` is tiny next to `th` (or exact, when `c` cancels `ph`), so
/// `th + s` rounds the same as the exact value unless `s` was inexact and
/// `th + s` lands exactly halfway between two f64 values. Those lanes, a
/// rounding-to-odd case (Boldo and Melquiond, IEEE TC 57(4), 2008), take
/// the scalar exact FMA, as do lanes outside the safe range (non-finite values,
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
                out[i] = crate::reference::fma(a[i], b[i], c[i]);
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
