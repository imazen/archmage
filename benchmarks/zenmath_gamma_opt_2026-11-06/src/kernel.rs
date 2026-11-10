use crate::generated as g;
use zenmath::scalar::reference;

#[derive(Clone, Copy, Debug)]
pub(crate) struct Evaluation {
    pub value: f32,
    pub rescued: bool,
    pub lower: f64,
    pub upper: f64,
}

#[inline(always)]
pub(crate) fn exponent<const K: usize>() -> f32 {
    f32::from_bits(if K == 0 {
        g::DECODE_P_BITS
    } else {
        g::ENCODE_P_BITS
    })
}
#[inline(always)]
pub(crate) fn domain<const K: usize>() -> (u32, u32) {
    (
        if K == 0 {
            g::DECODE_MIN_BITS
        } else {
            g::ENCODE_MIN_BITS
        },
        1.0f32.to_bits(),
    )
}
#[cold]
#[inline(never)]
fn rescue<const K: usize>(x: f32) -> f32 {
    reference::powf(x, exponent::<K>())
}

/// Exact CORE bits, using the generated enclosure only inside the admitted domain.
#[inline(always)]
pub(crate) fn evaluate<const K: usize, const N: usize, const L: usize>(x: f32) -> Evaluation {
    const {
        assert!(K < 2 && N >= 8 && N <= 20 && L < 2);
    }
    let bits = x.to_bits();
    let (minimum, maximum) = domain::<K>();
    if bits < minimum || bits > maximum {
        return Evaluation {
            value: rescue::<K>(x),
            rescued: true,
            lower: f64::NAN,
            upper: f64::NAN,
        };
    }
    let index = (bits >> 23) as usize - 118;
    let m = f64::from(f32::from_bits((bits & 0x007f_ffff) | 0x3f80_0000));
    let v = m - 1.5; // Exact for every admitted binary32 mantissa.
    let q = if L == 0 {
        let mut r = g::COEFF[K][N];
        let mut k = N;
        while k > 0 {
            k -= 1;
            r = r * v + g::COEFF[K][k];
        }
        r
    } else {
        g::estrin::<K, N>(v)
    };
    let y = g::FACTOR[K][index] * q;
    let delta = g::ERROR[K][L][N - 8][index];
    // Generation proves both rounded endpoints positive, finite and below64.
    // Adjacent binary64 bit patterns therefore move outward monotonically.
    let lower = f64::from_bits((y - delta).to_bits() - 1);
    let upper = f64::from_bits((y + delta).to_bits() + 1);
    let lo = lower as f32;
    let hi = upper as f32;
    if lo.to_bits() == hi.to_bits() {
        Evaluation {
            value: lo,
            rescued: false,
            lower,
            upper,
        }
    } else {
        Evaluation {
            value: rescue::<K>(x),
            rescued: true,
            lower,
            upper,
        }
    }
}

pub(crate) fn each<const K: usize>(x: f32, mut f: impl FnMut(usize, usize, Evaluation)) {
    macro_rules! one {
        ($n:literal) => {
            f($n, 0, evaluate::<K, $n, 0>(x));
            f($n, 1, evaluate::<K, $n, 1>(x));
        };
    }
    one!(8);
    one!(9);
    one!(10);
    one!(11);
    one!(12);
    one!(13);
    one!(14);
    one!(15);
    one!(16);
    one!(17);
    one!(18);
    one!(19);
    one!(20);
}

#[cfg(test)]
mod tests {
    use super::*;
    fn same(a: f32, b: f32) -> bool {
        a.to_bits() == b.to_bits() || a.is_nan() && b.is_nan()
    }
    #[test]
    fn actual_c0_facts() {
        let cutoff = 0.003_041_282_560_127_521f64;
        let srgb = (12.92f64 * cutoff) as f32;
        let a = 0.055_010_718_947_586_6f64 as f32;
        let ap1 = 1.055_010_718_947_586_6f64 as f32;
        assert_eq!((cutoff as f32).to_bits(), g::LINEAR_CUTOFF_BITS);
        assert_eq!(srgb.to_bits(), g::SRGB_CUTOFF_BITS);
        assert_eq!(((srgb + a) / ap1).to_bits(), g::DECODE_MIN_BITS);
        assert_eq!((2.4f32).to_bits(), g::DECODE_P_BITS);
        assert_eq!(((1.0f64 / 2.4f64) as f32).to_bits(), g::ENCODE_P_BITS);
    }
    fn check<const K: usize>() {
        let (start, end) = domain::<K>();
        let mut state = 0x31415926u32;
        let mut inputs = vec![];
        for i in 0..4096 {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            inputs.push(if i < 8 {
                start + i
            } else if i < 16 {
                end - (i - 8)
            } else {
                start + state % (end - start + 1)
            });
        }
        for e in -9..=0 {
            for mantissa in [0, 0x00400000] {
                let center = (((e + 127) as u32) << 23) | mantissa;
                for delta in -8i64..=8 {
                    inputs.push((i64::from(center) + delta) as u32);
                }
            }
        }
        inputs.extend([
            0,
            0x80000000,
            1,
            0x007fffff,
            0x80000001,
            0xbf800000,
            0x7f800000,
            0xff800000,
            0x7f800001,
            0x7fc01234,
            start - 1,
            end + 1,
        ]);
        for bits in inputs {
            let x = f32::from_bits(bits);
            let expected = reference::powf(x, exponent::<K>());
            each::<K>(x, |n, l, got| {
                assert!(
                    same(got.value, expected),
                    "K={K} N={n} L={l} x={bits:08x} actual={:08x} expected={:08x}",
                    got.value.to_bits(),
                    expected.to_bits()
                );
                if bits >= start && bits <= end {
                    assert!(got.lower > 0.0 && got.upper.is_finite() && got.lower <= got.upper);
                } else {
                    assert!(got.rescued);
                }
            });
        }
    }
    #[test]
    fn decode_all26() {
        check::<0>();
    }
    #[test]
    fn encode_all26() {
        check::<1>();
    }
}
