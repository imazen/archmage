//! Sixteen-center certificate; PIECEWISE-BOUND.md and generated per-center budgets.
use crate::{
    kernel::{self, Evaluation},
    piecewise_generated as g,
};

#[cold]
#[inline(never)]
fn rescue<const K: usize>(x: f32) -> f32 {
    zenmath::scalar::reference::powf(x, kernel::exponent::<K>())
}

#[inline(always)]
pub(crate) fn evaluate<const K: usize, const N: usize, const L: usize>(x: f32) -> Evaluation {
    const {
        assert!(K < 2 && N >= 3 && N <= 5 && L < 2);
    }
    let bits = x.to_bits();
    let (min, max) = kernel::domain::<K>();
    if bits < min || bits > max {
        return Evaluation {
            value: rescue::<K>(x),
            rescued: true,
            lower: f64::NAN,
            upper: f64::NAN,
        };
    }
    let e = (bits >> 23) as usize - 118;
    let center = ((bits >> 19) & 15) as usize;
    let m = f64::from(f32::from_bits((bits & 0x007f_ffff) | 0x3f80_0000));
    let c = (33 + 2 * center) as f64 * (1.0 / 32.0);
    let v = m - c; // Sterbenz-exact with the selected dyadic center.
    let q = if L == 0 {
        let mut q = g::COEFF[K][center][N];
        let mut k = N;
        while k > 0 {
            k -= 1;
            q = q * v + g::COEFF[K][center][k];
        }
        q
    } else {
        g::estrin::<K, N>(center, v)
    };
    let y = g::FACTOR[K][e][center] * q;
    let delta = g::ERROR[K][L][N - 3][e][center];
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
    one!(3);
    one!(4);
    one!(5);
}

#[cfg(test)]
mod tests {
    use super::*;
    fn check<const K: usize>() {
        let (start, end) = kernel::domain::<K>();
        let mut values = vec![
            0,
            0x80000000,
            0x7f800000,
            0xff800000,
            0x7fc00000,
            start - 1,
            start,
            end,
            end + 1,
        ];
        for e in -9..=0 {
            for split in 0..=16 {
                let bits = (((e + 127) as u32) << 23) + (split << 19);
                for d in -4i64..=4 {
                    values.push((i64::from(bits) + d) as u32);
                }
            }
        }
        let mut state = 271828183u32;
        for _ in 0..4096 {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            values.push(start + state % (end - start + 1));
        }
        for bits in values {
            let x = f32::from_bits(bits);
            let expected = zenmath::scalar::reference::powf(x, kernel::exponent::<K>());
            each::<K>(x, |n, l, got| {
                assert!(
                    got.value.to_bits() == expected.to_bits()
                        || got.value.is_nan() && expected.is_nan(),
                    "piecewise K={K} N={n} L={l} x={bits:08x}"
                )
            });
        }
    }
    #[test]
    fn decode_all6() {
        check::<0>();
    }
    #[test]
    fn encode_all6() {
        check::<1>();
    }
}
