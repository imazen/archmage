//! Intrinsic-only packed global Estrin-13/15 candidates, with per-lane rescue.
use super::packed::{Backend, V};
use crate::{gamma, gamma_tables as g};
#[path = "gamma_polynomials.rs"]
mod polynomials;

#[inline(always)]
pub(super) fn candidate<T: Backend<N>, const N: usize>(
    t: T,
    x: [f32; N],
    y: [f32; N],
) -> Option<[f32; N]> {
    // Uniform exponent guard is also required for dynamic-y callers.
    const {
        assert!(N > 0);
    }
    let bits = y[0].to_bits();
    const MASK: u32 = !(gamma::EXPONENT_BITS[0] ^ gamma::EXPONENT_BITS[1]);
    if bits & MASK != gamma::EXPONENT_BITS[0] & MASK {
        return None;
    }
    let kind = if bits == gamma::EXPONENT_BITS[0] {
        0
    } else if bits == gamma::EXPONENT_BITS[1] {
        1
    } else {
        return None;
    };
    if y.iter().any(|v| v.to_bits() != bits)
        || !x
            .iter()
            .all(|v| (gamma::MINIMUM[kind]..=gamma::MAXIMUM).contains(&v.to_bits()))
    {
        return None;
    }
    Some(if kind == 0 {
        global::<_, N, 0>(t, x)
    } else {
        global::<_, N, 1>(t, x)
    })
}

#[inline(always)]
fn global<T: Backend<N>, const N: usize, const K: usize>(t: T, x: [f32; N]) -> [f32; N] {
    let mut valid = [false; N];
    let mut reduced = [0.0; N];
    let mut factors = [0.0; N];
    let mut deltas = [0.0; N];
    for i in 0..N {
        let bits = x[i].to_bits();
        valid[i] = (gamma::MINIMUM[K]..=gamma::MAXIMUM).contains(&bits);
        let safe = if valid[i] { bits } else { gamma::MAXIMUM };
        let e = (safe >> 23) as usize - 118;
        reduced[i] = f64::from(f32::from_bits((safe & 0x007fffff) | 0x3f800000)) - 1.5;
        factors[i] = g::GLOBAL_FACTOR[K][e];
        deltas[i] = g::GLOBAL_ERROR[K][e];
    }
    let v = V::from_array_t(t, reduced);
    let q = if K == 0 {
        polynomials::decode(t, v)
    } else {
        polynomials::encode(t, v)
    };
    let values = (q * V::from_array_t(t, factors)).to_array();
    let mut out = [0.0; N];
    for i in 0..N {
        out[i] = if valid[i] {
            gamma::certificate(values[i], deltas[i]).unwrap_or_else(|| rescue::<K>(x[i]))
        } else {
            rescue::<K>(x[i])
        };
    }
    out
}
#[cold]
#[inline(never)]
fn rescue<const K: usize>(x: f32) -> f32 {
    crate::reference::powf(x, f32::from_bits(gamma::EXPONENT_BITS[K]))
}

#[cfg(test)]
mod tests {
    #[test]
    fn mixed_domain_and_exponents_use_full_core() {
        let t = archmage::ScalarToken;
        for k in 0..2 {
            let y = f32::from_bits(crate::gamma::EXPONENT_BITS[k]);
            for x in [
                [0.125, 0.5, 1.0, 2.0],
                [0.0, -0.0, -1.0, f32::NAN],
                [f32::from_bits(crate::gamma::MINIMUM[k] - 1), 0.5, 0.75, 1.0],
            ] {
                assert!(super::candidate(t, x, [y; 4]).is_none());
                let got = super::super::correct::pow(t, x, [y; 4]);
                for i in 0..4 {
                    let want = crate::reference::powf(x[i], y);
                    assert!(got[i].to_bits() == want.to_bits() || got[i].is_nan() && want.is_nan());
                }
            }
            assert!(
                super::candidate(t, [0.5; 4], [y, y, y, f32::from_bits(y.to_bits() + 1)]).is_none()
            );
        }
    }
}
