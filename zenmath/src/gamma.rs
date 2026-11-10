//! Certified fixed-exponent power candidates. The independent CORE stays untouched.
//!
//! Selected bit tables and DAG originate in the reviewed gamma study at
//! 1e6cc8e336ffb76727235d98c2aea988f4132bdd. See docs/GAMMA.md.
use crate::gamma_tables as g;

pub(crate) const EXPONENT_BITS: [u32; 2] = [0x4019_999a, 0x3ed5_5555];
pub(crate) const MINIMUM: [u32; 2] = [0x3db7_1074, 0x3b47_5041];
pub(crate) const MAXIMUM: u32 = 0x3f80_0000;

/// Only returns a value when outward-rounded enclosure endpoints round alike.
/// `None` always continues the original complete CORE algorithm.
#[inline(always)]
pub(crate) fn try_gamma(x: f32, y: f32) -> Option<f32> {
    let bits = y.to_bits();
    // A single common-bit rejection keeps general dynamic exponents cheap.
    // Exact exponent matching below is still required. Both admitted patterns
    // satisfy this mask by construction; no certificate/arithmetic changes.
    const MASK: u32 = !(EXPONENT_BITS[0] ^ EXPONENT_BITS[1]);
    if bits & MASK != EXPONENT_BITS[0] & MASK {
        return None;
    }
    match bits {
        bits if bits == EXPONENT_BITS[0] => piece::<0>(x),
        bits if bits == EXPONENT_BITS[1] => piece::<1>(x),
        _ => None,
    }
}

#[inline(always)]
fn piece<const K: usize>(x: f32) -> Option<f32> {
    let bits = x.to_bits();
    if bits < MINIMUM[K] || bits > MAXIMUM {
        return None;
    }
    let e = (bits >> 23) as usize - 118;
    let center = ((bits >> 19) & 15) as usize;
    let m = f64::from(f32::from_bits((bits & 0x007f_ffff) | 0x3f80_0000));
    let c = (33 + 2 * center) as f64 * (1.0 / 32.0);
    let v = m - c;
    let b = g::PIECE_COEFF[K][center];
    // Preserve separate operations and the certified Estrin-5 graph exactly.
    let v2 = v * v;
    let v4 = v2 * v2;
    let q01 = b[0] + b[1] * v;
    let q23 = b[2] + b[3] * v;
    let q45 = b[4] + b[5] * v;
    let q0123 = q01 + q23 * v2;
    let q = q0123 + q45 * v4;
    let y = g::PIECE_FACTOR[K][e][center] * q;
    certificate(y, g::PIECE_ERROR[K][e][center])
}

#[inline(always)]
pub(crate) fn certificate(y: f64, delta: f64) -> Option<f32> {
    // Positivity and finiteness of these endpoints are part of the certificate.
    // One adjacent binary64 value widens each rounded subtraction/addition.
    let lower = f64::from_bits((y - delta).to_bits() - 1) as f32;
    let upper = f64::from_bits((y + delta).to_bits() + 1) as f32;
    (lower.to_bits() == upper.to_bits()).then_some(lower)
}

#[cfg(test)]
mod tests {
    extern crate std;
    use super::*;
    fn same(a: f32, b: f32) -> bool {
        a.to_bits() == b.to_bits() || a.is_nan() && b.is_nan()
    }
    #[test]
    fn exact_guards_certificates_and_independent_rescue() {
        let mut inputs = std::vec![
            0, 0x80000000, 1, 0x80000001, 0x007fffff, 0x00800000, 0x7f7fffff, 0x7f800000,
            0xff800000, 0x7fc01234, 0x3f7fffff, 0x3f800000, 0x3f800001
        ];
        for k in 0..2 {
            inputs.extend([MINIMUM[k] - 1, MINIMUM[k], MINIMUM[k] + 1]);
            for exponent in 118..=127 {
                for center in 0..=16 {
                    let split = (exponent << 23) + (center << 19);
                    for d in -2i64..=2 {
                        inputs.push((i64::from(split) + d) as u32);
                    }
                }
            }
            let mut seed = 0x9e3779b9u32;
            for _ in 0..4096 {
                seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                inputs.push(seed);
                inputs.push(MINIMUM[k] + seed % (MAXIMUM - MINIMUM[k] + 1));
            }
            for ybits in [
                EXPONENT_BITS[k] - 1,
                EXPONENT_BITS[k],
                EXPONENT_BITS[k] + 1,
                0,
                0x80000000,
                0x3f800000,
                0x7f800000,
                0x7fc00123,
            ] {
                let y = f32::from_bits(ybits);
                for &bits in &inputs {
                    let x = f32::from_bits(bits);
                    let expected = crate::reference::powf(x, y);
                    if let Some(value) = try_gamma(x, y) {
                        assert!(
                            same(value, expected),
                            "certificate x={bits:08x} y={ybits:08x}"
                        );
                        assert!(ybits == EXPONENT_BITS[0] || ybits == EXPONENT_BITS[1]);
                    }
                    #[cfg(feature = "correct-rounding")]
                    assert!(
                        same(crate::scalar::powf(x, y), expected),
                        "ordinary x={bits:08x} y={ybits:08x}"
                    );
                }
            }
        }
    }
}
