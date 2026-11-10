//! Independent exact-rational FMA corpus reused from magetypes at f40f5d38.
#![forbid(unsafe_code)]
#[path = "data/fma_vectors.rs"]
mod vectors;
#[test]
fn exact_rational_fma() {
    for [a, b, c, want] in vectors::F64 {
        let got = zenmath::scalar::fma(f64::from_bits(a), f64::from_bits(b), f64::from_bits(c));
        if f64::from_bits(want).is_nan() {
            assert!(got.is_nan());
        } else {
            assert_eq!(got.to_bits(), want, "{a:x} {b:x} {c:x}");
        }
    }
    for [a, b, c, want] in vectors::F32 {
        let got = zenmath::scalar::fmaf(f32::from_bits(a), f32::from_bits(b), f32::from_bits(c));
        if f32::from_bits(want).is_nan() {
            assert!(got.is_nan());
        } else {
            assert_eq!(got.to_bits(), want, "{a:x} {b:x} {c:x}");
        }
    }
}
