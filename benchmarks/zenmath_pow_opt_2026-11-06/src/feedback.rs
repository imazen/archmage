//! Amortized dependent workload, including mapping and whole-call boundaries.
use crate::*;
pub fn map(d: Domain, value: f32) -> f32 {
    let low = if d == Domain::Decode {
        (DECODE_CUTOFF + A) / A1
    } else {
        CUTOFF
    };
    low + (1.0 - low) * (0.9 * (value.abs() / (1.0 + value.abs())))
}
pub fn step(d: Domain, out: &[f32], input: &mut [f32]) {
    assert_eq!(out.len(), input.len());
    for (&v, x) in out.iter().zip(input) {
        *x = map(d, v);
    }
}
#[test]
fn feedback_matches_frozen_reference() {
    for d in [Domain::Decode, Domain::Encode] {
        for n in [1, 8] {
            for case in domain_cases(d, true) {
                if case.name.starts_with("std") {
                    continue;
                }
                let (mut x, y) = inputs(d, n);
                let mut expected = x.clone();
                let mut out = vec![0.0; n];
                for _ in 0..1024 / n {
                    (case.run)(&x, &y, &mut out);
                    for i in 0..n {
                        let want = reference(case.policy, expected[i], y[i]);
                        assert!(
                            want.to_bits() == out[i].to_bits() || want.is_nan() && out[i].is_nan(),
                            "{}",
                            case.name
                        );
                        expected[i] = map(d, want);
                    }
                    step(d, &out, &mut x);
                }
            }
        }
    }
}
