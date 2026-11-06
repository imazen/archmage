//! Whole-transfer controls, separate from pow-only inputs and error claims.
use crate::*;
#[derive(Clone, Copy, Debug)]
pub enum Composition {
    Separate,
    Fused,
}
impl Composition {
    pub fn name(self) -> &'static str {
        match self {
            Self::Separate => "separate",
            Self::Fused => "fused",
        }
    }
}
pub fn cutoff(d: Domain) -> f32 {
    match d {
        Domain::Decode => DECODE_CUTOFF,
        Domain::Encode => CUTOFF,
        _ => panic!("gamma domain required"),
    }
}
pub fn transfer_inputs(d: Domain, distribution: Distribution, n: usize) -> Vec<f32> {
    let mut state = 0x4397d059u32;
    (0..n)
        .map(|i| {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            match distribution {
                Distribution::Uniform => (state >> 8) as f32 / 16777216.0,
                Distribution::Log => f32::from_bits(0x00800000 + state % (0x3f800001 - 0x00800000)),
                Distribution::Boundary => {
                    let c = cutoff(d).to_bits();
                    f32::from_bits([0, 1, c - 1, c, c + 1, 0x3f7fffff, 0x3f800000][i % 7])
                }
                Distribution::Historical => panic!("historical is pow-only"),
            }
        })
        .collect()
}
// Mask every inactive lane before invoking any eager SIMD precondition. The
// filler (1,0) is valid, including under debug precondition assertions.
pub fn prepare(d: Domain, input: &[f32], xs: &mut [f32], ys: &mut [f32]) {
    assert_eq!(input.len(), xs.len());
    assert_eq!(input.len(), ys.len());
    for ((&v, x), y) in input.iter().zip(xs).zip(ys) {
        if v < cutoff(d) || v >= 1.0 {
            *x = 1.0;
            *y = 0.0;
        } else {
            *x = if d == Domain::Decode { (v + A) / A1 } else { v };
            *y = if d == Domain::Decode {
                2.4
            } else {
                (1.0_f64 / 2.4) as f32
            };
        }
    }
}
pub fn compose(
    d: Domain,
    composition: Composition,
    input: &[f32],
    powers: &[f32],
    output: &mut [f32],
) {
    assert_eq!(input.len(), powers.len());
    assert_eq!(input.len(), output.len());
    for ((&v, &p), out) in input.iter().zip(powers).zip(output) {
        *out = if v < 0.0 {
            0.0
        } else if v < cutoff(d) {
            if d == Domain::Decode {
                v * ((1.0_f64 / 12.92) as f32)
            } else {
                v * 12.92
            }
        } else if v < 1.0 {
            if d == Domain::Decode {
                p
            } else {
                match composition {
                    Composition::Separate => A1 * p - A,
                    Composition::Fused => A1.mul_add(p, -A),
                }
            }
        } else {
            1.0
        };
    }
}
pub fn rational_scalar(d: Domain, input: &[f32], out: &mut [f32]) {
    assert_eq!(input.len(), out.len());
    for (&x, y) in input.iter().zip(out) {
        *y = if d == Domain::Decode {
            linear_srgb::default::srgb_to_linear(x)
        } else {
            linear_srgb::default::linear_to_srgb(x)
        };
    }
}
pub fn rational_slice(d: Domain, input: &[f32], out: &mut [f32]) {
    out.copy_from_slice(input);
    if d == Domain::Decode {
        linear_srgb::default::srgb_to_linear_slice(out)
    } else {
        linear_srgb::default::linear_to_srgb_slice(out)
    };
}
#[test]
fn masked_eager_adapters_preserve_frozen_policy_bits() {
    for d in [Domain::Decode, Domain::Encode] {
        for distribution in distributions(d) {
            for n in [1, 7, 16, 1024] {
                let input = transfer_inputs(d, *distribution, n);
                let (mut x, mut y) = (vec![0.0; n], vec![0.0; n]);
                prepare(d, &input, &mut x, &mut y);
                for case in domain_cases(d, false) {
                    if case.name.starts_with("std") {
                        continue;
                    }
                    let mut actual = vec![0.0; n];
                    (case.run)(&x, &y, &mut actual);
                    let want: Vec<_> = x
                        .iter()
                        .zip(&y)
                        .map(|(&x, &y)| reference(case.policy, x, y))
                        .collect();
                    for composition in [Composition::Separate, Composition::Fused] {
                        let (mut a, mut b) = (vec![0.0; n], vec![0.0; n]);
                        compose(d, composition, &input, &actual, &mut a);
                        compose(d, composition, &input, &want, &mut b);
                        assert!(
                            a.iter().zip(b).all(
                                |(a, b)| a.to_bits() == b.to_bits() || a.is_nan() && b.is_nan()
                            ),
                            "{} {:?}",
                            case.name,
                            distribution
                        );
                    }
                }
            }
        }
    }
}
