//! Polyfill kernels: the v3 tier agrees with the scalar tier at every width.
#![cfg(target_arch = "x86_64")]
use archmage::prelude::*;
use v4_context_asm::*;

#[test]
fn polyfill_widths_agree() {
    let Some(v3) = X64V3Token::summon() else { return };
    let input: Vec<f32> = (0..203).map(|i| (i as f32) * 0.25 - 7.0).collect();
    for (simd, scalar) in [
        (p4_gain_impl_v3 as fn(X64V3Token, &mut [f32], f32), p4_gain_impl_scalar as fn(ScalarToken, &mut [f32], f32)),
        (p8_gain_impl_v3, p8_gain_impl_scalar),
        (p16_gain_impl_v3, p16_gain_impl_scalar),
    ] {
        let (mut a, mut b) = (input.clone(), input.clone());
        simd(v3, &mut a, 0.5);
        scalar(ScalarToken, &mut b, 0.5);
        assert_eq!(a, b);
    }
    // Sums: the lane order differs between widths, so compare within a tolerance.
    let expect: f32 = input.iter().sum();
    for got in [p4_sum_impl_v3(v3, &input), p8_sum_impl_v3(v3, &input), p16_sum_impl_v3(v3, &input)] {
        assert!((got - expect).abs() <= 1e-3 * expect.abs().max(1.0), "{got} vs {expect}");
    }
}
