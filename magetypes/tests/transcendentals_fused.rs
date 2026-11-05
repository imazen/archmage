//! The native FMA implementation predates the fused scalar variants and is the
//! compatibility oracle. Exercise each width, including native/polyfill x16.
#![forbid(unsafe_code)]
use archmage::{ScalarToken, SimdToken, arcane};
#[cfg(feature = "w512")]
use magetypes::simd::generic::f32x16;
use magetypes::simd::generic::{f32x4, f32x8};

#[test]
fn scalar_rails_and_widths() {
    for x in [0.0, -0.0, -1.0, f32::INFINITY, f32::NEG_INFINITY] {
        let v = f32x4::splat_t(ScalarToken, x);
        let log = v.log2_midp_fused().to_array()[0];
        if x == 0.0 {
            assert_eq!(log, f32::NEG_INFINITY);
        } else if x < 0.0 {
            assert!(log.is_nan());
        } else {
            assert_eq!(log, f32::INFINITY);
        }
    }
    for x in [-200.0, -126.0, -0.5, 0.0, 0.5, 127.5, 128.0, 200.0] {
        let a = f32x4::splat_t(ScalarToken, x).exp2_midp_fused().to_array()[0];
        assert_eq!(
            f32x8::splat_t(ScalarToken, x).exp2_midp_fused().to_array(),
            [a; 8]
        );
        #[cfg(feature = "w512")]
        assert_eq!(
            f32x16::splat_t(ScalarToken, x).exp2_midp_fused().to_array(),
            [a; 16]
        );
        if x < -126.0 {
            assert_eq!(a, 0.0);
        }
        if x >= 128.0 {
            assert_eq!(a, f32::INFINITY);
        }
    }
}

macro_rules! compare {
    ($token:ident, $vector:ident, $width:literal) => {{
        for i in 0..4096u32 {
            let values = core::array::from_fn::<_, $width, _>(|lane| {
                // Positive normal values spanning the entire exponent range.
                let bits = 0x0080_0000 + ((i * $width + lane as u32) * 7919) % 0x7f00_0000;
                f32::from_bits(bits)
            });
            let scalar = $vector::from_array_t(ScalarToken, values);
            let native = $vector::from_array_t($token, values);
            assert_eq!(
                scalar.log2_midp_fused().to_array().map(f32::to_bits),
                native.log2_midp().to_array().map(f32::to_bits)
            );
            assert_eq!(
                native.log2_midp_fused().to_array().map(f32::to_bits),
                native.log2_midp().to_array().map(f32::to_bits)
            );
            let exponents =
                values.map(|v| v.to_bits() as f32 / 0x7f80_0000u32 as f32 * 254.0 - 126.0);
            let scalar_e = $vector::from_array_t(ScalarToken, exponents);
            let native_e = $vector::from_array_t($token, exponents);
            assert_eq!(
                scalar_e.exp2_midp_fused().to_array().map(f32::to_bits),
                native_e.exp2_midp().to_array().map(f32::to_bits)
            );
            for gamma in [0.25, 0.5, 1.8, 2.2] {
                assert_eq!(
                    scalar.pow_midp_fused(gamma).to_array().map(f32::to_bits),
                    native.pow_midp(gamma).to_array().map(f32::to_bits)
                );
            }
        }
    }};
}

#[arcane]
fn native_v3(token: archmage::X64V3Token) {
    compare!(token, f32x4, 4);
    compare!(token, f32x8, 8);
    #[cfg(feature = "w512")]
    compare!(token, f32x16, 16);
}

#[cfg(target_arch = "x86_64")]
#[test]
fn fused_scalar_matches_existing_native_fma() {
    if let Some(token) = archmage::X64V3Token::summon() {
        native_v3(token);
    }
}

#[test]
fn beta_relaxed_scalar_is_not_the_fused_contract() {
    assert!((1..1024).any(|i| {
        let v = f32x4::splat_t(ScalarToken, i as f32 / 1024.0);
        v.pow_midp(2.2).to_array()[0].to_bits() != v.pow_midp_fused(2.2).to_array()[0].to_bits()
    }));
}
