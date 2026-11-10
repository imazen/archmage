//! Independent composition oracle from the pre-remap power methods (main 6bfd83d7).
#![allow(deprecated)] // Intentional coverage of retained Magetypes compatibility APIs.
#![forbid(unsafe_code)]
#[cfg(any(
    target_arch = "x86_64",
    target_arch = "aarch64",
    target_arch = "wasm32"
))]
use archmage::SimdToken;
#[archmage::magetypes(v4x(cfg(avx512)), v3, neon, wasm128, scalar)]
fn check(token: Token) {
    macro_rules! width {
        ($ty:ident,$lanes:literal) => {{
            use magetypes::simd::generic::$ty as V;
            let sp = |x| V::splat_t(token, x);
            let mut state = 0x9e3779b9u32;
            for batch in 0..256 {
                let values = core::array::from_fn(|i| {
                    state ^= state << 13;
                    state ^= state >> 17;
                    state ^= state << 5;
                    if batch == 0 {
                        [
                            0.,
                            -0.,
                            f32::INFINITY,
                            f32::NAN,
                            -1.,
                            f32::from_bits(1),
                            1.,
                            65536.,
                        ][i % 8]
                    } else {
                        f32::from_bits(0x37800000 + state % 0x10000001)
                    }
                });
                let v = V::from_array_t(token, values);
                for n in [-4., -0., 0., 0.5, 2.4, 4.] {
                    let low = (v.log2_lowp() * sp(n)).exp2_lowp();
                    let low = V::blend(v.simd_eq(sp(0.)), sp(0.), low);
                    let mid = (v.log2_midp() * sp(n)).exp2_midp();
                    let unchecked = (v.log2_midp_unchecked() * sp(n)).exp2_midp_unchecked();
                    for (name, actual, expected) in [
                        ("pow_lowp", v.pow_lowp(n), low),
                        ("pow_lowp_unchecked", v.pow_lowp_unchecked(n), low),
                        ("pow_midp", v.pow_midp(n), mid),
                        ("pow_midp_precise", v.pow_midp_precise(n), mid),
                        ("pow_midp_unchecked", v.pow_midp_unchecked(n), unchecked),
                    ] {
                        for (a, b) in actual.to_array().into_iter().zip(expected.to_array()) {
                            if b.is_nan() {
                                assert!(a.is_nan());
                            } else {
                                assert_eq!(
                                    a.to_bits(),
                                    b.to_bits(),
                                    "{name}, n={n}, input={values:?}"
                                );
                            }
                        }
                    }
                }
            }
        }};
    }
    width!(f32x4, 4);
    width!(f32x8, 8);
    #[cfg(feature = "w512")]
    width!(f32x16, 16);
}
#[test]
fn published_power_matches_frozen_composition() {
    check_scalar(archmage::ScalarToken);
    #[cfg(target_arch = "x86_64")]
    {
        if let Some(t) = archmage::X64V3Token::summon() {
            check_v3(t);
        }
        #[cfg(feature = "avx512")]
        if let Some(t) = archmage::X64V4xToken::summon() {
            check_v4x(t);
        }
    }
    #[cfg(target_arch = "aarch64")]
    check_neon(archmage::NeonToken::summon().unwrap());
    #[cfg(target_arch = "wasm32")]
    check_wasm128(archmage::Wasm128Token::summon().unwrap());
}
