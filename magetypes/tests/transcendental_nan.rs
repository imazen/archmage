//! Exceptional-value contract at the clamped exp boundary. NaN payload/sign
//! are unspecified. This deliberately does not promise std pow semantics.
#![forbid(unsafe_code)]
use archmage::ScalarToken;
#[cfg(any(
    target_arch = "x86_64",
    target_arch = "aarch64",
    target_arch = "wasm32"
))]
use archmage::SimdToken;
use std::hint::black_box;

macro_rules! width {
    ($name:ident, $v:ident, $bound:ident, $n:literal) => {
        fn $name<T: magetypes::simd::backends::$bound>(t: T) {
            let cases = [
                0x7fc00000, 0xffc01234, 0x7f800001, 0xff800001, 0x7fffffff, 0xffffffff,
            ];
            for bits in cases {
                let input = core::array::from_fn(|i| {
                    if i % 2 == 0 {
                        f32::from_bits(bits)
                    } else {
                        0.25
                    }
                });
                let x = magetypes::simd::generic::$v::from_array_t(t, black_box(input));
                let outputs = [
                    x.exp2_lowp(),
                    x.exp2_lowp_unchecked(),
                    x.exp_lowp(),
                    x.exp_lowp_unchecked(),
                    x.exp2_midp(),
                    x.exp2_midp_precise(),
                    x.exp_midp(),
                    x.exp_midp_precise(),
                    x.sigmoid_midp(),
                    x.silu_midp(),
                ];
                for (op, out) in outputs.into_iter().enumerate() {
                    for (i, y) in out.to_array().into_iter().enumerate() {
                        assert_eq!(
                            y.is_nan(),
                            i % 2 == 0,
                            "{} {} op={op} lane={i} input={bits:08x} output={:08x}",
                            T::NAME,
                            stringify!($v),
                            y.to_bits()
                        );
                    }
                }
            }
            // Caller selects exhaustive payload enumeration through the recipe.
            // Default still covers both signs and quiet/signalling families.
            let stride: u32 = std::env::var("ARCHMAGE_NAN_STRIDE")
                .unwrap_or_else(|_| "65537".into())
                .parse()
                .unwrap();
            assert!(stride > 0);
            for payload in (1..=0x007f_ffff_u32).step_by(stride as usize) {
                for sign in [0, 0x8000_0000] {
                    let a = core::array::from_fn(|i| {
                        f32::from_bits(sign | 0x7f80_0000 | ((payload + i as u32).min(0x007f_ffff)))
                    });
                    let x = magetypes::simd::generic::$v::from_array_t(t, black_box(a));
                    for out in [x.exp2_lowp(), x.exp2_midp()] {
                        for y in out.to_array() {
                            assert!(
                                y.is_nan(),
                                "{} {} payload={payload:06x} sign={sign:08x}",
                                T::NAME,
                                stringify!($v)
                            );
                        }
                    }
                }
            }
            // A NaN created by the checked log must survive exp clamping.
            let x = magetypes::simd::generic::$v::splat_t(t, black_box(-1.0));
            for out in [x.pow_midp(2.4), x.pow_midp_precise(2.4)] {
                assert!(out.to_array().iter().all(|y| y.is_nan()));
            }
            for (base, exponent) in [
                (1.0, f32::INFINITY),
                (0.0, 0.0),
                (f32::INFINITY, 0.0),
                (-1.0, 0.0),
                (f32::NEG_INFINITY, 2.4),
            ] {
                let x = magetypes::simd::generic::$v::splat_t(t, black_box(base));
                assert!(
                    x.pow_midp(black_box(exponent))
                        .to_array()
                        .iter()
                        .all(|y| y.is_nan())
                );
            }
            // Exponent NaNs propagate, except the existing lowp zero mask wins.
            let x = magetypes::simd::generic::$v::splat_t(t, black_box(2.0));
            for out in [x.pow_lowp(f32::NAN), x.pow_midp(f32::NAN)] {
                assert!(out.to_array().iter().all(|y| y.is_nan()));
            }
            let zero = magetypes::simd::generic::$v::splat_t(t, 0.0);
            assert!(
                zero.pow_lowp(f32::NAN)
                    .to_array()
                    .iter()
                    .all(|y| y.to_bits() == 0)
            );
        }
    };
}
width!(x4, f32x4, F32x4Convert, 4);
width!(x8, f32x8, F32x8Convert, 8);
#[cfg(feature = "w512")]
width!(x16, f32x16, F32x16Convert, 16);

#[cfg(feature = "w512")]
fn check<
    T: magetypes::simd::backends::F32x4Convert
        + magetypes::simd::backends::F32x8Convert
        + magetypes::simd::backends::F32x16Convert,
>(
    t: T,
) {
    x4(t);
    x8(t);
    x16(t);
}

#[cfg(not(feature = "w512"))]
fn check<T: magetypes::simd::backends::F32x4Convert + magetypes::simd::backends::F32x8Convert>(
    t: T,
) {
    x4(t);
    x8(t);
}

#[test]
fn scalar() {
    check(ScalarToken);
}
#[test]
fn selected_backend() {
    let tier = std::env::var("ARCHMAGE_TEST_TIER").unwrap_or_else(|_| "scalar".into());
    match tier.as_str() {
        "scalar" => check(ScalarToken),
        #[cfg(target_arch = "x86_64")]
        "v3" => check(archmage::X64V3Token::summon().expect("required v3")),
        #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
        "v4" => check(archmage::X64V4Token::summon().expect("required v4")),
        #[cfg(target_arch = "aarch64")]
        "neon" => check(archmage::NeonToken::summon().expect("required neon")),
        #[cfg(target_arch = "wasm32")]
        "wasm128" => check(archmage::Wasm128Token::summon().expect("required wasm128")),
        _ => panic!("unsupported required tier {tier}"),
    }
}
