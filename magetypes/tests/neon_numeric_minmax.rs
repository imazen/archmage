//! NEON numeric min/max: quiet NaNs select the number, including reductions.
//! Signaling NaNs follow native FMINNM/FMAXNM rather than Rust's scalar max.
#![forbid(unsafe_code)]
#![cfg(target_arch = "aarch64")]
use archmage::{NeonToken, SimdToken};
use std::hint::black_box;

macro_rules! width {
    ($test:ident, $ty:ident, $n:literal, $float:ident, $uint:ident, $quiet:expr, $signal:expr) => {
        #[test]
        fn $test() {
            let t = NeonToken::summon().expect("NEON is required on aarch64");
            let quiet: $uint = $quiet;
            let signal: $uint = $signal;
            let sign: $uint = 1 << (<$uint>::BITS - 1);
            let numbers: [$float; 9] = [
                0.0, -0.0, 1.0, -1.0, <$float>::INFINITY,
                <$float>::NEG_INFINITY, <$float>::MIN_POSITIVE,
                <$float>::from_bits(1), -<$float>::from_bits(1),
            ];
            // Independent bit predicate and directed payloads. No std min/max oracle.
            for bits in [quiet, quiet | 123, quiet | sign, quiet | sign | 456] {
                let nan = <$float>::from_bits(black_box(bits));
                for number in numbers {
                    let a = magetypes::simd::generic::$ty::splat_t(t, nan);
                    let b = magetypes::simd::generic::$ty::splat_t(t, black_box(number));
                    for out in [a.min(b), b.min(a), a.max(b), b.max(a)] {
                        assert_eq!(out.to_array().map($float::to_bits), [number.to_bits(); $n]);
                    }
                    let mixed = magetypes::simd::generic::$ty::from_array_t(t,
                        core::array::from_fn(|i| if i == $n - 1 { number } else { nan }));
                    assert_eq!(mixed.reduce_min().to_bits(), number.to_bits());
                    assert_eq!(mixed.reduce_max().to_bits(), number.to_bits());
                }
                let a = magetypes::simd::generic::$ty::splat_t(t, nan);
                assert!(a.min(a).to_array().into_iter().all($float::is_nan));
                assert!(a.max(a).to_array().into_iter().all($float::is_nan));
                assert!(a.reduce_min().is_nan());
                assert!(a.reduce_max().is_nan());
            }
            let positive = magetypes::simd::generic::$ty::splat_t(t, black_box(0.0));
            let negative = magetypes::simd::generic::$ty::splat_t(t, black_box(-0.0));
            for out in [positive.min(negative), negative.min(positive)] {
                assert_eq!(out.to_array().map($float::to_bits), [sign; $n]);
            }
            for out in [positive.max(negative), negative.max(positive)] {
                assert_eq!(out.to_array().map($float::to_bits), [0; $n]);
            }
            for bits in [signal, signal | sign] {
                let a = magetypes::simd::generic::$ty::splat_t(t,
                    <$float>::from_bits(black_box(bits)));
                let b = magetypes::simd::generic::$ty::splat_t(t, black_box(1.0));
                for out in [a.min(b), b.min(a), a.max(b), b.max(a)] {
                    assert!(out.to_array().into_iter().all($float::is_nan));
                }
            }
        }
    };
}
width!(f32x4, f32x4, 4, f32, u32, 0x7fc0_0000, 0x7f80_0001);
width!(f32x8, f32x8, 8, f32, u32, 0x7fc0_0000, 0x7f80_0001);
width!(f64x2, f64x2, 2, f64, u64, 0x7ff8_0000_0000_0000, 0x7ff0_0000_0000_0001);
width!(f64x4, f64x4, 4, f64, u64, 0x7ff8_0000_0000_0000, 0x7ff0_0000_0000_0001);
#[cfg(feature = "w512")]
width!(f32x16, f32x16, 16, f32, u32, 0x7fc0_0000, 0x7f80_0001);
#[cfg(feature = "w512")]
width!(f64x8, f64x8, 8, f64, u64, 0x7ff8_0000_0000_0000, 0x7ff0_0000_0000_0001);
