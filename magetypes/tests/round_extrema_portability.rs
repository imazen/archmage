#![forbid(unsafe_code)]

use archmage::ScalarToken;
use magetypes::numerics::ExtremaOptions;
use magetypes::simd::{generic, scalar as single};
use magetypes::{nostd_math, scalar};

const PROPAGATE: u32 = ExtremaOptions::new().propagate_nan().bits();

macro_rules! checks {
    (
        $check:ident, $float:ty, $limit:expr, $nan:expr, $signaling:expr,
        $floor:ident, $ceil:ident, $round:ident, $trunc:ident,
        $floor_soft:ident, $ceil_soft:ident, $round_soft:ident,
        $min:ident, $max:ident, $min_with:ident, $max_with:ident,
        $single:ident, [$($vector:ident),+]
    ) => {
        fn $check() {
            let limit: $float = $limit;
            let cases: [$float; 22] = [
                0.0, -0.0, <$float>::from_bits(1), <$float>::MIN_POSITIVE.next_down(),
                <$float>::MIN_POSITIVE, 0.5, 1.5, 2.5,
                limit - 1.0, limit, limit + 1.0, <$float>::MAX,
                <$float>::INFINITY, <$float>::NEG_INFINITY,
                <$float>::from_bits($nan), <$float>::from_bits($signaling),
                0.5f32.next_down() as $float, 0.5f32.next_up() as $float,
                (0.5 as $float).next_down(), (0.5 as $float).next_up(),
                -(0.5 as $float).next_down(), -(0.5 as $float).next_up(),
            ];
            let inputs = cases.into_iter().chain(cases.map(|x| -x));
            let assert_value = |actual: $float, expected: $float| {
                if expected.is_nan() {
                    assert!(actual.is_nan(), "expected NaN, got {actual:?}");
                } else {
                    assert_eq!(actual.to_bits(), expected.to_bits());
                }
            };
            let rounding = |x: $float| {
                let expected = [
                    nostd_math::$floor_soft(x),
                    nostd_math::$ceil_soft(x),
                    nostd_math::$round_soft(x),
                    if x.is_nan() { x } else { x.trunc() },
                ];
                let actual = [scalar::$floor(x), scalar::$ceil(x), scalar::$round(x), scalar::$trunc(x)];
                for (actual, expected) in actual.into_iter().zip(expected) {
                    assert_value(actual, expected);
                }
                expected
            };
            for a in inputs.clone() {
                let [floor, ceil, round, trunc] = rounding(a);
                $(
                    let x = generic::$vector::splat_t(ScalarToken, a);
                    for (actual, expected) in [(x.floor(), floor), (x.ceil(), ceil), (x.round(), round), (x.trunc(), trunc)] {
                        for actual in actual.to_array() {
                            assert_value(actual, expected);
                        }
                    }
                )+
                for b in inputs.clone() {
                    let min = scalar::$min(a, b);
                    let max = scalar::$max(a, b);
                    let x = single::$single(a);
                    let y = single::$single(b);
                    assert_value(x.minimum(y).0, min);
                    assert_value(x.maximum(y).0, max);
                    assert_value(scalar::$min_with::<PROPAGATE>(a, b), min);
                    assert_value(scalar::$max_with::<PROPAGATE>(a, b), max);
                    $(
                        let x = generic::$vector::splat_t(ScalarToken, a);
                        let y = generic::$vector::splat_t(ScalarToken, b);
                        for actual in x.minimum(y).to_array() {
                            assert_value(actual, min);
                        }
                        for actual in x.maximum(y).to_array() {
                            assert_value(actual, max);
                        }
                    )+
                }
            }

            // Sample the bit space independently of the corner-case values.
            let mut bits = 0x243f_6a88_85a3_08d3u64;
            for _ in 0..65_536 {
                bits ^= bits << 13;
                bits ^= bits >> 7;
                bits ^= bits << 17;
                rounding(<$float>::from_bits(bits as _));
            }
        }
    };
}

checks!(
    check32,
    f32,
    8_388_608.0,
    0x7fc1_2345,
    0x7f81_2345,
    floor_f32,
    ceil_f32,
    round_f32,
    trunc_f32,
    floorf,
    ceilf,
    roundevenf,
    minimum_f32,
    maximum_f32,
    min_f32_with,
    max_f32_with,
    f32x1,
    [f32x4, f32x8]
);
checks!(
    check64,
    f64,
    4_503_599_627_370_496.0,
    0x7ff8_0000_0001_2345,
    0x7ff0_0000_0001_2345,
    floor_f64,
    ceil_f64,
    round_f64,
    trunc_f64,
    floor,
    ceil,
    roundeven,
    minimum_f64,
    maximum_f64,
    min_f64_with,
    max_f64_with,
    f64x1,
    [f64x2, f64x4]
);

#[test]
fn rounding_and_extrema_match_software() {
    check32();
    check64();

    // The runner selects this check with archmage/testable_dispatch enabled.
    // This binary has one test, so process-wide changes cannot race another test.
    if std::env::var_os("MAGETYPES_TEST_DISABLE_ROUND_EXTREMA").is_some() {
        #[cfg(target_arch = "x86_64")]
        check_disabled::<archmage::X64V2Token>(
            archmage::X64V2Token::dangerously_disable_token_process_wide,
        );
        #[cfg(target_arch = "aarch64")]
        check_disabled::<archmage::NeonToken>(
            archmage::NeonToken::dangerously_disable_token_process_wide,
        );
        #[cfg(target_arch = "wasm32")]
        check_disabled::<archmage::Wasm128Token>(
            archmage::Wasm128Token::dangerously_disable_token_process_wide,
        );
    }
}

#[cfg(any(
    target_arch = "x86_64",
    target_arch = "aarch64",
    target_arch = "wasm32"
))]
fn check_disabled<T: archmage::SimdToken>(
    disable: impl Fn(bool) -> Result<(), archmage::tokens::CompileTimeGuaranteedError>,
) {
    disable(true).expect("build with archmage/testable_dispatch");
    assert!(T::summon().is_none());
    check32();
    check64();
    disable(false).unwrap();
}
