//! Float comparisons treat NaN the same way on every backend, as Rust's
//! operators do: a NaN lane is unequal to everything (`simd_ne` sets it, as
//! `!=` is true) and every other comparison with it is false.
//!
//! x86-64-v3 used an ordered not-equal through 0.9.30, so `simd_ne` there was
//! false for NaN lanes while NEON, WASM, the scalar backend and the AVX-512
//! types returned true.

#![cfg(feature = "std")]

use archmage::{ScalarToken, SimdToken};
use magetypes::simd::generic::{f32x4, f32x8, f64x2, f64x4};

#[cfg(target_arch = "x86_64")]
type Native = archmage::X64V3Token;
#[cfg(target_arch = "aarch64")]
type Native = archmage::NeonToken;
#[cfg(target_arch = "wasm32")]
type Native = archmage::Wasm128Token;
#[cfg(not(any(
    target_arch = "x86_64",
    target_arch = "aarch64",
    target_arch = "wasm32"
)))]
type Native = archmage::ScalarToken;

/// Lane pairs: NaN vs NaN, 1 vs NaN, NaN vs 1, 0 vs 0.
const EXPECT: [[bool; 4]; 6] = [
    [true, true, true, false],    // ne
    [false, false, false, true],  // eq
    [false, false, false, false], // lt
    [false, false, false, true],  // le
    [false, false, false, false], // gt
    [false, false, false, true],  // ge
];

macro_rules! masks {
    ($ty:ident, $elem:ty, $lanes:literal, $tok:expr) => {{
        let n = <$elem>::NAN;
        let a: [$elem; $lanes] = core::array::from_fn(|i| [n, 1.0, n, 0.0][i % 4]);
        let b: [$elem; $lanes] = core::array::from_fn(|i| [n, n, 1.0, 0.0][i % 4]);
        let (a, b) = ($ty::from_array_t($tok, a), $ty::from_array_t($tok, b));
        let set = |m: $ty<_>| m.to_array().map(|x| x.to_bits() != 0);
        [
            set(a.simd_ne(b)),
            set(a.simd_eq(b)),
            set(a.simd_lt(b)),
            set(a.simd_le(b)),
            set(a.simd_gt(b)),
            set(a.simd_ge(b)),
        ]
    }};
}

macro_rules! check {
    ($ty:ident, $elem:ty, $lanes:literal, $tok:expr, $what:expr) => {{
        let got = masks!($ty, $elem, $lanes, $tok);
        for (op, (g, want)) in ["ne", "eq", "lt", "le", "gt", "ge"]
            .iter()
            .zip(got.iter().zip(EXPECT))
        {
            for (i, &x) in g.iter().enumerate() {
                assert_eq!(
                    x,
                    want[i % 4],
                    "{} {} simd_{op}: lane {i} {}",
                    $what,
                    stringify!($ty),
                    ["NaN vs NaN", "1 vs NaN", "NaN vs 1", "0 vs 0"][i % 4]
                );
            }
        }
    }};
}

#[test]
fn nan_comparisons_agree_across_backends() {
    check!(f32x4, f32, 4, ScalarToken, "scalar");
    check!(f32x8, f32, 8, ScalarToken, "scalar");
    check!(f64x2, f64, 2, ScalarToken, "scalar");
    check!(f64x4, f64, 4, ScalarToken, "scalar");
    if let Some(t) = Native::summon() {
        check!(f32x4, f32, 4, t, "native");
        check!(f32x8, f32, 8, t, "native");
        check!(f64x2, f64, 2, t, "native");
        check!(f64x4, f64, 4, t, "native");
    }
    #[cfg(feature = "w512")]
    {
        use magetypes::simd::generic::{f32x16, f64x8};
        check!(f32x16, f32, 16, ScalarToken, "scalar");
        check!(f64x8, f64, 8, ScalarToken, "scalar");
        if let Some(t) = Native::summon() {
            check!(f32x16, f32, 16, t, "native");
            check!(f64x8, f64, 8, t, "native");
        }
        #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
        if let Some(t) = archmage::X64V4Token::summon() {
            check!(f32x4, f32, 4, t, "v4");
            check!(f32x8, f32, 8, t, "v4");
            check!(f32x16, f32, 16, t, "v4");
            check!(f64x8, f64, 8, t, "v4");
        }
    }
}
