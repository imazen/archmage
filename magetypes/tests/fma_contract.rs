//! Fixed, exact-rational oracles: no dependency on the host's std/libm FMA.
//! Portable results are bit-exact except NaNs, under round-to-nearest with
//! gradual underflow. Bare arithmetic follows the backend's native contract.
#![forbid(unsafe_code)]
#[cfg(any(
    target_arch = "x86_64",
    target_arch = "aarch64",
    target_arch = "wasm32"
))]
use archmage::SimdToken;
use archmage::{ScalarToken, magetypes};
#[path = "common/fma_vectors.rs"]
mod fixtures;
#[path = "common/fma_expected.rs"]
mod native;
use core::any::Any;
use native::FmaExpected;

macro_rules! same {
    ($got:expr, $want:expr) => {{
        let (got, want) = ($got, $want);
        if want.is_nan() {
            assert!(got.is_nan(), "expected NaN, got {got:?}");
        } else {
            assert_eq!(got.to_bits(), want.to_bits(), "{got:?} != {want:?}");
        }
    }};
}

macro_rules! check {
    ($token:expr, $ty:ident, $float:ident, $n:expr, $cases:expr) => {
        for rows in $cases.chunks_exact($n) {
            let a = core::array::from_fn(|i| $float::from_bits(rows[i][0]));
            let b = core::array::from_fn(|i| $float::from_bits(rows[i][1]));
            let c = core::array::from_fn(|i| $float::from_bits(rows[i][2]));
            let x = magetypes::simd::generic::$ty::from_array_t($token, a);
            let y = magetypes::simd::generic::$ty::from_array_t($token, b);
            let z = magetypes::simd::generic::$ty::from_array_t($token, c);
            let minus_z = magetypes::simd::generic::$ty::from_array_t($token, c.map(|v| -v));
            let add = x.mul_add_portable(y, z).to_array();
            let sub = x.mul_sub_portable(y, minus_z).to_array();
            let fast_add = x.mul_add_relaxed(y, z).to_array();
            let fast_sub = x.mul_sub_relaxed(y, minus_z).to_array();
            for i in 0..$n {
                let expected = $float::from_bits(rows[i][3]);
                same!(add[i], expected);
                same!(sub[i], expected);
                // Hardware native paths share the exact oracle. Scalar and WASM
                // use separate multiply/add or the engine's relaxed instruction.
                let fast = if ($token).type_id() == core::any::TypeId::of::<ScalarToken>()
                    || cfg!(target_arch = "wasm32")
                {
                    a[i].mul_add_expected(b[i], c[i], $token)
                } else {
                    expected
                };
                same!(fast_add[i], fast);
                same!(fast_sub[i], fast);
            }
        }
    };
}

#[magetypes(v3, v4, v4x, neon, wasm128, scalar)]
fn vectors(token: Token) {
    check!(token, f32x4, f32, 4, fixtures::F32);
    check!(token, f32x8, f32, 8, fixtures::F32);
    check!(token, f64x2, f64, 2, fixtures::F64);
    check!(token, f64x4, f64, 4, fixtures::F64);
    #[cfg(feature = "w512")]
    {
        check!(token, f32x16, f32, 16, fixtures::F32);
        check!(token, f64x8, f64, 8, fixtures::F64);
    }
}

#[test]
fn scalar_and_software_contract() {
    vectors_scalar(ScalarToken);
    for [a, b, c, expected] in fixtures::F32.map(|row| row.map(f32::from_bits)) {
        same!(magetypes::nostd_math::fmaf(a, b, c), expected);
        same!(magetypes::nostd_math::fmaf_soft(a, b, c), expected);
        use magetypes::simd::f32x1 as V;
        same!(V(a).mul_add_portable(V(b), V(c)).0, expected);
        same!(V(a).mul_sub_portable(V(b), V(-c)).0, expected);
        same!(V(a).mul_add_relaxed(V(b), V(c)).0, a * b + c);
    }
    for [a, b, c, expected] in fixtures::F64.map(|row| row.map(f64::from_bits)) {
        same!(magetypes::nostd_math::fma(a, b, c), expected);
        use magetypes::simd::f64x1 as V;
        same!(V(a).mul_add_portable(V(b), V(c)).0, expected);
        same!(V(a).mul_add_relaxed(V(b), V(c)).0, a * b + c);
    }
}

// Selection is explicit: requested tiers must execute, never silently skip.
// ARCHMAGE_TEST_TIER forces a tier; auto selects the best available backend.
#[test]
fn selected_backend_contract() {
    match std::env::var("ARCHMAGE_TEST_TIER")
        .as_deref()
        .unwrap_or("auto")
    {
        "scalar" => vectors_scalar(ScalarToken),
        #[cfg(target_arch = "x86_64")]
        "v3" => vectors_v3(archmage::X64V3Token::summon().expect("requested v3")),
        #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
        "v4" => vectors_v4(archmage::X64V4Token::summon().expect("requested v4")),
        #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
        "v4x" => vectors_v4x(archmage::X64V4xToken::summon().expect("requested v4x")),
        #[cfg(target_arch = "aarch64")]
        "neon" => vectors_neon(archmage::NeonToken::summon().expect("requested neon")),
        #[cfg(target_arch = "wasm32")]
        "wasm128" => vectors_wasm128(archmage::Wasm128Token::summon().expect("requested wasm128")),
        "auto" => {
            archmage::incant!(vectors(), [v4x, v4, v3, neon, wasm128, scalar]);
        }
        tier => panic!("unsupported requested tier {tier}"),
    }
}
