//! Uniform-saturation pins for `to_i32_saturating` — the issue #80 fix — and
//! a document-the-divergence check on bare `to_i32`.
//!
//! `to_i32_saturating` must equal Rust scalar `as` (saturate to
//! `i32::MIN`/`i32::MAX`, NaN → 0) lane-for-lane on EVERY backend. Bare
//! `to_i32` keeps each backend's native single-instruction behavior — the
//! x86 `cvttps` `i32::MIN` sentinel included — which this file also pins so
//! the documented divergence can't silently change shape.

use archmage::{ScalarToken, SimdToken};
use magetypes::simd::generic::{f32x4, f32x8, f64x2, f64x4, i32x4};
#[cfg(feature = "w512")]
use magetypes::simd::generic::{f32x16, f64x8, i32x8};

/// Every interesting class: in-range (incl. fractional + negative), the exact
/// f32 values nearest the i32 boundary from both sides, overflow both ways,
/// infinities, NaN, and signed zero.
fn inputs() -> Vec<f32> {
    vec![
        0.0,
        -0.0,
        0.5,
        -0.5,
        1.5,
        -1.5,
        123456.78,
        -123456.78,
        2_147_483_520.0,  // largest f32 < 2^31
        -2_147_483_648.0, // exactly i32::MIN
        2_147_483_648.0,  // 2^31 — overflows
        -2_147_483_904.0, // below i32::MIN — overflows
        3.0e9,
        -3.0e9,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::NAN,
        f32::MAX,
        f32::MIN,
    ]
}

macro_rules! check_saturating {
    ($ty:ident, $lanes:literal, $t:expr, $tier:expr) => {{
        let (t, tier) = ($t, $tier);
        for chunk in inputs().chunks($lanes) {
            let mut arr = [0.0f32; $lanes];
            arr[..chunk.len()].copy_from_slice(chunk);
            let got = $ty::from_array_t(t, arr).to_i32_saturating().to_array();
            for (i, &x) in arr.iter().enumerate() {
                let want = x as i32; // Rust `as` IS the contract
                assert_eq!(
                    got[i],
                    want,
                    "{tier} {}::to_i32_saturating lane {i}: input {x:?} gave {} want {want}",
                    stringify!($ty),
                    got[i]
                );
            }
            // Bare to_i32 must AGREE with saturating wherever lanes are
            // in-range — the divergence is confined to OOR/NaN lanes.
            let bare = $ty::from_array_t(t, arr).to_i32().to_array();
            for (i, &x) in arr.iter().enumerate() {
                if x.is_finite() && (-2_147_483_648.0..2_147_483_648.0).contains(&x) {
                    assert_eq!(
                        bare[i],
                        x as i32,
                        "{tier} {}::to_i32 in-range lane {i} diverged on {x:?}",
                        stringify!($ty)
                    );
                }
            }
        }
    }};
}

#[test]
fn scalar_backend_saturating() {
    let t = ScalarToken::summon().unwrap();
    check_saturating!(f32x4, 4, t, "scalar");
    check_saturating!(f32x8, 8, t, "scalar");
    #[cfg(feature = "w512")]
    check_saturating!(f32x16, 16, t, "scalar");
}

#[cfg(target_arch = "x86_64")]
#[test]
fn x64v3_backend_saturating() {
    match archmage::X64V3Token::summon() {
        Some(t) => {
            check_saturating!(f32x4, 4, t, "x86_v3");
            check_saturating!(f32x8, 8, t, "x86_v3");
            #[cfg(feature = "w512")]
            check_saturating!(f32x16, 16, t, "x86_v3");

            // Pin the DOCUMENTED bare-op divergence so it cannot drift: x86
            // cvttps gives the i32::MIN sentinel for +overflow AND NaN.
            let v = f32x4::from_array_t(t, [3.0e9, -3.0e9, f32::NAN, 1.5]);
            assert_eq!(v.to_i32().to_array(), [i32::MIN, i32::MIN, i32::MIN, 1]);
        }
        None => {
            #[cfg(feature = "std")]
            assert!(
                !std::arch::is_x86_feature_detected!("avx2"),
                "host reports AVX2 but X64V3Token did not summon"
            );
        }
    }
}

#[cfg(all(target_arch = "x86_64", feature = "avx512", feature = "w512"))]
#[test]
fn x64v4_backend_saturating() {
    if let Some(t) = archmage::X64V4Token::summon() {
        check_saturating!(f32x16, 16, t, "x86_v4");
    }
    if let Some(t) = archmage::X64V4xToken::summon() {
        check_saturating!(f32x16, 16, t, "x86_v4x");
    }
}

#[cfg(target_arch = "aarch64")]
#[test]
fn neon_backend_saturating() {
    if let Some(t) = archmage::NeonToken::summon() {
        check_saturating!(f32x4, 4, t, "neon");
        check_saturating!(f32x8, 8, t, "neon");
        #[cfg(feature = "w512")]
        check_saturating!(f32x16, 16, t, "neon");

        // NEON's native op already saturates with NaN→0 — bare == saturating.
        let v = f32x4::from_array_t(t, [3.0e9, -3.0e9, f32::NAN, 1.5]);
        assert_eq!(v.to_i32().to_array(), [i32::MAX, i32::MIN, 0, 1]);
    }
}

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
#[test]
fn wasm128_backend_saturating() {
    if let Some(t) = archmage::Wasm128Token::summon() {
        check_saturating!(f32x4, 4, t, "wasm128");
        check_saturating!(f32x8, 8, t, "wasm128");
        #[cfg(feature = "w512")]
        check_saturating!(f32x16, 16, t, "wasm128");
    }
}

fn f64_inputs() -> Vec<f64> {
    vec![
        0.0,
        -0.0,
        0.5,
        -0.5,
        1.5,
        -1.5,
        2.5,
        -2.5,
        123456.78,
        -123456.78,
        2_147_483_647.0,     // exact i32::MAX
        2_147_483_647.999_9, // truncates to i32::MAX; rounds to 2^31
        -2_147_483_648.0,    // exact i32::MIN
        -2_147_483_648.9,    // truncates toward zero to i32::MIN; rounds below
        2_147_483_648.0,     // 2^31 — overflows i32
        -2_147_483_649.0,    // below i32::MIN — overflows i32
        1e20,
        -1e20,
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::NAN,
    ]
}

fn check_f64_i32_tier<
    T: magetypes::simd::backends::F64x2Backend
        + magetypes::simd::backends::F64x2Convert
        + magetypes::simd::backends::F64x4Backend
        + magetypes::simd::backends::F64x4Convert
        + magetypes::simd::backends::I32x4Backend
        + archmage::SimdToken
        + Copy,
>(
    t: T,
    tier: &str,
) {
    let vals = f64_inputs();
    for chunk in vals.chunks(2) {
        let arr = [chunk[0], *chunk.get(1).unwrap_or(&0.0)];
        let v = f64x2::from_array_t(t, arr);
        let sat = v.to_i32_low_saturating().to_array();
        assert_eq!(sat[0], arr[0] as i32, "{tier} f64x2 sat[0] on {:?}", arr[0]);
        assert_eq!(sat[1], arr[1] as i32, "{tier} f64x2 sat[1] on {:?}", arr[1]);
        assert_eq!(sat[2], 0, "{tier} f64x2 sat high lane 2 must be zero");
        assert_eq!(sat[3], 0, "{tier} f64x2 sat high lane 3 must be zero");

        let bare = v.to_i32_low().to_array();
        let rnd = v.to_i32_low_round().to_array();
        assert_eq!([bare[2], bare[3]], [0, 0]);
        assert_eq!([rnd[2], rnd[3]], [0, 0]);
        for i in 0..2 {
            let x = arr[i];
            if x.is_finite() && x > -2_147_483_649.0 && x < 2_147_483_648.0 {
                assert_eq!(bare[i], x as i32, "{tier} f64x2 bare[{i}] on {x:?}");
            }
            let xr = x.round_ties_even();
            if xr.is_finite() && (-2_147_483_648.0..=2_147_483_647.0).contains(&xr) {
                assert_eq!(rnd[i], xr as i32, "{tier} f64x2 round[{i}] on {x:?}");
            }
        }
    }

    for chunk in vals.chunks(4) {
        let mut arr = [0.0f64; 4];
        arr[..chunk.len()].copy_from_slice(chunk);
        let v = f64x4::from_array_t(t, arr);
        let sat = v.to_i32_saturating().to_array();
        let bare = v.to_i32().to_array();
        let rnd = v.to_i32_round().to_array();
        for i in 0..4 {
            let x = arr[i];
            assert_eq!(sat[i], x as i32, "{tier} f64x4 sat[{i}] on {x:?}");
            if x.is_finite() && x > -2_147_483_649.0 && x < 2_147_483_648.0 {
                assert_eq!(bare[i], x as i32, "{tier} f64x4 bare[{i}] on {x:?}");
            }
            let xr = x.round_ties_even();
            if xr.is_finite() && (-2_147_483_648.0..=2_147_483_647.0).contains(&xr) {
                assert_eq!(rnd[i], xr as i32, "{tier} f64x4 round[{i}] on {x:?}");
            }
        }
    }

    // Roundtrip i32 -> f64
    let i4 = i32x4::from_array_t(t, [-1_000_000_007, 0, 42, i32::MAX]);
    assert_eq!(
        i4.to_f64x2_low().to_array(),
        [-1_000_000_007.0, 0.0],
        "{tier} i32x4::to_f64x2_low"
    );
    assert_eq!(
        i4.to_f64().to_array(),
        [-1_000_000_007.0, 0.0, 42.0, i32::MAX as f64],
        "{tier} i32x4::to_f64"
    );
    assert_eq!(
        f64x2::from_i32_low_t(t, i4).to_array(),
        [-1_000_000_007.0, 0.0]
    );
    assert_eq!(
        f64x4::from_i32_t(t, i4).to_array(),
        [-1_000_000_007.0, 0.0, 42.0, i32::MAX as f64]
    );
}

#[cfg(feature = "w512")]
fn check_f64x8_i32<
    T: magetypes::simd::backends::F64x8Backend
        + magetypes::simd::backends::F64x8Convert
        + magetypes::simd::backends::I32x8Backend
        + archmage::SimdToken
        + Copy,
>(
    t: T,
    tier: &str,
) {
    for chunk in f64_inputs().chunks(8) {
        let mut arr = [0.0f64; 8];
        arr[..chunk.len()].copy_from_slice(chunk);
        let v = f64x8::from_array_t(t, arr);
        let sat = v.to_i32_saturating().to_array();
        let bare = v.to_i32().to_array();
        let rnd = v.to_i32_round().to_array();
        for i in 0..8 {
            let x = arr[i];
            assert_eq!(sat[i], x as i32, "{tier} f64x8 sat[{i}] on {x:?}");
            if x.is_finite() && x > -2_147_483_649.0 && x < 2_147_483_648.0 {
                assert_eq!(bare[i], x as i32, "{tier} f64x8 bare[{i}] on {x:?}");
            }
            let xr = x.round_ties_even();
            if xr.is_finite() && (-2_147_483_648.0..=2_147_483_647.0).contains(&xr) {
                assert_eq!(rnd[i], xr as i32, "{tier} f64x8 round[{i}] on {x:?}");
            }
        }
    }
    let i8v = i32x8::from_array_t(t, [i32::MIN, -7, 0, 1, 2, 99, -123456, i32::MAX]);
    assert_eq!(
        i8v.to_f64().to_array(),
        [
            i32::MIN as f64,
            -7.0,
            0.0,
            1.0,
            2.0,
            99.0,
            -123456.0,
            i32::MAX as f64
        ],
        "{tier} i32x8::to_f64"
    );
    assert_eq!(
        f64x8::from_i32_t(t, i8v).to_array(),
        i8v.to_f64().to_array()
    );
}

#[test]
fn scalar_f64_to_i32_saturating() {
    let t = ScalarToken::summon().unwrap();
    check_f64_i32_tier(t, "scalar");
    #[cfg(feature = "w512")]
    check_f64x8_i32(t, "scalar");
}

#[cfg(target_arch = "x86_64")]
#[test]
fn x64_f64_to_i32_saturating() {
    if let Some(t) = archmage::X64V3Token::summon() {
        check_f64_i32_tier(t, "x86_v3");
        #[cfg(feature = "w512")]
        check_f64x8_i32(t, "x86_v3");
    }
    #[cfg(all(feature = "avx512", feature = "w512"))]
    if let Some(t) = archmage::X64V4Token::summon() {
        check_f64_i32_tier(t, "x86_v4");
        check_f64x8_i32(t, "x86_v4");
    }
}

#[cfg(target_arch = "aarch64")]
#[test]
fn neon_f64_to_i32_saturating() {
    if let Some(t) = archmage::NeonToken::summon() {
        check_f64_i32_tier(t, "neon");
        #[cfg(feature = "w512")]
        check_f64x8_i32(t, "neon");
    }
}

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
#[test]
fn wasm128_f64_to_i32_saturating() {
    if let Some(t) = archmage::Wasm128Token::summon() {
        check_f64_i32_tier(t, "wasm128");
        #[cfg(feature = "w512")]
        check_f64x8_i32(t, "wasm128");
    }
}
