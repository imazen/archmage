//! Uniform-saturation pins for `to_i32_saturating` — the issue #80 fix — and
//! a document-the-divergence check on bare `to_i32`.
//!
//! `to_i32_saturating` must equal Rust scalar `as` (saturate to
//! `i32::MIN`/`i32::MAX`, NaN → 0) lane-for-lane on EVERY backend. Bare
//! `to_i32` keeps each backend's native single-instruction behavior — the
//! x86 `cvttps` `i32::MIN` sentinel included — which this file also pins so
//! the documented divergence can't silently change shape.

use archmage::{ScalarToken, SimdToken};
use magetypes::simd::backends::{F32x4Convert, F32x8Convert};
use magetypes::simd::generic::{f32x4, f32x8};
#[cfg(feature = "w512")]
use magetypes::simd::{backends::F32x16Convert, generic::f32x16};

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
            let got = $ty::from_array(t, arr).to_i32_saturating().to_array();
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
            let bare = $ty::from_array(t, arr).to_i32().to_array();
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
            let v = f32x4::from_array(t, [3.0e9, -3.0e9, f32::NAN, 1.5]);
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
        let v = f32x4::from_array(t, [3.0e9, -3.0e9, f32::NAN, 1.5]);
        assert_eq!(v.to_i32().to_array(), [i32::MAX, i32::MIN, 0, 1]);
    }
}

#[cfg(target_arch = "wasm32")]
#[test]
fn wasm128_backend_saturating() {
    if let Some(t) = archmage::Wasm128Token::summon() {
        check_saturating!(f32x4, 4, t, "wasm128");
        check_saturating!(f32x8, 8, t, "wasm128");
        #[cfg(feature = "w512")]
        check_saturating!(f32x16, 16, t, "wasm128");
    }
}
