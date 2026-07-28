//! `recip()` is contracted as "precise reciprocal (1/x), full f32 precision".
//!
//! This pins that contract. The ARM NEON backend previously implemented it as
//! `vrecpeq_f32` + two Newton steps, which missed exactness by ~2 ULP *and*
//! was 1.35x slower than the hardware divide on Apple Silicon (measured over
//! 1 M elements: vdivq 0.107 ms/Melem exact, vrecpe+2N 0.145 ms/Melem ~2 ULP).
//!
//! The estimate-and-refine form is what `rcp_approx()` is for — it documents a
//! >=12-bit floor and is free to be inexact. `recip()` is not.
//!
//! If a backend regresses to an estimate, this test fails on the first input
//! whose reciprocal is not bit-exact.

use magetypes::simd::generic::{f32x4, f32x8};
use magetypes::simd::backends::{F32x4Convert, F32x8Convert};

/// Inputs spanning normal range plus the awkward cases: powers of two (exact
/// reciprocals), values near 1, very large and very small magnitudes, and
/// negatives.
fn inputs() -> Vec<f32> {
    let mut v = vec![
        1.0, 2.0, 4.0, 0.5, 0.25, 3.0, 7.0, 10.0, 100.0, 1e6, 1e-6, 1e12, 1e-12,
        -1.0, -2.0, -3.0, -0.125, 255.0, 1.0 / 3.0, core::f32::consts::PI,
    ];
    // Plus a deterministic spread so this is not just hand-picked cases.
    let mut s = 0x9e37_79b9u32;
    for _ in 0..4096 {
        s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        let x = (s >> 8) as f32 / 16_777_216.0;
        v.push(x * 200.0 - 100.0);
    }
    v.retain(|x| x.abs() > 1e-20 && x.is_finite());
    v
}

fn check_x4<T: F32x4Convert>(token: T) {
    for chunk in inputs().chunks(4) {
        let mut arr = [1.0f32; 4];
        arr[..chunk.len()].copy_from_slice(chunk);
        let got = f32x4::from_array(token, arr).recip().to_array();
        for (i, (&x, &g)) in arr.iter().zip(got.iter()).enumerate() {
            let want = 1.0f32 / x;
            assert_eq!(
                g.to_bits(),
                want.to_bits(),
                "f32x4::recip not exact at lane {i}: 1/{x} gave {g}, want {want} \
                 ({} ULP) — a backend has regressed to an estimate; use rcp_approx \
                 if an approximation is intended",
                (g.to_bits() as i64 - want.to_bits() as i64).abs()
            );
        }
    }
}

fn check_x8<T: F32x8Convert>(token: T) {
    for chunk in inputs().chunks(8) {
        let mut arr = [1.0f32; 8];
        arr[..chunk.len()].copy_from_slice(chunk);
        let got = f32x8::from_array(token, arr).recip().to_array();
        for (i, (&x, &g)) in arr.iter().zip(got.iter()).enumerate() {
            let want = 1.0f32 / x;
            assert_eq!(
                g.to_bits(),
                want.to_bits(),
                "f32x8::recip not exact at lane {i}: 1/{x} gave {g}, want {want} \
                 ({} ULP)",
                (g.to_bits() as i64 - want.to_bits() as i64).abs()
            );
        }
    }
}

#[test]
fn recip_is_exact_on_every_available_tier() {
    use archmage::SimdToken;

    #[cfg(target_arch = "aarch64")]
    if let Some(t) = archmage::NeonToken::summon() {
        check_x4(t);
        check_x8(t);
    }
    #[cfg(target_arch = "x86_64")]
    if let Some(t) = archmage::X64V3Token::summon() {
        check_x4(t);
        check_x8(t);
    }
    let t = archmage::ScalarToken::summon().expect("scalar tier always available");
    check_x4(t);
    check_x8(t);
}
