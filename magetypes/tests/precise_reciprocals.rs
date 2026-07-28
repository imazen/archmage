//! `recip()` and `rsqrt()` are contracted as full f32 precision.
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

fn check_x4<T: F32x4Convert>(token: T, max_ulp: i64, tier: &str) {
    // rsqrt is only defined for non-negative input.
    let pos: Vec<f32> = inputs().iter().map(|x| x.abs() + 1e-6).collect();
    for chunk in pos.chunks(4) {
        let mut arr = [1.0f32; 4];
        arr[..chunk.len()].copy_from_slice(chunk);
        let got = f32x4::from_array(token, arr).rsqrt().to_array();
        for (i, (&x, &g)) in arr.iter().zip(got.iter()).enumerate() {
            let want = 1.0f32 / x.sqrt();
            let ulp = (g.to_bits() as i64 - want.to_bits() as i64).abs();
            assert!(
                ulp <= max_ulp,
                "{tier} f32x4::rsqrt exceeded its pinned bound at lane {i}: \
                 1/sqrt({x}) gave {g}, want {want} ({ulp} ULP > {max_ulp}) — a backend \
                 has regressed; use rsqrt_approx if an approximation is intended"
            );
        }
    }
    for chunk in inputs().chunks(4) {
        let mut arr = [1.0f32; 4];
        arr[..chunk.len()].copy_from_slice(chunk);
        let got = f32x4::from_array(token, arr).recip().to_array();
        for (i, (&x, &g)) in arr.iter().zip(got.iter()).enumerate() {
            let want = 1.0f32 / x;
            let ulp = (g.to_bits() as i64 - want.to_bits() as i64).abs();
            assert!(
                ulp <= max_ulp,
                "{tier} f32x4::recip exceeded its pinned bound at lane {i}: \
                 1/{x} gave {g}, want {want} ({ulp} ULP > {max_ulp}) — a backend has \
                 regressed; use rcp_approx if an approximation is intended"
            );
        }
    }
}

fn check_x8<T: F32x8Convert>(token: T, max_ulp: i64, tier: &str) {
    for chunk in inputs().chunks(8) {
        let mut arr = [1.0f32; 8];
        arr[..chunk.len()].copy_from_slice(chunk);
        let got = f32x8::from_array(token, arr).recip().to_array();
        for (i, (&x, &g)) in arr.iter().zip(got.iter()).enumerate() {
            let want = 1.0f32 / x;
            let ulp = (g.to_bits() as i64 - want.to_bits() as i64).abs();
            assert!(
                ulp <= max_ulp,
                "{tier} f32x8::recip exceeded its pinned bound at lane {i}: \
                 1/{x} gave {g}, want {want} ({ulp} ULP > {max_ulp})"
            );
        }
    }
}

/// Exactness is asserted where the backend computes these with exact ops, and
/// a PINNED bound is asserted where it does not — rather than skipping the
/// non-conforming tier, which would let it drift unnoticed.
///
/// x86 is the non-conforming one: `x86_v3.rs` implements `recip` as
/// `rcp_approx` + one Newton step and `rsqrt` likewise, so neither is
/// bit-exact despite both being documented "full f32 precision". That is the
/// same defect fixed on ARM here, but the ARM fix was justified by a
/// measurement showing the exact op is also FASTER on that core; on x86 the
/// divide/estimate tradeoff genuinely differs (higher divide latency) and is
/// unmeasurable from this machine. Changing it unmeasured would be trading a
/// known-correct answer for an unknown regression, so the bound is pinned at
/// the observed value instead and the decision left to an x86 run.
#[test]
fn recip_and_rsqrt_meet_their_precision_contract() {
    use archmage::SimdToken;

    // Exact: the backend divides.
    #[cfg(target_arch = "aarch64")]
    if let Some(t) = archmage::NeonToken::summon() {
        check_x4(t, 0, "neon");
        check_x8(t, 0, "neon");
    }

    // NOT exact — pinned, see the doc comment above. 4 ULP is a snug bound on
    // rcp_approx + 1 Newton; a regression to the raw ~12-bit estimate would be
    // orders of magnitude worse and would trip this.
    #[cfg(target_arch = "x86_64")]
    if let Some(t) = archmage::X64V3Token::summon() {
        check_x4(t, 4, "x86_v3");
        check_x8(t, 4, "x86_v3");
    }

    // Scalar computes 1.0 / x directly.
    let t = archmage::ScalarToken::summon().expect("scalar tier always available");
    check_x4(t, 0, "scalar");
    check_x8(t, 0, "scalar");
}
