//! `recip()` and `rsqrt()` are contracted as full precision, f32 *and* f64.
//!
//! This pins that contract. The ARM NEON backend previously implemented it as
//! `vrecpeq` + Newton steps, which missed exactness *and* was slower than the
//! hardware divide on Apple Silicon:
//!
//! - f32, measured over 1 M elements: vdivq 0.107 ms/Melem exact,
//!   vrecpe+2N 0.145 ms/Melem ~2 ULP (`recip`); rsqrt likewise ~3 ULP.
//! - f64, measured by reverting this file's backend: `vrecpeq_f64`/`vrsqrteq_f64`
//!   + three Newton steps lands 1 ULP off (`1/sqrt(1.000001)` →
//!   `0.9999995000003751`, want `0.999999500000375`).
//!
//! The estimate-and-refine form is what `rcp_approx()` is for — it documents a
//! >=12-bit floor and is free to be inexact. `recip()` is not.
//!
//! If a backend regresses to an estimate, this test fails on the first input
//! whose reciprocal is not bit-exact.
//!
//! **This test is the runtime half of a two-part guard.** The other half is
//! CI's `generate-check`: these NEON bodies are GENERATED, so a fix applied to
//! `magetypes/src/simd/impls/arm_neon.rs` alone is reverted by the next
//! `just generate`. The fix belongs in
//! `xtask/src/simd_types/backend_gen.rs`, and `just check-generated` catches
//! the mistake locally. See CLAUDE.md, "FIX THE GENERATOR, NOT ITS OUTPUT".

use magetypes::simd::backends::{F32x4Convert, F32x8Convert, F64x2Backend, F64x4Backend};
use magetypes::simd::generic::{f32x4, f32x8, f64x2, f64x4};
#[cfg(feature = "w512")]
use magetypes::simd::{
    backends::{F32x16Backend, F64x8Backend},
    generic::{f32x16, f64x8},
};

/// Inputs spanning normal range plus the awkward cases: powers of two (exact
/// reciprocals), values near 1, very large and very small magnitudes, and
/// negatives.
fn inputs() -> Vec<f32> {
    let mut v = vec![
        1.0,
        2.0,
        4.0,
        0.5,
        0.25,
        3.0,
        7.0,
        10.0,
        100.0,
        1e6,
        1e-6,
        1e12,
        1e-12,
        -1.0,
        -2.0,
        -3.0,
        -0.125,
        255.0,
        1.0 / 3.0,
        core::f32::consts::PI,
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

fn inputs_f64() -> Vec<f64> {
    inputs().iter().map(|&x| x as f64).collect()
}

fn check_f64x2<T: F64x2Backend>(token: T, max_ulp: i64, tier: &str) {
    let pos: Vec<f64> = inputs_f64().iter().map(|x| x.abs() + 1e-6).collect();
    for chunk in pos.chunks(2) {
        let mut arr = [1.0f64; 2];
        arr[..chunk.len()].copy_from_slice(chunk);
        let got = f64x2::from_array(token, arr).rsqrt().to_array();
        for (i, (&x, &g)) in arr.iter().zip(got.iter()).enumerate() {
            let want = 1.0f64 / x.sqrt();
            let ulp = (g.to_bits() as i64 - want.to_bits() as i64).abs();
            assert!(
                ulp <= max_ulp,
                "{tier} f64x2::rsqrt exceeded its pinned bound at lane {i}: \
                 1/sqrt({x}) gave {g}, want {want} ({ulp} ULP > {max_ulp}) — a backend \
                 has regressed; use rsqrt_approx if an approximation is intended"
            );
        }
    }
    for chunk in inputs_f64().chunks(2) {
        let mut arr = [1.0f64; 2];
        arr[..chunk.len()].copy_from_slice(chunk);
        let got = f64x2::from_array(token, arr).recip().to_array();
        for (i, (&x, &g)) in arr.iter().zip(got.iter()).enumerate() {
            let want = 1.0f64 / x;
            let ulp = (g.to_bits() as i64 - want.to_bits() as i64).abs();
            assert!(
                ulp <= max_ulp,
                "{tier} f64x2::recip exceeded its pinned bound at lane {i}: \
                 1/{x} gave {g}, want {want} ({ulp} ULP > {max_ulp}) — a backend has \
                 regressed; use rcp_approx if an approximation is intended"
            );
        }
    }
}

fn check_f64x4<T: F64x4Backend>(token: T, max_ulp: i64, tier: &str) {
    for chunk in inputs_f64().chunks(4) {
        let mut arr = [1.0f64; 4];
        arr[..chunk.len()].copy_from_slice(chunk);
        let got = f64x4::from_array(token, arr).recip().to_array();
        for (i, (&x, &g)) in arr.iter().zip(got.iter()).enumerate() {
            let want = 1.0f64 / x;
            let ulp = (g.to_bits() as i64 - want.to_bits() as i64).abs();
            assert!(
                ulp <= max_ulp,
                "{tier} f64x4::recip exceeded its pinned bound at lane {i}: \
                 1/{x} gave {g}, want {want} ({ulp} ULP > {max_ulp})"
            );
        }
    }
}

/// The 512-bit widths, gated on `w512` (polyfill on v3/NEON/WASM/scalar,
/// native on `X64V4Token`/`X64V4xToken` under `avx512`).
#[cfg(feature = "w512")]
fn check_f32x16<T: F32x16Backend>(token: T, max_ulp: i64, tier: &str) {
    for chunk in inputs().chunks(16) {
        let mut arr = [1.0f32; 16];
        arr[..chunk.len()].copy_from_slice(chunk);
        let got = f32x16::from_array(token, arr).recip().to_array();
        for (i, (&x, &g)) in arr.iter().zip(got.iter()).enumerate() {
            let want = 1.0f32 / x;
            let ulp = (g.to_bits() as i64 - want.to_bits() as i64).abs();
            assert!(
                ulp <= max_ulp,
                "{tier} f32x16::recip exceeded its pinned bound at lane {i}: \
                 1/{x} gave {g}, want {want} ({ulp} ULP > {max_ulp})"
            );
        }
        let pos = arr.map(|x| x.abs() + 1e-6);
        let got = f32x16::from_array(token, pos).rsqrt().to_array();
        for (i, (&x, &g)) in pos.iter().zip(got.iter()).enumerate() {
            let want = 1.0f32 / x.sqrt();
            let ulp = (g.to_bits() as i64 - want.to_bits() as i64).abs();
            assert!(
                ulp <= max_ulp,
                "{tier} f32x16::rsqrt exceeded its pinned bound at lane {i}: \
                 1/sqrt({x}) gave {g}, want {want} ({ulp} ULP > {max_ulp})"
            );
        }
    }
}

#[cfg(feature = "w512")]
fn check_f64x8<T: F64x8Backend>(token: T, max_ulp: i64, tier: &str) {
    for chunk in inputs_f64().chunks(8) {
        let mut arr = [1.0f64; 8];
        arr[..chunk.len()].copy_from_slice(chunk);
        let got = f64x8::from_array(token, arr).recip().to_array();
        for (i, (&x, &g)) in arr.iter().zip(got.iter()).enumerate() {
            let want = 1.0f64 / x;
            let ulp = (g.to_bits() as i64 - want.to_bits() as i64).abs();
            assert!(
                ulp <= max_ulp,
                "{tier} f64x8::recip exceeded its pinned bound at lane {i}: \
                 1/{x} gave {g}, want {want} ({ulp} ULP > {max_ulp})"
            );
        }
        let pos = arr.map(|x| x.abs() + 1e-6);
        let got = f64x8::from_array(token, pos).rsqrt().to_array();
        for (i, (&x, &g)) in pos.iter().zip(got.iter()).enumerate() {
            let want = 1.0f64 / x.sqrt();
            let ulp = (g.to_bits() as i64 - want.to_bits() as i64).abs();
            assert!(
                ulp <= max_ulp,
                "{tier} f64x8::rsqrt exceeded its pinned bound at lane {i}: \
                 1/sqrt({x}) gave {g}, want {want} ({ulp} ULP > {max_ulp})"
            );
        }
    }
}

/// IEEE rail behavior (issue #64): `recip(±0) = ±inf`, `recip(±inf) = ±0`,
/// `rsqrt(+0) = +inf`, `rsqrt(-0) = -inf`, `rsqrt(+inf) = +0`, `rsqrt(x<0)` is
/// NaN — bit-compared against the scalar `1.0 / x` / `1.0 / x.sqrt()`
/// reference on every backend. This is what the estimate-and-refine forms got
/// wrong: refining `r ≈ 1/a` computes `a*r = inf*0 = NaN` at the rails, which
/// is how `((-v).exp_midp() + one).recip()` NaN'd 97% of a sigmoid tensor.
macro_rules! rails_checker {
    ($fname:ident, $ty:ident, $elem:ty, $lanes:expr, $Trait:ident $(, $cfg:meta)?) => {
        $(#[$cfg])?
        fn $fname<T: $Trait>(token: T, tier: &str) {
            let recip_in: [$elem; $lanes] = core::array::from_fn(|i| {
                [0.0, -0.0, <$elem>::INFINITY, <$elem>::NEG_INFINITY][i % 4]
            });
            let got = $ty::from_array(token, recip_in).recip().to_array();
            for (i, (&x, &g)) in recip_in.iter().zip(got.iter()).enumerate() {
                let want = (1.0 as $elem) / x;
                assert!(
                    g.to_bits() == want.to_bits(),
                    "{tier} {}::recip rail at lane {i}: recip({x:?}) gave {g:?}, \
                     want {want:?} (issue #64)",
                    stringify!($ty)
                );
            }
            let rsqrt_in: [$elem; $lanes] = core::array::from_fn(|i| {
                [0.0, -0.0, <$elem>::INFINITY, -1.0][i % 4]
            });
            let got = $ty::from_array(token, rsqrt_in).rsqrt().to_array();
            for (i, (&x, &g)) in rsqrt_in.iter().zip(got.iter()).enumerate() {
                let want = (1.0 as $elem) / x.sqrt();
                if want.is_nan() {
                    assert!(
                        g.is_nan(),
                        "{tier} {}::rsqrt rail at lane {i}: rsqrt({x:?}) gave {g:?}, want NaN",
                        stringify!($ty)
                    );
                } else {
                    assert!(
                        g.to_bits() == want.to_bits(),
                        "{tier} {}::rsqrt rail at lane {i}: rsqrt({x:?}) gave {g:?}, \
                         want {want:?} (issue #64)",
                        stringify!($ty)
                    );
                }
            }
        }
    };
}

rails_checker!(rails_f32x4, f32x4, f32, 4, F32x4Convert);
rails_checker!(rails_f32x8, f32x8, f32, 8, F32x8Convert);
rails_checker!(rails_f64x2, f64x2, f64, 2, F64x2Backend);
rails_checker!(rails_f64x4, f64x4, f64, 4, F64x4Backend);
rails_checker!(
    rails_f32x16,
    f32x16,
    f32,
    16,
    F32x16Backend,
    cfg(feature = "w512")
);
rails_checker!(
    rails_f64x8,
    f64x8,
    f64,
    8,
    F64x8Backend,
    cfg(feature = "w512")
);

/// Every width on one token: full-precision sweep at `max_ulp` + exact rails.
macro_rules! check_all_widths {
    ($t:expr, $max_ulp:expr, $tier:expr) => {{
        let (t, ulp, tier) = ($t, $max_ulp, $tier);
        check_x4(t, ulp, tier);
        check_x8(t, ulp, tier);
        check_f64x2(t, ulp, tier);
        check_f64x4(t, ulp, tier);
        rails_f32x4(t, tier);
        rails_f32x8(t, tier);
        rails_f64x2(t, tier);
        rails_f64x4(t, tier);
        #[cfg(feature = "w512")]
        {
            check_f32x16(t, ulp, tier);
            check_f64x8(t, ulp, tier);
            rails_f32x16(t, tier);
            rails_f64x8(t, tier);
        }
    }};
}

/// Exactness is asserted on every backend: `recip`/`rsqrt` are contracted as
/// exact IEEE division (and sqrt), 0 ULP against the scalar reference, rails
/// included.
///
/// History: NEON shipped exact division first (justified by aarch64
/// measurements showing exact is also FASTER there), while x86 f32 stayed on
/// `rcp_approx` + one Newton step with a pinned 4 ULP bound and "the decision
/// left to an x86 run" — the divide/estimate tradeoff was unmeasurable from
/// that machine. That run exists now
/// (`benchmarks/recip_x86_zen5-9950x3d_2026-09-03.md`, Zen 5): exact division
/// costs ~1.9x (recip) / ~3.6x (rsqrt) in an L1 throughput microbench, and it
/// ships anyway — the 0 ULP contract, the IEEE rails (issue #64), and
/// cross-backend consistency win in the full-precision tier. The estimate
/// tier (`rcp_approx`/`rsqrt_approx`) keeps the throughput path.
#[test]
fn recip_and_rsqrt_meet_their_precision_contract() {
    use archmage::SimdToken;

    #[cfg(target_arch = "aarch64")]
    if let Some(t) = archmage::NeonToken::summon() {
        check_all_widths!(t, 0, "neon");
    }

    #[cfg(target_arch = "x86_64")]
    if let Some(t) = archmage::X64V3Token::summon() {
        check_all_widths!(t, 0, "x86_v3");
    }

    // Native AVX-512 512-bit impls (128/256-bit go through .v3() downcast).
    #[cfg(all(target_arch = "x86_64", feature = "avx512", feature = "w512"))]
    if let Some(t) = archmage::X64V4Token::summon() {
        check_f32x16(t, 0, "x86_v4");
        check_f64x8(t, 0, "x86_v4");
        rails_f32x16(t, "x86_v4");
        rails_f64x8(t, "x86_v4");
    }
    #[cfg(all(target_arch = "x86_64", feature = "avx512", feature = "w512"))]
    if let Some(t) = archmage::X64V4xToken::summon() {
        check_f32x16(t, 0, "x86_v4x");
        check_f64x8(t, 0, "x86_v4x");
        rails_f32x16(t, "x86_v4x");
        rails_f64x8(t, "x86_v4x");
    }

    #[cfg(target_arch = "wasm32")]
    if let Some(t) = archmage::Wasm128Token::summon() {
        check_all_widths!(t, 0, "wasm128");
    }

    // Scalar computes 1.0 / x directly.
    let t = archmage::ScalarToken::summon().expect("scalar tier always available");
    check_all_widths!(t, 0, "scalar");
}
