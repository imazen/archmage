//! Contract pins for the reciprocal tier grid.
//!
//! Three tiers, three contracts (see the generic type docs):
//!
//! - `rcp_approx`/`rsqrt_approx` — raw-estimate tier, >= ~12-bit floor
//!   (pinned by `reciprocal_precision.rs`, not here).
//! - `recip()`/`rsqrt()` — the WORKING tier: <= 4 ULP AND exact IEEE rails
//!   at ±0/±inf/NaN, by the backend's fastest conforming path. x86 f32:
//!   estimate + FMA-Newton + a branchless blendv rescue to the (rail-exact)
//!   raw estimate; native AVX-512 512-bit: rcp14/rsqrt14 + Newton + a
//!   VFIXUPIMM patch (tables probe-verified on Zen 5 silicon); NEON f32:
//!   FRECPS/FRSQRTS whose hardware inf*0 special case passes rails through
//!   (rsqrt formed as y*FRSQRTS(a, y*y) — a plain a*y mul would NaN);
//!   WASM/scalar and ALL f64: exact division, so those ULP pins are 0.
//!   SUBNORMAL inputs are the one unspecified case at this tier.
//! - `recip_portable()`/`rsqrt_portable()` — the PRECISE tier: exact IEEE
//!   division, 0 ULP, IEEE rails (`recip(±inf) = ±0`, issue #64), and
//!   bit-identical cross-arch for free. The rails checks below run against
//!   bit-identical cross-arch for free.
//!
//! Rails are pinned below on BOTH tiers (bare names and `_portable`).
//!
//! **This test is the runtime half of a two-part guard.** The other half is
//! CI's `generate-check`: the backend bodies are GENERATED, so a fix applied
//! to `magetypes/src/simd/impls/*.rs` alone is reverted by the next
//! `just generate`. Fixes belong in `xtask/src/simd_types/backend_gen*.rs`;
//! `just check-generated` catches the mistake locally. See CLAUDE.md,
//! "FIX THE GENERATOR, NOT ITS OUTPUT".

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
    ($fname:ident, $ty:ident, $elem:ty, $lanes:expr, $Trait:ident, $recip:ident, $rsqrt:ident $(, $cfg:meta)?) => {
        $(#[$cfg])?
        fn $fname<T: $Trait>(token: T, tier: &str) {
            let recip_in: [$elem; $lanes] = core::array::from_fn(|i| {
                [0.0, -0.0, <$elem>::INFINITY, <$elem>::NEG_INFINITY][i % 4]
            });
            let got = $ty::from_array(token, recip_in).$recip().to_array();
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
            let got = $ty::from_array(token, rsqrt_in).$rsqrt().to_array();
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

rails_checker!(
    rails_f32x4,
    f32x4,
    f32,
    4,
    F32x4Convert,
    recip_portable,
    rsqrt_portable
);
rails_checker!(
    rails_f32x8,
    f32x8,
    f32,
    8,
    F32x8Convert,
    recip_portable,
    rsqrt_portable
);
rails_checker!(rails_bare_f32x4, f32x4, f32, 4, F32x4Convert, recip, rsqrt);
rails_checker!(rails_bare_f32x8, f32x8, f32, 8, F32x8Convert, recip, rsqrt);
rails_checker!(rails_f64x2, f64x2, f64, 2, F64x2Backend, recip, rsqrt);
rails_checker!(rails_f64x4, f64x4, f64, 4, F64x4Backend, recip, rsqrt);
rails_checker!(
    rails_f32x16,
    f32x16,
    f32,
    16,
    F32x16Backend,
    recip_portable,
    rsqrt_portable,
    cfg(feature = "w512")
);
rails_checker!(
    rails_bare_f32x16,
    f32x16,
    f32,
    16,
    F32x16Backend,
    recip,
    rsqrt,
    cfg(feature = "w512")
);
rails_checker!(
    rails_f64x8,
    f64x8,
    f64,
    8,
    F64x8Backend,
    recip,
    rsqrt,
    cfg(feature = "w512")
);

/// Every width on one token: working-tier sweep (f32 at `$f32_ulp`, f64 at
/// `$f64_ulp`) + rails on the precise tier (f32 `_portable`, f64 bare).
macro_rules! check_all_widths {
    ($t:expr, $f32_ulp:expr, $f64_ulp:expr, $tier:expr) => {{
        let (t, f32_ulp, f64_ulp, tier) = ($t, $f32_ulp, $f64_ulp, $tier);
        check_x4(t, f32_ulp, tier);
        check_x8(t, f32_ulp, tier);
        check_f64x2(t, f64_ulp, tier);
        check_f64x4(t, f64_ulp, tier);
        rails_f32x4(t, tier);
        rails_f32x8(t, tier);
        rails_bare_f32x4(t, tier);
        rails_bare_f32x8(t, tier);
        rails_f64x2(t, tier);
        rails_f64x4(t, tier);
        #[cfg(feature = "w512")]
        {
            check_f32x16(t, f32_ulp, tier);
            check_f64x8(t, f64_ulp, tier);
            rails_f32x16(t, tier);
            rails_bare_f32x16(t, tier);
            rails_f64x8(t, tier);
        }
    }};
}

/// The working-tier pins: f32 bare names at <= 4 ULP where the backend
/// refines an estimate (x86, NEON — the published 0.9.28 lowerings, kept
/// because they are the FAST forms there), 0 ULP where it divides
/// (WASM/scalar f32, all f64). Rails are pinned on the precise tier
/// (`_portable` for f32, bare names for f64). History: the bare names were
/// briefly flipped to exact division everywhere post-0.9.28, which fixed
/// the #64 rails under the default name but cost a measured ~1.9x/~3.6x on
/// Zen-class x86 with no fast escape hatch — the tier split restores the
/// published speed and gives the rails an explicit home.
#[test]
fn recip_and_rsqrt_meet_their_precision_contract() {
    use archmage::SimdToken;

    #[cfg(target_arch = "aarch64")]
    if let Some(t) = archmage::NeonToken::summon() {
        // f32: vrecpe/vrsqrte + fused refine steps, ~2-3 ULP measured.
        check_all_widths!(t, 4, 0, "neon");
    }

    #[cfg(target_arch = "x86_64")]
    if let Some(t) = archmage::X64V3Token::summon() {
        // f32: rcpps/rsqrtps + one Newton step — 4 ULP is the snug bound the
        // pre-flip pin used. f64 divides (no _mm_rcp_pd exists).
        check_all_widths!(t, 4, 0, "x86_v3");
    }

    // Native AVX-512 512-bit impls (128/256-bit go through .v3() downcast).
    // f32x16: rcp14/rsqrt14 + one Newton step (14 -> ~28 bits, inside 4 ULP).
    #[cfg(all(target_arch = "x86_64", feature = "avx512", feature = "w512"))]
    if let Some(t) = archmage::X64V4Token::summon() {
        check_f32x16(t, 4, "x86_v4");
        check_f64x8(t, 0, "x86_v4");
        rails_f32x16(t, "x86_v4");
        rails_bare_f32x16(t, "x86_v4");
        rails_f64x8(t, "x86_v4");
    }
    #[cfg(all(target_arch = "x86_64", feature = "avx512", feature = "w512"))]
    if let Some(t) = archmage::X64V4xToken::summon() {
        check_f32x16(t, 4, "x86_v4x");
        check_f64x8(t, 0, "x86_v4x");
        rails_f32x16(t, "x86_v4x");
        rails_bare_f32x16(t, "x86_v4x");
        rails_f64x8(t, "x86_v4x");
    }

    #[cfg(target_arch = "wasm32")]
    if let Some(t) = archmage::Wasm128Token::summon() {
        check_all_widths!(t, 0, 0, "wasm128");
    }

    // Scalar computes 1.0 / x directly.
    let t = archmage::ScalarToken::summon().expect("scalar tier always available");
    check_all_widths!(t, 0, 0, "scalar");
}
