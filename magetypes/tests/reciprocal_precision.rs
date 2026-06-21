//! Precision regression guards for the reciprocal / reciprocal-sqrt families.
//!
//! Two properties, checked on x86_64, aarch64 (via `cross`) and wasm32 (wasmtime):
//!
//!  1. **f64 `recip`/`rsqrt` reach (near-)full precision** (≥ 50 of 53 bits) on
//!     every width and backend — `ScalarToken` included. This caught several real
//!     identity/half-precision bugs: x86 `f64x2`/`f64x4` and the scalar
//!     `f32x16`/`f64x8` fell through to the trait-default `recip = rcp_approx = a`
//!     (the identity); the ARM/WASM `f64x8` polyfill omitted the reciprocal
//!     methods (identity again); the AVX-512 `f64x8` did a single Newton step
//!     (~28 bits).
//!
//!  2. **f32 `rcp_approx`/`rsqrt_approx` hold a ≥ 10-bit floor** by the cheapest
//!     path per platform (x86 ~12-bit hardware estimate, ARM ~16-bit; WASM/scalar
//!     use exact division for rcp and a bit-hack for rsqrt). Sampled (not
//!     exhaustive) so it stays cheap under emulation.
// Not under Miri: the hardware reciprocal-estimate intrinsics (rcpps/rsqrtps,
// rcp14/rsqrt14, vrecpe/vrsqrte) are implementation-defined approximations that
// Miri cannot reproduce bit-for-bit, so a *precision* assertion is meaningless
// there. SIMD UB is covered by `miri_boundary_tests.rs`; this test validates
// achieved precision on real silicon / faithful emulation (QEMU, wasmtime).
#![cfg(all(
    feature = "std",
    not(miri),
    any(
        target_arch = "x86_64",
        target_arch = "aarch64",
        target_arch = "wasm32"
    )
))]

use archmage::SimdToken;
use magetypes::simd::generic::{f32x4, f32x8, f32x16, f64x2, f64x4, f64x8};

/// Correct bits = -log2(max relative error). Full f64 ≈ 52-53, full f32 ≈ 24.
fn bits(max_rel: f64) -> f64 {
    if max_rel <= 0.0 {
        53.0
    } else {
        -max_rel.log2()
    }
}

fn inputs() -> Vec<f64> {
    (0..64).map(|i| 0.25 + i as f64 * 0.37).collect()
}

/// Newton from a hardware estimate lands at ~52 (not bit-exact like division),
/// so 50 is a safe floor that still rejects a missing step (~28) or identity (<0).
const F64_FULL_FLOOR: f64 = 50.0;
/// x86 `rcpps` is the weakest estimate at ~11.9 bits; 10 leaves margin while
/// still rejecting a wrong tier (~8) or the identity.
const F32_APPROX_FLOOR: f64 = 10.0;

macro_rules! check_f64_full {
    ($ty:ident, $tok:expr, $toktype:ty, $lanes:literal, $label:literal, $xs:expr) => {{
        let tok = $tok;
        let (mut wr, mut ws) = (0.0f64, 0.0f64);
        for chunk in $xs.chunks($lanes) {
            if chunk.len() < $lanes {
                break;
            }
            let mut arr = [0.0f64; $lanes];
            arr.copy_from_slice(chunk);
            let v = $ty::<$toktype>::from_array(tok, arr);
            let (mut r, mut rs) = ([0.0f64; $lanes], [0.0f64; $lanes]);
            v.recip().store(&mut r);
            v.rsqrt().store(&mut rs);
            for k in 0..$lanes {
                let x = arr[k];
                wr = wr.max(((r[k] - 1.0 / x) / (1.0 / x)).abs());
                ws = ws.max(((rs[k] - 1.0 / x.sqrt()) / (1.0 / x.sqrt())).abs());
            }
        }
        let (rb, sb) = (bits(wr), bits(ws));
        println!("{:<24} recip={:5.1}b rsqrt={:5.1}b", $label, rb, sb);
        assert!(rb >= F64_FULL_FLOOR, "{} recip only {:.1} bits", $label, rb);
        assert!(sb >= F64_FULL_FLOOR, "{} rsqrt only {:.1} bits", $label, sb);
    }};
}

macro_rules! check_f32_approx {
    ($ty:ident, $tok:expr, $toktype:ty, $lanes:literal, $label:literal, $xs:expr) => {{
        let tok = $tok;
        let (mut wra, mut wsa) = (0.0f64, 0.0f64);
        for chunk in $xs.chunks($lanes) {
            if chunk.len() < $lanes {
                break;
            }
            let mut arr = [0.0f32; $lanes];
            for (d, s) in arr.iter_mut().zip(chunk) {
                *d = *s as f32;
            }
            let v = $ty::<$toktype>::from_array(tok, arr);
            let (mut ra, mut rsa) = ([0.0f32; $lanes], [0.0f32; $lanes]);
            v.rcp_approx().store(&mut ra);
            v.rsqrt_approx().store(&mut rsa);
            for k in 0..$lanes {
                let x = arr[k] as f64;
                wra = wra.max(((ra[k] as f64 - 1.0 / x) / (1.0 / x)).abs());
                wsa = wsa.max(((rsa[k] as f64 - 1.0 / x.sqrt()) / (1.0 / x.sqrt())).abs());
            }
        }
        let (ab, sb) = (bits(wra), bits(wsa));
        println!(
            "{:<24} rcp_approx={:5.1}b rsqrt_approx={:5.1}b",
            $label, ab, sb
        );
        assert!(
            ab >= F32_APPROX_FLOOR,
            "{} rcp_approx only {:.1} bits",
            $label,
            ab
        );
        assert!(
            sb >= F32_APPROX_FLOOR,
            "{} rsqrt_approx only {:.1} bits",
            $label,
            sb
        );
    }};
}

#[test]
fn f64_recip_rsqrt_is_full_precision() {
    let xs = inputs();
    println!("\n=== f64 full-precision recip/rsqrt (floor {F64_FULL_FLOOR} bits) ===");

    #[cfg(target_arch = "x86_64")]
    {
        use archmage::{X64V3Token, X64V4Token};
        let v3 = X64V3Token::summon().expect("x86_64 baseline");
        check_f64_full!(f64x2, v3, X64V3Token, 2, "f64x2<V3>", xs);
        check_f64_full!(f64x4, v3, X64V3Token, 4, "f64x4<V3>", xs);
        check_f64_full!(f64x8, v3, X64V3Token, 8, "f64x8<V3 polyfill>", xs);
        if let Some(v4) = X64V4Token::summon() {
            check_f64_full!(f64x8, v4, X64V4Token, 8, "f64x8<V4 native>", xs);
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        use archmage::NeonToken;
        let n = NeonToken::summon().expect("aarch64 NEON");
        check_f64_full!(f64x2, n, NeonToken, 2, "f64x2<Neon>", xs);
        check_f64_full!(f64x4, n, NeonToken, 4, "f64x4<Neon polyfill>", xs);
        check_f64_full!(f64x8, n, NeonToken, 8, "f64x8<Neon polyfill>", xs);
    }
    #[cfg(target_arch = "wasm32")]
    {
        use archmage::Wasm128Token;
        let w = Wasm128Token::summon().expect("wasm simd128");
        check_f64_full!(f64x2, w, Wasm128Token, 2, "f64x2<Wasm>", xs);
        check_f64_full!(f64x4, w, Wasm128Token, 4, "f64x4<Wasm polyfill>", xs);
        check_f64_full!(f64x8, w, Wasm128Token, 8, "f64x8<Wasm polyfill>", xs);
    }
    // ScalarToken runs on every arch — the fallback whose W512 f32x16/f64x8
    // reciprocals previously fell through to the identity (no override).
    {
        use archmage::ScalarToken;
        let s = ScalarToken::summon().expect("scalar always available");
        check_f64_full!(f64x2, s, ScalarToken, 2, "f64x2<Scalar>", xs);
        check_f64_full!(f64x4, s, ScalarToken, 4, "f64x4<Scalar>", xs);
        check_f64_full!(f64x8, s, ScalarToken, 8, "f64x8<Scalar>", xs);
    }
    println!();
}

#[test]
fn f32_approx_holds_floor() {
    let xs = inputs();
    println!("\n=== f32 approx floor (≥ {F32_APPROX_FLOOR} bits, cheapest path per platform) ===");

    #[cfg(target_arch = "x86_64")]
    {
        use archmage::X64V3Token;
        let v3 = X64V3Token::summon().expect("x86_64 baseline");
        check_f32_approx!(f32x4, v3, X64V3Token, 4, "f32x4<V3>", xs);
        check_f32_approx!(f32x8, v3, X64V3Token, 8, "f32x8<V3>", xs);
    }
    #[cfg(target_arch = "aarch64")]
    {
        use archmage::NeonToken;
        let n = NeonToken::summon().expect("aarch64 NEON");
        check_f32_approx!(f32x4, n, NeonToken, 4, "f32x4<Neon>", xs);
        check_f32_approx!(f32x8, n, NeonToken, 8, "f32x8<Neon>", xs);
    }
    #[cfg(target_arch = "wasm32")]
    {
        use archmage::Wasm128Token;
        let w = Wasm128Token::summon().expect("wasm simd128");
        check_f32_approx!(f32x4, w, Wasm128Token, 4, "f32x4<Wasm>", xs);
        check_f32_approx!(f32x8, w, Wasm128Token, 8, "f32x8<Wasm>", xs);
    }
    {
        use archmage::ScalarToken;
        let s = ScalarToken::summon().expect("scalar always available");
        check_f32_approx!(f32x4, s, ScalarToken, 4, "f32x4<Scalar>", xs);
        check_f32_approx!(f32x8, s, ScalarToken, 8, "f32x8<Scalar>", xs);
        check_f32_approx!(f32x16, s, ScalarToken, 16, "f32x16<Scalar>", xs);
    }
    println!();
}

/// On WASM/scalar the fast `rsqrt_approx` IS the bit-hack (seed + 2 Newton
/// steps), since the full sqrt+div is expensive. It must equal
/// `rsqrt_approx_portable` then one more `rsqrt_newton_portable` **bit-for-bit** —
/// this guards the hand-written backend estimate against drifting from the
/// proven, cross-platform-tested portable one (a wrong magic constant, a
/// non-logical shift, or a different mul/sub associativity). `rcp_approx` is
/// exact division on these backends (a bit-hack is no faster there), so it is
/// not checked here.
#[test]
fn wasm_scalar_rsqrt_approx_is_portable_bitexact() {
    let xs = inputs();

    macro_rules! check_bitexact {
        ($ty:ident, $tok:expr, $toktype:ty, $lanes:literal, $label:literal) => {{
            let tok = $tok;
            for chunk in xs.chunks($lanes) {
                if chunk.len() < $lanes {
                    break;
                }
                let mut arr = [0.0f32; $lanes];
                for (d, s) in arr.iter_mut().zip(chunk) {
                    *d = *s as f32;
                }
                let v = $ty::<$toktype>::from_array(tok, arr);
                // Backend rsqrt_approx is the bit-hack seed + 2 Newton steps; the
                // `_portable` estimate is seed + 1 step, so the reference applies
                // one extra `rsqrt_newton_portable`. Must match bit-for-bit.
                let (mut b1, mut b2) = ([0.0f32; $lanes], [0.0f32; $lanes]);
                v.rsqrt_approx().store(&mut b1);
                v.rsqrt_approx_portable()
                    .rsqrt_newton_portable(v)
                    .store(&mut b2);
                assert_eq!(
                    b1.map(f32::to_bits),
                    b2.map(f32::to_bits),
                    "{} rsqrt_approx != portable seed+2",
                    $label
                );
            }
        }};
    }

    {
        use archmage::ScalarToken;
        let s = ScalarToken::summon().expect("scalar always available");
        check_bitexact!(f32x4, s, ScalarToken, 4, "f32x4<Scalar>");
        check_bitexact!(f32x8, s, ScalarToken, 8, "f32x8<Scalar>");
        check_bitexact!(f32x16, s, ScalarToken, 16, "f32x16<Scalar>");
    }
    #[cfg(target_arch = "wasm32")]
    {
        use archmage::Wasm128Token;
        let w = Wasm128Token::summon().expect("wasm simd128");
        check_bitexact!(f32x4, w, Wasm128Token, 4, "f32x4<Wasm>");
        check_bitexact!(f32x8, w, Wasm128Token, 8, "f32x8<Wasm>");
        check_bitexact!(f32x16, w, Wasm128Token, 16, "f32x16<Wasm>");
    }
}

/// The legacy concrete `arm::w128::f32x4` and the modern generic
/// `f32x4<NeonToken>` must produce **bit-identical** `rcp_approx` / `rsqrt_approx`
/// / `recip` / `rsqrt` — they are the same raw-vrecpe + FRECPS/FRSQRTS sequence
/// and must never drift (this guards the divergence that prompted the fix: the
/// legacy `rcp_approx` used to be the bare ~8-bit estimate while the generic one
/// already refined to ~16-bit).
#[cfg(target_arch = "aarch64")]
#[test]
fn arm_legacy_concrete_matches_generic() {
    use archmage::NeonToken;
    use magetypes::simd::arm::w128::f32x4 as Legacy;
    use magetypes::simd::generic::f32x4 as Generic;

    let t = NeonToken::summon().expect("neon");
    for chunk in inputs().chunks(4) {
        if chunk.len() < 4 {
            continue;
        }
        let mut arr = [0.0f32; 4];
        for (d, s) in arr.iter_mut().zip(chunk) {
            *d = *s as f32;
        }
        let g = Generic::<NeonToken>::from_array(t, arr);
        let l = Legacy::from_array(t, arr);
        macro_rules! eq {
            ($m:ident) => {{
                let mut go = [0.0f32; 4];
                g.$m().store(&mut go);
                assert_eq!(
                    go.map(f32::to_bits),
                    l.$m().to_array().map(f32::to_bits),
                    "legacy vs generic f32x4::{} differ",
                    stringify!($m)
                );
            }};
        }
        eq!(rcp_approx);
        eq!(rsqrt_approx);
        eq!(recip);
        eq!(rsqrt);
    }
}
