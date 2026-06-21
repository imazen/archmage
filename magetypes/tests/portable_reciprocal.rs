//! Cross-platform **bit-identity** + precision for the deterministic `_portable`
//! reciprocal / reciprocal-sqrt methods on the actual `f32x4` API.
//!
//! The whole point of these methods is "same bits on every machine". This test
//! folds every output bit pattern into an FNV-1a checksum: run it on x86 and on
//! aarch64 (via `cross`) and the checksums must match exactly. Precision bounds
//! for the 8-bit / refined / full tiers are asserted too.
//!
//!   cargo test --test portable_reciprocal -- --nocapture
//!   cross test --test portable_reciprocal --target aarch64-unknown-linux-gnu -- --nocapture

#![cfg(feature = "std")]

use archmage::SimdToken;
use magetypes::simd::generic::f32x4;

#[cfg(target_arch = "x86_64")]
type Tok = archmage::X64V3Token;
#[cfg(target_arch = "aarch64")]
type Tok = archmage::NeonToken;
#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
type Tok = archmage::ScalarToken;

#[cfg(target_arch = "x86_64")]
const ARCH: &str = "x86_64";
#[cfg(target_arch = "aarch64")]
const ARCH: &str = "aarch64";
#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
const ARCH: &str = "other";

/// Deterministic integer-domain sweep across many octaves; length a multiple of 4.
fn inputs() -> Vec<f32> {
    let mut v = Vec::new();
    for exp in 110u32..=145 {
        let mut m = 0u32;
        while m < (1 << 23) {
            v.push(f32::from_bits((exp << 23) | m));
            m += 4099;
        }
    }
    while v.len() % 16 != 0 {
        v.push(1.0);
    }
    v
}

fn fnv(h: &mut u64, b: u32) {
    *h ^= b as u64;
    *h = h.wrapping_mul(0x0000_0100_0000_01b3);
}

fn bits_of(rel: f64) -> f64 {
    if rel <= 0.0 {
        f64::INFINITY
    } else {
        -rel.log2()
    }
}

/// Run a 4-lane op over the sweep; return (checksum of output bits, max rel err).
fn measure(
    token: Tok,
    data: &[f32],
    op: impl Fn(f32x4<Tok>) -> f32x4<Tok>,
    reference: impl Fn(f64) -> f64,
) -> (u64, f64) {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    let mut err = 0.0f64;
    for chunk in data.chunks_exact(4) {
        let arr: [f32; 4] = chunk.try_into().unwrap();
        let out = op(f32x4::<Tok>::from_array(token, arr)).to_array();
        for (&x, &y) in arr.iter().zip(out.iter()) {
            fnv(&mut h, y.to_bits());
            let r = reference(x as f64);
            err = err.max(((y as f64 - r) / r).abs());
        }
    }
    (h, err)
}

#[test]
fn portable_reciprocal_bit_identity_and_precision() {
    let Some(token) = Tok::summon() else {
        // No native tier on this build/CPU (e.g. a no_std lib build has no
        // runtime detection). Nothing to characterize here.
        eprintln!("portable_reciprocal: native {ARCH} token unavailable — skipping");
        return;
    };
    let data = inputs();
    let rsqrt_ref = |x: f64| 1.0 / x.sqrt();
    let recip_ref = |x: f64| 1.0 / x;

    let rs8 = measure(token, &data, |v| v.rsqrt_approx_portable(), rsqrt_ref);
    let rs16 = measure(
        token,
        &data,
        |v| v.rsqrt_approx_portable().rsqrt_newton_portable(v),
        rsqrt_ref,
    );
    let rsf = measure(token, &data, |v| v.rsqrt_portable(), rsqrt_ref);
    let rc8 = measure(token, &data, |v| v.rcp_approx_portable(), recip_ref);
    let rc16 = measure(
        token,
        &data,
        |v| v.rcp_approx_portable().recip_newton_portable(v),
        recip_ref,
    );
    let rcf = measure(token, &data, |v| v.recip_portable(), recip_ref);

    eprintln!(
        "\n=== portable reciprocals on {ARCH} ({} inputs) ===",
        data.len()
    );
    eprintln!(
        "  {:<28}{:>18}  {:>11}  {:>5}",
        "method", "checksum", "max rel err", "bits"
    );
    for (name, (h, e)) in [
        ("rsqrt_approx_portable", rs8),
        ("  + rsqrt_newton_portable", rs16),
        ("rsqrt_portable (full)", rsf),
        ("rcp_approx_portable", rc8),
        ("  + recip_newton_portable", rc16),
        ("recip_portable (full)", rcf),
    ] {
        eprintln!("  {name:<28}{h:>#18x}  {e:>11.3e}  {:>5.1}", bits_of(e));
    }
    eprintln!("  (checksums must match x86 vs aarch64 to prove bit-identity)\n");

    // Precision contract for the three tiers.
    assert!(rs8.1 < 6e-3, "rsqrt_approx_portable err {:.3e}", rs8.1);
    assert!(rs16.1 < 1e-4, "rsqrt + 1 refine err {:.3e}", rs16.1);
    assert!(rsf.1 < 1e-6, "rsqrt_portable err {:.3e}", rsf.1);
    assert!(rc8.1 < 6e-3, "rcp_approx_portable err {:.3e}", rc8.1);
    assert!(rc16.1 < 1e-4, "rcp + 1 refine err {:.3e}", rc16.1);
    assert!(rcf.1 < 1e-6, "recip_portable err {:.3e}", rcf.1);
}

/// Collect every output bit pattern of `$method` over `$data`, processed
/// `$lanes` at a time through the `$ty` vector.
macro_rules! collect_bits {
    ($ty:ty, $lanes:literal, $token:expr, $data:expr, $method:ident) => {{
        let mut out: Vec<u32> = Vec::with_capacity($data.len());
        for chunk in $data.chunks_exact($lanes) {
            let arr: [f32; $lanes] = chunk.try_into().unwrap();
            let r = <$ty>::from_array($token, arr).$method().to_array();
            out.extend(r.iter().map(|v| v.to_bits()));
        }
        out
    }};
}

/// The wider widths are generated from the identical template (only the lane
/// count differs), so they must produce **exactly** the same bits as `f32x4`,
/// lane for lane. Combined with the cross-architecture proof above, this
/// transitively makes `f32x8`/`f32x16` bit-identical across machines too.
#[test]
fn portable_cross_width_consistency() {
    let Some(token) = Tok::summon() else {
        eprintln!("portable_cross_width_consistency: native {ARCH} token unavailable — skipping");
        return;
    };
    let data = inputs(); // length is a multiple of 16

    let rs4 = collect_bits!(f32x4<Tok>, 4, token, &data, rsqrt_approx_portable);
    let rc4 = collect_bits!(f32x4<Tok>, 4, token, &data, rcp_approx_portable);
    let rs8 = collect_bits!(
        magetypes::simd::generic::f32x8<Tok>,
        8,
        token,
        &data,
        rsqrt_approx_portable
    );
    let rc8 = collect_bits!(
        magetypes::simd::generic::f32x8<Tok>,
        8,
        token,
        &data,
        rcp_approx_portable
    );
    assert_eq!(rs4, rs8, "f32x8 rsqrt_approx_portable diverges from f32x4");
    assert_eq!(rc4, rc8, "f32x8 rcp_approx_portable diverges from f32x4");

    #[cfg(feature = "w512")]
    {
        let rs16 = collect_bits!(
            magetypes::simd::generic::f32x16<Tok>,
            16,
            token,
            &data,
            rsqrt_approx_portable
        );
        let rc16 = collect_bits!(
            magetypes::simd::generic::f32x16<Tok>,
            16,
            token,
            &data,
            rcp_approx_portable
        );
        assert_eq!(
            rs4, rs16,
            "f32x16 rsqrt_approx_portable diverges from f32x4"
        );
        assert_eq!(rc4, rc16, "f32x16 rcp_approx_portable diverges from f32x4");
    }

    eprintln!(
        "portable_cross_width_consistency on {ARCH}: f32x4 == f32x8{} (bit-for-bit) ✓",
        if cfg!(feature = "w512") {
            " == f32x16"
        } else {
            ""
        }
    );
}
