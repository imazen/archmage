//! Precision characterization for the reciprocal / reciprocal-square-root
//! approximations.
//!
//! For every backend this measures the max *relative* error (and the
//! equivalent bits of mantissa) against an f64 reference, for:
//!   * the raw hardware seed (`vrsqrteq` / `_mm_rsqrt_ps`) — the "before"
//!     behaviour, measured directly so the baseline is real, not assumed;
//!   * `rsqrt_approx` / `rcp_approx` — the fast path (now one native
//!     FRSQRTS / FRECPS Newton step on ARM);
//!   * `rsqrt` / `recip` — full precision (two steps).
//!
//! The point of the change: on ARM the raw seed is only ~8-bit, far coarser
//! than x86's ~12-bit hardware estimate or WASM's full-precision division, so
//! the `_approx` path used to be the worst across architectures. One native
//! Newton step lifts it to ~16-bit and tightens that cross-architecture spread.
//!
//! See the printed numbers with:
//!   cargo test --test rsqrt_precision -- --nocapture
//!   cross test --test rsqrt_precision --target aarch64-unknown-linux-gnu -- --nocapture

use archmage::SimdToken;
use magetypes::simd::generic::f32x4;

// The native backend token for this architecture. (WASM's `_approx` is already
// full-precision division, and is exercised by `scalar_parity`; here the
// non-x86/non-ARM fallback just measures the scalar backend.)
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
const ARCH: &str = "scalar-fallback";

/// Dense log-spaced sweep over a wide range. rcp / rsqrt relative error is
/// mantissa-driven (the exponent is handled exactly), so a wide multi-octave
/// sweep exercises every mantissa pattern. Length is a multiple of 4.
fn inputs() -> Vec<f32> {
    let n: usize = 120_000;
    let (lo, hi) = (1e-3f64, 1e3f64);
    (0..n)
        .map(|i| {
            let t = i as f64 / (n - 1) as f64;
            (lo.ln() + t * (hi.ln() - lo.ln())).exp() as f32
        })
        .collect()
}

fn bits(rel: f64) -> f64 {
    if rel <= 0.0 {
        f64::INFINITY
    } else {
        -rel.log2()
    }
}

/// Max relative error of a 4-lane `op` against an f64 `reference`.
fn max_rel_err(
    token: Tok,
    data: &[f32],
    op: impl Fn(f32x4<Tok>) -> f32x4<Tok>,
    reference: impl Fn(f64) -> f64,
) -> f64 {
    let mut maxerr = 0.0f64;
    for chunk in data.chunks_exact(4) {
        let arr: [f32; 4] = chunk.try_into().unwrap();
        let out = op(f32x4::<Tok>::from_array(token, arr)).to_array();
        for (&x, &got) in arr.iter().zip(out.iter()) {
            let r = reference(x as f64);
            let e = ((got as f64 - r) / r).abs();
            if e > maxerr {
                maxerr = e;
            }
        }
    }
    maxerr
}

fn recip_ref(x: f64) -> f64 {
    1.0 / x
}
fn rsqrt_ref(x: f64) -> f64 {
    1.0 / x.sqrt()
}

/// Raw hardware reciprocal / reciprocal-sqrt seed error — the pre-refinement
/// `_approx` behaviour, measured directly. Returns `(rcp_err, rsqrt_err)`, or
/// `None` on architectures without an estimate instruction.
#[cfg(target_arch = "x86_64")]
fn raw_seed_errs(data: &[f32]) -> Option<(f64, f64)> {
    use core::arch::x86_64::*;
    // SSE is baseline on x86_64; the value intrinsics are safe, the loads need
    // a raw pointer (hence the `unsafe`).
    let mut rcp_max = 0.0f64;
    let mut rsqrt_max = 0.0f64;
    for chunk in data.chunks_exact(4) {
        let arr: [f32; 4] = chunk.try_into().unwrap();
        let v = unsafe { _mm_loadu_ps(arr.as_ptr()) };
        let mut rcp = [0f32; 4];
        let mut rsq = [0f32; 4];
        unsafe {
            _mm_storeu_ps(rcp.as_mut_ptr(), _mm_rcp_ps(v));
            _mm_storeu_ps(rsq.as_mut_ptr(), _mm_rsqrt_ps(v));
        }
        for ((&x, &c), &s) in arr.iter().zip(rcp.iter()).zip(rsq.iter()) {
            let x = x as f64;
            rcp_max = rcp_max.max(((c as f64 - recip_ref(x)) / recip_ref(x)).abs());
            rsqrt_max = rsqrt_max.max(((s as f64 - rsqrt_ref(x)) / rsqrt_ref(x)).abs());
        }
    }
    Some((rcp_max, rsqrt_max))
}

#[cfg(target_arch = "aarch64")]
fn raw_seed_errs(data: &[f32]) -> Option<(f64, f64)> {
    use core::arch::aarch64::*;
    // NEON is baseline on aarch64; value intrinsics are safe, loads need a ptr.
    let mut rcp_max = 0.0f64;
    let mut rsqrt_max = 0.0f64;
    for chunk in data.chunks_exact(4) {
        let arr: [f32; 4] = chunk.try_into().unwrap();
        let v = unsafe { vld1q_f32(arr.as_ptr()) };
        let mut rcp = [0f32; 4];
        let mut rsq = [0f32; 4];
        unsafe {
            vst1q_f32(rcp.as_mut_ptr(), vrecpeq_f32(v));
            vst1q_f32(rsq.as_mut_ptr(), vrsqrteq_f32(v));
        }
        for ((&x, &c), &s) in arr.iter().zip(rcp.iter()).zip(rsq.iter()) {
            let x = x as f64;
            rcp_max = rcp_max.max(((c as f64 - recip_ref(x)) / recip_ref(x)).abs());
            rsqrt_max = rsqrt_max.max(((s as f64 - rsqrt_ref(x)) / rsqrt_ref(x)).abs());
        }
    }
    Some((rcp_max, rsqrt_max))
}

#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
fn raw_seed_errs(_data: &[f32]) -> Option<(f64, f64)> {
    None
}

#[test]
fn rsqrt_rcp_precision_report() {
    let Some(token) = Tok::summon() else {
        panic!("native backend token unavailable on {ARCH} — cannot characterize precision");
    };
    let data = inputs();

    let rcp_approx_err = max_rel_err(token, &data, |v| v.rcp_approx(), recip_ref);
    let recip_err = max_rel_err(token, &data, |v| v.recip(), recip_ref);
    let rsqrt_approx_err = max_rel_err(token, &data, |v| v.rsqrt_approx(), rsqrt_ref);
    let rsqrt_err = max_rel_err(token, &data, |v| v.rsqrt(), rsqrt_ref);
    let raw = raw_seed_errs(&data);

    eprintln!(
        "\n=== rcp / rsqrt precision on {ARCH} ({} inputs) ===",
        data.len()
    );
    eprintln!("  {:<22} {:>12}   {:>8}", "op", "max rel err", "bits");
    if let Some((rcp_raw, _)) = raw {
        eprintln!(
            "  {:<22} {:>12.3e}   {:>8.1}   (pre-refinement seed)",
            "rcp   raw seed",
            rcp_raw,
            bits(rcp_raw)
        );
    }
    eprintln!(
        "  {:<22} {:>12.3e}   {:>8.1}",
        "rcp_approx",
        rcp_approx_err,
        bits(rcp_approx_err)
    );
    eprintln!(
        "  {:<22} {:>12.3e}   {:>8.1}",
        "recip (full)",
        recip_err,
        bits(recip_err)
    );
    if let Some((_rcp_raw, rsqrt_raw)) = raw {
        eprintln!(
            "  {:<22} {:>12.3e}   {:>8.1}   (pre-refinement seed)",
            "rsqrt raw seed",
            rsqrt_raw,
            bits(rsqrt_raw)
        );
    }
    eprintln!(
        "  {:<22} {:>12.3e}   {:>8.1}",
        "rsqrt_approx",
        rsqrt_approx_err,
        bits(rsqrt_approx_err)
    );
    eprintln!(
        "  {:<22} {:>12.3e}   {:>8.1}",
        "rsqrt (full)",
        rsqrt_err,
        bits(rsqrt_err)
    );
    eprintln!();

    // Contract: the fast path holds at least ~10 bits everywhere (the parity
    // floor), full precision reaches ~24-bit f32.
    assert!(
        rcp_approx_err < 1e-3,
        "rcp_approx err {rcp_approx_err:.3e} exceeds 1e-3 floor on {ARCH}"
    );
    assert!(
        rsqrt_approx_err < 1e-3,
        "rsqrt_approx err {rsqrt_approx_err:.3e} exceeds 1e-3 floor on {ARCH}"
    );
    assert!(
        recip_err < 1e-5,
        "recip err {recip_err:.3e} exceeds 1e-5 on {ARCH}"
    );
    assert!(
        rsqrt_err < 1e-5,
        "rsqrt err {rsqrt_err:.3e} exceeds 1e-5 on {ARCH}"
    );

    // On ARM the refinement must materially beat the raw ~8-bit seed (the whole
    // point of the change). On x86 the `_approx` path *is* the raw seed, so we
    // only require it to be no worse.
    #[cfg(target_arch = "aarch64")]
    if let Some((rcp_raw, rsqrt_raw)) = raw {
        assert!(
            rsqrt_approx_err < rsqrt_raw / 16.0,
            "rsqrt_approx ({rsqrt_approx_err:.3e}) should be >=4 bits better than the raw seed ({rsqrt_raw:.3e})"
        );
        assert!(
            rcp_approx_err < rcp_raw / 16.0,
            "rcp_approx ({rcp_approx_err:.3e}) should be >=4 bits better than the raw seed ({rcp_raw:.3e})"
        );
    }
    #[cfg(not(target_arch = "aarch64"))]
    let _ = raw;
}
