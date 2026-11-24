//! WASM relaxed-SIMD fusion probes and probe-free packed software `f32` FMA.
//!
//! Each operation's dispatcher probes once (`powf_f32_fuses` / `powf_f64_fuses`)
//! to select either relaxed-SIMD fused multiply-add or the probe-free software
//! path (`software_lanes`), keeping the Horner loop free of runtime probes.
#![forbid(unsafe_code)]

use archmage::Wasm128Token;
use core::arch::wasm32::{self as w, v128};

// Pow chooses one complete graph after this check; no probe in its Horner loop.
pub(crate) fn powf_f32_fuses() -> bool {
    #[cfg(target_feature = "relaxed-simd")]
    {
        relaxed_probe::fuses_f32()
    }
    #[cfg(not(target_feature = "relaxed-simd"))]
    {
        false
    }
}

// Canonical HIGHP uses binary64 intermediates; screen that operator separately.
pub(crate) fn powf_f64_fuses() -> bool {
    #[cfg(target_feature = "relaxed-simd")]
    {
        relaxed_probe::fuses_f64()
    }
    #[cfg(not(target_feature = "relaxed-simd"))]
    {
        false
    }
}

/// Whether this engine's relaxed madd rounds once. The relaxed-SIMD proposal
/// lets each environment choose fused or unfused per operator but requires a
/// fixed choice, so one cached probe per type decides: fusing engines get the
/// single instruction for the `_portable` forms, others the software path.
///
/// Keep every step inline, with no call: a call site in the caller's loop
/// (even a cold one) made wasmtime keep loop state in memory. With a cold
/// probe call, f32x4 chains took 3.2x and streams 2.0x the time of this form
/// on a fusing engine; the call-free form matches plain software on engines
/// that round twice (`benchmarks/soft_fma_simd_2026-10-24.md`).
#[cfg(target_feature = "relaxed-simd")]
mod relaxed_probe {
    use core::arch::wasm32 as w;
    use core::hint::black_box;
    use core::sync::atomic::{AtomicU8, Ordering};

    /// 0 = not probed, 1 = rounds twice, 2 = fuses.
    static F32: AtomicU8 = AtomicU8::new(0);
    static F64: AtomicU8 = AtomicU8::new(0);

    #[inline(always)]
    pub(crate) fn fuses_f32() -> bool {
        let state = F32.load(Ordering::Relaxed);
        state == 2 || (state == 0 && store(&F32, probe_f32()))
    }

    #[inline(always)]
    pub(crate) fn fuses_f64() -> bool {
        let state = F64.load(Ordering::Relaxed);
        state == 2 || (state == 0 && store(&F64, probe_f64()))
    }

    #[inline(always)]
    fn store(cache: &AtomicU8, fused: bool) -> bool {
        cache.store(if fused { 2 } else { 1 }, Ordering::Relaxed);
        fused
    }

    /// (1 + 2^-12)^2 - (1 + 2^-11) is 2^-24 with one rounding; the rounded
    /// product ties to 1 + 2^-11, so two roundings give 0.
    #[inline(always)]
    fn probe_f32() -> bool {
        let x = w::f32x4_splat(black_box(1.0 + 1.0 / 4096.0));
        let c = w::f32x4_splat(black_box(-(1.0 + 1.0 / 2048.0)));
        let r = w::f32x4_relaxed_madd(x, x, c);
        w::f32x4_extract_lane::<0>(r) == 1.0 / 16_777_216.0
            && w::f32x4_extract_lane::<3>(r) == 1.0 / 16_777_216.0
    }

    /// (1 + 2^-27)^2 - (1 + 2^-26) is 2^-54 with one rounding, 0 with two.
    #[inline(always)]
    fn probe_f64() -> bool {
        let e = 1.0 / 134_217_728.0;
        let x = w::f64x2_splat(black_box(1.0 + e));
        let c = w::f64x2_splat(black_box(-(1.0 + 2.0 * e)));
        let r = w::f64x2_relaxed_madd(x, x, c);
        w::f64x2_extract_lane::<0>(r) == e * e && w::f64x2_extract_lane::<1>(r) == e * e
    }
}

/// Vector form of nostd_math::fmaf: exact f64 product, TwoSum, round to odd.
#[archmage::attune(wasm128)]
fn wide_sum(_token: Wasm128Token, a: v128, b: v128, c: v128) -> v128 {
    let product = w::f64x2_mul(w::f64x2_promote_low_f32x4(a), w::f64x2_promote_low_f32x4(b));
    let addend = w::f64x2_promote_low_f32x4(c);
    let sum = w::f64x2_add(product, addend);
    let virtual_addend = w::f64x2_sub(sum, product);
    let error = w::f64x2_add(
        w::f64x2_sub(product, w::f64x2_sub(sum, virtual_addend)),
        w::f64x2_sub(addend, virtual_addend),
    );
    let zero = w::f64x2_splat(0.0);
    let one = w::i64x2_splat(1);
    let even = w::i64x2_eq(w::v128_and(sum, one), w::i64x2_splat(0));
    let finite = w::f64x2_lt(w::f64x2_abs(sum), w::f64x2_splat(f64::INFINITY));
    let adjust = w::v128_and(finite, w::v128_and(even, w::f64x2_ne(error, zero)));
    let same_sign = w::i64x2_eq(w::f64x2_gt(error, zero), w::f64x2_gt(sum, zero));
    let step = w::v128_bitselect(one, w::i64x2_splat(-1), same_sign);
    let odd = w::i64x2_add(sum, step);
    w::f32x4_demote_f64x2_zero(w::v128_bitselect(odd, sum, adjust))
}

/// Probe-free packed software `f32x4` FMA (dispatcher selects software once
/// for the complete expression graph).
#[archmage::attune(wasm128)]
fn powf_soft_f32x4(token: Wasm128Token, a: v128, b: v128, c: v128) -> v128 {
    let lo = wide_sum(token, a, b, c);
    let hi = wide_sum(
        token,
        w::i32x4_shuffle::<2, 3, 2, 3>(a, a),
        w::i32x4_shuffle::<2, 3, 2, 3>(b, b),
        w::i32x4_shuffle::<2, 3, 2, 3>(c, c),
    );
    w::i32x4_shuffle::<0, 1, 4, 5>(lo, hi)
}

/// Packed probe-free software `f32` FMA for whole-operation `WASM_SOFTWARE`
/// kernels (`powf`, `expf`, `logf`) when the WASM engine does not fuse
/// `f32x4.relaxed_madd`.
#[archmage::attune(wasm128)]
pub(crate) fn software_lanes<const N: usize>(
    token: Wasm128Token,
    a: [f32; N],
    b: [f32; N],
    c: [f32; N],
) -> [f32; N] {
    assert_eq!(
        N % 4,
        0,
        "private packed software adapter requires whole vectors"
    );
    let mut output = [0.0; N];
    for (((a, b), c), out) in a
        .as_chunks::<4>()
        .0
        .iter()
        .zip(b.as_chunks::<4>().0)
        .zip(c.as_chunks::<4>().0)
        .zip(output.as_chunks_mut::<4>().0)
    {
        let a = w::f32x4(a[0], a[1], a[2], a[3]);
        let b = w::f32x4(b[0], b[1], b[2], b[3]);
        let c = w::f32x4(c[0], c[1], c[2], c[3]);
        let r = powf_soft_f32x4(token, a, b, c);
        *out = [
            w::f32x4_extract_lane::<0>(r),
            w::f32x4_extract_lane::<1>(r),
            w::f32x4_extract_lane::<2>(r),
            w::f32x4_extract_lane::<3>(r),
        ];
    }
    output
}

/// The cached answer must match the engine. CI sets
/// `ARCHMAGE_EXPECT_RELAXED_FUSION` per runner (`yes` with host FMA, `no` with
/// `cranelift-has-fma=false`); fused and software results are bit-identical,
/// so the arithmetic tests cannot tell which branch ran.
#[cfg(all(test, feature = "std", target_feature = "relaxed-simd"))]
mod tests {
    #[test]
    fn relaxed_probe_matches_engine() {
        let (f32_fuses, f64_fuses) = (
            super::relaxed_probe::fuses_f32(),
            super::relaxed_probe::fuses_f64(),
        );
        // Cached: later calls return the first answer.
        assert_eq!(super::relaxed_probe::fuses_f32(), f32_fuses);
        assert_eq!(super::relaxed_probe::fuses_f64(), f64_fuses);
        std::eprintln!("relaxed madd probe: f32 fuses={f32_fuses}, f64 fuses={f64_fuses}");
        if let Ok(expected) = std::env::var("ARCHMAGE_EXPECT_RELAXED_FUSION") {
            assert!(matches!(expected.as_str(), "yes" | "no"));
            let expected = expected == "yes";
            assert_eq!((f32_fuses, f64_fuses), (expected, expected));
        }
    }
}
