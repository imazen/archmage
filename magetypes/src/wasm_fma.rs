//! WASM multiply-add. `madd_*`/`msub_*` back `mul_add`/`mul_sub`: the engine's
//! relaxed madd in relaxed-SIMD builds, a multiply then an add otherwise.
//! `fused_*` back the `_portable` forms, which round once: through relaxed madd
//! where a cached probe shows the engine fuses it, in software otherwise
//! (strict builds, and engines that round twice).
#![forbid(unsafe_code)]

use archmage::Wasm128Token;
use core::arch::wasm32::{self as w, v128};

// Feature detection folds away in normal builds and honors token suppression
// in testable_dispatch builds.
#[cfg(target_feature = "relaxed-simd")]
#[inline(always)]
fn relaxed_token() -> Option<archmage::Wasm128RelaxedToken> {
    use archmage::SimdToken;
    archmage::Wasm128RelaxedToken::summon()
}

#[cfg(not(target_feature = "relaxed-simd"))]
#[archmage::rite]
pub(crate) fn madd_f32x4(_token: Wasm128Token, a: v128, b: v128, c: v128) -> v128 {
    w::f32x4_add(w::f32x4_mul(a, b), c)
}

#[cfg(not(target_feature = "relaxed-simd"))]
#[archmage::rite]
pub(crate) fn msub_f32x4(_token: Wasm128Token, a: v128, b: v128, c: v128) -> v128 {
    w::f32x4_sub(w::f32x4_mul(a, b), c)
}

#[cfg(not(target_feature = "relaxed-simd"))]
#[archmage::rite]
pub(crate) fn madd_f64x2(_token: Wasm128Token, a: v128, b: v128, c: v128) -> v128 {
    w::f64x2_add(w::f64x2_mul(a, b), c)
}

#[cfg(not(target_feature = "relaxed-simd"))]
#[archmage::rite]
pub(crate) fn msub_f64x2(_token: Wasm128Token, a: v128, b: v128, c: v128) -> v128 {
    w::f64x2_sub(w::f64x2_mul(a, b), c)
}

#[cfg(target_feature = "relaxed-simd")]
#[archmage::attune(wasm128)]
pub(crate) fn madd_f32x4(_token: Wasm128Token, a: v128, b: v128, c: v128) -> v128 {
    if relaxed_token().is_some() {
        w::f32x4_relaxed_madd(a, b, c)
    } else {
        w::f32x4_add(w::f32x4_mul(a, b), c)
    }
}

#[cfg(target_feature = "relaxed-simd")]
#[archmage::attune(wasm128)]
pub(crate) fn msub_f32x4(_token: Wasm128Token, a: v128, b: v128, c: v128) -> v128 {
    if relaxed_token().is_some() {
        w::f32x4_relaxed_madd(a, b, w::f32x4_neg(c))
    } else {
        w::f32x4_sub(w::f32x4_mul(a, b), c)
    }
}

#[cfg(target_feature = "relaxed-simd")]
#[archmage::attune(wasm128)]
pub(crate) fn madd_f64x2(_token: Wasm128Token, a: v128, b: v128, c: v128) -> v128 {
    if relaxed_token().is_some() {
        w::f64x2_relaxed_madd(a, b, c)
    } else {
        w::f64x2_add(w::f64x2_mul(a, b), c)
    }
}

#[cfg(target_feature = "relaxed-simd")]
#[archmage::attune(wasm128)]
pub(crate) fn msub_f64x2(_token: Wasm128Token, a: v128, b: v128, c: v128) -> v128 {
    if relaxed_token().is_some() {
        w::f64x2_relaxed_madd(a, b, w::f64x2_neg(c))
    } else {
        w::f64x2_sub(w::f64x2_mul(a, b), c)
    }
}

#[cfg(not(target_feature = "relaxed-simd"))]
#[archmage::rite]
pub(crate) fn trunc_f32x4_fast(_token: Wasm128Token, a: v128) -> v128 {
    w::i32x4_trunc_sat_f32x4(a)
}

#[cfg(target_feature = "relaxed-simd")]
#[archmage::attune(wasm128)]
pub(crate) fn trunc_f32x4_fast(_token: Wasm128Token, a: v128) -> v128 {
    if relaxed_token().is_some() {
        w::i32x4_relaxed_trunc_f32x4(a)
    } else {
        w::i32x4_trunc_sat_f32x4(a)
    }
}

#[cfg(not(target_feature = "relaxed-simd"))]
#[archmage::rite]
pub(crate) fn min_f32x4_fast(_token: Wasm128Token, a: v128, b: v128) -> v128 {
    w::f32x4_pmin(b, a)
}

#[cfg(not(target_feature = "relaxed-simd"))]
#[archmage::rite]
pub(crate) fn max_f32x4_fast(_token: Wasm128Token, a: v128, b: v128) -> v128 {
    w::f32x4_pmax(b, a)
}

#[cfg(not(target_feature = "relaxed-simd"))]
#[archmage::rite]
pub(crate) fn min_f64x2_fast(_token: Wasm128Token, a: v128, b: v128) -> v128 {
    w::f64x2_pmin(b, a)
}

#[cfg(not(target_feature = "relaxed-simd"))]
#[archmage::rite]
pub(crate) fn max_f64x2_fast(_token: Wasm128Token, a: v128, b: v128) -> v128 {
    w::f64x2_pmax(b, a)
}

#[cfg(target_feature = "relaxed-simd")]
#[archmage::attune(wasm128)]
pub(crate) fn min_f32x4_fast(_token: Wasm128Token, a: v128, b: v128) -> v128 {
    if relaxed_token().is_some() {
        w::f32x4_relaxed_min(a, b)
    } else {
        w::f32x4_pmin(b, a)
    }
}

#[cfg(target_feature = "relaxed-simd")]
#[archmage::attune(wasm128)]
pub(crate) fn max_f32x4_fast(_token: Wasm128Token, a: v128, b: v128) -> v128 {
    if relaxed_token().is_some() {
        w::f32x4_relaxed_max(a, b)
    } else {
        w::f32x4_pmax(b, a)
    }
}

#[cfg(target_feature = "relaxed-simd")]
#[archmage::attune(wasm128)]
pub(crate) fn min_f64x2_fast(_token: Wasm128Token, a: v128, b: v128) -> v128 {
    if relaxed_token().is_some() {
        w::f64x2_relaxed_min(a, b)
    } else {
        w::f64x2_pmin(b, a)
    }
}

#[cfg(target_feature = "relaxed-simd")]
#[archmage::attune(wasm128)]
pub(crate) fn max_f64x2_fast(_token: Wasm128Token, a: v128, b: v128) -> v128 {
    if relaxed_token().is_some() {
        w::f64x2_relaxed_max(a, b)
    } else {
        w::f64x2_pmax(b, a)
    }
}

#[cfg(target_feature = "relaxed-simd")]
#[archmage::attune(wasm128)]
pub(crate) fn min_f32_fast_scalar(token: Wasm128Token, a: f32, b: f32) -> f32 {
    if relaxed_token().is_some() {
        w::f32x4_extract_lane::<0>(min_f32x4_fast(token, w::f32x4_splat(a), w::f32x4_splat(b)))
    } else {
        w::f32x4_extract_lane::<0>(min_f32x4_fast(token, w::f32x4_splat(a), w::f32x4_splat(b)))
    }
}

#[cfg(target_feature = "relaxed-simd")]
#[archmage::attune(wasm128)]
pub(crate) fn max_f32_fast_scalar(token: Wasm128Token, a: f32, b: f32) -> f32 {
    if relaxed_token().is_some() {
        w::f32x4_extract_lane::<0>(max_f32x4_fast(token, w::f32x4_splat(a), w::f32x4_splat(b)))
    } else {
        w::f32x4_extract_lane::<0>(max_f32x4_fast(token, w::f32x4_splat(a), w::f32x4_splat(b)))
    }
}

#[cfg(target_feature = "relaxed-simd")]
#[archmage::rite]
pub(crate) fn madd_f32_scalar(token: Wasm128Token, a: f32, b: f32, c: f32) -> f32 {
    w::f32x4_extract_lane::<0>(madd_f32x4(
        token,
        w::f32x4_splat(a),
        w::f32x4_splat(b),
        w::f32x4_splat(c),
    ))
}

#[cfg(target_feature = "relaxed-simd")]
#[archmage::rite]
pub(crate) fn msub_f32_scalar(token: Wasm128Token, a: f32, b: f32, c: f32) -> f32 {
    w::f32x4_extract_lane::<0>(msub_f32x4(
        token,
        w::f32x4_splat(a),
        w::f32x4_splat(b),
        w::f32x4_splat(c),
    ))
}

#[cfg(target_feature = "relaxed-simd")]
#[archmage::attune(wasm128)]
pub(crate) fn trunc_f32_fast_scalar(token: Wasm128Token, a: f32) -> i32 {
    if relaxed_token().is_some() {
        w::i32x4_extract_lane::<0>(trunc_f32x4_fast(token, w::f32x4_splat(a)))
    } else {
        w::i32x4_extract_lane::<0>(trunc_f32x4_fast(token, w::f32x4_splat(a)))
    }
}

#[cfg(target_feature = "relaxed-simd")]
#[archmage::attune(wasm128)]
pub(crate) fn min_f64_fast_scalar(token: Wasm128Token, a: f64, b: f64) -> f64 {
    if relaxed_token().is_some() {
        w::f64x2_extract_lane::<0>(min_f64x2_fast(token, w::f64x2_splat(a), w::f64x2_splat(b)))
    } else {
        w::f64x2_extract_lane::<0>(min_f64x2_fast(token, w::f64x2_splat(a), w::f64x2_splat(b)))
    }
}

#[cfg(target_feature = "relaxed-simd")]
#[archmage::attune(wasm128)]
pub(crate) fn max_f64_fast_scalar(token: Wasm128Token, a: f64, b: f64) -> f64 {
    if relaxed_token().is_some() {
        w::f64x2_extract_lane::<0>(max_f64x2_fast(token, w::f64x2_splat(a), w::f64x2_splat(b)))
    } else {
        w::f64x2_extract_lane::<0>(max_f64x2_fast(token, w::f64x2_splat(a), w::f64x2_splat(b)))
    }
}

#[cfg(target_feature = "relaxed-simd")]
#[archmage::rite]
pub(crate) fn madd_f64_scalar(token: Wasm128Token, a: f64, b: f64, c: f64) -> f64 {
    w::f64x2_extract_lane::<0>(madd_f64x2(
        token,
        w::f64x2_splat(a),
        w::f64x2_splat(b),
        w::f64x2_splat(c),
    ))
}

#[cfg(target_feature = "relaxed-simd")]
#[archmage::rite]
pub(crate) fn msub_f64_scalar(token: Wasm128Token, a: f64, b: f64, c: f64) -> f64 {
    w::f64x2_extract_lane::<0>(msub_f64x2(
        token,
        w::f64x2_splat(a),
        w::f64x2_splat(b),
        w::f64x2_splat(c),
    ))
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

#[archmage::rite]
pub(crate) fn fused_f32x4(token: Wasm128Token, a: v128, b: v128, c: v128) -> v128 {
    #[cfg(target_feature = "relaxed-simd")]
    if relaxed_token().is_some() && relaxed_probe::fuses_f32() {
        return w::f32x4_relaxed_madd(a, b, c);
    }
    let lo = wide_sum(token, a, b, c);
    let hi = wide_sum(
        token,
        w::i32x4_shuffle::<2, 3, 2, 3>(a, a),
        w::i32x4_shuffle::<2, 3, 2, 3>(b, b),
        w::i32x4_shuffle::<2, 3, 2, 3>(c, c),
    );
    w::i32x4_shuffle::<0, 1, 4, 5>(lo, hi)
}

/// Vector form of nostd_math::fmaf: exact f64 product, TwoSum, round to odd.
#[archmage::rite]
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

#[archmage::rite]
pub(crate) fn fused_f64x2(_token: Wasm128Token, a: v128, b: v128, c: v128) -> v128 {
    #[cfg(target_feature = "relaxed-simd")]
    if relaxed_token().is_some() && relaxed_probe::fuses_f64() {
        return w::f64x2_relaxed_madd(a, b, c);
    }
    #[cfg(target_feature = "relaxed-simd")]
    return soft_f64x2(a, b, c);
    #[cfg(not(target_feature = "relaxed-simd"))]
    soft_f64x2_inline(a, b, c)
}

/// With relaxed SIMD the software path sits behind one call, so a fusing
/// engine's loop holds one call site instead of the exact path's per-lane
/// calls (`v128` arguments and results travel in registers).
#[cfg(target_feature = "relaxed-simd")]
#[inline(never)]
fn soft_f64x2(a: v128, b: v128, c: v128) -> v128 {
    soft_f64x2_inline(a, b, c)
}

#[inline(always)]
fn soft_f64x2_inline(a: v128, b: v128, c: v128) -> v128 {
    let lanes = |v: v128| [w::f64x2_extract_lane::<0>(v), w::f64x2_extract_lane::<1>(v)];
    let [lo, hi] = crate::soft_fma::fma_lanes(lanes(a), lanes(b), lanes(c));
    w::f64x2(lo, hi)
}

/// The cached answer must match the engine. CI sets
/// `ARCHMAGE_EXPECT_RELAXED_FUSION` per runner (`yes` with host FMA, `no` with
/// `cranelift-has-fma=false`); fused and software results are bit-identical,
/// so the arithmetic tests cannot tell which branch ran.
#[cfg(all(test, feature = "std", target_feature = "relaxed-simd"))]
mod tests {
    #[test]
    fn relaxed_probe_matches_engine() {
        let _guard = archmage::testing::lock_token_testing();
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

    #[test]
    fn relaxed_selection_respects_token_disabling() {
        use archmage::{SimdToken, Wasm128RelaxedToken};
        if Wasm128RelaxedToken::compiled_with().is_some() {
            return;
        }
        let _guard = archmage::testing::lock_token_testing();
        let token = super::relaxed_token().unwrap().wasm128();
        Wasm128RelaxedToken::dangerously_disable_token_process_wide(true).unwrap();
        assert!(super::relaxed_token().is_none());
        disabled_results(token);
        Wasm128RelaxedToken::dangerously_disable_token_process_wide(false).unwrap();
        assert!(super::relaxed_token().is_some());
    }

    #[archmage::attune(wrap)]
    fn disabled_results(token: archmage::Wasm128Token) {
        use core::arch::wasm32 as w;
        let e = 1.0 / 4096.0;
        let x = w::f32x4_splat(1.0 + e);
        let c = w::f32x4_splat(1.0 + 2.0 * e);
        assert_eq!(
            w::f32x4_extract_lane::<0>(super::msub_f32x4(token, x, x, c)),
            0.0
        );
        assert_eq!(
            w::f32x4_extract_lane::<0>(super::fused_f32x4(token, x, x, w::f32x4_neg(c))),
            e * e
        );

        let e = 1.0 / 134_217_728.0;
        let x = w::f64x2_splat(1.0 + e);
        let c = w::f64x2_splat(-(1.0 + 2.0 * e));
        assert_eq!(
            w::f64x2_extract_lane::<0>(super::madd_f64x2(token, x, x, c)),
            0.0
        );
        assert_eq!(
            w::f64x2_extract_lane::<0>(super::fused_f64x2(token, x, x, c)),
            e * e
        );
    }
}

#[cfg(not(target_feature = "relaxed-simd"))]
#[archmage::rite]
pub(crate) fn nmadd_f32x4(_token: Wasm128Token, a: v128, b: v128, c: v128) -> v128 {
    w::f32x4_add(w::f32x4_mul(w::f32x4_neg(a), b), c)
}

#[cfg(target_feature = "relaxed-simd")]
#[archmage::rite]
pub(crate) fn nmadd_f32x4(_token: Wasm128Token, a: v128, b: v128, c: v128) -> v128 {
    if relaxed_token().is_some() {
        w::f32x4_relaxed_nmadd(a, b, c)
    } else {
        w::f32x4_add(w::f32x4_mul(w::f32x4_neg(a), b), c)
    }
}

#[cfg(not(target_feature = "relaxed-simd"))]
#[archmage::rite]
pub(crate) fn nmadd_f64x2(_token: Wasm128Token, a: v128, b: v128, c: v128) -> v128 {
    w::f64x2_add(w::f64x2_mul(w::f64x2_neg(a), b), c)
}

#[cfg(target_feature = "relaxed-simd")]
#[archmage::rite]
pub(crate) fn nmadd_f64x2(_token: Wasm128Token, a: v128, b: v128, c: v128) -> v128 {
    if relaxed_token().is_some() {
        w::f64x2_relaxed_nmadd(a, b, c)
    } else {
        w::f64x2_add(w::f64x2_mul(w::f64x2_neg(a), b), c)
    }
}

#[cfg(target_feature = "relaxed-simd")]
#[archmage::attune(wasm128)]
pub(crate) fn trunc_u32_f32x4_fast(_token: Wasm128Token, a: v128) -> v128 {
    if relaxed_token().is_some() {
        w::u32x4_relaxed_trunc_f32x4(a)
    } else {
        w::u32x4_trunc_sat_f32x4(a)
    }
}
