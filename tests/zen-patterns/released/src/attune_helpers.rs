//! Known-tier helpers, proof wrappers, raw intrinsics and token idioms
//! (zenmetrics `mdctpsnr`, `vmaf`, `iwssim`, `vsi`; rav1d-safe; zensim).
//!
//! `#[attune]` and `#[arcane]`/`#[rite]` emit `#[cfg(target_arch)]` on these
//! functions themselves; only un-attributed call sites carry a cfg.

#[cfg(target_arch = "x86_64")]
use archmage::X64V3Token;
use archmage::attune;
#[cfg(target_arch = "x86_64")]
use magetypes::simd::generic::f32x8;

/// `#[attune(v3)]` known-tier helper using a flat transcendental
/// (zenmetrics `mdctpsnr/src/lib.rs:395`).
#[attune(v3)]
pub(crate) fn powf_mapped_v3(token: X64V3Token, out: &mut [f32], inp: &[f32], vis: f32, e: f32) {
    type V = f32x8<X64V3Token>;
    let visv = V::splat_t(token, vis);
    let (ic, it) = V::partition_slice_t(token, inp);
    let (oc, ot) = V::partition_slice_mut_t(token, out);
    for (i, o) in ic.iter().zip(oc.iter_mut()) {
        (V::load_t(token, i).abs() * visv).pow_midp(e).store(o);
    }
    for (o, &v) in ot.iter_mut().zip(it) {
        *o = (v.abs() * vis).powf(e);
    }
}

/// `#[attune(wrap, import_intrinsics)]` proof wrapper that calls the v3
/// helper with no extra boundary and uses a raw intrinsic
/// (zenmetrics `mdctpsnr/src/lib.rs:312`).
#[attune(wrap, import_intrinsics)]
pub fn powf_mapped_sum_v3(token: X64V3Token, out: &mut [f32], inp: &[f32], e: f32) -> f32 {
    powf_mapped_v3(token, out, inp, 1.0, e);
    let mut acc = _mm256_setzero_ps();
    let (chunks, tail) = f32x8::<X64V3Token>::partition_slice_t(token, out);
    for c in chunks {
        acc = _mm256_add_ps(acc, f32x8::load_t(token, c).into_repr());
    }
    f32x8::from_repr_t(token, acc).reduce_add() + tail.iter().sum::<f32>()
}

/// Bare `#[attune]` inferring a proof wrapper from the `_v3_t` suffix
/// (218 sites; zenmetrics `iwssim/src/simd_kernels.rs:1625`).
#[archmage::attune]
pub fn clamp01_v3_t(token: X64V3Token, data: &mut [f32]) {
    let (chunks, tail) = f32x8::<X64V3Token>::partition_slice_mut_t(token, data);
    let (lo, hi) = (f32x8::splat_t(token, 0.0), f32x8::splat_t(token, 1.0));
    for c in chunks {
        f32x8::load_t(token, c).max(lo).min(hi).store(c);
    }
    for v in tail {
        *v = v.clamp(0.0, 1.0);
    }
}

/// `#[attune(import_intrinsics)]` on a `_v3_t` body with raw AVX2 integer
/// intrinsics (52 sites; zenmetrics `vmaf/src/adm.rs:607`).
#[attune(import_intrinsics)]
pub fn add_shift_i32_v3_t(_token: X64V3Token, data: &mut [i32; 8], add: i32, shift: i32) {
    let v = _mm256_loadu_si256(&*data);
    let v = _mm256_srav_epi32(
        _mm256_add_epi32(v, _mm256_set1_epi32(add)),
        _mm256_set1_epi32(shift),
    );
    _mm256_storeu_si256(data, v);
}

/// `X64V3Token::from_context()` inside a v3 feature body: constructs the
/// proof without runtime detection (253 sites, mostly rav1d-safe).
#[archmage::rite(v3)]
fn halve_in_context(data: &mut [f32]) {
    let token = X64V3Token::from_context();
    let (chunks, tail) = f32x8::<X64V3Token>::partition_slice_mut_t(token, data);
    let half = f32x8::splat_t(token, 0.5);
    for c in chunks {
        (f32x8::load_t(token, c) * half).store(c);
    }
    for v in tail {
        *v *= 0.5;
    }
}

#[archmage::arcane]
pub fn halve_v3(_token: X64V3Token, data: &mut [f32]) {
    halve_in_context(data);
}

/// `.v3()` downcast of an AVX-512 proof to call V3 kernels (104 sites).
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
pub fn clamp01_from_v4(token: archmage::X64V4Token, data: &mut [f32]) {
    clamp01_v3_t(token.v3(), data);
}

/// `#[attune(wasm128)]` known-tier body (zenmetrics `vsi/src/kernel.rs:365`).
#[attune(wasm128)]
pub fn negate_wasm128(token: archmage::Wasm128Token, data: &mut [f32]) {
    use magetypes::simd::generic::f32x4;
    let (chunks, tail) = f32x4::<archmage::Wasm128Token>::partition_slice_mut_t(token, data);
    for c in chunks {
        (-f32x4::load_t(token, c)).store(c);
    }
    for v in tail {
        *v = -*v;
    }
}

#[cfg(all(test, target_arch = "x86_64"))]
mod tests {
    use super::*;
    use crate::reference::{close, ramp};
    use archmage::SimdToken;

    #[test]
    fn known_tier_wrappers_and_intrinsics() {
        // Every x86-64 CI host has AVX2+FMA; a missing tier is a failure, not a skip.
        let token = X64V3Token::summon().expect("x86-64 test host without AVX2+FMA");
        let inp = ramp(29, 0.01, 2.0);
        let mut out = vec![0.0; 29];
        let sum = powf_mapped_sum_v3(token, &mut out, &inp, 2.2);
        for (o, i) in out.iter().zip(&inp) {
            assert!(close(*o, i.powf(2.2), 2e-5), "{o} vs {}", i.powf(2.2));
        }
        assert!(close(sum, out.iter().sum(), 1e-5));

        let mut d = ramp(19, -1.0, 2.0);
        clamp01_v3_t(token, &mut d);
        assert!(d.iter().all(|v| (0.0..=1.0).contains(v)));

        let mut ints = [1, 2, 3, 4, -5, -6, 7, 8];
        add_shift_i32_v3_t(token, &mut ints, 3, 1);
        assert_eq!(ints, [2, 2, 3, 3, -1, -2, 5, 5]);

        let mut h = ramp(11, 2.0, 4.0);
        let want: Vec<f32> = h.iter().map(|v| v * 0.5).collect();
        halve_v3(token, &mut h);
        assert_eq!(h, want);
    }
}
