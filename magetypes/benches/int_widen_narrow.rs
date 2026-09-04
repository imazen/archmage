//! What integer widening and saturating narrowing cost against the only
//! portable alternative a `#[magetypes]` body has without them: a round trip
//! through `to_array()` / `from_array()`.
//!
//! The motivating case is a byte-exact SVT-AV1 port whose kernels (`me_sad`,
//! the directional predictors, `residual_i16`, `variance::sse`, the SATD
//! accumulate) are hand-written per ISA today precisely because the
//! `u8 -> u16` / `i16 -> u8` steps had no portable spelling. This bench
//! measures the price of the portable spelling.
//!
//! `widen/*`            — `widen_low()` + `widen_high()`, both halves stored.
//! `widen_via_array/*`  — the same result via `to_array()`, per-lane `as`, and
//!                        `from_array()`: what the body would otherwise do.
//! `narrow/*`           — `narrow_saturating_u8(hi)`, one call per output vector.
//! `narrow_via_array/*` — the same result via arrays and an explicit clamp.
//! `widen_chain/*`      — `u8 -> u16 -> u32` accumulate, i.e. both widening
//!                        pairs in one dependency chain (the `variance::sse`
//!                        shape), against the same chain through arrays.
//!
//! Run: `cargo bench -p magetypes --bench int_widen_narrow`
//!
//! Sizes span tiny (call-overhead-dominated) through large
//! (memory-bandwidth-bound) so the fixed per-call cost and the per-element cost
//! can be separated rather than blended into one number.

// The loop shape is deliberately `chunks_exact`, identical in the primitive and
// the baseline arm and identical to the sibling `int_uniform_shift` bench, so
// the two families' numbers are comparable. Switching to `as_chunks` would
// change what is measured, not just how it is spelled.
#![allow(clippy::chunks_exact_to_as_chunks)]

use archmage::SimdToken;
use magetypes::simd::backends::{I16x8Narrow, U8x16Widen, U16x8Widen, U32x4Backend};
use magetypes::simd::generic::{i16x8, u8x16, u16x8, u32x4};
use zenbench::criterion_compat::*;
use zenbench::{criterion_group, criterion_main};

/// Element counts: tiny → large. 16 u8 lanes / 8 i16 lanes per vector.
const SIZES: &[usize] = &[16, 256, 4096, 65536, 1_048_576];

fn u8_input(n: usize) -> Vec<u8> {
    (0..n)
        .map(|i| (i.wrapping_mul(2_654_435_761) & 0xFF) as u8)
        .collect()
}

fn i16_input(n: usize) -> Vec<i16> {
    (0..n)
        .map(|i| ((i.wrapping_mul(2_654_435_761) & 0xFFFF) as u16 as i16) / 3)
        .collect()
}

// ============================================================================
// Widening
// ============================================================================

/// `widen_low` + `widen_high` over the slice — one instruction per half.
#[inline(always)]
fn widen_slice<T: U8x16Widen>(t: T, src: &[u8], dst: &mut [u16]) {
    for (s, d) in src.chunks_exact(16).zip(dst.chunks_exact_mut(16)) {
        let arr: [u8; 16] = s.try_into().unwrap();
        let v = u8x16::<T>::from_array(t, arr);
        d[..8].copy_from_slice(&v.widen_low().to_array());
        d[8..].copy_from_slice(&v.widen_high().to_array());
    }
}

/// The array round trip a portable body needs without the primitive.
#[inline(always)]
fn widen_via_array_slice<T: U8x16Widen>(t: T, src: &[u8], dst: &mut [u16]) {
    for (s, d) in src.chunks_exact(16).zip(dst.chunks_exact_mut(16)) {
        let arr: [u8; 16] = s.try_into().unwrap();
        let v = u8x16::<T>::from_array(t, arr);
        let a = v.to_array();
        let lo = u16x8::<T>::from_array(t, core::array::from_fn(|i| a[i] as u16));
        let hi = u16x8::<T>::from_array(t, core::array::from_fn(|i| a[i + 8] as u16));
        d[..8].copy_from_slice(&lo.to_array());
        d[8..].copy_from_slice(&hi.to_array());
    }
}

/// `u8 -> u16 -> u32` accumulate: both widening pairs in one chain.
#[inline(always)]
fn widen_chain_slice<T>(t: T, src: &[u8]) -> u32
where
    T: U8x16Widen + U16x8Widen + U32x4Backend,
{
    let mut acc = u32x4::<T>::zero(t);
    for s in src.chunks_exact(16) {
        let arr: [u8; 16] = s.try_into().unwrap();
        let v = u8x16::<T>::from_array(t, arr);
        for half in [v.widen_low(), v.widen_high()] {
            acc = acc + half.widen_low() + half.widen_high();
        }
    }
    acc.to_array().iter().fold(0u32, |a, b| a.wrapping_add(*b))
}

/// The same chain without the primitive: every widening step through arrays.
#[inline(always)]
fn widen_chain_via_array_slice<T>(t: T, src: &[u8]) -> u32
where
    T: U8x16Widen + U16x8Widen + U32x4Backend,
{
    let mut acc = u32x4::<T>::zero(t);
    for s in src.chunks_exact(16) {
        let arr: [u8; 16] = s.try_into().unwrap();
        let a = u8x16::<T>::from_array(t, arr).to_array();
        for off8 in [0usize, 8] {
            let h =
                u16x8::<T>::from_array(t, core::array::from_fn(|i| a[i + off8] as u16)).to_array();
            for off4 in [0usize, 4] {
                acc += u32x4::<T>::from_array(t, core::array::from_fn(|i| h[i + off4] as u32));
            }
        }
    }
    acc.to_array().iter().fold(0u32, |a, b| a.wrapping_add(*b))
}

// ============================================================================
// Saturating narrowing
// ============================================================================

/// `narrow_saturating_u8` over the slice — one call per 16 output bytes.
#[inline(always)]
fn narrow_slice<T: I16x8Narrow>(t: T, src: &[i16], dst: &mut [u8]) {
    for (s, d) in src.chunks_exact(16).zip(dst.chunks_exact_mut(16)) {
        let lo = i16x8::<T>::from_array(t, s[..8].try_into().unwrap());
        let hi = i16x8::<T>::from_array(t, s[8..].try_into().unwrap());
        d.copy_from_slice(&lo.narrow_saturating_u8(hi).to_array());
    }
}

/// The array round trip with an explicit clamp.
#[inline(always)]
fn narrow_via_array_slice<T: I16x8Narrow>(t: T, src: &[i16], dst: &mut [u8]) {
    for (s, d) in src.chunks_exact(16).zip(dst.chunks_exact_mut(16)) {
        let lo = i16x8::<T>::from_array(t, s[..8].try_into().unwrap());
        let hi = i16x8::<T>::from_array(t, s[8..].try_into().unwrap());
        let (a, b) = (lo.to_array(), hi.to_array());
        let out: [u8; 16] = core::array::from_fn(|i| {
            let x = if i < 8 { a[i] } else { b[i - 8] };
            x.clamp(0, 255) as u8
        });
        d.copy_from_slice(&u8x16::<T>::from_array(t, out).to_array());
    }
}

/// x86 `#[arcane]` arm: the same six generic kernels, called inside one
/// AVX2 `#[target_feature]` region — the way a real `#[magetypes]`/`#[arcane]`
/// consumer runs them. Necessary for a fair x86 measurement: `widen_low`
/// lowers to `_mm_cvtepu8_epi16`, which is SSE4.1 — NOT in the x86-64
/// baseline — so outside a target-feature region every widen is an outlined
/// `#[target_feature]` call while the `via_array` casts inline and
/// auto-vectorize at SSE2. The bare-generic `v3` label measures that call
/// boundary (widen 4-5x slower than via_array on Zen 5); this label measures
/// the op.
#[cfg(target_arch = "x86_64")]
mod v3_arcane {
    use super::*;
    use archmage::{X64V3Token, arcane};
    use magetypes::simd::backends::I16x16Narrow;
    use magetypes::simd::generic::i16x16;

    /// 256-bit narrow — the width that carries the `permute4x64::<0xD8>`
    /// lane-order fixup. This is the case whose cost the 128-bit kernels
    /// cannot show.
    #[inline(always)]
    fn narrow256_slice<T: I16x16Narrow>(t: T, src: &[i16], dst: &mut [u8]) {
        for (s, d) in src.chunks_exact(32).zip(dst.chunks_exact_mut(32)) {
            let lo = i16x16::<T>::from_array(t, s[..16].try_into().unwrap());
            let hi = i16x16::<T>::from_array(t, s[16..].try_into().unwrap());
            d.copy_from_slice(&lo.narrow_saturating_u8(hi).to_array());
        }
    }

    #[inline(always)]
    fn narrow256_via_array_slice<T: I16x16Narrow>(t: T, src: &[i16], dst: &mut [u8]) {
        for (s, d) in src.chunks_exact(32).zip(dst.chunks_exact_mut(32)) {
            let lo = i16x16::<T>::from_array(t, s[..16].try_into().unwrap());
            let hi = i16x16::<T>::from_array(t, s[16..].try_into().unwrap());
            let (a, b) = (lo.to_array(), hi.to_array());
            let out: [u8; 32] = core::array::from_fn(|i| {
                let x = if i < 16 { a[i] } else { b[i - 16] };
                x.clamp(0, 255) as u8
            });
            d.copy_from_slice(&out);
        }
    }

    #[arcane]
    pub fn narrow256_slice_a(t: X64V3Token, src: &[i16], dst: &mut [u8]) {
        narrow256_slice::<X64V3Token>(t, src, dst)
    }
    #[arcane]
    pub fn narrow256_via_array_slice_a(t: X64V3Token, src: &[i16], dst: &mut [u8]) {
        narrow256_via_array_slice::<X64V3Token>(t, src, dst)
    }

    #[arcane]
    pub fn widen_slice_a(t: X64V3Token, src: &[u8], dst: &mut [u16]) {
        widen_slice::<X64V3Token>(t, src, dst)
    }
    #[arcane]
    pub fn widen_via_array_slice_a(t: X64V3Token, src: &[u8], dst: &mut [u16]) {
        widen_via_array_slice::<X64V3Token>(t, src, dst)
    }
    #[arcane]
    pub fn widen_chain_slice_a(t: X64V3Token, src: &[u8]) -> u32 {
        widen_chain_slice::<X64V3Token>(t, src)
    }
    #[arcane]
    pub fn widen_chain_via_array_slice_a(t: X64V3Token, src: &[u8]) -> u32 {
        widen_chain_via_array_slice::<X64V3Token>(t, src)
    }
    #[arcane]
    pub fn narrow_slice_a(t: X64V3Token, src: &[i16], dst: &mut [u8]) {
        narrow_slice::<X64V3Token>(t, src, dst)
    }
    #[arcane]
    pub fn narrow_via_array_slice_a(t: X64V3Token, src: &[i16], dst: &mut [u8]) {
        narrow_via_array_slice::<X64V3Token>(t, src, dst)
    }
}

#[cfg(target_arch = "x86_64")]
macro_rules! bench_backend_v3_arcane {
    ($c:expr, $t:expr) => {{
        let t = $t;
        for &n in SIZES {
            let src8 = u8_input(n);
            let src16 = i16_input(n);
            let mut dst16 = vec![0u16; n];
            let mut dst8 = vec![0u8; n];

            $c.bench_function(&format!("widen/v3_arcane/{n}"), |b| {
                b.iter(|| v3_arcane::widen_slice_a(t, black_box(&src8), black_box(&mut dst16)))
            });
            $c.bench_function(&format!("widen_via_array/v3_arcane/{n}"), |b| {
                b.iter(|| {
                    v3_arcane::widen_via_array_slice_a(t, black_box(&src8), black_box(&mut dst16))
                })
            });
            $c.bench_function(&format!("widen_chain/v3_arcane/{n}"), |b| {
                b.iter(|| v3_arcane::widen_chain_slice_a(t, black_box(&src8)))
            });
            $c.bench_function(&format!("widen_chain_via_array/v3_arcane/{n}"), |b| {
                b.iter(|| v3_arcane::widen_chain_via_array_slice_a(t, black_box(&src8)))
            });
            $c.bench_function(&format!("narrow/v3_arcane/{n}"), |b| {
                b.iter(|| v3_arcane::narrow_slice_a(t, black_box(&src16), black_box(&mut dst8)))
            });
            $c.bench_function(&format!("narrow_via_array/v3_arcane/{n}"), |b| {
                b.iter(|| {
                    v3_arcane::narrow_via_array_slice_a(t, black_box(&src16), black_box(&mut dst8))
                })
            });
            $c.bench_function(&format!("narrow256/v3_arcane/{n}"), |b| {
                b.iter(|| v3_arcane::narrow256_slice_a(t, black_box(&src16), black_box(&mut dst8)))
            });
            $c.bench_function(&format!("narrow256_via_array/v3_arcane/{n}"), |b| {
                b.iter(|| {
                    v3_arcane::narrow256_via_array_slice_a(
                        t,
                        black_box(&src16),
                        black_box(&mut dst8),
                    )
                })
            });
        }
    }};
}

macro_rules! bench_backend {
    ($c:expr, $label:literal, $Tok:ty, $t:expr) => {{
        let t = $t;
        for &n in SIZES {
            let src8 = u8_input(n);
            let src16 = i16_input(n);
            let mut dst16 = vec![0u16; n];
            let mut dst8 = vec![0u8; n];

            $c.bench_function(&format!("widen/{}/{n}", $label), |b| {
                b.iter(|| widen_slice::<$Tok>(t, black_box(&src8), black_box(&mut dst16)))
            });
            $c.bench_function(&format!("widen_via_array/{}/{n}", $label), |b| {
                b.iter(|| widen_via_array_slice::<$Tok>(t, black_box(&src8), black_box(&mut dst16)))
            });
            $c.bench_function(&format!("widen_chain/{}/{n}", $label), |b| {
                b.iter(|| widen_chain_slice::<$Tok>(t, black_box(&src8)))
            });
            $c.bench_function(&format!("widen_chain_via_array/{}/{n}", $label), |b| {
                b.iter(|| widen_chain_via_array_slice::<$Tok>(t, black_box(&src8)))
            });
            $c.bench_function(&format!("narrow/{}/{n}", $label), |b| {
                b.iter(|| narrow_slice::<$Tok>(t, black_box(&src16), black_box(&mut dst8)))
            });
            $c.bench_function(&format!("narrow_via_array/{}/{n}", $label), |b| {
                b.iter(|| {
                    narrow_via_array_slice::<$Tok>(t, black_box(&src16), black_box(&mut dst8))
                })
            });
        }
    }};
}

fn bench_all(c: &mut Criterion) {
    bench_backend!(
        c,
        "scalar",
        archmage::ScalarToken,
        archmage::ScalarToken::summon().unwrap()
    );

    #[cfg(target_arch = "aarch64")]
    if let Some(t) = archmage::NeonToken::summon() {
        bench_backend!(c, "neon", archmage::NeonToken, t);
    }

    #[cfg(target_arch = "x86_64")]
    if let Some(t) = archmage::X64V3Token::summon() {
        bench_backend!(c, "v3", archmage::X64V3Token, t);
        bench_backend_v3_arcane!(c, t);
    }
}

criterion_group!(benches, bench_all);
criterion_main!(benches);
