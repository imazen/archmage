//! Legacy attribute families still carrying most zen* volume:
//! `#[magetypes]` + `incant!`, `#[arcane]`/`#[rite]` per-tier functions,
//! `#[autoversion]`, generic backend bounds and raw-interop constructors.

use archmage::{ScalarToken, incant};
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::__m128i;
use magetypes::simd::backends::F32x8Backend;
use magetypes::simd::generic::f32x8;

// ---------------------------------------------------------------------------
// `#[magetypes]` with a const generic, legacy token-taking constructors and an
// opt-in AVX-512 tier (zenpixels-convert `convert_kernels.rs:3253`).
// ---------------------------------------------------------------------------

#[archmage::magetypes(define(f32x8), v4(cfg(avx512)), v3, neon, wasm128, scalar)]
fn gain_channels_tier<const CHANNELS: usize>(token: Token, data: &mut [f32], gains: &[f32; 4]) {
    let lanes: [f32; 8] = core::array::from_fn(|i| gains[i % CHANNELS]);
    #[allow(deprecated)] // 0.9.31 still supports the token-taking spelling.
    let g = f32x8::from_array(token, lanes);
    // Eight lanes cycle through the channels only when 8 % CHANNELS == 0.
    let step = if 8 % CHANNELS == 0 { 8 } else { usize::MAX };
    let mut i = 0;
    while step != usize::MAX && i + 8 <= data.len() {
        let chunk: &mut [f32; 8] = (&mut data[i..i + 8]).try_into().unwrap();
        #[allow(deprecated)]
        let v = f32x8::load(token, chunk);
        (v * g).store(chunk);
        i += 8;
    }
    for (j, v) in data.iter_mut().enumerate().skip(i) {
        *v *= gains[j % CHANNELS];
    }
}

pub fn gain_channels<const CHANNELS: usize>(data: &mut [f32], gains: &[f32; 4]) {
    incant!(
        gain_channels_tier::<CHANNELS>(data, gains),
        [v4(cfg(avx512)), v3, neon, wasm128, scalar]
    )
}

// ---------------------------------------------------------------------------
// Hand-written per-tier `#[arcane]` entries over one generic body bounded on
// a backend trait (BRAG `brag-art`, linear-srgb `src/simd.rs:48`).
// ---------------------------------------------------------------------------

#[inline(always)]
fn premultiply_body<T: F32x8Backend>(token: T, rgba: &mut [f32]) {
    let (chunks, tail) = f32x8::<T>::partition_slice_mut_t(token, rgba);
    for c in chunks {
        // Two RGBA pixels per vector: broadcast each alpha over its pixel.
        let a = f32x8::from_array_t(token, [c[3], c[3], c[3], 1.0, c[7], c[7], c[7], 1.0]);
        (f32x8::load_t(token, c) * a).store(c);
    }
    for px in tail.as_chunks_mut::<4>().0 {
        let a = px[3];
        px[0] *= a;
        px[1] *= a;
        px[2] *= a;
    }
}

#[archmage::arcane]
fn premultiply_v3(token: archmage::X64V3Token, rgba: &mut [f32]) {
    premultiply_body(token, rgba)
}

#[archmage::arcane]
fn premultiply_neon(token: archmage::NeonToken, rgba: &mut [f32]) {
    premultiply_body(token, rgba)
}

#[archmage::arcane]
fn premultiply_wasm128(token: archmage::Wasm128Token, rgba: &mut [f32]) {
    premultiply_body(token, rgba)
}

fn premultiply_scalar(token: ScalarToken, rgba: &mut [f32]) {
    premultiply_body(token, rgba)
}

pub fn premultiply(rgba: &mut [f32]) {
    incant!(premultiply(rgba), [v3, neon, wasm128, scalar])
}

// ---------------------------------------------------------------------------
// `#[rite]` helpers called from an `#[arcane]` entry: tier-based with raw
// intrinsics (zenwebp `decoder/yuv_fused.rs:38`) and token-based.
// ---------------------------------------------------------------------------

#[archmage::rite(v3, import_intrinsics)]
fn avg_u8_16(a: __m128i, b: __m128i) -> __m128i {
    _mm_avg_epu8(a, b)
}

#[archmage::rite]
fn avg_rows_body(token: archmage::X64V3Token, a: &[u8; 16], b: &[u8; 16], out: &mut [u8; 16]) {
    let _ = token;
    use archmage::intrinsics::x86_64::{_mm_loadu_si128, _mm_storeu_si128};
    _mm_storeu_si128(out, avg_u8_16(_mm_loadu_si128(a), _mm_loadu_si128(b)));
}

#[archmage::arcane]
pub fn avg_rows_v3(token: archmage::X64V3Token, a: &[u8; 16], b: &[u8; 16], out: &mut [u8; 16]) {
    avg_rows_body(token, a, b, out)
}

// ---------------------------------------------------------------------------
// `#[autoversion]`: bare (309 sites; zensim `streaming.rs:112`) and with an
// explicit tier list (59 sites).
// ---------------------------------------------------------------------------

#[archmage::autoversion]
pub fn sum_abs(x: &[f32]) -> f32 {
    x.iter().map(|v| v.abs()).sum()
}

#[archmage::autoversion(v3, neon, wasm128)]
pub fn saxpy(a: f32, x: &[f32], y: &mut [f32]) {
    for (yi, xi) in y.iter_mut().zip(x) {
        *yi += a * xi;
    }
}

// ---------------------------------------------------------------------------
// Deprecated platform-named raw constructors (jxl-encoder `transpose.rs:25`,
// 111 sites): still supported through 0.9.x.
// ---------------------------------------------------------------------------

#[archmage::arcane(import_intrinsics)]
#[allow(deprecated)]
pub fn from_m256_roundtrip_v3(token: archmage::X64V3Token, x: [f32; 8]) -> [f32; 8] {
    let raw = _mm256_loadu_ps(&x);
    f32x8::<archmage::X64V3Token>::from_m256(token, raw).to_array()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::reference::ramp;
    use archmage::testing::{CompileTimePolicy, for_each_token_permutation};

    #[test]
    fn magetypes_family_with_const_generic() {
        let gains = [0.5, 2.0, 4.0, 1.0];
        let base = ramp(43, 1.0, 9.0);
        for channels in [1usize, 2, 3, 4] {
            let want: Vec<f32> = base
                .iter()
                .enumerate()
                .map(|(i, v)| v * gains[i % channels])
                .collect();
            let _ = for_each_token_permutation(CompileTimePolicy::Warn, |_| {
                let mut d = base.clone();
                match channels {
                    1 => gain_channels::<1>(&mut d, &gains),
                    2 => gain_channels::<2>(&mut d, &gains),
                    3 => gain_channels::<3>(&mut d, &gains),
                    _ => gain_channels::<4>(&mut d, &gains),
                }
                assert_eq!(d, want, "channels={channels}");
            });
        }
    }

    #[test]
    fn per_tier_arcane_family_and_autoversion() {
        let base: Vec<f32> = (0..36).map(|i| (i % 7) as f32 * 0.25).collect();
        let mut want = base.clone();
        for px in want.as_chunks_mut::<4>().0 {
            let a = px[3];
            px[..3].iter_mut().for_each(|v| *v *= a);
        }
        let x = ramp(41, -2.0, 2.0);
        let _ = for_each_token_permutation(CompileTimePolicy::Warn, |_| {
            let mut d = base.clone();
            premultiply(&mut d);
            assert_eq!(d, want);
            assert!((sum_abs(&x) - x.iter().map(|v| v.abs()).sum::<f32>()).abs() < 1e-4);
            let mut y = vec![1.0; 41];
            saxpy(2.0, &x, &mut y);
            assert!(y.iter().zip(&x).all(|(yi, xi)| *yi == 1.0 + 2.0 * xi));
        });
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn rite_helpers_and_raw_constructors() {
        use archmage::SimdToken;
        let _lock = archmage::testing::lock_token_testing();
        let token = archmage::X64V3Token::summon().expect("x86-64 test host without AVX2+FMA");
        let a: [u8; 16] = core::array::from_fn(|i| i as u8 * 10);
        let b: [u8; 16] = core::array::from_fn(|i| 255 - i as u8);
        let mut out = [0u8; 16];
        avg_rows_v3(token, &a, &b, &mut out);
        for i in 0..16 {
            assert_eq!(out[i] as u16, (a[i] as u16 + b[i] as u16).div_ceil(2));
        }
        let x = [1.0, -2.0, 3.5, 0.0, 9.0, -0.5, 7.0, 8.0];
        assert_eq!(from_m256_roundtrip_v3(token, x), x);
    }
}
