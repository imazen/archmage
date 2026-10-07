//! `to_u8` and the RGBA stores round half to even and clamp to `0..=255` on
//! every backend, including inputs the x86 `cvtps` conversion cannot represent:
//! large finite values, ±inf, 2^31 and NaN (NaN → 0, matching the scalar
//! reference). The x86 V4/V4x f32x4/f32x8 paths delegate to V3, so each is
//! checked separately.

use archmage::{ScalarToken, SimdToken};
use magetypes::simd::backends::{F32x4Backend, F32x8Backend};
use magetypes::simd::generic::{f32x4, f32x8};

const EDGES: [f32; 16] = [
    255.0,
    256.0,
    3.0e9,
    f32::INFINITY,
    f32::NEG_INFINITY,
    f32::NAN,
    -0.5,
    0.5,
    1.5,
    254.5,
    255.49,
    255.5,
    -3.0e9,
    f32::MAX,
    -f32::MAX,
    2_147_483_648.0,
];

/// The scalar contract: round half to even, clamp to 0..=255, NaN → 0.
fn expected(x: f32) -> u8 {
    x.round_ties_even().clamp(0.0, 255.0) as u8
}

/// Channel `c` of rotation `k`: every edge value lands in every channel and lane.
fn plane<const N: usize>(k: usize, c: usize) -> [f32; N] {
    core::array::from_fn(|i| EDGES[(k + c * 5 + i) % EDGES.len()])
}

fn check4<T: F32x4Backend>(t: T, name: &str) {
    for chunk in EDGES.chunks_exact(4) {
        let lanes: [f32; 4] = chunk.try_into().unwrap();
        let want: [u8; 4] = core::array::from_fn(|i| expected(lanes[i]));
        assert_eq!(
            f32x4::<T>::from_array_t(t, lanes).to_u8(),
            want,
            "{name} f32x4::to_u8 {lanes:?}"
        );
    }
    for k in 0..EDGES.len() {
        let p: [[f32; 4]; 4] = core::array::from_fn(|c| plane::<4>(k, c));
        let v = |c: usize| f32x4::<T>::from_array_t(t, p[c]);
        let got = f32x4::<T>::store_4_rgba_u8(v(0), v(1), v(2), v(3));
        let want: [u8; 16] = core::array::from_fn(|j| expected(p[j % 4][j / 4]));
        assert_eq!(got, want, "{name} f32x4::store_4_rgba_u8 rotation {k}");
    }
}

fn check8<T: F32x8Backend>(t: T, name: &str) {
    for chunk in EDGES.chunks_exact(8) {
        let lanes: [f32; 8] = chunk.try_into().unwrap();
        let want: [u8; 8] = core::array::from_fn(|i| expected(lanes[i]));
        assert_eq!(
            f32x8::<T>::from_array_t(t, lanes).to_u8(),
            want,
            "{name} f32x8::to_u8 {lanes:?}"
        );
    }
    for k in 0..EDGES.len() {
        let p: [[f32; 8]; 4] = core::array::from_fn(|c| plane::<8>(k, c));
        let v = |c: usize| f32x8::<T>::from_array_t(t, p[c]);
        let got = f32x8::<T>::store_8_rgba_u8(v(0), v(1), v(2), v(3));
        let want: [u8; 32] = core::array::from_fn(|j| expected(p[j % 4][j / 4]));
        assert_eq!(got, want, "{name} f32x8::store_8_rgba_u8 rotation {k}");
    }
}

#[test]
fn scalar() {
    check4(ScalarToken, "scalar");
    check8(ScalarToken, "scalar");
}

#[cfg(target_arch = "x86_64")]
#[test]
fn x64v3() {
    match archmage::X64V3Token::summon() {
        Some(t) => {
            check4(t, "v3");
            check8(t, "v3");
        }
        None => eprintln!("skipped: this CPU lacks x86-64-v3"),
    }
}

#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
#[test]
fn x64v4() {
    match archmage::X64V4Token::summon() {
        Some(t) => {
            check4(t, "v4");
            check8(t, "v4");
        }
        None => eprintln!("skipped: this CPU lacks AVX-512"),
    }
}

#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
#[test]
fn x64v4x() {
    match archmage::X64V4xToken::summon() {
        Some(t) => {
            check4(t, "v4x");
            check8(t, "v4x");
        }
        None => eprintln!("skipped: this CPU lacks AVX-512 v4x"),
    }
}

#[cfg(target_arch = "aarch64")]
#[test]
fn neon() {
    match archmage::NeonToken::summon() {
        Some(t) => {
            check4(t, "neon");
            check8(t, "neon");
        }
        None => eprintln!("skipped: no NEON"),
    }
}

#[cfg(target_arch = "wasm32")]
#[test]
fn wasm128() {
    match archmage::Wasm128Token::summon() {
        Some(t) => {
            check4(t, "wasm128");
            check8(t, "wasm128");
        }
        None => eprintln!("skipped: no simd128"),
    }
}
