//! `concat_shift` — every backend, every element type, against the reference.
//!
//! The backend traits carry a portable lane-gather default and every ISA that
//! has a funnel shift overrides it, so what has to be tested is that the
//! overrides agree with the default lane-for-lane, for every shift amount, on
//! every token this machine can summon. A wrong `palignr` immediate or a wrong
//! `vperm2f128` selector produces plausible-looking output that is silently off
//! by one lane, which is exactly the class of bug a reference catches and eyes
//! do not.
//!
//! Shifts are exhaustive up to 16 lanes. On the 32- and 64-lane types they are
//! sampled at the boundaries that matter — 0, the first few, and either side of
//! every 128-bit lane crossing — because those are where a per-lane instruction
//! (`palignr`, `vpalignr`) stops matching a full-width shift.
//!
//! Values are `i` and `lanes + i`, which stay distinct per lane and inside
//! `i8::MAX` for every width, so one scheme covers f32 through u8.

use archmage::SimdToken;

/// Lanes `n..n+LANES` of the concatenation `[a, b]`.
fn reference<T: Copy>(a: &[T], b: &[T], n: usize) -> Vec<T> {
    let lanes = a.len();
    (0..lanes)
        .map(|i| {
            if n + i < lanes {
                a[n + i]
            } else {
                b[n + i - lanes]
            }
        })
        .collect()
}

/// One (type, token, shift) case.
macro_rules! check {
    ($ty:ident, $elem:ty, $lanes:expr, $token:path, $shift:expr) => {{
        if let Some(t) = <$token>::summon() {
            let a: [$elem; $lanes] = core::array::from_fn(|i| i as $elem);
            let b: [$elem; $lanes] = core::array::from_fn(|i| ($lanes + i) as $elem);
            let lo = <magetypes::simd::generic::$ty<$token>>::from_array(t, a);
            let hi = <magetypes::simd::generic::$ty<$token>>::from_array(t, b);
            let got = lo.concat_shift::<{ $shift }>(hi).to_array();
            assert_eq!(
                got.to_vec(),
                reference(&a, &b, $shift as usize),
                "{} on {} at shift {}",
                stringify!($ty),
                stringify!($token),
                $shift
            );
        }
    }};
}

/// Every shift, for a type with 2, 4, 8 or 16 lanes.
macro_rules! all_small {
    ($ty:ident, $elem:ty, 2, $token:path) => {{
        check!($ty, $elem, 2, $token, 0);
        check!($ty, $elem, 2, $token, 1);
    }};
    ($ty:ident, $elem:ty, 4, $token:path) => {{
        check!($ty, $elem, 4, $token, 0);
        check!($ty, $elem, 4, $token, 1);
        check!($ty, $elem, 4, $token, 2);
        check!($ty, $elem, 4, $token, 3);
    }};
    ($ty:ident, $elem:ty, 8, $token:path) => {{
        check!($ty, $elem, 8, $token, 0);
        check!($ty, $elem, 8, $token, 1);
        check!($ty, $elem, 8, $token, 2);
        check!($ty, $elem, 8, $token, 3);
        check!($ty, $elem, 8, $token, 4);
        check!($ty, $elem, 8, $token, 5);
        check!($ty, $elem, 8, $token, 6);
        check!($ty, $elem, 8, $token, 7);
    }};
    ($ty:ident, $elem:ty, 16, $token:path) => {{
        check!($ty, $elem, 16, $token, 0);
        check!($ty, $elem, 16, $token, 1);
        check!($ty, $elem, 16, $token, 2);
        check!($ty, $elem, 16, $token, 3);
        check!($ty, $elem, 16, $token, 4);
        check!($ty, $elem, 16, $token, 5);
        check!($ty, $elem, 16, $token, 6);
        check!($ty, $elem, 16, $token, 7);
        check!($ty, $elem, 16, $token, 8);
        check!($ty, $elem, 16, $token, 9);
        check!($ty, $elem, 16, $token, 10);
        check!($ty, $elem, 16, $token, 11);
        check!($ty, $elem, 16, $token, 12);
        check!($ty, $elem, 16, $token, 13);
        check!($ty, $elem, 16, $token, 14);
        check!($ty, $elem, 16, $token, 15);
    }};
}

/// Boundary shifts for a 32-lane type: 128-bit lane crossings sit at 8 and 24.
macro_rules! all_32 {
    ($ty:ident, $elem:ty, $token:path) => {{
        check!($ty, $elem, 32, $token, 0);
        check!($ty, $elem, 32, $token, 1);
        check!($ty, $elem, 32, $token, 2);
        check!($ty, $elem, 32, $token, 7);
        check!($ty, $elem, 32, $token, 8);
        check!($ty, $elem, 32, $token, 9);
        check!($ty, $elem, 32, $token, 15);
        check!($ty, $elem, 32, $token, 16);
        check!($ty, $elem, 32, $token, 17);
        check!($ty, $elem, 32, $token, 23);
        check!($ty, $elem, 32, $token, 24);
        check!($ty, $elem, 32, $token, 31);
    }};
}

/// Boundary shifts for a 64-lane type: crossings at 16, 32, 48.
macro_rules! all_64 {
    ($ty:ident, $elem:ty, $token:path) => {{
        check!($ty, $elem, 64, $token, 0);
        check!($ty, $elem, 64, $token, 1);
        check!($ty, $elem, 64, $token, 2);
        check!($ty, $elem, 64, $token, 3);
        check!($ty, $elem, 64, $token, 15);
        check!($ty, $elem, 64, $token, 16);
        check!($ty, $elem, 64, $token, 17);
        check!($ty, $elem, 64, $token, 31);
        check!($ty, $elem, 64, $token, 32);
        check!($ty, $elem, 64, $token, 33);
        check!($ty, $elem, 64, $token, 47);
        check!($ty, $elem, 64, $token, 48);
        check!($ty, $elem, 64, $token, 63);
    }};
}

/// Every 128- and 256-bit type, for one token.
macro_rules! all_types {
    ($token:path) => {{
        all_small!(f32x4, f32, 4, $token);
        all_small!(f32x8, f32, 8, $token);
        all_small!(f64x2, f64, 2, $token);
        all_small!(f64x4, f64, 4, $token);
        all_small!(i8x16, i8, 16, $token);
        all_small!(u8x16, u8, 16, $token);
        all_32!(i8x32, i8, $token);
        all_32!(u8x32, u8, $token);
        all_small!(i16x8, i16, 8, $token);
        all_small!(u16x8, u16, 8, $token);
        all_small!(i16x16, i16, 16, $token);
        all_small!(u16x16, u16, 16, $token);
        all_small!(i32x4, i32, 4, $token);
        all_small!(u32x4, u32, 4, $token);
        all_small!(i32x8, i32, 8, $token);
        all_small!(u32x8, u32, 8, $token);
        all_small!(i64x2, i64, 2, $token);
        all_small!(u64x2, u64, 2, $token);
        all_small!(i64x4, i64, 4, $token);
        all_small!(u64x4, u64, 4, $token);
    }};
}

/// Every 512-bit type, for one token. Gated on `w512`, which is what defines
/// them at all.
#[cfg(feature = "w512")]
macro_rules! all_types_512 {
    ($token:path) => {{
        all_small!(f32x16, f32, 16, $token);
        all_small!(f64x8, f64, 8, $token);
        all_64!(i8x64, i8, $token);
        all_64!(u8x64, u8, $token);
        all_32!(i16x32, i16, $token);
        all_32!(u16x32, u16, $token);
        all_small!(i32x16, i32, 16, $token);
        all_small!(u32x16, u32, 16, $token);
        all_small!(i64x8, i64, 8, $token);
        all_small!(u64x8, u64, 8, $token);
    }};
}

/// The scalar backend uses the trait's default body, so this pins the reference
/// every native path is checked against.
#[test]
fn scalar_matches_reference() {
    all_types!(archmage::ScalarToken);
    #[cfg(feature = "w512")]
    all_types_512!(archmage::ScalarToken);
}

/// AVX2 and SSSE3: native 128- and 256-bit paths, plus the 2x256 512-bit
/// polyfills.
#[cfg(target_arch = "x86_64")]
#[test]
fn x86_v3_matches_reference() {
    all_types!(archmage::X64V3Token);
    #[cfg(feature = "w512")]
    all_types_512!(archmage::X64V3Token);
}

/// AVX-512: the native 512-bit paths, plus f32x4/f32x8 which these tokens reach
/// through the V3 delegation in `impls/x86_v4_f32_delegated.rs`.
///
/// The other narrow types are deliberately absent: V4 and V4x implement only the
/// ten 512-bit backends and the two delegated f32 ones, so asking for `u64x4` on
/// a V4 token does not compile. That is the crate's shape, not an omission here.
#[cfg(all(target_arch = "x86_64", feature = "avx512", feature = "w512"))]
#[test]
fn x86_v4_matches_reference() {
    all_types_512!(archmage::X64V4Token);
    all_small!(f32x4, f32, 4, archmage::X64V4Token);
    all_small!(f32x8, f32, 8, archmage::X64V4Token);
    all_types_512!(archmage::X64V4xToken);
    all_small!(f32x4, f32, 4, archmage::X64V4xToken);
    all_small!(f32x8, f32, 8, archmage::X64V4xToken);
}

/// NEON: native `vextq_*` at 128 bits, delegated by the 2x and 4x polyfills.
#[cfg(target_arch = "aarch64")]
#[test]
fn neon_matches_reference() {
    all_types!(archmage::NeonToken);
    #[cfg(feature = "w512")]
    all_types_512!(archmage::NeonToken);
}

/// wasm SIMD128: native `i8x16.shuffle` at 128 bits, delegated above it.
#[cfg(target_arch = "wasm32")]
#[test]
fn wasm_matches_reference() {
    all_types!(archmage::Wasm128Token);
    #[cfg(feature = "w512")]
    all_types_512!(archmage::Wasm128Token);
}
