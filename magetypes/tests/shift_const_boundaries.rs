//! Boundary coverage for the right-shift-by-const methods: `N == 0` (identity)
//! and `N == lane_bits - 1` (maximum in-contract shift) on every backend
//! reachable from the host arch.
//!
//! Regression test for <https://github.com/imazen/archmage/issues/63>: the NEON
//! backend used to forward `N` straight to `vshrq_n_*`, whose immediate
//! encoding rejects 0, so `shr_*_const::<0>` was a post-monomorphization
//! compile error on aarch64 while every other backend accepted it.

use archmage::{ScalarToken, SimdToken};
use magetypes::simd::generic::{
    i8x16, i8x32, i16x8, i16x16, i32x4, i32x8, i64x2, i64x4, u8x16, u8x32, u16x8, u16x16, u32x4,
    u32x8, u64x2, u64x4,
};
#[cfg(feature = "w512")]
use magetypes::simd::generic::{i8x64, i16x32, i32x16, i64x8, u8x64, u16x32, u32x16, u64x8};

/// Deterministic per-element-type test pattern hitting sign/magnitude edges.
macro_rules! pattern {
    ($elem:ty) => {
        core::array::from_fn(|i| match i % 6 {
            0 => 0,
            1 => 1,
            2 => <$elem>::MAX,
            3 => <$elem>::MIN,
            4 => 37 as $elem,
            _ => <$elem>::MIN.wrapping_add(1),
        })
    };
}

macro_rules! check_signed {
    ($Tok:ty, $t:expr, $ty:ident, $elem:ty, $uelem:ty, $max:literal) => {{
        let arr: [$elem; _] = pattern!($elem);
        let v = $ty::<$Tok>::from_array($t, arr);
        // N == 0 is the identity shift for both flavors.
        assert_eq!(v.shr_arithmetic_const::<0>().to_array(), arr);
        assert_eq!(v.shr_logical_const::<0>().to_array(), arr);
        // N == lane_bits - 1 against a scalar-computed reference.
        assert_eq!(
            v.shr_arithmetic_const::<$max>().to_array(),
            arr.map(|x| x >> $max)
        );
        assert_eq!(
            v.shr_logical_const::<$max>().to_array(),
            arr.map(|x| ((x as $uelem) >> $max) as $elem)
        );
    }};
}

macro_rules! check_unsigned {
    ($Tok:ty, $t:expr, $ty:ident, $elem:ty, $max:literal) => {{
        let arr: [$elem; _] = pattern!($elem);
        let v = $ty::<$Tok>::from_array($t, arr);
        assert_eq!(v.shr_logical_const::<0>().to_array(), arr);
        assert_eq!(
            v.shr_logical_const::<$max>().to_array(),
            arr.map(|x| x >> $max)
        );
    }};
}

/// Exercises every integer type at 128-bit and x2-polyfill widths — the x2
/// forms hit the NEON `[vreg; 2]` array impls, which had their own emission
/// path for the defect.
macro_rules! run_all {
    ($Tok:ty, $t:expr) => {{
        let t = $t;
        check_signed!($Tok, t, i8x16, i8, u8, 7);
        check_signed!($Tok, t, i8x32, i8, u8, 7);
        check_signed!($Tok, t, i16x8, i16, u16, 15);
        check_signed!($Tok, t, i16x16, i16, u16, 15);
        check_signed!($Tok, t, i32x4, i32, u32, 31);
        check_signed!($Tok, t, i32x8, i32, u32, 31);
        check_signed!($Tok, t, i64x2, i64, u64, 63);
        check_signed!($Tok, t, i64x4, i64, u64, 63);
        check_unsigned!($Tok, t, u8x16, u8, 7);
        check_unsigned!($Tok, t, u8x32, u8, 7);
        check_unsigned!($Tok, t, u16x8, u16, 15);
        check_unsigned!($Tok, t, u16x16, u16, 15);
        check_unsigned!($Tok, t, u32x4, u32, 31);
        check_unsigned!($Tok, t, u32x8, u32, 31);
        check_unsigned!($Tok, t, u64x2, u64, 63);
        check_unsigned!($Tok, t, u64x4, u64, 63);
    }};
}

/// The 512-bit widths — on most backends these are 2x-W256 polyfills, but on
/// `X64V4Token`/`X64V4xToken` (with the `avx512` feature) they are native
/// AVX-512 impls with their own byte-shift polyfills.
#[cfg(feature = "w512")]
macro_rules! run_all_512 {
    ($Tok:ty, $t:expr) => {{
        let t = $t;
        check_signed!($Tok, t, i8x64, i8, u8, 7);
        check_signed!($Tok, t, i16x32, i16, u16, 15);
        check_signed!($Tok, t, i32x16, i32, u32, 31);
        check_signed!($Tok, t, i64x8, i64, u64, 63);
        check_unsigned!($Tok, t, u8x64, u8, 7);
        check_unsigned!($Tok, t, u16x32, u16, 15);
        check_unsigned!($Tok, t, u32x16, u32, 31);
        check_unsigned!($Tok, t, u64x8, u64, 63);
    }};
}

#[cfg(not(feature = "w512"))]
macro_rules! run_all_512 {
    ($Tok:ty, $t:expr) => {{
        let _ = $t;
    }};
}

#[test]
fn scalar_backend_shift_boundaries() {
    run_all!(ScalarToken, ScalarToken::summon().unwrap());
    run_all_512!(ScalarToken, ScalarToken::summon().unwrap());
}

#[cfg(target_arch = "x86_64")]
#[test]
fn x64v3_backend_shift_boundaries() {
    if let Some(t) = archmage::X64V3Token::summon() {
        run_all!(archmage::X64V3Token, t);
        run_all_512!(archmage::X64V3Token, t);
    }
}

/// `X64V4Token` natively backs only the 512-bit widths (128/256-bit work goes
/// through a `.v3()` downcast), so only the 512 matrix runs here.
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
#[test]
fn x64v4_backend_shift_boundaries() {
    if let Some(t) = archmage::X64V4Token::summon() {
        run_all_512!(archmage::X64V4Token, t);
    }
}

#[cfg(target_arch = "aarch64")]
#[test]
fn neon_backend_shift_boundaries() {
    if let Some(t) = archmage::NeonToken::summon() {
        run_all!(archmage::NeonToken, t);
        run_all_512!(archmage::NeonToken, t);
    }
}

#[cfg(target_arch = "wasm32")]
#[test]
fn wasm128_backend_shift_boundaries() {
    if let Some(t) = archmage::Wasm128Token::summon() {
        run_all!(archmage::Wasm128Token, t);
        run_all_512!(archmage::Wasm128Token, t);
    }
}
