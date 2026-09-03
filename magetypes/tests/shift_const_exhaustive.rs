//! Exhaustive per-`N` coverage for the shift-by-const contract: every legal
//! `N` in `0..=lane_bits-1` for `shl_const`, `shr_arithmetic_const` (signed
//! types), and `shr_logical_const`, on every integer element type and width,
//! on every backend reachable from the host arch, checked lane-by-lane
//! against a scalar reference.
//!
//! Regression coverage for <https://github.com/imazen/archmage/issues/62>
//! (same defect as #63): the NEON backend used to forward `N` straight to
//! the `vshrq_n_*` immediate intrinsics, whose encoding rejects 0, so
//! `shr_*_const::<0>` was a post-monomorphization compile error on aarch64
//! while the identical code compiled — as the identity shift — on x86, WASM,
//! and scalar. Real const-generic kernels (AV1-style rounding dequants of
//! the shape `(x + (1 << (N - 1))) >> N`) legitimately instantiate `N == 0`.
//! The fix lowers NEON right-shifts to `vshlq_*(a, vdupq_n_*(-N))` and pins
//! a uniform compile-time contract (`N` in `0..=lane_bits-1`) on every
//! backend. `shift_const_boundaries.rs` pins the `N == 0` / `N == max`
//! edges; this file walks the full range so a backend that mis-lowers any
//! intermediate immediate (or regresses at the edges) fails loudly, with
//! the offending type, method, and `N` in the panic message.
//!
//! The out-of-range side of the contract (`N == lane_bits`, negative `N`
//! fail to compile identically on every backend) is pinned by the
//! `compile_fail` doctests in `magetypes/src/simd/mod.rs`.

use archmage::{ScalarToken, SimdToken};
use magetypes::simd::generic::{
    i8x16, i8x32, i16x8, i16x16, i32x4, i32x8, i64x2, i64x4, u8x16, u8x32, u16x8, u16x16, u32x4,
    u32x8, u64x2, u64x4,
};
#[cfg(feature = "w512")]
use magetypes::simd::generic::{i8x64, i16x32, i32x16, i64x8, u8x64, u16x32, u32x16, u64x8};

/// Deterministic per-element-type pattern. Alternating-bit values (0x55…,
/// 0xAA…) make sign-extension vs zero-fill differences visible at every
/// shift distance, not just at the sign bit; MIN/MAX/±1 hit the edges.
macro_rules! pattern {
    ($elem:ty) => {
        core::array::from_fn(|i| match i % 8 {
            0 => 0,
            1 => 1,
            2 => <$elem>::MAX,
            3 => <$elem>::MIN,
            4 => 0x5555_5555_5555_5555u64 as $elem,
            5 => 0xAAAA_AAAA_AAAA_AAAAu64 as $elem,
            6 => 0x8000_0000_0000_0001u64 as $elem,
            _ => 37 as $elem,
        })
    };
}

/// Check one `(type, N)` combination on a signed element type: all three
/// shift flavors against the scalar reference.
///
/// The body runs inside an immediately-invoked closure so each check gets
/// its own stack frame: expanding thousands of checks inline into one test
/// function overflows the 1 MB default WASM stack in debug builds (debug
/// codegen gives every expansion's temporaries a distinct slot in the
/// enclosing frame).
macro_rules! check_n_signed {
    ($n:literal, $Tok:ty, $t:expr, $ty:ident, $elem:ty, $uelem:ty) => {
        (|| {
            let arr: [$elem; _] = pattern!($elem);
            let v = $ty::<$Tok>::from_array($t, arr);
            assert_eq!(
                v.shl_const::<$n>().to_array(),
                arr.map(|x| x << $n),
                concat!(stringify!($ty), "::shl_const::<", $n, ">")
            );
            assert_eq!(
                v.shr_arithmetic_const::<$n>().to_array(),
                arr.map(|x| x >> $n),
                concat!(stringify!($ty), "::shr_arithmetic_const::<", $n, ">")
            );
            assert_eq!(
                v.shr_logical_const::<$n>().to_array(),
                arr.map(|x| ((x as $uelem) >> $n) as $elem),
                concat!(stringify!($ty), "::shr_logical_const::<", $n, ">")
            );
        })()
    };
}

/// Check one `(type, N)` combination on an unsigned element type (unsigned
/// types expose `shl_const` and `shr_logical_const` only). Closure-wrapped
/// for the same WASM stack reason as [`check_n_signed`].
macro_rules! check_n_unsigned {
    ($n:literal, $Tok:ty, $t:expr, $ty:ident, $elem:ty, $uelem:ty) => {
        (|| {
            let arr: [$elem; _] = pattern!($elem);
            let v = $ty::<$Tok>::from_array($t, arr);
            assert_eq!(
                v.shl_const::<$n>().to_array(),
                arr.map(|x| x << $n),
                concat!(stringify!($ty), "::shl_const::<", $n, ">")
            );
            assert_eq!(
                v.shr_logical_const::<$n>().to_array(),
                arr.map(|x| x >> $n),
                concat!(stringify!($ty), "::shr_logical_const::<", $n, ">")
            );
        })()
    };
}

/// Expand a check macro for every listed `N`.
macro_rules! for_each_n {
    ($check:ident, $Tok:ty, $t:expr, $ty:ident, $elem:ty, $uelem:ty; $($n:literal)+) => {
        $( $check!($n, $Tok, $t, $ty, $elem, $uelem); )+
    };
}

macro_rules! all_n_8 {
    ($check:ident, $Tok:ty, $t:expr, $ty:ident, $elem:ty, $uelem:ty) => {
        for_each_n!($check, $Tok, $t, $ty, $elem, $uelem;
            0 1 2 3 4 5 6 7);
    };
}

macro_rules! all_n_16 {
    ($check:ident, $Tok:ty, $t:expr, $ty:ident, $elem:ty, $uelem:ty) => {
        for_each_n!($check, $Tok, $t, $ty, $elem, $uelem;
            0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15);
    };
}

macro_rules! all_n_32 {
    ($check:ident, $Tok:ty, $t:expr, $ty:ident, $elem:ty, $uelem:ty) => {
        for_each_n!($check, $Tok, $t, $ty, $elem, $uelem;
            0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15
            16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31);
    };
}

macro_rules! all_n_64 {
    ($check:ident, $Tok:ty, $t:expr, $ty:ident, $elem:ty, $uelem:ty) => {
        for_each_n!($check, $Tok, $t, $ty, $elem, $uelem;
            0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15
            16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31
            32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47
            48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63);
    };
}

/// Every integer type at the 128-bit and x2-polyfill (256-bit) widths. The
/// x2 forms hit the `[vreg; 2]` array impls, which had their own emission
/// path for the #62/#63 defect.
macro_rules! run_all {
    ($Tok:ty, $t:expr) => {{
        let t = $t;
        all_n_8!(check_n_signed, $Tok, t, i8x16, i8, u8);
        all_n_8!(check_n_signed, $Tok, t, i8x32, i8, u8);
        all_n_16!(check_n_signed, $Tok, t, i16x8, i16, u16);
        all_n_16!(check_n_signed, $Tok, t, i16x16, i16, u16);
        all_n_32!(check_n_signed, $Tok, t, i32x4, i32, u32);
        all_n_32!(check_n_signed, $Tok, t, i32x8, i32, u32);
        all_n_64!(check_n_signed, $Tok, t, i64x2, i64, u64);
        all_n_64!(check_n_signed, $Tok, t, i64x4, i64, u64);
        all_n_8!(check_n_unsigned, $Tok, t, u8x16, u8, u8);
        all_n_8!(check_n_unsigned, $Tok, t, u8x32, u8, u8);
        all_n_16!(check_n_unsigned, $Tok, t, u16x8, u16, u16);
        all_n_16!(check_n_unsigned, $Tok, t, u16x16, u16, u16);
        all_n_32!(check_n_unsigned, $Tok, t, u32x4, u32, u32);
        all_n_32!(check_n_unsigned, $Tok, t, u32x8, u32, u32);
        all_n_64!(check_n_unsigned, $Tok, t, u64x2, u64, u64);
        all_n_64!(check_n_unsigned, $Tok, t, u64x4, u64, u64);
    }};
}

/// The 512-bit widths — 2x-W256 polyfills on most backends, native AVX-512
/// impls (with their own byte-shift polyfills) on `X64V4Token`/`X64V4xToken`
/// when the `avx512` feature is on.
#[cfg(feature = "w512")]
macro_rules! run_all_512 {
    ($Tok:ty, $t:expr) => {{
        let t = $t;
        all_n_8!(check_n_signed, $Tok, t, i8x64, i8, u8);
        all_n_16!(check_n_signed, $Tok, t, i16x32, i16, u16);
        all_n_32!(check_n_signed, $Tok, t, i32x16, i32, u32);
        all_n_64!(check_n_signed, $Tok, t, i64x8, i64, u64);
        all_n_8!(check_n_unsigned, $Tok, t, u8x64, u8, u8);
        all_n_16!(check_n_unsigned, $Tok, t, u16x32, u16, u16);
        all_n_32!(check_n_unsigned, $Tok, t, u32x16, u32, u32);
        all_n_64!(check_n_unsigned, $Tok, t, u64x8, u64, u64);
    }};
}

#[cfg(not(feature = "w512"))]
macro_rules! run_all_512 {
    ($Tok:ty, $t:expr) => {{
        let _ = $t;
    }};
}

#[test]
fn scalar_backend_shift_exhaustive() {
    run_all!(ScalarToken, ScalarToken::summon().unwrap());
    run_all_512!(ScalarToken, ScalarToken::summon().unwrap());
}

#[cfg(target_arch = "x86_64")]
#[test]
fn x64v3_backend_shift_exhaustive() {
    if let Some(t) = archmage::X64V3Token::summon() {
        run_all!(archmage::X64V3Token, t);
        run_all_512!(archmage::X64V3Token, t);
    }
}

/// `X64V4Token` natively backs only the 512-bit widths (128/256-bit work
/// goes through a `.v3()` downcast), so only the 512 matrix runs here.
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
#[test]
fn x64v4_backend_shift_exhaustive() {
    if let Some(t) = archmage::X64V4Token::summon() {
        run_all_512!(archmage::X64V4Token, t);
    }
}

/// `X64V4xToken` has its own 512-bit backend impls (not shared with
/// `X64V4Token`), so exercise them separately.
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
#[test]
fn x64v4x_backend_shift_exhaustive() {
    if let Some(t) = archmage::X64V4xToken::summon() {
        run_all_512!(archmage::X64V4xToken, t);
    }
}

#[cfg(target_arch = "aarch64")]
#[test]
fn neon_backend_shift_exhaustive() {
    if let Some(t) = archmage::NeonToken::summon() {
        run_all!(archmage::NeonToken, t);
        run_all_512!(archmage::NeonToken, t);
    }
}

#[cfg(target_arch = "wasm32")]
#[test]
fn wasm128_backend_shift_exhaustive() {
    if let Some(t) = archmage::Wasm128Token::summon() {
        run_all!(archmage::Wasm128Token, t);
        run_all_512!(archmage::Wasm128Token, t);
    }
}
