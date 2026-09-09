//! `all_true` / `any_true` answer one question on every backend and every
//! width: **is the lane's sign bit set?**
//!
//! This is not a restatement of the obvious. Before this test existed the
//! operation meant four different things:
//!
//! - `x86_v3` read sign bits (`movemask`),
//! - `x86_v4` tested lanes for nonzero (`cmpneq` against a zero vector),
//! - NEON and wasm tested for nonzero, except NEON `u32x4`, which tested for
//!   `u32::MAX`,
//! - and the 512-bit polyfills folded their halves with `AND` before
//!   reducing — correct under the sign-bit rule, but inconsistent with the
//!   nonzero rule the narrower widths on those same backends were using, so
//!   NEON's `i32x16::all_true` disagreed with its own `i32x4::all_true` on
//!   the same logical data.
//!
//! Every assertion below uses inputs that are *not* comparison masks, because
//! that is the only place the rules differ — an all-ones or all-zeros vector
//! cannot tell them apart, and every pre-existing test used exactly those.
use archmage::SimdToken;
#[cfg(feature = "w512")]
use magetypes::simd::backends::I32x16Backend;
use magetypes::simd::backends::{
    I8x16Backend, I16x8Backend, I32x4Backend, I32x8Backend, I64x2Backend, U8x16Backend,
    U32x4Backend,
};
#[cfg(feature = "w512")]
use magetypes::simd::generic::i32x16;
use magetypes::simd::generic::{i8x16, i16x8, i32x4, i32x8, i64x2, u8x16, u32x4};

/// Lanes that are nonzero with the sign bit clear. Sign-bit rule: false.
/// Nonzero rule: true.
macro_rules! assert_nonzero_is_not_true {
    ($ty:ident, $tok:expr, $one:expr) => {{
        let v = $ty::splat($tok, $one);
        assert!(
            !v.all_true(),
            concat!(
                stringify!($ty),
                "::all_true must test the sign bit, not nonzero"
            )
        );
        assert!(
            !v.any_true(),
            concat!(
                stringify!($ty),
                "::any_true must test the sign bit, not nonzero"
            )
        );
    }};
}

/// A real comparison mask still behaves.
macro_rules! assert_mask_behaves {
    ($ty:ident, $tok:expr, $ones:expr, $zero:expr) => {{
        assert!($ty::splat($tok, $ones).all_true());
        assert!($ty::splat($tok, $ones).any_true());
        assert!(!$ty::splat($tok, $zero).all_true());
        assert!(!$ty::splat($tok, $zero).any_true());
    }};
}

fn check<T>(token: T)
where
    T: Copy
        + I32x4Backend
        + I32x8Backend
        + I8x16Backend
        + I16x8Backend
        + I64x2Backend
        + U32x4Backend
        + U8x16Backend,
{
    assert_nonzero_is_not_true!(i32x4, token, 1i32);
    assert_nonzero_is_not_true!(i32x8, token, 1i32);
    assert_nonzero_is_not_true!(i8x16, token, 1i8);
    assert_nonzero_is_not_true!(i16x8, token, 1i16);
    assert_nonzero_is_not_true!(i64x2, token, 1i64);
    // Unsigned lanes use the same rule: the top bit. Not `!= 0`, and not the
    // `== MAX` that NEON's `u32x4` used to apply.
    assert_nonzero_is_not_true!(u32x4, token, 1u32);
    assert_nonzero_is_not_true!(u8x16, token, 1u8);
    assert!(u32x4::splat(token, 0x8000_0000u32).all_true());
    assert!(u8x16::splat(token, 0x80u8).all_true());

    assert_mask_behaves!(i32x4, token, -1i32, 0i32);
    assert_mask_behaves!(i32x8, token, -1i32, 0i32);
    assert_mask_behaves!(i8x16, token, -1i8, 0i8);
    assert_mask_behaves!(u32x4, token, u32::MAX, 0u32);
}

/// A width whose backend folds its halves must agree with one that does not,
/// on data where folding could change the answer: disjoint nonzero bits per
/// half, every sign bit clear.
fn check_cross_width<T>(token: T)
where
    T: Copy + I32x4Backend + I32x8Backend,
{
    let a4 = i32x4::from_array(token, [0b01; 4]);
    let a8 = i32x8::from_array(
        token,
        core::array::from_fn(|i| if i < 4 { 0b01 } else { 0b10 }),
    );
    assert!(!a4.all_true());
    assert_eq!(a4.all_true(), a8.all_true(), "i32x4 and i32x8 disagree");
    assert_eq!(a4.any_true(), a8.any_true(), "i32x4 and i32x8 disagree");

    // Sign bits set in one half only: `any_true` true, `all_true` false.
    let b8 = i32x8::from_array(token, core::array::from_fn(|i| if i < 4 { -1 } else { 1 }));
    assert!(b8.any_true() && !b8.all_true());
}

/// The 512-bit types are `w512`-gated, and they matter most here: fold-first
/// landed at 512-bit first, which is what made a backend disagree with itself.
#[cfg(feature = "w512")]
fn check_w512<T>(token: T)
where
    T: Copy + I32x4Backend + I32x8Backend + I32x16Backend,
{
    assert_nonzero_is_not_true!(i32x16, token, 1i32);
    assert_mask_behaves!(i32x16, token, -1i32, 0i32);

    let a8 = i32x8::from_array(
        token,
        core::array::from_fn(|i| if i < 4 { 0b01 } else { 0b10 }),
    );
    let a16 = i32x16::from_array(
        token,
        core::array::from_fn(|i| if i / 4 < 2 { 0b01 } else { 0b10 }),
    );
    assert_eq!(a8.all_true(), a16.all_true(), "i32x8 and i32x16 disagree");
    assert_eq!(a8.any_true(), a16.any_true(), "i32x8 and i32x16 disagree");

    let b16 = i32x16::from_array(token, core::array::from_fn(|i| if i < 8 { -1 } else { 1 }));
    assert!(b16.any_true() && !b16.all_true());
}

/// `X64V4Token` implements only the 512-bit backend traits (it has no
/// `I32x4Backend`/`I32x8Backend` — see the AVX-512 narrow-width gap tracked
/// separately), so its arm can only exercise `i32x16`.
#[cfg(feature = "w512")]
fn check_w512_only<T>(token: T)
where
    T: Copy + I32x16Backend,
{
    assert_nonzero_is_not_true!(i32x16, token, 1i32);
    assert_mask_behaves!(i32x16, token, -1i32, 0i32);

    // Disjoint nonzero bits across the folded halves, sign bits clear.
    let a16 = i32x16::from_array(
        token,
        core::array::from_fn(|i| if i / 4 < 2 { 0b01 } else { 0b10 }),
    );
    assert!(!a16.all_true());
    assert!(!a16.any_true());

    let b16 = i32x16::from_array(token, core::array::from_fn(|i| if i < 8 { -1 } else { 1 }));
    assert!(b16.any_true() && !b16.all_true());
}

macro_rules! check_all {
    ($token:expr) => {{
        let token = $token;
        check(token);
        check_cross_width(token);
        #[cfg(feature = "w512")]
        check_w512(token);
    }};
}

#[test]
fn scalar_backend_uses_the_sign_bit_contract() {
    check_all!(archmage::ScalarToken);
}

#[cfg(target_arch = "x86_64")]
#[test]
fn x86_v3_uses_the_sign_bit_contract() {
    // Honesty guard: under `--no-default-features` archmage has no runtime
    // CPUID path, so `summon()` correctly returns `None` on every host and
    // this arm has nothing to run. With `std` it must not be `None` on a
    // capable host — that would silently skip the x86 coverage.
    #[cfg(feature = "std")]
    let host_has_v3 = std::arch::is_x86_feature_detected!("avx2")
        && std::arch::is_x86_feature_detected!("fma")
        && std::arch::is_x86_feature_detected!("sse4.2");
    match archmage::X64V3Token::summon() {
        Some(token) => check_all!(token),
        None => {
            #[cfg(feature = "std")]
            assert!(
                !host_has_v3,
                "the host reports AVX2+FMA+SSE4.2 but X64V3Token did not summon — \
                 the v3 arm would have been silently skipped"
            );
        }
    }
}

/// AVX-512 had its own contract before this (`cmpneq` against zero, i.e.
/// nonzero), so it needs its own arm rather than riding on the v3 one.
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
#[test]
fn x86_v4_uses_the_sign_bit_contract() {
    #[cfg(feature = "std")]
    let host_has_v4 = std::arch::is_x86_feature_detected!("avx512f")
        && std::arch::is_x86_feature_detected!("avx512bw")
        && std::arch::is_x86_feature_detected!("avx512vl")
        && std::arch::is_x86_feature_detected!("avx512dq");
    match archmage::X64V4Token::summon() {
        Some(token) => {
            #[cfg(feature = "w512")]
            check_w512_only(token);
        }
        None => {
            #[cfg(feature = "std")]
            assert!(
                !host_has_v4,
                "the host reports the AVX-512 F/BW/VL/DQ set but X64V4Token did \
                 not summon — the v4 arm would have been silently skipped"
            );
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[test]
fn neon_uses_the_sign_bit_contract() {
    let Some(token) = archmage::NeonToken::summon() else {
        // NEON is architectural on aarch64; a None here is a detection bug,
        // not an absent feature, so fail rather than skip.
        panic!("NeonToken did not summon on aarch64 — NEON is baseline there");
    };
    check_all!(token);
}

#[cfg(target_arch = "wasm32")]
#[test]
fn wasm128_uses_the_sign_bit_contract() {
    let Some(token) = archmage::Wasm128Token::summon() else {
        panic!("simd128 is required; build with -Ctarget-feature=+simd128");
    };
    check_all!(token);
}
