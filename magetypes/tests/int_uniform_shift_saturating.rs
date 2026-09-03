//! Differential coverage for the uniform variable shifts and saturating
//! add/sub added to the 8- and 16-bit integer types.
//!
//! Every backend reachable from the host arch is compared against a scalar
//! reference computed here in the test (not borrowed from the scalar backend,
//! so a generator bug in that backend is caught too), at every width the types
//! come in.
//!
//! Three things this file is deliberately strict about:
//!
//! 1. **Boundary counts.** Shifts are exercised at `0`, `1`, `bits/2`,
//!    `bits - 1`, `bits`, `bits + 1`, `2 * bits`, `255`, `256`, `257` and
//!    `u32::MAX`. `256` is the NEON trap (USHL/SSHL read only the low signed
//!    byte of each amount lane, so an unclamped 256 wraps to a no-op) and
//!    `u32::MAX` is the x86 trap (the count is read as a 64-bit unsigned from
//!    the low quadword of an XMM). `bits` and above must produce the
//!    contracted zero / sign fill, identically everywhere.
//!
//! 2. **Lane order.** [`lane_order_cases`] uses operands whose value is
//!    distinct in every lane and compares element by element, so a permuted
//!    result fails — a checksum or a `reduce_add` would not catch it.
//!
//! 3. **The arm actually ran.** Each backend test asserts the token summons
//!    whenever the host reports the features for it, and every runner returns
//!    the number of comparisons it made so the caller can assert a floor. A
//!    macro that silently expanded to nothing, or a `summon()` that quietly
//!    returned `None` on a capable CPU, fails instead of passing green.

use archmage::{ScalarToken, SimdToken};
use magetypes::simd::generic::{i8x16, i8x32, i16x8, i16x16, u8x16, u8x32, u16x8, u16x16};
#[cfg(feature = "w512")]
use magetypes::simd::generic::{i8x64, i16x32, u8x64, u16x32};

/// Shift counts covering in-range, exactly-at-width, and the two
/// wrap-the-count traps described in the module docs.
const fn counts(bits: u32) -> [u32; 11] {
    [
        0,
        1,
        bits / 2,
        bits - 1,
        bits,
        bits + 1,
        bits * 2,
        255,
        256,
        257,
        u32::MAX,
    ]
}

/// Deterministic operand pattern hitting sign and magnitude edges.
macro_rules! pattern_a {
    ($elem:ty) => {
        core::array::from_fn(|i| match i % 8 {
            0 => 0,
            1 => 1,
            2 => <$elem>::MAX,
            3 => <$elem>::MIN,
            4 => 37 as $elem,
            5 => <$elem>::MIN.wrapping_add(1),
            6 => <$elem>::MAX.wrapping_sub(1),
            _ => 0x5A as $elem,
        })
    };
}

/// Second operand chosen so the saturating ops overflow in both directions.
macro_rules! pattern_b {
    ($elem:ty) => {
        core::array::from_fn(|i| match i % 8 {
            0 => <$elem>::MAX,
            1 => <$elem>::MIN,
            2 => <$elem>::MAX,
            3 => <$elem>::MIN,
            4 => 1,
            5 => <$elem>::MAX,
            6 => 2,
            _ => <$elem>::MIN,
        })
    };
}

// ============================================================================
// Per-type checks
// ============================================================================

/// Signed 8/16-bit: all three shift flavours plus saturating add/sub.
macro_rules! check_signed {
    ($Tok:ty, $t:expr, $ty:ident, $elem:ty, $uelem:ty, $bits:literal, $n:expr) => {{
        let t = $t;
        let mut cases = 0usize;

        let a: [$elem; $n] = pattern_a!($elem);
        let b: [$elem; $n] = pattern_b!($elem);
        let va = $ty::<$Tok>::from_array(t, a);
        let vb = $ty::<$Tok>::from_array(t, b);

        for count in counts($bits) {
            let want_shl: [$elem; $n] = a.map(|x| {
                if count >= $bits {
                    0
                } else {
                    (((x as $uelem) << count) as $elem)
                }
            });
            assert_eq!(
                va.shl_uniform(count).to_array(),
                want_shl,
                "{}::shl_uniform({count})",
                stringify!($ty)
            );

            let want_shr_l: [$elem; $n] = a.map(|x| {
                if count >= $bits {
                    0
                } else {
                    (((x as $uelem) >> count) as $elem)
                }
            });
            assert_eq!(
                va.shr_logical_uniform(count).to_array(),
                want_shr_l,
                "{}::shr_logical_uniform({count})",
                stringify!($ty)
            );

            // count >= bits is a sign fill, i.e. a shift by bits - 1.
            let clamped = if count > $bits - 1 { $bits - 1 } else { count };
            let want_shr_a: [$elem; $n] = a.map(|x| x >> clamped);
            assert_eq!(
                va.shr_arithmetic_uniform(count).to_array(),
                want_shr_a,
                "{}::shr_arithmetic_uniform({count})",
                stringify!($ty)
            );
            cases += 3;
        }

        let want_add: [$elem; $n] = core::array::from_fn(|i| a[i].saturating_add(b[i]));
        assert_eq!(
            va.saturating_add(vb).to_array(),
            want_add,
            "{}::saturating_add",
            stringify!($ty)
        );
        let want_sub: [$elem; $n] = core::array::from_fn(|i| a[i].saturating_sub(b[i]));
        assert_eq!(
            va.saturating_sub(vb).to_array(),
            want_sub,
            "{}::saturating_sub",
            stringify!($ty)
        );
        cases += 2;

        cases += lane_order_cases!($Tok, t, $ty, $elem, $n);
        cases
    }};
}

/// Unsigned 8/16-bit: no arithmetic shift on the generic surface.
macro_rules! check_unsigned {
    ($Tok:ty, $t:expr, $ty:ident, $elem:ty, $bits:literal, $n:expr) => {{
        let t = $t;
        let mut cases = 0usize;

        let a: [$elem; $n] = pattern_a!($elem);
        let b: [$elem; $n] = pattern_b!($elem);
        let va = $ty::<$Tok>::from_array(t, a);
        let vb = $ty::<$Tok>::from_array(t, b);

        for count in counts($bits) {
            let want_shl: [$elem; $n] = a.map(|x| if count >= $bits { 0 } else { x << count });
            assert_eq!(
                va.shl_uniform(count).to_array(),
                want_shl,
                "{}::shl_uniform({count})",
                stringify!($ty)
            );

            let want_shr: [$elem; $n] = a.map(|x| if count >= $bits { 0 } else { x >> count });
            assert_eq!(
                va.shr_logical_uniform(count).to_array(),
                want_shr,
                "{}::shr_logical_uniform({count})",
                stringify!($ty)
            );
            cases += 2;
        }

        let want_add: [$elem; $n] = core::array::from_fn(|i| a[i].saturating_add(b[i]));
        assert_eq!(
            va.saturating_add(vb).to_array(),
            want_add,
            "{}::saturating_add",
            stringify!($ty)
        );
        let want_sub: [$elem; $n] = core::array::from_fn(|i| a[i].saturating_sub(b[i]));
        assert_eq!(
            va.saturating_sub(vb).to_array(),
            want_sub,
            "{}::saturating_sub",
            stringify!($ty)
        );
        cases += 2;

        cases += lane_order_cases!($Tok, t, $ty, $elem, $n);
        cases
    }};
}

/// Lane-order pin: every lane of both operands holds a distinct value, so any
/// permutation of the result (an AVX2-style in-lane pack, a swapped polyfill
/// half, a reversed array impl) fails the element-wise comparison.
macro_rules! lane_order_cases {
    ($Tok:ty, $t:expr, $ty:ident, $elem:ty, $n:expr) => {{
        let t = $t;
        let a: [$elem; $n] = core::array::from_fn(|i| (i as $elem).wrapping_mul(3).wrapping_add(1));
        let b: [$elem; $n] = core::array::from_fn(|i| (i as $elem).wrapping_mul(7));
        let va = $ty::<$Tok>::from_array(t, a);
        let vb = $ty::<$Tok>::from_array(t, b);

        // A shift by 0 is the identity: the result must be the input, lane for
        // lane, in the original order.
        assert_eq!(
            va.shl_uniform(0).to_array(),
            a,
            "{} lane order: shl_uniform(0) is not the identity",
            stringify!($ty)
        );
        assert_eq!(
            va.shr_logical_uniform(0).to_array(),
            a,
            "{} lane order: shr_logical_uniform(0) is not the identity",
            stringify!($ty)
        );
        let want: [$elem; $n] = core::array::from_fn(|i| a[i].saturating_add(b[i]));
        assert_eq!(
            va.saturating_add(vb).to_array(),
            want,
            "{} lane order: saturating_add pairs the wrong lanes",
            stringify!($ty)
        );
        3usize
    }};
}

// ============================================================================
// Exhaustive 8-bit sweeps
// ============================================================================

/// Every one of the 65 536 `(u8, u8)` operand pairs through `saturating_add`
/// and `saturating_sub`, and every one of the 256 `u8` values through every
/// boundary shift count. 8-bit is small enough that "sampled" is not an
/// excuse.
macro_rules! exhaustive_u8 {
    ($Tok:ty, $t:expr) => {{
        let t = $t;
        let mut cases = 0usize;
        for a_hi in 0..16u16 {
            let a: [u8; 16] = core::array::from_fn(|i| (a_hi * 16 + i as u16) as u8);
            let va = u8x16::<$Tok>::from_array(t, a);
            for b_hi in 0..16u16 {
                // Rotating b's low nibble against a's makes the sweep cover all
                // 256 x 256 operand pairs, not just the ones that share a low
                // nibble.
                for rot in 0..16usize {
                    let b: [u8; 16] =
                        core::array::from_fn(|i| (b_hi * 16 + ((i + rot) % 16) as u16) as u8);
                    let vb = u8x16::<$Tok>::from_array(t, b);
                    let want_add: [u8; 16] = core::array::from_fn(|i| a[i].saturating_add(b[i]));
                    let want_sub: [u8; 16] = core::array::from_fn(|i| a[i].saturating_sub(b[i]));
                    assert_eq!(va.saturating_add(vb).to_array(), want_add);
                    assert_eq!(va.saturating_sub(vb).to_array(), want_sub);
                    cases += 2;
                }
            }
            for count in counts(8) {
                let want_shl: [u8; 16] = a.map(|x| if count >= 8 { 0 } else { x << count });
                let want_shr: [u8; 16] = a.map(|x| if count >= 8 { 0 } else { x >> count });
                assert_eq!(va.shl_uniform(count).to_array(), want_shl, "u8 shl {count}");
                assert_eq!(
                    va.shr_logical_uniform(count).to_array(),
                    want_shr,
                    "u8 shr {count}"
                );
                cases += 2;
            }
        }
        cases
    }};
}

/// The signed twin of [`exhaustive_u8`], including the arithmetic shift (the
/// flavour with the sign-fill contract and, on x86, the most involved
/// polyfill).
macro_rules! exhaustive_i8 {
    ($Tok:ty, $t:expr) => {{
        let t = $t;
        let mut cases = 0usize;
        for a_hi in 0..16i32 {
            let a: [i8; 16] = core::array::from_fn(|i| (a_hi * 16 + i as i32 - 128) as i8);
            let va = i8x16::<$Tok>::from_array(t, a);
            for b_hi in 0..16i32 {
                for rot in 0..16usize {
                    let b: [i8; 16] =
                        core::array::from_fn(|i| (b_hi * 16 + ((i + rot) % 16) as i32 - 128) as i8);
                    let vb = i8x16::<$Tok>::from_array(t, b);
                    let want_add: [i8; 16] = core::array::from_fn(|i| a[i].saturating_add(b[i]));
                    let want_sub: [i8; 16] = core::array::from_fn(|i| a[i].saturating_sub(b[i]));
                    assert_eq!(va.saturating_add(vb).to_array(), want_add);
                    assert_eq!(va.saturating_sub(vb).to_array(), want_sub);
                    cases += 2;
                }
            }
            for count in counts(8) {
                let want_shl: [i8; 16] = a.map(|x| {
                    if count >= 8 {
                        0
                    } else {
                        ((x as u8) << count) as i8
                    }
                });
                let want_shr_l: [i8; 16] = a.map(|x| {
                    if count >= 8 {
                        0
                    } else {
                        ((x as u8) >> count) as i8
                    }
                });
                let clamped = if count > 7 { 7 } else { count };
                let want_shr_a: [i8; 16] = a.map(|x| x >> clamped);
                assert_eq!(va.shl_uniform(count).to_array(), want_shl, "i8 shl {count}");
                assert_eq!(
                    va.shr_logical_uniform(count).to_array(),
                    want_shr_l,
                    "i8 shr_l {count}"
                );
                assert_eq!(
                    va.shr_arithmetic_uniform(count).to_array(),
                    want_shr_a,
                    "i8 shr_a {count}"
                );
                cases += 3;
            }
        }
        cases
    }};
}

// ============================================================================
// Per-backend runners
// ============================================================================

/// The 128-bit and 2x-polyfill widths.
macro_rules! run_all {
    ($Tok:ty, $t:expr) => {{
        let t = $t;
        let mut n = 0usize;
        n += check_signed!($Tok, t, i8x16, i8, u8, 8, 16);
        n += check_signed!($Tok, t, i8x32, i8, u8, 8, 32);
        n += check_signed!($Tok, t, i16x8, i16, u16, 16, 8);
        n += check_signed!($Tok, t, i16x16, i16, u16, 16, 16);
        n += check_unsigned!($Tok, t, u8x16, u8, 8, 16);
        n += check_unsigned!($Tok, t, u8x32, u8, 8, 32);
        n += check_unsigned!($Tok, t, u16x8, u16, 16, 8);
        n += check_unsigned!($Tok, t, u16x16, u16, 16, 16);
        n += exhaustive_u8!($Tok, t);
        n += exhaustive_i8!($Tok, t);
        n
    }};
}

/// The 512-bit widths: a 4x/2x polyfill on most backends, native AVX-512 on
/// `X64V4Token`/`X64V4xToken`.
#[cfg(feature = "w512")]
macro_rules! run_all_512 {
    ($Tok:ty, $t:expr) => {{
        let t = $t;
        let mut n = 0usize;
        n += check_signed!($Tok, t, i8x64, i8, u8, 8, 64);
        n += check_signed!($Tok, t, i16x32, i16, u16, 16, 32);
        n += check_unsigned!($Tok, t, u8x64, u8, 8, 64);
        n += check_unsigned!($Tok, t, u16x32, u16, 16, 32);
        n
    }};
}

#[cfg(not(feature = "w512"))]
macro_rules! run_all_512 {
    ($Tok:ty, $t:expr) => {{
        let _ = $t;
        0usize
    }};
}

/// Floors for the "did the arm actually run" assertions. Deliberately close to
/// the real counts so a macro that expands to fewer cases trips them.
const MIN_CASES_W128: usize = 17_000;
#[cfg(feature = "w512")]
const MIN_CASES_W512: usize = 250;
#[cfg(not(feature = "w512"))]
const MIN_CASES_W512: usize = 0;

#[test]
fn scalar_backend() {
    let t = ScalarToken::summon().expect("ScalarToken must always summon");
    let n = run_all!(ScalarToken, t) + run_all_512!(ScalarToken, t);
    assert!(
        n >= MIN_CASES_W128 + MIN_CASES_W512,
        "scalar arm ran only {n} comparisons"
    );
}

#[cfg(target_arch = "x86_64")]
#[test]
fn x64v3_backend() {
    // X64V3Token is the x86-64-v3 level: AVX2 is the discriminating feature.
    let host_has_v3 = std::arch::is_x86_feature_detected!("avx2")
        && std::arch::is_x86_feature_detected!("fma")
        && std::arch::is_x86_feature_detected!("sse4.2");
    match archmage::X64V3Token::summon() {
        Some(t) => {
            let n = run_all!(archmage::X64V3Token, t) + run_all_512!(archmage::X64V3Token, t);
            assert!(n >= MIN_CASES_W128, "v3 arm ran only {n} comparisons");
        }
        None => assert!(
            !host_has_v3,
            "the host reports AVX2+FMA+SSE4.2 but X64V3Token did not summon — \
             the v3 arm would have been silently skipped"
        ),
    }
}

/// `X64V4Token` natively backs only the 512-bit widths; 128/256-bit work on a
/// V4 machine goes through a `.v3()` downcast, so only the 512 matrix runs.
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
#[test]
fn x64v4_backend() {
    let host_has_v4 = std::arch::is_x86_feature_detected!("avx512f")
        && std::arch::is_x86_feature_detected!("avx512bw")
        && std::arch::is_x86_feature_detected!("avx512vl")
        && std::arch::is_x86_feature_detected!("avx512dq")
        && std::arch::is_x86_feature_detected!("avx512cd");
    match archmage::X64V4Token::summon() {
        Some(t) => {
            let n = run_all_512!(archmage::X64V4Token, t);
            assert!(n >= MIN_CASES_W512, "v4 arm ran only {n} comparisons");
        }
        None => assert!(
            !host_has_v4,
            "the host reports the AVX-512 F/BW/VL/DQ/CD set but X64V4Token did \
             not summon — the v4 arm would have been silently skipped"
        ),
    }
}

#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
#[test]
fn x64v4x_backend() {
    match archmage::X64V4xToken::summon() {
        Some(t) => {
            let n = run_all_512!(archmage::X64V4xToken, t);
            assert!(n >= MIN_CASES_W512, "v4x arm ran only {n} comparisons");
        }
        None => {
            // v4x adds VBMI/VBMI2/VNNI/BITALG/IFMA/VPOPCNTDQ/GFNI/VAES on top
            // of v4; absence is ordinary, so there is nothing to assert beyond
            // "the v4 arm above covers the shared 512-bit paths".
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[test]
fn neon_backend() {
    // AdvSIMD is baseline on aarch64: a None here is a defect, not a skip.
    let t = archmage::NeonToken::summon().expect("NEON is baseline on aarch64");
    let n = run_all!(archmage::NeonToken, t) + run_all_512!(archmage::NeonToken, t);
    assert!(
        n >= MIN_CASES_W128 + MIN_CASES_W512,
        "NEON arm ran only {n} comparisons"
    );
}

#[cfg(target_arch = "wasm32")]
#[test]
fn wasm128_backend() {
    let built_with_simd128 = cfg!(target_feature = "simd128");
    match archmage::Wasm128Token::summon() {
        Some(t) => {
            let n = run_all!(archmage::Wasm128Token, t) + run_all_512!(archmage::Wasm128Token, t);
            assert!(n >= MIN_CASES_W128, "wasm128 arm ran only {n} comparisons");
        }
        None => assert!(
            !built_with_simd128,
            "built with target-feature=+simd128 but Wasm128Token did not summon \
             — the wasm arm would have been silently skipped"
        ),
    }
}
