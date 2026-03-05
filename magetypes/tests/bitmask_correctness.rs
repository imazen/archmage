//! Comprehensive bitmask correctness tests for ALL generic SIMD integer types.
//!
//! Tests every type through the generic layer (backends + generic wrappers),
//! which is what users actually call. Covers:
//!   - W128: i8x16, u8x16, i16x8, u16x8, i32x4, u32x4, i64x2, u64x2
//!   - W256: i8x32, u8x32, i16x16, u16x16, i32x8, u32x8, i64x4, u64x4
//!   - W512: i8x64, u8x64, i16x32, u16x32, i32x16, u32x16, i64x8, u64x8
//!
//! For each type, tests:
//!   1. Individual lane isolation (one lane set, verify exact bit)
//!   2. Cross-boundary pattern (different patterns in low vs high half)
//!   3. All-set and all-clear
//!
//! The i16x16/u16x16 tests are regression tests for a bug where
//! _mm256_packs_epi16 lane interleaving caused lanes 8-15 to be dropped.

/// Compute a bitmask with the low `n` bits set, without overflowing
/// when `n` equals the bit width of T (e.g. 32 for u32, 64 for u64).
fn mask_n_u32(n: u32) -> u32 {
    if n >= 32 { u32::MAX } else { (1u32 << n) - 1 }
}

fn mask_n_u64(n: u32) -> u64 {
    if n >= 64 { u64::MAX } else { (1u64 << n) - 1 }
}

// ============================================================================
// Single macro that works for both signed and unsigned types.
// $set_val is the value that has the MSB set (e.g. -1i8, 0x80u8, -1i16, 0x8000u16).
// $mask_fn is either mask_n_u32 or mask_n_u64.
// ============================================================================
macro_rules! bitmask_tests {
    (
        $mod_name:ident, $ty:ident, $token_ty:ty, $token_fn:path,
        $elem_ty:ty, $lanes:expr, $set_val:expr,
        $mask_ty:ty, $mask_fn:path
    ) => {
        mod $mod_name {
            use super::*;
            use magetypes::simd::generic::$ty;

            fn make(arr: [$elem_ty; $lanes]) -> Option<$ty<$token_ty>> {
                let token = $token_fn()?;
                Some($ty::from_array(token, arr))
            }

            #[test]
            fn individual_lanes() {
                for lane in 0..$lanes {
                    let mut arr = [0 as $elem_ty; $lanes];
                    arr[lane] = $set_val;
                    let Some(v) = make(arr) else { return };
                    let mask = v.bitmask();
                    let expected: $mask_ty = 1 << lane;
                    assert_eq!(
                        mask,
                        expected,
                        "{}: lane {lane} should give {expected:#x}, got {mask:#x}",
                        stringify!($ty)
                    );
                }
            }

            #[test]
            fn all_set() {
                let arr = [$set_val as $elem_ty; $lanes];
                let Some(v) = make(arr) else { return };
                let mask = v.bitmask();
                let expected: $mask_ty = $mask_fn($lanes as u32);
                assert_eq!(
                    mask,
                    expected,
                    "{}: all-set should give {expected:#x}, got {mask:#x}",
                    stringify!($ty)
                );
            }

            #[test]
            fn all_clear() {
                let arr = [0 as $elem_ty; $lanes];
                let Some(v) = make(arr) else { return };
                let mask = v.bitmask();
                assert_eq!(
                    mask,
                    0,
                    "{}: all-clear should give 0, got {mask:#x}",
                    stringify!($ty)
                );
            }

            #[test]
            fn cross_boundary() {
                let half = $lanes / 2;

                // Low half set, high half clear
                let mut arr = [0 as $elem_ty; $lanes];
                for i in 0..half {
                    arr[i] = $set_val;
                }
                let Some(v) = make(arr) else { return };
                let mask = v.bitmask();
                let expected: $mask_ty = $mask_fn(half as u32);
                assert_eq!(
                    mask,
                    expected,
                    "{}: low-half set should give {expected:#x}, got {mask:#x}",
                    stringify!($ty)
                );

                // High half set, low half clear
                let mut arr = [0 as $elem_ty; $lanes];
                for i in half..$lanes {
                    arr[i] = $set_val;
                }
                let Some(v) = make(arr) else { return };
                let mask = v.bitmask();
                let expected: $mask_ty = $mask_fn(half as u32) << (half as u32);
                assert_eq!(
                    mask,
                    expected,
                    "{}: high-half set should give {expected:#x}, got {mask:#x}",
                    stringify!($ty)
                );
            }
        }
    };
}

// ============================================================================
// x86_64 tests (X64V3Token / AVX2 backend)
// ============================================================================
#[cfg(test)]
#[cfg(target_arch = "x86_64")]
#[cfg(not(miri))]
mod x86_tests {
    use super::*;
    #[allow(unused_imports)]
    use super::{mask_n_u32, mask_n_u64};
    use archmage::{SimdToken, X64V3Token};

    fn token() -> Option<X64V3Token> {
        X64V3Token::summon()
    }

    // W128
    bitmask_tests!(
        i8x16, i8x16, X64V3Token, token, i8, 16, -1i8, u32, mask_n_u32
    );
    bitmask_tests!(
        u8x16, u8x16, X64V3Token, token, u8, 16, 0x80u8, u32, mask_n_u32
    );
    bitmask_tests!(
        i16x8, i16x8, X64V3Token, token, i16, 8, -1i16, u32, mask_n_u32
    );
    bitmask_tests!(
        u16x8, u16x8, X64V3Token, token, u16, 8, 0x8000u16, u32, mask_n_u32
    );
    bitmask_tests!(
        i32x4, i32x4, X64V3Token, token, i32, 4, -1i32, u32, mask_n_u32
    );
    bitmask_tests!(
        u32x4,
        u32x4,
        X64V3Token,
        token,
        u32,
        4,
        0x8000_0000u32,
        u32,
        mask_n_u32
    );
    bitmask_tests!(
        i64x2, i64x2, X64V3Token, token, i64, 2, -1i64, u32, mask_n_u32
    );
    bitmask_tests!(
        u64x2,
        u64x2,
        X64V3Token,
        token,
        u64,
        2,
        0x8000_0000_0000_0000u64,
        u32,
        mask_n_u32
    );

    // W256
    bitmask_tests!(
        i8x32, i8x32, X64V3Token, token, i8, 32, -1i8, u32, mask_n_u32
    );
    bitmask_tests!(
        u8x32, u8x32, X64V3Token, token, u8, 32, 0x80u8, u32, mask_n_u32
    );
    bitmask_tests!(
        i16x16, i16x16, X64V3Token, token, i16, 16, -1i16, u32, mask_n_u32
    );
    bitmask_tests!(
        u16x16, u16x16, X64V3Token, token, u16, 16, 0x8000u16, u32, mask_n_u32
    );
    bitmask_tests!(
        i32x8, i32x8, X64V3Token, token, i32, 8, -1i32, u32, mask_n_u32
    );
    bitmask_tests!(
        u32x8,
        u32x8,
        X64V3Token,
        token,
        u32,
        8,
        0x8000_0000u32,
        u32,
        mask_n_u32
    );
    bitmask_tests!(
        i64x4, i64x4, X64V3Token, token, i64, 4, -1i64, u32, mask_n_u32
    );
    bitmask_tests!(
        u64x4,
        u64x4,
        X64V3Token,
        token,
        u64,
        4,
        0x8000_0000_0000_0000u64,
        u32,
        mask_n_u32
    );

    // W512 (polyfill: 2× W256 on V3)
    bitmask_tests!(
        i8x64, i8x64, X64V3Token, token, i8, 64, -1i8, u64, mask_n_u64
    );
    bitmask_tests!(
        u8x64, u8x64, X64V3Token, token, u8, 64, 0x80u8, u64, mask_n_u64
    );
    bitmask_tests!(
        i16x32, i16x32, X64V3Token, token, i16, 32, -1i16, u64, mask_n_u64
    );
    bitmask_tests!(
        u16x32, u16x32, X64V3Token, token, u16, 32, 0x8000u16, u64, mask_n_u64
    );
    bitmask_tests!(
        i32x16, i32x16, X64V3Token, token, i32, 16, -1i32, u64, mask_n_u64
    );
    bitmask_tests!(
        u32x16,
        u32x16,
        X64V3Token,
        token,
        u32,
        16,
        0x8000_0000u32,
        u64,
        mask_n_u64
    );
    bitmask_tests!(
        i64x8, i64x8, X64V3Token, token, i64, 8, -1i64, u64, mask_n_u64
    );
    bitmask_tests!(
        u64x8,
        u64x8,
        X64V3Token,
        token,
        u64,
        8,
        0x8000_0000_0000_0000u64,
        u64,
        mask_n_u64
    );
}

// ============================================================================
// Scalar backend tests — runs on any platform
// ============================================================================
#[cfg(test)]
mod scalar_tests {
    use super::*;
    #[allow(unused_imports)]
    use super::{mask_n_u32, mask_n_u64};
    use archmage::{ScalarToken, SimdToken};

    fn token() -> Option<ScalarToken> {
        ScalarToken::summon()
    }

    // W128
    bitmask_tests!(
        i8x16,
        i8x16,
        ScalarToken,
        token,
        i8,
        16,
        -1i8,
        u32,
        mask_n_u32
    );
    bitmask_tests!(
        u8x16,
        u8x16,
        ScalarToken,
        token,
        u8,
        16,
        0x80u8,
        u32,
        mask_n_u32
    );
    bitmask_tests!(
        i16x8,
        i16x8,
        ScalarToken,
        token,
        i16,
        8,
        -1i16,
        u32,
        mask_n_u32
    );
    bitmask_tests!(
        u16x8,
        u16x8,
        ScalarToken,
        token,
        u16,
        8,
        0x8000u16,
        u32,
        mask_n_u32
    );
    bitmask_tests!(
        i32x4,
        i32x4,
        ScalarToken,
        token,
        i32,
        4,
        -1i32,
        u32,
        mask_n_u32
    );
    bitmask_tests!(
        u32x4,
        u32x4,
        ScalarToken,
        token,
        u32,
        4,
        0x8000_0000u32,
        u32,
        mask_n_u32
    );
    bitmask_tests!(
        i64x2,
        i64x2,
        ScalarToken,
        token,
        i64,
        2,
        -1i64,
        u32,
        mask_n_u32
    );
    bitmask_tests!(
        u64x2,
        u64x2,
        ScalarToken,
        token,
        u64,
        2,
        0x8000_0000_0000_0000u64,
        u32,
        mask_n_u32
    );

    // W256
    bitmask_tests!(
        i8x32,
        i8x32,
        ScalarToken,
        token,
        i8,
        32,
        -1i8,
        u32,
        mask_n_u32
    );
    bitmask_tests!(
        u8x32,
        u8x32,
        ScalarToken,
        token,
        u8,
        32,
        0x80u8,
        u32,
        mask_n_u32
    );
    bitmask_tests!(
        i16x16,
        i16x16,
        ScalarToken,
        token,
        i16,
        16,
        -1i16,
        u32,
        mask_n_u32
    );
    bitmask_tests!(
        u16x16,
        u16x16,
        ScalarToken,
        token,
        u16,
        16,
        0x8000u16,
        u32,
        mask_n_u32
    );
    bitmask_tests!(
        i32x8,
        i32x8,
        ScalarToken,
        token,
        i32,
        8,
        -1i32,
        u32,
        mask_n_u32
    );
    bitmask_tests!(
        u32x8,
        u32x8,
        ScalarToken,
        token,
        u32,
        8,
        0x8000_0000u32,
        u32,
        mask_n_u32
    );
    bitmask_tests!(
        i64x4,
        i64x4,
        ScalarToken,
        token,
        i64,
        4,
        -1i64,
        u32,
        mask_n_u32
    );
    bitmask_tests!(
        u64x4,
        u64x4,
        ScalarToken,
        token,
        u64,
        4,
        0x8000_0000_0000_0000u64,
        u32,
        mask_n_u32
    );

    // W512
    bitmask_tests!(
        i8x64,
        i8x64,
        ScalarToken,
        token,
        i8,
        64,
        -1i8,
        u64,
        mask_n_u64
    );
    bitmask_tests!(
        u8x64,
        u8x64,
        ScalarToken,
        token,
        u8,
        64,
        0x80u8,
        u64,
        mask_n_u64
    );
    bitmask_tests!(
        i16x32,
        i16x32,
        ScalarToken,
        token,
        i16,
        32,
        -1i16,
        u64,
        mask_n_u64
    );
    bitmask_tests!(
        u16x32,
        u16x32,
        ScalarToken,
        token,
        u16,
        32,
        0x8000u16,
        u64,
        mask_n_u64
    );
    bitmask_tests!(
        i32x16,
        i32x16,
        ScalarToken,
        token,
        i32,
        16,
        -1i32,
        u64,
        mask_n_u64
    );
    bitmask_tests!(
        u32x16,
        u32x16,
        ScalarToken,
        token,
        u32,
        16,
        0x8000_0000u32,
        u64,
        mask_n_u64
    );
    bitmask_tests!(
        i64x8,
        i64x8,
        ScalarToken,
        token,
        i64,
        8,
        -1i64,
        u64,
        mask_n_u64
    );
    bitmask_tests!(
        u64x8,
        u64x8,
        ScalarToken,
        token,
        u64,
        8,
        0x8000_0000_0000_0000u64,
        u64,
        mask_n_u64
    );
}
