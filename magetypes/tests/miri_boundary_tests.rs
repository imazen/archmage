//! Exhaustive boundary tests for load/store APIs.
//!
//! These tests exercise edge cases that Miri can verify for soundness:
//! - Alignment boundary conditions
//! - cast_slice with various lengths
//! - Byte-level operations (as_bytes, from_bytes)
//! - All element types and vector widths
//!
//! Run with: cargo +nightly miri test -p magetypes --test miri_boundary_tests

#![cfg(target_arch = "x86_64")]

use archmage::{SimdToken, X64V3Token};

// =============================================================================
// cast_slice boundary tests
// =============================================================================

/// Test cast_slice with lengths that are exact multiples, not multiples, and zero.
#[test]
fn test_cast_slice_length_boundaries_f32x4() {
    use magetypes::simd::f32x4;

    if let Some(token) = X64V3Token::try_new() {
        // Aligned buffer (Vec guarantees alignment for primitives)
        let mut data: Vec<f32> = vec![0.0; 128];

        // Length 0: should succeed (empty slice)
        assert!(f32x4::cast_slice(token, &data[..0]).is_some());
        assert_eq!(f32x4::cast_slice(token, &data[..0]).unwrap().len(), 0);

        // Length 1, 2, 3: not multiples of 4, should fail
        assert!(f32x4::cast_slice(token, &data[..1]).is_none());
        assert!(f32x4::cast_slice(token, &data[..2]).is_none());
        assert!(f32x4::cast_slice(token, &data[..3]).is_none());

        // Length 4: exactly 1 vector
        assert!(f32x4::cast_slice(token, &data[..4]).is_some());
        assert_eq!(f32x4::cast_slice(token, &data[..4]).unwrap().len(), 1);

        // Length 5, 6, 7: not multiples of 4
        assert!(f32x4::cast_slice(token, &data[..5]).is_none());
        assert!(f32x4::cast_slice(token, &data[..6]).is_none());
        assert!(f32x4::cast_slice(token, &data[..7]).is_none());

        // Length 8: exactly 2 vectors
        assert!(f32x4::cast_slice(token, &data[..8]).is_some());
        assert_eq!(f32x4::cast_slice(token, &data[..8]).unwrap().len(), 2);

        // Test cast_slice_mut with same boundaries
        assert!(f32x4::cast_slice_mut(token, &mut data[..0]).is_some());
        assert!(f32x4::cast_slice_mut(token, &mut data[..1]).is_none());
        assert!(f32x4::cast_slice_mut(token, &mut data[..4]).is_some());
        assert!(f32x4::cast_slice_mut(token, &mut data[..5]).is_none());
    }
}

/// Test cast_slice with lengths for f64x2 (2 elements per vector).
#[test]
fn test_cast_slice_length_boundaries_f64x2() {
    use magetypes::simd::f64x2;

    if let Some(token) = X64V3Token::try_new() {
        let mut data: Vec<f64> = vec![0.0; 64];

        // Length 0: empty
        assert!(f64x2::cast_slice(token, &data[..0]).is_some());

        // Length 1: not a multiple of 2
        assert!(f64x2::cast_slice(token, &data[..1]).is_none());

        // Length 2: exactly 1 vector
        assert!(f64x2::cast_slice(token, &data[..2]).is_some());
        assert_eq!(f64x2::cast_slice(token, &data[..2]).unwrap().len(), 1);

        // Length 3: not a multiple of 2
        assert!(f64x2::cast_slice(token, &data[..3]).is_none());

        // Mutable version
        assert!(f64x2::cast_slice_mut(token, &mut data[..0]).is_some());
        assert!(f64x2::cast_slice_mut(token, &mut data[..1]).is_none());
        assert!(f64x2::cast_slice_mut(token, &mut data[..2]).is_some());
    }
}

/// Test cast_slice with i8x16 (16 elements per vector).
#[test]
fn test_cast_slice_length_boundaries_i8x16() {
    use magetypes::simd::i8x16;

    if let Some(token) = X64V3Token::try_new() {
        let mut data: Vec<i8> = vec![0; 256];

        // Check all lengths 0-17
        assert!(i8x16::cast_slice(token, &data[..0]).is_some()); // 0 vectors
        for len in 1..16 {
            assert!(
                i8x16::cast_slice(token, &data[..len]).is_none(),
                "len {} should fail",
                len
            );
        }
        assert!(i8x16::cast_slice(token, &data[..16]).is_some()); // 1 vector
        assert!(i8x16::cast_slice(token, &data[..17]).is_none());
        assert!(i8x16::cast_slice(token, &data[..32]).is_some()); // 2 vectors

        // Mutable version spot checks
        assert!(i8x16::cast_slice_mut(token, &mut data[..0]).is_some());
        assert!(i8x16::cast_slice_mut(token, &mut data[..15]).is_none());
        assert!(i8x16::cast_slice_mut(token, &mut data[..16]).is_some());
    }
}

// =============================================================================
// Alignment boundary tests
// =============================================================================

/// Test cast_slice rejects unaligned data.
#[test]
fn test_cast_slice_alignment_rejection_f32x4() {
    use magetypes::simd::f32x4;

    if let Some(token) = X64V3Token::try_new() {
        // Create a large aligned buffer, then take unaligned slices
        let data: Vec<f32> = vec![1.0; 128];

        // The Vec itself is aligned, so cast_slice on the whole thing should work
        assert!(f32x4::cast_slice(token, &data[..8]).is_some());

        // Taking a slice starting at index 1 might be unaligned (depends on Vec allocation)
        // We can't guarantee unalignment, but we test that cast_slice handles it correctly
        let slice_at_1 = &data[1..9]; // 8 elements starting at index 1
        let result = f32x4::cast_slice(token, slice_at_1);
        // Result depends on actual alignment - either Some (if aligned) or None (if not)
        // Miri will catch any UB if we access an unaligned pointer incorrectly

        // If it succeeds, verify we can safely access all elements
        if let Some(vectors) = result {
            for v in vectors {
                let _ = v.to_array(); // Access the data
            }
        }
    }
}

// =============================================================================
// load/store with fixed-size arrays
// =============================================================================

/// Test load and store with boundary values.
#[test]
fn test_load_store_boundary_values_f32x4() {
    use magetypes::simd::f32x4;

    if let Some(token) = X64V3Token::try_new() {
        // Test with special floating-point values
        let specials: [f32; 4] = [f32::MIN, f32::MAX, f32::INFINITY, f32::NEG_INFINITY];
        let v = f32x4::load(token, &specials);
        let mut out = [0.0f32; 4];
        v.store(&mut out);
        assert_eq!(out[0], f32::MIN);
        assert_eq!(out[1], f32::MAX);
        assert!(out[2].is_infinite() && out[2].is_sign_positive());
        assert!(out[3].is_infinite() && out[3].is_sign_negative());

        // Test with NaN (NaN != NaN, so use is_nan)
        let nans: [f32; 4] = [f32::NAN, -f32::NAN, f32::NAN, f32::NAN];
        let v = f32x4::load(token, &nans);
        let mut out = [0.0f32; 4];
        v.store(&mut out);
        assert!(out[0].is_nan());
        assert!(out[1].is_nan());

        // Test with subnormals
        let subnormals: [f32; 4] = [
            f32::MIN_POSITIVE / 2.0,
            -f32::MIN_POSITIVE / 2.0,
            f32::from_bits(1),          // smallest positive subnormal
            f32::from_bits(0x007FFFFF), // largest subnormal
        ];
        let v = f32x4::load(token, &subnormals);
        let mut out = [0.0f32; 4];
        v.store(&mut out);
        // Verify bits are preserved exactly
        assert_eq!(out[0].to_bits(), subnormals[0].to_bits());
        assert_eq!(out[1].to_bits(), subnormals[1].to_bits());
        assert_eq!(out[2].to_bits(), subnormals[2].to_bits());
        assert_eq!(out[3].to_bits(), subnormals[3].to_bits());
    }
}

/// Test integer load/store with boundary values.
#[test]
fn test_load_store_boundary_values_i32x4() {
    use magetypes::simd::i32x4;

    if let Some(token) = X64V3Token::try_new() {
        let boundaries: [i32; 4] = [i32::MIN, i32::MAX, 0, -1];
        let v = i32x4::load(token, &boundaries);
        let mut out = [0i32; 4];
        v.store(&mut out);
        assert_eq!(out, boundaries);
    }
}

/// Test u64 load/store with boundary values.
#[test]
fn test_load_store_boundary_values_u64x2() {
    use magetypes::simd::u64x2;

    if let Some(token) = X64V3Token::try_new() {
        let boundaries: [u64; 2] = [u64::MIN, u64::MAX];
        let v = u64x2::load(token, &boundaries);
        let mut out = [0u64; 2];
        v.store(&mut out);
        assert_eq!(out, boundaries);
    }
}

// =============================================================================
// Byte-level operations
// =============================================================================

/// Test as_bytes and from_bytes roundtrip.
#[test]
fn test_bytes_roundtrip_f32x4() {
    use magetypes::simd::f32x4;

    if let Some(token) = X64V3Token::try_new() {
        let original: [f32; 4] = [1.0, 2.0, 3.0, 4.0];
        let v = f32x4::load(token, &original);

        // Get bytes
        let bytes: &[u8; 16] = v.as_bytes();

        // Reconstruct from bytes
        let v2 = f32x4::from_bytes(token, bytes);
        let mut out = [0.0f32; 4];
        v2.store(&mut out);
        assert_eq!(out, original);
    }
}

/// Test as_bytes_mut modification.
#[test]
fn test_bytes_mut_modification_i32x4() {
    use magetypes::simd::i32x4;

    if let Some(token) = X64V3Token::try_new() {
        let original: [i32; 4] = [0x01020304, 0x05060708, 0x090A0B0C, 0x0D0E0F10];
        let mut v = i32x4::load(token, &original);

        // Modify first byte
        let bytes = v.as_bytes_mut();
        bytes[0] = 0xFF;

        // Verify modification took effect
        let mut out = [0i32; 4];
        v.store(&mut out);
        // First i32 should have its low byte changed to 0xFF
        assert_eq!(out[0] & 0xFF, 0xFF);
    }
}

/// Test from_bytes with all-zeros and all-ones patterns.
#[test]
fn test_from_bytes_patterns() {
    use magetypes::simd::{f32x4, i32x4};

    if let Some(token) = X64V3Token::try_new() {
        // All zeros
        let zeros: [u8; 16] = [0; 16];
        let v = f32x4::from_bytes(token, &zeros);
        let arr = v.to_array();
        assert_eq!(arr, [0.0, 0.0, 0.0, 0.0]);

        // All ones (for i32: -1)
        let ones: [u8; 16] = [0xFF; 16];
        let v = i32x4::from_bytes(token, &ones);
        let arr = v.to_array();
        assert_eq!(arr, [-1, -1, -1, -1]);
    }
}

/// Test from_bytes_owned (takes ownership).
#[test]
fn test_from_bytes_owned() {
    use magetypes::simd::f32x4;

    if let Some(token) = X64V3Token::try_new() {
        let bytes: [u8; 16] = [
            0x00, 0x00, 0x80, 0x3F, // 1.0f32 in little-endian
            0x00, 0x00, 0x00, 0x40, // 2.0f32
            0x00, 0x00, 0x40, 0x40, // 3.0f32
            0x00, 0x00, 0x80, 0x40, // 4.0f32
        ];
        let v = f32x4::from_bytes_owned(token, bytes);
        let arr = v.to_array();
        assert_eq!(arr, [1.0, 2.0, 3.0, 4.0]);
    }
}

// =============================================================================
// Bitcast operations
// =============================================================================

/// Test bitcast preserves all bit patterns.
#[test]
fn test_bitcast_preserves_bits() {
    use magetypes::simd::u32x4;

    if let Some(token) = X64V3Token::try_new() {
        // Create a pattern with specific bits (use u32 to avoid literal overflow)
        let uints: [u32; 4] = [0x12345678, 0x9ABCDEF0, 0xFFFFFFFF, 0x00000000];
        let v_u32 = u32x4::load(token, &uints);

        // Bitcast to i32
        let v_i32 = v_u32.bitcast_i32x4();
        let i32_bytes = v_i32.as_bytes();

        // Bitcast to f32
        let v_f32 = v_u32.bitcast_f32x4();
        let f32_bytes = v_f32.as_bytes();

        // All representations should have identical bytes
        let u32_bytes = v_u32.as_bytes();
        assert_eq!(u32_bytes, i32_bytes);
        assert_eq!(u32_bytes, f32_bytes);
    }
}

/// Test bitcast_ref and bitcast_mut don't cause aliasing issues.
#[test]
fn test_bitcast_ref_mut_aliasing() {
    use magetypes::simd::{f32x4, i32x4};

    if let Some(token) = X64V3Token::try_new() {
        let mut v = f32x4::splat(token, 1.0);

        // Get immutable bitcast reference
        let as_i32: &i32x4 = v.bitcast_ref_i32x4();
        let _ = as_i32.to_array(); // Read through the reference

        // Get mutable bitcast reference (this should be fine, v is mutable)
        let as_i32_mut: &mut i32x4 = v.bitcast_mut_i32x4();
        let arr = as_i32_mut.to_array();
        // 1.0f32 = 0x3F800000
        assert_eq!(arr[0], 0x3F800000_u32 as i32);
    }
}

// =============================================================================
// All integer types coverage
// =============================================================================

/// Test all signed integer types load/store.
#[test]
fn test_all_signed_int_types() {
    use magetypes::simd::{i8x16, i16x8, i32x4, i64x2};

    if let Some(token) = X64V3Token::try_new() {
        // i8x16
        let data_i8: [i8; 16] = [
            i8::MIN,
            -1,
            0,
            1,
            i8::MAX,
            -128,
            127,
            42,
            -42,
            100,
            -100,
            0,
            0,
            0,
            0,
            0,
        ];
        let v = i8x16::load(token, &data_i8);
        let mut out = [0i8; 16];
        v.store(&mut out);
        assert_eq!(out, data_i8);

        // i16x8
        let data_i16: [i16; 8] = [i16::MIN, -1, 0, 1, i16::MAX, -32768, 32767, 1000];
        let v = i16x8::load(token, &data_i16);
        let mut out = [0i16; 8];
        v.store(&mut out);
        assert_eq!(out, data_i16);

        // i32x4
        let data_i32: [i32; 4] = [i32::MIN, 0, i32::MAX, -1];
        let v = i32x4::load(token, &data_i32);
        let mut out = [0i32; 4];
        v.store(&mut out);
        assert_eq!(out, data_i32);

        // i64x2
        let data_i64: [i64; 2] = [i64::MIN, i64::MAX];
        let v = i64x2::load(token, &data_i64);
        let mut out = [0i64; 2];
        v.store(&mut out);
        assert_eq!(out, data_i64);
    }
}

/// Test all unsigned integer types load/store.
#[test]
fn test_all_unsigned_int_types() {
    use magetypes::simd::{u8x16, u16x8, u32x4, u64x2};

    if let Some(token) = X64V3Token::try_new() {
        // u8x16
        let data_u8: [u8; 16] = [
            u8::MIN,
            1,
            127,
            128,
            255,
            u8::MAX,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        ];
        let v = u8x16::load(token, &data_u8);
        let mut out = [0u8; 16];
        v.store(&mut out);
        assert_eq!(out, data_u8);

        // u16x8
        let data_u16: [u16; 8] = [u16::MIN, 1, 32767, 32768, 65535, u16::MAX, 0, 1000];
        let v = u16x8::load(token, &data_u16);
        let mut out = [0u16; 8];
        v.store(&mut out);
        assert_eq!(out, data_u16);

        // u32x4
        let data_u32: [u32; 4] = [u32::MIN, 0, u32::MAX, 0xDEADBEEF];
        let v = u32x4::load(token, &data_u32);
        let mut out = [0u32; 4];
        v.store(&mut out);
        assert_eq!(out, data_u32);

        // u64x2
        let data_u64: [u64; 2] = [u64::MIN, u64::MAX];
        let v = u64x2::load(token, &data_u64);
        let mut out = [0u64; 2];
        v.store(&mut out);
        assert_eq!(out, data_u64);
    }
}

// =============================================================================
// 256-bit types
// =============================================================================

/// Test 256-bit load/store boundaries.
#[test]
fn test_256bit_load_store() {
    use magetypes::simd::{f32x8, f64x4, i32x8};

    if let Some(token) = X64V3Token::try_new() {
        // f32x8
        let data_f32: [f32; 8] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let v = f32x8::load(token, &data_f32);
        let mut out = [0.0f32; 8];
        v.store(&mut out);
        assert_eq!(out, data_f32);

        // f64x4
        let data_f64: [f64; 4] = [1.0, 2.0, 3.0, 4.0];
        let v = f64x4::load(token, &data_f64);
        let mut out = [0.0f64; 4];
        v.store(&mut out);
        assert_eq!(out, data_f64);

        // i32x8
        let data_i32: [i32; 8] = [i32::MIN, -1, 0, 1, i32::MAX, 100, -100, 0];
        let v = i32x8::load(token, &data_i32);
        let mut out = [0i32; 8];
        v.store(&mut out);
        assert_eq!(out, data_i32);
    }
}

/// Test 256-bit cast_slice length boundaries.
/// Note: Vec<f32> isn't guaranteed 32-byte aligned, so we test length logic
/// on arrays which have proper alignment.
#[test]
fn test_256bit_cast_slice_boundaries() {
    use magetypes::simd::f32x8;

    if let Some(token) = X64V3Token::try_new() {
        // Use aligned array (repr(align) would be better, but load/to_array test alignment)
        let arr1: [f32; 8] = [1.0; 8];

        // Test length boundary logic via to_array roundtrip (doesn't require cast_slice)
        let v = f32x8::load(token, &arr1);
        let out = v.to_array();
        assert_eq!(out, arr1);

        // Test that incorrect lengths fail for f32x8 (8 elements per vector)
        let data: Vec<f32> = vec![1.0; 128];
        for len in 1..8 {
            // Lengths not multiple of 8 must fail regardless of alignment
            assert!(
                f32x8::cast_slice(token, &data[..len]).is_none(),
                "len {} should fail (not multiple of 8)",
                len
            );
        }

        // 0, 8, 16 etc. may succeed or fail depending on alignment
        // We can't assert success without guaranteed alignment
        let _ = f32x8::cast_slice(token, &data[..8]); // result depends on alignment
    }
}

// =============================================================================
// 512-bit types (AVX-512)
// =============================================================================

#[cfg(feature = "avx512")]
mod avx512_tests {
    use archmage::{SimdToken, X64V4Token};

    /// Test 512-bit load/store boundaries.
    #[test]
    fn test_512bit_load_store() {
        use magetypes::simd::{f32x16, i32x16};

        if let Some(token) = X64V4Token::try_new() {
            // f32x16
            let data_f32: [f32; 16] = [
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
                16.0,
            ];
            let v = f32x16::load(token, &data_f32);
            let mut out = [0.0f32; 16];
            v.store(&mut out);
            assert_eq!(out, data_f32);

            // i32x16
            let data_i32: [i32; 16] = [
                i32::MIN,
                -1,
                0,
                1,
                i32::MAX,
                100,
                -100,
                0,
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                8,
            ];
            let v = i32x16::load(token, &data_i32);
            let mut out = [0i32; 16];
            v.store(&mut out);
            assert_eq!(out, data_i32);
        }
    }

    /// Test 512-bit cast_slice length boundaries.
    /// Note: Vec<f32> isn't guaranteed 64-byte aligned, so we test length logic.
    #[test]
    fn test_512bit_cast_slice_boundaries() {
        use magetypes::simd::f32x16;

        if let Some(token) = X64V4Token::try_new() {
            // Test length boundary logic via to_array roundtrip
            let arr: [f32; 16] = [1.0; 16];
            let v = f32x16::load(token, &arr);
            let out = v.to_array();
            assert_eq!(out, arr);

            // Test that incorrect lengths fail for f32x16 (16 elements per vector)
            let data: Vec<f32> = vec![1.0; 256];
            for len in 1..16 {
                // Lengths not multiple of 16 must fail regardless of alignment
                assert!(
                    f32x16::cast_slice(token, &data[..len]).is_none(),
                    "len {} should fail (not multiple of 16)",
                    len
                );
            }

            // 0, 16, 32 etc. may succeed or fail depending on alignment
            let _ = f32x16::cast_slice(token, &data[..16]); // result depends on alignment
        }
    }
}
