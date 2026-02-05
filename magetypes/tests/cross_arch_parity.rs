//! Cross-architecture parity tests for W128 SIMD types.
//!
//! **Auto-generated** by `cargo xtask generate` - do not edit manually.
//!
//! These tests verify that all 10 W128 types produce identical results
//! across x86_64, aarch64, and wasm32 architectures.

#![allow(unused_imports)]
#![allow(clippy::approx_constant)]
#![allow(clippy::excessive_precision)]

// ============================================================================
// Test Suite Macro
// ============================================================================

/// Macro that generates parity tests for all W128 types given a token type.
///
/// The token type varies by architecture:
/// - x86_64: X64V3Token
/// - aarch64: NeonToken
/// - wasm32: Wasm128Token
macro_rules! w128_parity_suite {
    ($token_ty:ty) => {
        use archmage::SimdToken;
        use magetypes::simd::*;

        // ========== f32x4 tests ==========

        #[test]
        fn test_f32x4_from_array_to_array() {
            if let Some(token) = <$token_ty>::summon() {
                let input = [1.5f32, 2.25, -3.75, 4.0];
                let v = f32x4::from_array(token, input);
                assert_eq!(v.to_array(), input);
            }
        }

        #[test]
        fn test_f32x4_splat() {
            if let Some(token) = <$token_ty>::summon() {
                let v = f32x4::splat(token, 42.5);
                assert_eq!(v.to_array(), [42.5, 42.5, 42.5, 42.5]);
            }
        }

        #[test]
        fn test_f32x4_add_sub() {
            if let Some(token) = <$token_ty>::summon() {
                let a = f32x4::from_array(token, [1.0, 2.0, 3.0, 4.0]);
                let b = f32x4::from_array(token, [0.5, 1.5, 2.5, 3.5]);
                assert_eq!((a + b).to_array(), [1.5, 3.5, 5.5, 7.5]);
                assert_eq!((a - b).to_array(), [0.5, 0.5, 0.5, 0.5]);
            }
        }

        #[test]
        fn test_f32x4_mul_div() {
            if let Some(token) = <$token_ty>::summon() {
                let a = f32x4::from_array(token, [2.0, 4.0, 6.0, 8.0]);
                let b = f32x4::from_array(token, [2.0, 2.0, 2.0, 2.0]);
                assert_eq!((a * b).to_array(), [4.0, 8.0, 12.0, 16.0]);
                assert_eq!((a / b).to_array(), [1.0, 2.0, 3.0, 4.0]);
            }
        }

        #[test]
        fn test_f32x4_neg_abs() {
            if let Some(token) = <$token_ty>::summon() {
                let v = f32x4::from_array(token, [1.0, -2.0, 3.0, -4.0]);
                assert_eq!((-v).to_array(), [-1.0, 2.0, -3.0, 4.0]);
                assert_eq!(v.abs().to_array(), [1.0, 2.0, 3.0, 4.0]);
            }
        }

        #[test]
        fn test_f32x4_min_max() {
            if let Some(token) = <$token_ty>::summon() {
                let a = f32x4::from_array(token, [1.0, 5.0, 3.0, 7.0]);
                let b = f32x4::from_array(token, [2.0, 4.0, 6.0, 8.0]);
                assert_eq!(a.min(b).to_array(), [1.0, 4.0, 3.0, 7.0]);
                assert_eq!(a.max(b).to_array(), [2.0, 5.0, 6.0, 8.0]);
            }
        }

        #[test]
        fn test_f32x4_floor_ceil_round() {
            if let Some(token) = <$token_ty>::summon() {
                let v = f32x4::from_array(token, [1.3, 2.7, -1.3, -2.7]);
                assert_eq!(v.floor().to_array(), [1.0, 2.0, -2.0, -3.0]);
                assert_eq!(v.ceil().to_array(), [2.0, 3.0, -1.0, -2.0]);
                assert_eq!(v.round().to_array(), [1.0, 3.0, -1.0, -3.0]);
            }
        }

        #[test]
        fn test_f32x4_sqrt() {
            if let Some(token) = <$token_ty>::summon() {
                let v = f32x4::from_array(token, [1.0, 4.0, 9.0, 16.0]);
                assert_eq!(v.sqrt().to_array(), [1.0, 2.0, 3.0, 4.0]);
            }
        }

        #[test]
        fn test_f32x4_mul_add() {
            if let Some(token) = <$token_ty>::summon() {
                let a = f32x4::from_array(token, [1.0, 2.0, 3.0, 4.0]);
                let b = f32x4::from_array(token, [2.0, 3.0, 4.0, 5.0]);
                let c = f32x4::from_array(token, [0.5, 0.5, 0.5, 0.5]);
                // a * b + c
                let result = a.mul_add(b, c);
                assert_eq!(result.to_array(), [2.5, 6.5, 12.5, 20.5]);
            }
        }

        #[test]
        fn test_f32x4_reduce() {
            if let Some(token) = <$token_ty>::summon() {
                let v = f32x4::from_array(token, [1.0, 2.0, 3.0, 4.0]);
                assert_eq!(v.reduce_add(), 10.0);
                assert_eq!(v.reduce_min(), 1.0);
                assert_eq!(v.reduce_max(), 4.0);
            }
        }

        #[test]
        fn test_f32x4_comparison() {
            if let Some(token) = <$token_ty>::summon() {
                let a = f32x4::from_array(token, [1.0, 2.0, 3.0, 4.0]);
                let b = f32x4::from_array(token, [2.0, 2.0, 2.0, 2.0]);

                // simd_lt: a < b => [true, false, false, false]
                // Comparisons return the same type with all-1s or all-0s per lane
                let lt_mask = a.simd_lt(b);

                // Verify via bitcast to i32x4 and checking values
                // All-1s (0xFFFFFFFF) as i32 is -1, so we check via bits
                let lt_bits = lt_mask.bitcast_i32x4();
                // lane 0 should be all-1s (-1 as signed), others should be 0
                let lt_arr = lt_bits.to_array();
                assert_eq!(lt_arr[0], -1, "lane 0 should be true");
                assert_eq!(lt_arr[1], 0, "lane 1 should be false");
                assert_eq!(lt_arr[2], 0, "lane 2 should be false");
                assert_eq!(lt_arr[3], 0, "lane 3 should be false");

                // simd_eq: a == b => [false, true, false, false]
                let eq_mask = a.simd_eq(b);
                let eq_bits = eq_mask.bitcast_i32x4().to_array();
                assert_eq!(eq_bits[0], 0, "lane 0 should be false");
                assert_eq!(eq_bits[1], -1, "lane 1 should be true");
                assert_eq!(eq_bits[2], 0, "lane 2 should be false");
                assert_eq!(eq_bits[3], 0, "lane 3 should be false");

                // simd_ge: a >= b => [false, true, true, true]
                let ge_mask = a.simd_ge(b);
                let ge_bits = ge_mask.bitcast_i32x4().to_array();
                assert_eq!(ge_bits[0], 0, "lane 0 should be false");
                assert_eq!(ge_bits[1], -1, "lane 1 should be true");
                assert_eq!(ge_bits[2], -1, "lane 2 should be true");
                assert_eq!(ge_bits[3], -1, "lane 3 should be true");
            }
        }

        // ========== f64x2 tests ==========

        #[test]
        fn test_f64x2_from_array_to_array() {
            if let Some(token) = <$token_ty>::summon() {
                let input = [1.5f64, -2.25];
                let v = f64x2::from_array(token, input);
                assert_eq!(v.to_array(), input);
            }
        }

        #[test]
        fn test_f64x2_add_sub_mul_div() {
            if let Some(token) = <$token_ty>::summon() {
                let a = f64x2::from_array(token, [4.0, 8.0]);
                let b = f64x2::from_array(token, [2.0, 2.0]);
                assert_eq!((a + b).to_array(), [6.0, 10.0]);
                assert_eq!((a - b).to_array(), [2.0, 6.0]);
                assert_eq!((a * b).to_array(), [8.0, 16.0]);
                assert_eq!((a / b).to_array(), [2.0, 4.0]);
            }
        }

        #[test]
        fn test_f64x2_sqrt() {
            if let Some(token) = <$token_ty>::summon() {
                let v = f64x2::from_array(token, [4.0, 9.0]);
                assert_eq!(v.sqrt().to_array(), [2.0, 3.0]);
            }
        }

        // ========== i32x4 tests ==========

        #[test]
        fn test_i32x4_from_array_to_array() {
            if let Some(token) = <$token_ty>::summon() {
                let input = [1i32, -2, 3, -4];
                let v = i32x4::from_array(token, input);
                assert_eq!(v.to_array(), input);
            }
        }

        #[test]
        fn test_i32x4_add_sub() {
            if let Some(token) = <$token_ty>::summon() {
                let a = i32x4::from_array(token, [10, 20, 30, 40]);
                let b = i32x4::from_array(token, [1, 2, 3, 4]);
                assert_eq!((a + b).to_array(), [11, 22, 33, 44]);
                assert_eq!((a - b).to_array(), [9, 18, 27, 36]);
            }
        }

        #[test]
        fn test_i32x4_min_max_abs() {
            if let Some(token) = <$token_ty>::summon() {
                let a = i32x4::from_array(token, [1, -5, 3, -7]);
                let b = i32x4::from_array(token, [2, -4, 2, -8]);
                assert_eq!(a.min(b).to_array(), [1, -5, 2, -8]);
                assert_eq!(a.max(b).to_array(), [2, -4, 3, -7]);
                assert_eq!(a.abs().to_array(), [1, 5, 3, 7]);
            }
        }

        #[test]
        fn test_i32x4_shift() {
            if let Some(token) = <$token_ty>::summon() {
                let v = i32x4::from_array(token, [4, 8, 16, -32]);
                assert_eq!(v.shl::<1>().to_array(), [8, 16, 32, -64]);
                // Note: shr behavior differs by architecture
                // - x86: logical shift (zero-fill)
                // - ARM: arithmetic shift (sign-extend) for signed types
                // shr_arithmetic is consistent across architectures
                assert_eq!(v.shr_arithmetic::<1>().to_array(), [2, 4, 8, -16]); // arithmetic shift
            }
        }

        // Logical shift test (x86 only - ARM shr is arithmetic for signed types)
        #[cfg(target_arch = "x86_64")]
        #[test]
        fn test_i32x4_shr_logical() {
            if let Some(token) = <$token_ty>::summon() {
                let v = i32x4::from_array(token, [4, 8, 16, -32]);
                assert_eq!(v.shr::<1>().to_array(), [2, 4, 8, 2147483632]); // logical shift on x86
            }
        }

        // Bitwise operator tests (x86 only - ARM uses method calls instead)
        #[cfg(target_arch = "x86_64")]
        #[test]
        fn test_i32x4_bitwise() {
            if let Some(token) = <$token_ty>::summon() {
                let a = i32x4::from_array(token, [0b1010, 0b1100, 0b1111, 0b0000]);
                let b = i32x4::from_array(token, [0b1100, 0b1010, 0b0101, 0b1111]);
                assert_eq!((a & b).to_array(), [0b1000, 0b1000, 0b0101, 0b0000]);
                assert_eq!((a | b).to_array(), [0b1110, 0b1110, 0b1111, 0b1111]);
                assert_eq!((a ^ b).to_array(), [0b0110, 0b0110, 0b1010, 0b1111]);
                assert_eq!(a.not().to_array(), [!0b1010i32, !0b1100, !0b1111, !0b0000]);
            }
        }

        #[test]
        fn test_i32x4_comparison() {
            if let Some(token) = <$token_ty>::summon() {
                let a = i32x4::from_array(token, [1, 2, 3, 4]);
                let b = i32x4::from_array(token, [2, 2, 2, 2]);

                let lt_mask = a.simd_lt(b);
                assert!(lt_mask.any_true());

                let eq_mask = a.simd_eq(b);
                assert!(eq_mask.any_true());
            }
        }

        #[test]
        fn test_i32x4_bitmask() {
            if let Some(token) = <$token_ty>::summon() {
                let a = i32x4::from_array(token, [-1, 0, -1, 0]); // MSB: 1, 0, 1, 0
                let mask = a.bitmask();
                assert_eq!(mask, 0b0101); // lanes 0 and 2 have MSB set
            }
        }

        // ========== u32x4 tests ==========

        #[test]
        fn test_u32x4_from_array_to_array() {
            if let Some(token) = <$token_ty>::summon() {
                let input = [1u32, 2, 3, 4];
                let v = u32x4::from_array(token, input);
                assert_eq!(v.to_array(), input);
            }
        }

        #[test]
        fn test_u32x4_min_max() {
            if let Some(token) = <$token_ty>::summon() {
                let a = u32x4::from_array(token, [1, 5, 3, 7]);
                let b = u32x4::from_array(token, [2, 4, 6, 8]);
                assert_eq!(a.min(b).to_array(), [1, 4, 3, 7]);
                assert_eq!(a.max(b).to_array(), [2, 5, 6, 8]);
            }
        }

        // ========== i64x2 tests ==========

        #[test]
        fn test_i64x2_from_array_to_array() {
            if let Some(token) = <$token_ty>::summon() {
                let input = [1i64, -2];
                let v = i64x2::from_array(token, input);
                assert_eq!(v.to_array(), input);
            }
        }

        #[test]
        fn test_i64x2_add_sub() {
            if let Some(token) = <$token_ty>::summon() {
                let a = i64x2::from_array(token, [100, 200]);
                let b = i64x2::from_array(token, [10, 20]);
                assert_eq!((a + b).to_array(), [110, 220]);
                assert_eq!((a - b).to_array(), [90, 180]);
            }
        }

        // ========== u64x2 tests ==========

        #[test]
        fn test_u64x2_from_array_to_array() {
            if let Some(token) = <$token_ty>::summon() {
                let input = [1u64, 2];
                let v = u64x2::from_array(token, input);
                assert_eq!(v.to_array(), input);
            }
        }

        // ========== i16x8 tests ==========

        #[test]
        fn test_i16x8_from_array_to_array() {
            if let Some(token) = <$token_ty>::summon() {
                let input = [1i16, -2, 3, -4, 5, -6, 7, -8];
                let v = i16x8::from_array(token, input);
                assert_eq!(v.to_array(), input);
            }
        }

        #[test]
        fn test_i16x8_add_sub() {
            if let Some(token) = <$token_ty>::summon() {
                let a = i16x8::from_array(token, [1, 2, 3, 4, 5, 6, 7, 8]);
                let b = i16x8::from_array(token, [1, 1, 1, 1, 1, 1, 1, 1]);
                assert_eq!((a + b).to_array(), [2, 3, 4, 5, 6, 7, 8, 9]);
                assert_eq!((a - b).to_array(), [0, 1, 2, 3, 4, 5, 6, 7]);
            }
        }

        #[test]
        fn test_i16x8_min_max_abs() {
            if let Some(token) = <$token_ty>::summon() {
                let a = i16x8::from_array(token, [1, -5, 3, -7, 2, -4, 6, -8]);
                let b = i16x8::from_array(token, [2, -4, 2, -8, 1, -5, 5, -9]);
                assert_eq!(a.min(b).to_array(), [1, -5, 2, -8, 1, -5, 5, -9]);
                assert_eq!(a.max(b).to_array(), [2, -4, 3, -7, 2, -4, 6, -8]);
                assert_eq!(a.abs().to_array(), [1, 5, 3, 7, 2, 4, 6, 8]);
            }
        }

        // ========== u16x8 tests ==========

        #[test]
        fn test_u16x8_from_array_to_array() {
            if let Some(token) = <$token_ty>::summon() {
                let input = [1u16, 2, 3, 4, 5, 6, 7, 8];
                let v = u16x8::from_array(token, input);
                assert_eq!(v.to_array(), input);
            }
        }

        // ========== i8x16 tests ==========

        #[test]
        fn test_i8x16_from_array_to_array() {
            if let Some(token) = <$token_ty>::summon() {
                let input = [
                    1i8, -2, 3, -4, 5, -6, 7, -8, 9, -10, 11, -12, 13, -14, 15, -16,
                ];
                let v = i8x16::from_array(token, input);
                assert_eq!(v.to_array(), input);
            }
        }

        #[test]
        fn test_i8x16_add_sub() {
            if let Some(token) = <$token_ty>::summon() {
                let a = i8x16::from_array(
                    token,
                    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
                );
                let b = i8x16::splat(token, 1);
                assert_eq!(
                    (a + b).to_array(),
                    [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
                );
                assert_eq!(
                    (a - b).to_array(),
                    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
                );
            }
        }

        // ========== u8x16 tests ==========

        #[test]
        fn test_u8x16_from_array_to_array() {
            if let Some(token) = <$token_ty>::summon() {
                let input = [1u8, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
                let v = u8x16::from_array(token, input);
                assert_eq!(v.to_array(), input);
            }
        }

        // ========== Type conversion tests (f32x4 <-> i32x4) ==========

        #[test]
        fn test_f32x4_to_i32x4() {
            if let Some(token) = <$token_ty>::summon() {
                let f = f32x4::from_array(token, [1.0, 2.5, -3.7, 4.9]);
                // Truncation
                let i = f.to_i32x4();
                assert_eq!(i.to_array(), [1, 2, -3, 4]);
                // Rounding
                let i_round = f.to_i32x4_round();
                assert_eq!(i_round.to_array(), [1, 2, -4, 5]);
            }
        }

        #[test]
        fn test_i32x4_to_f32x4() {
            if let Some(token) = <$token_ty>::summon() {
                let i = i32x4::from_array(token, [1, 2, -3, 4]);
                let f = i.to_f32x4();
                assert_eq!(f.to_array(), [1.0, 2.0, -3.0, 4.0]);
            }
        }

        // ========== Block ops tests ==========

        #[test]
        fn test_f32x4_interleave_deinterleave_4ch() {
            if let Some(token) = <$token_ty>::summon() {
                let r = f32x4::from_array(token, [1.0, 2.0, 3.0, 4.0]);
                let g = f32x4::from_array(token, [10.0, 20.0, 30.0, 40.0]);
                let b = f32x4::from_array(token, [100.0, 200.0, 300.0, 400.0]);
                let a = f32x4::from_array(token, [255.0, 255.0, 255.0, 255.0]);

                // Interleave SoA -> AoS
                let aos = f32x4::interleave_4ch([r, g, b, a]);
                assert_eq!(aos[0].to_array(), [1.0, 10.0, 100.0, 255.0]);
                assert_eq!(aos[1].to_array(), [2.0, 20.0, 200.0, 255.0]);
                assert_eq!(aos[2].to_array(), [3.0, 30.0, 300.0, 255.0]);
                assert_eq!(aos[3].to_array(), [4.0, 40.0, 400.0, 255.0]);

                // Deinterleave back
                let [r2, g2, b2, a2] = f32x4::deinterleave_4ch(aos);
                assert_eq!(r2.to_array(), r.to_array());
                assert_eq!(g2.to_array(), g.to_array());
                assert_eq!(b2.to_array(), b.to_array());
                assert_eq!(a2.to_array(), a.to_array());
            }
        }

        #[test]
        fn test_f32x4_interleave_lo_hi() {
            if let Some(token) = <$token_ty>::summon() {
                let a = f32x4::from_array(token, [1.0, 2.0, 3.0, 4.0]);
                let b = f32x4::from_array(token, [5.0, 6.0, 7.0, 8.0]);

                let lo = a.interleave_lo(b);
                let hi = a.interleave_hi(b);

                assert_eq!(lo.to_array(), [1.0, 5.0, 2.0, 6.0]);
                assert_eq!(hi.to_array(), [3.0, 7.0, 4.0, 8.0]);
            }
        }
    };
}

// ============================================================================
// Architecture-specific modules
// ============================================================================

#[cfg(target_arch = "x86_64")]
mod x86 {
    w128_parity_suite!(archmage::X64V3Token);
}

#[cfg(target_arch = "aarch64")]
mod arm {
    w128_parity_suite!(archmage::NeonToken);
}

#[cfg(target_arch = "wasm32")]
mod wasm {
    w128_parity_suite!(archmage::Wasm128Token);
}
