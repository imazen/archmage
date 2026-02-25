//! Tests for `partition_slice()` and `partition_slice_mut()` on magetypes SIMD types.

use archmage::SimdToken;

#[cfg(target_arch = "x86_64")]
mod x86_tests {
    use super::*;
    use magetypes::simd::f32x8;

    #[test]
    fn partition_exact_multiple() {
        if let Some(token) = archmage::X64V3Token::summon() {
            let data: Vec<f32> = (0..24).map(|i| i as f32).collect();
            let (chunks, remainder) = f32x8::partition_slice(token, &data);
            assert_eq!(chunks.len(), 3); // 24 / 8 = 3
            assert_eq!(remainder.len(), 0);
            assert_eq!(chunks[0], [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]);
            assert_eq!(chunks[1], [8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0]);
            assert_eq!(chunks[2], [16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0]);
        }
    }

    #[test]
    fn partition_with_remainder() {
        if let Some(token) = archmage::X64V3Token::summon() {
            let data: Vec<f32> = (0..19).map(|i| i as f32).collect();
            let (chunks, remainder) = f32x8::partition_slice(token, &data);
            assert_eq!(chunks.len(), 2); // 19 / 8 = 2
            assert_eq!(remainder.len(), 3); // 19 % 8 = 3
            assert_eq!(chunks[0], [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]);
            assert_eq!(chunks[1], [8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0]);
            assert_eq!(remainder, &[16.0, 17.0, 18.0]);
        }
    }

    #[test]
    fn partition_smaller_than_lanes() {
        if let Some(token) = archmage::X64V3Token::summon() {
            let data = [1.0f32, 2.0, 3.0];
            let (chunks, remainder) = f32x8::partition_slice(token, &data);
            assert_eq!(chunks.len(), 0);
            assert_eq!(remainder.len(), 3);
            assert_eq!(remainder, &[1.0, 2.0, 3.0]);
        }
    }

    #[test]
    fn partition_empty() {
        if let Some(token) = archmage::X64V3Token::summon() {
            let data: &[f32] = &[];
            let (chunks, remainder) = f32x8::partition_slice(token, data);
            assert_eq!(chunks.len(), 0);
            assert_eq!(remainder.len(), 0);
        }
    }

    #[test]
    fn partition_mut_modifies_in_place() {
        if let Some(token) = archmage::X64V3Token::summon() {
            let mut data: Vec<f32> = (0..20).map(|i| i as f32).collect();
            let (chunks, remainder) = f32x8::partition_slice_mut(token, &mut data);

            // Modify first chunk
            for x in chunks[0].iter_mut() {
                *x *= 2.0;
            }
            // Modify remainder
            for x in remainder.iter_mut() {
                *x += 100.0;
            }

            // Verify modifications applied to original data
            assert_eq!(data[0], 0.0); // 0 * 2
            assert_eq!(data[1], 2.0); // 1 * 2
            assert_eq!(data[7], 14.0); // 7 * 2
            assert_eq!(data[8], 8.0); // second chunk, untouched
            assert_eq!(data[16], 116.0); // remainder: 16 + 100
            assert_eq!(data[19], 119.0); // remainder: 19 + 100
        }
    }

    #[test]
    fn partition_chunks_loadable() {
        if let Some(token) = archmage::X64V3Token::summon() {
            let data: Vec<f32> = (0..16).map(|i| i as f32).collect();
            let (chunks, _) = f32x8::partition_slice(token, &data);

            // Each chunk can be passed directly to f32x8::load
            let v = f32x8::load(token, &chunks[0]);
            let arr = v.to_array();
            assert_eq!(arr, [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]);
        }
    }

    #[test]
    fn partition_i32x8() {
        if let Some(token) = archmage::X64V3Token::summon() {
            use magetypes::simd::i32x8;
            let data: Vec<i32> = (0..20).collect();
            let (chunks, remainder) = i32x8::partition_slice(token, &data);
            assert_eq!(chunks.len(), 2);
            assert_eq!(remainder.len(), 4);
            assert_eq!(chunks[0], [0, 1, 2, 3, 4, 5, 6, 7]);
        }
    }

    #[test]
    fn partition_f32x4() {
        if let Some(token) = archmage::X64V3Token::summon() {
            use magetypes::simd::f32x4;
            let data: Vec<f32> = (0..10).map(|i| i as f32).collect();
            let (chunks, remainder) = f32x4::partition_slice(token, &data);
            assert_eq!(chunks.len(), 2); // 10 / 4 = 2
            assert_eq!(remainder.len(), 2); // 10 % 4 = 2
            assert_eq!(chunks[0], [0.0, 1.0, 2.0, 3.0]);
            assert_eq!(chunks[1], [4.0, 5.0, 6.0, 7.0]);
            assert_eq!(remainder, &[8.0, 9.0]);
        }
    }
}
