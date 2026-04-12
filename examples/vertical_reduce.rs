//! Fixed-point vertical image reduction using raw AVX2 intrinsics
//!
//! Demonstrates a production image resampling pattern: reducing N input rows
//! to one output row using 15-bit fixed-point weights. This is the core
//! operation in separable image resizers (vertical pass).
//!
//! Run: `cargo run --example vertical_reduce --release`
//!
//! For the f32 version using magetypes' ergonomic API, see:
//! <https://github.com/imazen/archmage/blob/main/magetypes/examples/convolution.rs>

#[cfg(not(target_arch = "x86_64"))]
fn main() {
    println!("This example requires x86_64 with AVX2");
}

#[cfg(target_arch = "x86_64")]
fn main() {
    x86_impl::run();
}

#[cfg(target_arch = "x86_64")]
mod x86_impl {
    use archmage::{SimdToken, X64V3Token, arcane};

    const SHIFT: i32 = 15;
    const HALF: i32 = 1 << (SHIFT - 1);

    /// Reduce N input rows to 1 output row using fixed-point weighted sum.
    ///
    /// Processes 16 u8 pixels per iteration. Weights are 15-bit fixed-point
    /// (multiply by 32768 to convert from float).
    ///
    /// Formula: out[x] = clamp((sum(input[i][x] * weight[i]) + HALF) >> 15, 0, 255)
    #[arcane(import_intrinsics)]
    pub fn reduce_vertical_u8(
        _token: X64V3Token,
        inputs: &[&[u8]],
        weights: &[i16],
        output: &mut [u8],
    ) {
        debug_assert_eq!(inputs.len(), weights.len());

        let len = output.len();
        let chunks = len / 16;

        for chunk_idx in 0..chunks {
            let base = chunk_idx * 16;

            // Initialize accumulators with rounding bias
            let mut acc_lo = _mm256_set1_epi32(HALF);
            let mut acc_hi = _mm256_set1_epi32(HALF);

            for (input, &w) in inputs.iter().zip(weights.iter()) {
                // Load 16 bytes, extend to i16, then i32
                let chunk: &[u8; 16] = input[base..base + 16].try_into().unwrap();
                let bytes = _mm_loadu_si128(chunk);
                let words = _mm256_cvtepu8_epi16(bytes);

                let lo = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(words));
                let hi = _mm256_cvtepi16_epi32(_mm256_extracti128_si256::<1>(words));
                let w_vec = _mm256_set1_epi32(w as i32);

                acc_lo = _mm256_add_epi32(acc_lo, _mm256_mullo_epi32(lo, w_vec));
                acc_hi = _mm256_add_epi32(acc_hi, _mm256_mullo_epi32(hi, w_vec));
            }

            // Shift, clamp, pack: i32 → i16 → u8
            acc_lo = _mm256_srai_epi32::<SHIFT>(acc_lo);
            acc_hi = _mm256_srai_epi32::<SHIFT>(acc_hi);

            let zero = _mm256_setzero_si256();
            let max_val = _mm256_set1_epi32(255);
            acc_lo = _mm256_min_epi32(_mm256_max_epi32(acc_lo, zero), max_val);
            acc_hi = _mm256_min_epi32(_mm256_max_epi32(acc_hi, zero), max_val);

            let packed16 =
                _mm256_permute4x64_epi64::<0b11011000>(_mm256_packs_epi32(acc_lo, acc_hi));
            let packed8 =
                _mm256_permute4x64_epi64::<0b11011000>(_mm256_packus_epi16(packed16, packed16));

            let out_chunk: &mut [u8; 16] = (&mut output[base..base + 16]).try_into().unwrap();
            _mm_storeu_si128(out_chunk, _mm256_castsi256_si128(packed8));
        }

        // Scalar remainder
        for x in (chunks * 16)..len {
            let mut sum = HALF;
            for (input, &w) in inputs.iter().zip(weights.iter()) {
                sum += input[x] as i32 * w as i32;
            }
            output[x] = (sum >> SHIFT).clamp(0, 255) as u8;
        }
    }

    pub fn run() {
        let Some(token) = X64V3Token::summon() else {
            println!("AVX2+FMA not available");
            return;
        };

        let width = 64;

        // 3 rows of gradient data
        let row0: Vec<u8> = (0..width).map(|i| (i * 4).min(255) as u8).collect();
        let row1: Vec<u8> = (0..width).map(|_| 128u8).collect();
        let row2: Vec<u8> = (0..width).map(|i| (255 - i * 4).max(0) as u8).collect();
        let inputs: Vec<&[u8]> = vec![&row0, &row1, &row2];

        // Equal weights for box filter: 1/3 ≈ 10923/32768
        let weights = [10923i16, 10923, 10923];
        let mut output = vec![0u8; width];

        reduce_vertical_u8(token, &inputs, &weights, &mut output);

        println!("Row 0 (gradient up):   {:?}", &row0[..16]);
        println!("Row 1 (constant 128):  {:?}", &row1[..16]);
        println!("Row 2 (gradient down): {:?}", &row2[..16]);
        println!("Averaged output:       {:?}", &output[..16]);
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn test_uniform_input() {
            let Some(token) = X64V3Token::summon() else {
                return;
            };
            let row: Vec<u8> = vec![100; 32];
            let inputs: Vec<&[u8]> = vec![&row, &row, &row];
            let weights = [10923i16, 10923, 10923]; // ~1/3 each
            let mut output = vec![0u8; 32];
            reduce_vertical_u8(token, &inputs, &weights, &mut output);
            for &v in &output {
                assert!((v as i32 - 100).abs() <= 1, "expected ~100, got {v}");
            }
        }

        #[test]
        fn test_identity_weight() {
            let Some(token) = X64V3Token::summon() else {
                return;
            };
            let row: Vec<u8> = (0..32).map(|i| (i * 8) as u8).collect();
            let zero: Vec<u8> = vec![0; 32];
            let inputs: Vec<&[u8]> = vec![&row, &zero, &zero];
            let weights = [32767i16, 0, 0]; // ~1.0, 0, 0
            let mut output = vec![0u8; 32];
            reduce_vertical_u8(token, &inputs, &weights, &mut output);
            for (i, (&expected, &actual)) in row.iter().zip(output.iter()).enumerate() {
                assert!(
                    (expected as i32 - actual as i32).abs() <= 1,
                    "idx {i}: expected {expected}, got {actual}"
                );
            }
        }
    }
}
