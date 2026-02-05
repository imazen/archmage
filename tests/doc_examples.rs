//! Test that doc examples from SAFETY-AND-IDIOMS.md compile.
//!
//! This file verifies that the examples in the documentation are syntactically
//! correct and type-check. Not all examples are runnable (some are fragments).

#![allow(dead_code)]
#![allow(unused_variables)]
#![allow(unused_imports)]

use archmage::prelude::*;
use magetypes::simd::f32x8;

// ============================================================================
// Pattern 1: #[rite] Inside, #[arcane] at the Boundary
// ============================================================================

mod pattern1 {
    use super::*;

    pub fn public_api(data: &[f32]) -> f32 {
        if let Some(token) = Desktop64::summon() {
            process_simd(token, data)
        } else {
            data.iter().sum()
        }
    }

    #[arcane]
    fn process_simd(token: Desktop64, data: &[f32]) -> f32 {
        let mut sum = 0.0;
        for chunk in data.chunks_exact(8) {
            sum += process_chunk(token, chunk.try_into().unwrap());
        }
        sum
    }

    #[rite]
    fn process_chunk(token: Desktop64, chunk: &[f32; 8]) -> f32 {
        let v = f32x8::from_array(token, *chunk);
        v.reduce_add()
    }

    #[test]
    fn test_pattern1() {
        let data = [1.0f32; 16];
        let result = public_api(&data);
        // Should be 16.0 (sum of 16 ones)
        assert!((result - 16.0).abs() < 0.001);
    }
}

// ============================================================================
// Pattern 2: Summon Once, Pass Everywhere (correct version)
// ============================================================================

mod pattern2_correct {
    use super::*;

    fn process_all(pairs: &[([f32; 8], [f32; 8])]) -> f32 {
        if let Some(token) = Desktop64::summon() {
            process_all_simd(token, pairs)
        } else {
            process_all_scalar(pairs)
        }
    }

    #[arcane]
    fn process_all_simd(token: Desktop64, pairs: &[([f32; 8], [f32; 8])]) -> f32 {
        pairs
            .iter()
            .map(|(a, b)| process_pair_simd(token, a, b))
            .sum()
    }

    #[rite]
    fn process_pair_simd(token: Desktop64, a: &[f32; 8], b: &[f32; 8]) -> f32 {
        let va = f32x8::from_array(token, *a);
        let vb = f32x8::from_array(token, *b);
        (va * vb).reduce_add()
    }

    fn process_all_scalar(pairs: &[([f32; 8], [f32; 8])]) -> f32 {
        pairs
            .iter()
            .map(|(a, b)| a.iter().zip(b.iter()).map(|(x, y)| x * y).sum::<f32>())
            .sum()
    }

    #[test]
    fn test_pattern2() {
        let pairs = vec![([1.0f32; 8], [2.0f32; 8]), ([3.0f32; 8], [4.0f32; 8])];
        let result = process_all(&pairs);
        // First pair: 8 * (1*2) = 16
        // Second pair: 8 * (3*4) = 96
        // Total: 112
        assert!((result - 112.0).abs() < 0.001);
    }
}

// ============================================================================
// Pattern 4: Memory Operations via safe_unaligned_simd
// ============================================================================

mod pattern4 {
    use super::*;
    // Explicit import of safe version (shadows core::arch)
    use safe_unaligned_simd::x86_64::_mm256_loadu_ps;

    #[arcane]
    fn load_and_square_intrinsics(token: Desktop64, data: &[f32; 8]) -> __m256 {
        let v = _mm256_loadu_ps(data);
        _mm256_mul_ps(v, v)
    }

    #[arcane]
    fn load_and_square_magetypes(token: Desktop64, data: &[f32; 8]) -> f32x8 {
        let v = f32x8::from_array(token, *data);
        v * v
    }

    #[test]
    fn test_pattern4() {
        if let Some(token) = Desktop64::summon() {
            let data = [2.0f32; 8];
            let result = load_and_square_magetypes(token, &data);
            let arr = result.to_array();
            for &v in &arr {
                assert!((v - 4.0).abs() < 0.001);
            }
        }
    }
}

// ============================================================================
// Mistake 5: Forgetting Token (correct version)
// ============================================================================

mod mistake5_correct {
    use super::*;

    pub fn api(data: &[f32]) -> f32 {
        if let Some(token) = Desktop64::summon() {
            process_simd(token, data)
        } else {
            process_scalar(data)
        }
    }

    #[arcane]
    fn process_simd(token: Desktop64, data: &[f32]) -> f32 {
        let mut sum = f32x8::zero(token);
        for chunk in data.chunks_exact(8) {
            let v = f32x8::from_array(token, chunk.try_into().unwrap());
            sum = sum + v;
        }
        sum.reduce_add()
    }

    fn process_scalar(data: &[f32]) -> f32 {
        data.iter().sum()
    }

    #[test]
    fn test_api() {
        let data = [1.0f32; 24];
        let result = api(&data);
        assert!((result - 24.0).abs() < 0.001);
    }
}

// ============================================================================
// Explicit Types, Explicit Dispatch (banned prelude aliases)
// ============================================================================

mod explicit_dispatch {
    use super::*;
    use magetypes::simd::f32x4;

    pub fn process(data: &[f32]) -> f32 {
        if let Some(token) = Desktop64::summon() {
            process_avx2(token, data)
        } else if let Some(token) = Arm64::summon() {
            process_neon(token, data)
        } else {
            process_scalar(data)
        }
    }

    #[arcane]
    fn process_avx2(token: Desktop64, data: &[f32]) -> f32 {
        let mut sum = f32x8::zero(token);
        for chunk in data.chunks_exact(8) {
            let v = f32x8::from_array(token, chunk.try_into().unwrap());
            sum = sum + v;
        }
        sum.reduce_add() + data.chunks_exact(8).remainder().iter().sum::<f32>()
    }

    #[arcane]
    fn process_neon(token: Arm64, data: &[f32]) -> f32 {
        let mut sum = f32x4::zero(token);
        for chunk in data.chunks_exact(4) {
            let v = f32x4::from_array(token, chunk.try_into().unwrap());
            sum = sum + v;
        }
        sum.reduce_add() + data.chunks_exact(4).remainder().iter().sum::<f32>()
    }

    fn process_scalar(data: &[f32]) -> f32 {
        data.iter().sum()
    }

    #[test]
    fn test_explicit_dispatch() {
        let data = [1.0f32; 17];
        let result = process(&data);
        assert!((result - 17.0).abs() < 0.001);
    }
}

// ============================================================================
// Cross-Architecture: x86 kernel compiles on all platforms
// ============================================================================

mod cross_arch {
    use super::*;

    #[arcane]
    fn x86_kernel(token: X64V3Token, data: &[f32; 8]) -> f32 {
        let v = f32x8::from_array(token, *data);
        v.reduce_add()
    }

    pub fn dispatch(data: &[f32; 8]) -> f32 {
        if let Some(token) = X64V3Token::summon() {
            x86_kernel(token, data)
        } else {
            data.iter().sum()
        }
    }

    #[test]
    fn test_cross_arch() {
        let data = [1.0f32; 8];
        let result = dispatch(&data);
        assert!((result - 8.0).abs() < 0.001);
    }
}

// ============================================================================
// Nontrivial Algorithm: RMS Normalization
// ============================================================================

mod rms_normalize {
    use super::*;

    pub fn rms_normalize(samples: &mut [f32]) {
        if samples.is_empty() {
            return;
        }

        let sum_sq: f32 = if let Some(token) = Desktop64::summon() {
            sum_squares_avx2(token, samples)
        } else {
            samples.iter().map(|x| x * x).sum()
        };

        let rms = (sum_sq / samples.len() as f32).sqrt();
        if rms < 1e-10 {
            return;
        }
        let scale = 1.0 / rms;

        if let Some(token) = Desktop64::summon() {
            scale_avx2(token, samples, scale);
        } else {
            samples.iter_mut().for_each(|x| *x *= scale);
        }
    }

    #[arcane]
    fn sum_squares_avx2(token: Desktop64, data: &[f32]) -> f32 {
        let mut acc = f32x8::zero(token);

        for chunk in data.chunks_exact(8) {
            let v = f32x8::from_array(token, chunk.try_into().unwrap());
            acc = v.mul_add(v, acc);
        }

        let mut sum = acc.reduce_add();
        for &x in data.chunks_exact(8).remainder() {
            sum += x * x;
        }
        sum
    }

    #[arcane]
    fn scale_avx2(token: Desktop64, data: &mut [f32], scale: f32) {
        let s = f32x8::splat(token, scale);

        let chunks = data.len() / 8;
        for i in 0..chunks {
            let chunk = &mut data[i * 8..(i + 1) * 8];
            let v = f32x8::from_array(token, chunk.as_ref().try_into().unwrap());
            (v * s).store(chunk.try_into().unwrap());
        }
        for x in data.chunks_exact_mut(8).into_remainder() {
            *x *= scale;
        }
    }

    #[test]
    fn test_rms_normalize() {
        let mut samples = vec![3.0f32; 16];
        rms_normalize(&mut samples);
        // RMS of [3,3,3,...] is 3, so normalized should be [1,1,1,...]
        for &s in &samples {
            assert!((s - 1.0).abs() < 0.001, "Expected 1.0, got {}", s);
        }
    }
}

// ============================================================================
// Nontrivial Algorithm: Softmax
// ============================================================================

mod softmax {
    use super::*;

    pub fn softmax(logits: &mut [f32]) {
        if logits.is_empty() {
            return;
        }

        if let Some(token) = Desktop64::summon() {
            softmax_avx2(token, logits);
        } else {
            softmax_scalar(logits);
        }
    }

    #[arcane]
    fn softmax_avx2(token: Desktop64, data: &mut [f32]) {
        // Find max
        let max_val = reduce_max(token, data);

        // exp(x - max) and sum
        let max_v = f32x8::splat(token, max_val);
        let mut sum_vec = f32x8::zero(token);

        let chunks = data.len() / 8;
        for i in 0..chunks {
            let chunk = &mut data[i * 8..(i + 1) * 8];
            let v = f32x8::from_array(token, chunk.as_ref().try_into().unwrap());
            let e = (v - max_v).exp_lowp();
            e.store(chunk.try_into().unwrap());
            sum_vec = sum_vec + e;
        }

        let mut sum = sum_vec.reduce_add();
        for x in data.chunks_exact_mut(8).into_remainder() {
            *x = (*x - max_val).exp();
            sum += *x;
        }

        // Normalize
        let inv = f32x8::splat(token, 1.0 / sum);
        for i in 0..chunks {
            let chunk = &mut data[i * 8..(i + 1) * 8];
            let v = f32x8::from_array(token, chunk.as_ref().try_into().unwrap());
            (v * inv).store(chunk.try_into().unwrap());
        }
        for x in data.chunks_exact_mut(8).into_remainder() {
            *x /= sum;
        }
    }

    #[rite]
    fn reduce_max(token: Desktop64, data: &[f32]) -> f32 {
        let mut max_vec = f32x8::splat(token, f32::NEG_INFINITY);

        for chunk in data.chunks_exact(8) {
            let v = f32x8::from_array(token, chunk.try_into().unwrap());
            max_vec = max_vec.max(v);
        }

        let mut max_val = max_vec.reduce_max();
        for &x in data.chunks_exact(8).remainder() {
            max_val = max_val.max(x);
        }
        max_val
    }

    fn softmax_scalar(data: &mut [f32]) {
        let max_val = data.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0;
        for x in data.iter_mut() {
            *x = (*x - max_val).exp();
            sum += *x;
        }
        for x in data.iter_mut() {
            *x /= sum;
        }
    }

    #[test]
    fn test_softmax() {
        let mut logits = vec![1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0];
        softmax(&mut logits);

        // Check probabilities sum to 1
        let sum: f32 = logits.iter().sum();
        assert!(
            (sum - 1.0).abs() < 0.01,
            "Softmax should sum to 1, got {}",
            sum
        );

        // Check monotonicity (higher input = higher probability)
        assert!(logits[3] > logits[2]);
        assert!(logits[2] > logits[1]);
        assert!(logits[1] > logits[0]);
    }
}

// ============================================================================
// Nontrivial Algorithm: Dot Product with explicit dispatch
// ============================================================================

mod dot_product {
    use super::*;
    use magetypes::simd::f32x4;

    pub fn dot(a: &[f32], b: &[f32]) -> f32 {
        assert_eq!(a.len(), b.len());

        if let Some(token) = Desktop64::summon() {
            dot_avx2(token, a, b)
        } else if let Some(token) = Arm64::summon() {
            dot_neon(token, a, b)
        } else {
            dot_scalar(a, b)
        }
    }

    #[arcane]
    fn dot_avx2(token: Desktop64, a: &[f32], b: &[f32]) -> f32 {
        let mut acc = f32x8::zero(token);

        for (a_chunk, b_chunk) in a.chunks_exact(8).zip(b.chunks_exact(8)) {
            let va = f32x8::from_array(token, a_chunk.try_into().unwrap());
            let vb = f32x8::from_array(token, b_chunk.try_into().unwrap());
            acc = va.mul_add(vb, acc);
        }

        let mut sum = acc.reduce_add();
        for (x, y) in a
            .chunks_exact(8)
            .remainder()
            .iter()
            .zip(b.chunks_exact(8).remainder())
        {
            sum += x * y;
        }
        sum
    }

    #[arcane]
    fn dot_neon(token: Arm64, a: &[f32], b: &[f32]) -> f32 {
        let mut acc = f32x4::zero(token);

        for (a_chunk, b_chunk) in a.chunks_exact(4).zip(b.chunks_exact(4)) {
            let va = f32x4::from_array(token, a_chunk.try_into().unwrap());
            let vb = f32x4::from_array(token, b_chunk.try_into().unwrap());
            acc = va.mul_add(vb, acc);
        }

        let mut sum = acc.reduce_add();
        for (x, y) in a
            .chunks_exact(4)
            .remainder()
            .iter()
            .zip(b.chunks_exact(4).remainder())
        {
            sum += x * y;
        }
        sum
    }

    fn dot_scalar(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b).map(|(x, y)| x * y).sum()
    }

    #[test]
    fn test_dot() {
        let a = vec![1.0f32; 100];
        let b = vec![2.0f32; 100];
        let result = dot(&a, &b);
        // 100 * (1 * 2) = 200
        assert!((result - 200.0).abs() < 0.001);
    }
}
