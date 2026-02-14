//! Magetypes Showcase: Boilerplate Elimination
//!
//! Demonstrates how magetypes reduces SIMD code from verbose intrinsic calls
//! to clean, readable Rust with operators and methods.
//!
//! Run with: cargo run --example magetypes_showcase

#![allow(dead_code)]

// This example is x86_64-only (uses AVX2 intrinsics + magetypes).
// On other architectures it compiles but does nothing.

#[cfg(not(target_arch = "x86_64"))]
fn main() {
    eprintln!("This example requires x86_64.");
}

#[cfg(target_arch = "x86_64")]
use archmage::prelude::*;
#[cfg(target_arch = "x86_64")]
use magetypes::simd::{f32x8, i32x8};
#[cfg(target_arch = "x86_64")]
use safe_unaligned_simd::x86_64::{_mm256_loadu_ps, _mm256_storeu_ps};

// ============================================================================
// COMPARISON: Raw Intrinsics vs Magetypes
// ============================================================================

#[cfg(target_arch = "x86_64")]
mod comparison {
    use super::*;

    /// Raw intrinsics: verbose, error-prone, hard to read
    #[arcane]
    pub fn dot_product_raw(_token: Desktop64, a: &[f32; 8], b: &[f32; 8]) -> f32 {
        let va = _mm256_loadu_ps(a);
        let vb = _mm256_loadu_ps(b);
        let prod = _mm256_mul_ps(va, vb);

        // Horizontal sum - the painful part
        let sum1 = _mm256_hadd_ps(prod, prod);
        let sum2 = _mm256_hadd_ps(sum1, sum1);
        let low = _mm256_castps256_ps128(sum2);
        let high = _mm256_extractf128_ps::<1>(sum2);
        let result = _mm_add_ss(low, high);
        _mm_cvtss_f32(result)
    }

    /// Magetypes: clean, readable, same codegen
    #[arcane]
    pub fn dot_product_clean(token: Desktop64, a: &[f32; 8], b: &[f32; 8]) -> f32 {
        let va = f32x8::from_array(token, *a);
        let vb = f32x8::from_array(token, *b);
        (va * vb).reduce_add()
    }

    /// FMA with raw intrinsics
    #[arcane]
    pub fn fma_raw(_token: Desktop64, a: &[f32; 8], b: &[f32; 8], c: &[f32; 8]) -> [f32; 8] {
        let va = _mm256_loadu_ps(a);
        let vb = _mm256_loadu_ps(b);
        let vc = _mm256_loadu_ps(c);
        let result = _mm256_fmadd_ps(va, vb, vc);
        let mut out = [0.0f32; 8];
        _mm256_storeu_ps(&mut out, result);
        out
    }

    /// FMA with magetypes: a * b + c
    #[arcane]
    pub fn fma_clean(token: Desktop64, a: &[f32; 8], b: &[f32; 8], c: &[f32; 8]) -> [f32; 8] {
        let va = f32x8::from_array(token, *a);
        let vb = f32x8::from_array(token, *b);
        let vc = f32x8::from_array(token, *c);
        va.mul_add(vb, vc).to_array()
    }
}

// ============================================================================
// OPERATORS: +, -, *, / just work
// ============================================================================

#[cfg(target_arch = "x86_64")]
mod operators {
    use super::*;

    #[arcane]
    pub fn vector_math(token: Desktop64, a: &[f32; 8], b: &[f32; 8]) -> [f32; 8] {
        let va = f32x8::from_array(token, *a);
        let vb = f32x8::from_array(token, *b);

        // Natural operators - no _mm256_add_ps, _mm256_mul_ps, etc.
        let sum = va + vb;
        let diff = va - vb;
        let prod = va * vb;
        let quot = va / vb;

        // Chain them naturally
        let result = (sum * diff + prod) / quot;
        result.to_array()
    }

    #[arcane]
    pub fn integer_ops(token: Desktop64, a: &[i32; 8], b: &[i32; 8]) -> [i32; 8] {
        let va = i32x8::from_array(token, *a);
        let vb = i32x8::from_array(token, *b);

        // Integer operators
        let sum = va + vb;
        let diff = va - vb;
        let prod = va * vb;

        (sum + diff + prod).to_array()
    }
}

// ============================================================================
// METHODS: Readable names for complex operations
// ============================================================================

#[cfg(target_arch = "x86_64")]
mod methods {
    use super::*;

    #[arcane]
    pub fn statistics(token: Desktop64, data: &[f32; 8]) -> (f32, f32, f32) {
        let v = f32x8::from_array(token, *data);

        let sum = v.reduce_add();
        let min = v.reduce_min();
        let max = v.reduce_max();

        (sum, min, max)
    }

    #[arcane]
    pub fn clamped_normalize(token: Desktop64, data: &[f32; 8], lo: f32, hi: f32) -> [f32; 8] {
        let v = f32x8::from_array(token, *data);
        let lo_v = f32x8::splat(token, lo);
        let hi_v = f32x8::splat(token, hi);

        // Clamp then normalize to [0, 1]
        let clamped = v.max(lo_v).min(hi_v);
        let range = hi - lo;
        let normalized = (clamped - lo_v) / f32x8::splat(token, range);

        normalized.to_array()
    }

    #[arcane]
    pub fn abs_values(token: Desktop64, data: &[f32; 8]) -> [f32; 8] {
        let v = f32x8::from_array(token, *data);
        v.abs().to_array()
    }

    #[arcane]
    pub fn sqrt_and_reciprocals(
        token: Desktop64,
        data: &[f32; 8],
    ) -> ([f32; 8], [f32; 8], [f32; 8]) {
        let v = f32x8::from_array(token, *data);

        let sqrt = v.sqrt();
        let rsqrt = v.rsqrt_approx(); // 1/sqrt(x), fast approximation
        let rcp = v.rcp_approx(); // 1/x, fast approximation

        (sqrt.to_array(), rsqrt.to_array(), rcp.to_array())
    }

    #[arcane]
    pub fn floor_ceil_round(token: Desktop64, data: &[f32; 8]) -> ([f32; 8], [f32; 8], [f32; 8]) {
        let v = f32x8::from_array(token, *data);

        (
            v.floor().to_array(),
            v.ceil().to_array(),
            v.round().to_array(),
        )
    }
}

// ============================================================================
// TRANSCENDENTALS: exp, log, pow
// ============================================================================

#[cfg(target_arch = "x86_64")]
mod transcendentals {
    use super::*;

    /// Softmax activation function (one chunk)
    #[arcane]
    pub fn softmax_chunk(token: Desktop64, logits: &[f32; 8], max_val: f32) -> ([f32; 8], f32) {
        let v = f32x8::from_array(token, *logits);
        let max_v = f32x8::splat(token, max_val);

        // exp(x - max) for numerical stability
        let shifted = v - max_v;
        let exp_vals = shifted.exp_lowp();

        let sum = exp_vals.reduce_add();
        (exp_vals.to_array(), sum)
    }

    /// Power function for gamma correction
    #[arcane]
    pub fn gamma_correction(token: Desktop64, pixels: &[f32; 8], gamma: f32) -> [f32; 8] {
        let v = f32x8::from_array(token, *pixels);
        v.pow_lowp(gamma).to_array()
    }

    /// Log-sum-exp (numerically stable)
    #[arcane]
    pub fn log_sum_exp(token: Desktop64, data: &[f32; 8]) -> f32 {
        let v = f32x8::from_array(token, *data);

        let max_val = v.reduce_max();
        let max_v = f32x8::splat(token, max_val);

        let shifted = v - max_v;
        let exp_sum = shifted.exp_lowp().reduce_add();

        max_val + exp_sum.ln()
    }

    /// Natural log
    #[arcane]
    pub fn log_values(token: Desktop64, data: &[f32; 8]) -> [f32; 8] {
        let v = f32x8::from_array(token, *data);
        v.ln_lowp().to_array()
    }

    /// Base-2 logarithm
    #[arcane]
    pub fn log2_values(token: Desktop64, data: &[f32; 8]) -> [f32; 8] {
        let v = f32x8::from_array(token, *data);
        v.log2_lowp().to_array()
    }
}

// ============================================================================
// COMPARISONS AND BLENDING: Conditional operations without branches
// ============================================================================

#[cfg(target_arch = "x86_64")]
mod conditionals {
    use super::*;

    /// ReLU activation: max(0, x)
    #[arcane]
    pub fn relu(token: Desktop64, x: &[f32; 8]) -> [f32; 8] {
        let v = f32x8::from_array(token, *x);
        let zero = f32x8::zero(token);
        v.max(zero).to_array()
    }

    /// Leaky ReLU: x if x > 0 else alpha * x
    #[arcane]
    pub fn leaky_relu(token: Desktop64, x: &[f32; 8], alpha: f32) -> [f32; 8] {
        let v = f32x8::from_array(token, *x);
        let zero = f32x8::zero(token);
        let alpha_v = f32x8::splat(token, alpha);

        // mask where x > 0
        let mask = v.simd_gt(zero);
        let scaled = v * alpha_v;

        // blend: where mask is true use v, else use scaled
        f32x8::blend(mask, v, scaled).to_array()
    }

    /// Threshold: 1.0 if x > threshold else 0.0
    #[arcane]
    pub fn threshold(token: Desktop64, x: &[f32; 8], thresh: f32) -> [f32; 8] {
        let v = f32x8::from_array(token, *x);
        let thresh_v = f32x8::splat(token, thresh);
        let one = f32x8::splat(token, 1.0);
        let zero = f32x8::zero(token);

        let mask = v.simd_gt(thresh_v);
        f32x8::blend(mask, one, zero).to_array()
    }

    /// Clamp to range using the dedicated method
    #[arcane]
    pub fn clamp(token: Desktop64, x: &[f32; 8], lo: f32, hi: f32) -> [f32; 8] {
        let v = f32x8::from_array(token, *x);
        let lo_v = f32x8::splat(token, lo);
        let hi_v = f32x8::splat(token, hi);
        v.clamp(lo_v, hi_v).to_array()
    }
}

// ============================================================================
// REAL ALGORITHM: Batch Normalization
// ============================================================================

#[cfg(target_arch = "x86_64")]
mod batch_norm {
    use super::*;

    /// Batch normalization: (x - mean) / sqrt(var + eps) * gamma + beta
    pub fn batch_norm(data: &mut [f32], mean: f32, var: f32, gamma: f32, beta: f32, eps: f32) {
        if let Some(token) = Desktop64::summon() {
            batch_norm_avx2(token, data, mean, var, gamma, beta, eps);
        } else {
            batch_norm_scalar(data, mean, var, gamma, beta, eps);
        }
    }

    #[arcane]
    fn batch_norm_avx2(
        token: Desktop64,
        data: &mut [f32],
        mean: f32,
        var: f32,
        gamma: f32,
        beta: f32,
        eps: f32,
    ) {
        let mean_v = f32x8::splat(token, mean);
        let inv_std = f32x8::splat(token, gamma / (var + eps).sqrt());
        let beta_v = f32x8::splat(token, beta);

        for chunk in data.chunks_exact_mut(8) {
            let x = f32x8::from_array(token, chunk.as_ref().try_into().unwrap());
            // (x - mean) * inv_std + beta
            let normalized = (x - mean_v).mul_add(inv_std, beta_v);
            normalized.store(chunk.try_into().unwrap());
        }

        let inv_std_scalar = gamma / (var + eps).sqrt();
        for x in data.chunks_exact_mut(8).into_remainder() {
            *x = (*x - mean) * inv_std_scalar + beta;
        }
    }

    fn batch_norm_scalar(data: &mut [f32], mean: f32, var: f32, gamma: f32, beta: f32, eps: f32) {
        let inv_std = gamma / (var + eps).sqrt();
        for x in data {
            *x = (*x - mean) * inv_std + beta;
        }
    }
}

// ============================================================================
// REAL ALGORITHM: Layer Normalization
// ============================================================================

#[cfg(target_arch = "x86_64")]
mod layer_norm {
    use super::*;

    pub fn layer_norm(data: &mut [f32], gamma: f32, beta: f32, eps: f32) {
        if let Some(token) = Desktop64::summon() {
            layer_norm_avx2(token, data, gamma, beta, eps);
        } else {
            layer_norm_scalar(data, gamma, beta, eps);
        }
    }

    #[arcane]
    fn layer_norm_avx2(token: Desktop64, data: &mut [f32], gamma: f32, beta: f32, eps: f32) {
        // Pass 1: Compute mean
        let mean = compute_mean(token, data);

        // Pass 2: Compute variance
        let var = compute_variance(token, data, mean);

        // Pass 3: Normalize
        let mean_v = f32x8::splat(token, mean);
        let inv_std = f32x8::splat(token, gamma / (var + eps).sqrt());
        let beta_v = f32x8::splat(token, beta);

        for chunk in data.chunks_exact_mut(8) {
            let x = f32x8::from_array(token, chunk.as_ref().try_into().unwrap());
            let normalized = (x - mean_v).mul_add(inv_std, beta_v);
            normalized.store(chunk.try_into().unwrap());
        }

        let inv_std_scalar = gamma / (var + eps).sqrt();
        for x in data.chunks_exact_mut(8).into_remainder() {
            *x = (*x - mean) * inv_std_scalar + beta;
        }
    }

    #[rite]
    fn compute_mean(token: Desktop64, data: &[f32]) -> f32 {
        let mut sum = f32x8::zero(token);
        for chunk in data.chunks_exact(8) {
            let v = f32x8::from_array(token, chunk.try_into().unwrap());
            sum += v;
        }
        let mut total = sum.reduce_add();
        for &x in data.chunks_exact(8).remainder() {
            total += x;
        }
        total / data.len() as f32
    }

    #[rite]
    fn compute_variance(token: Desktop64, data: &[f32], mean: f32) -> f32 {
        let mean_v = f32x8::splat(token, mean);
        let mut sum_sq = f32x8::zero(token);

        for chunk in data.chunks_exact(8) {
            let v = f32x8::from_array(token, chunk.try_into().unwrap());
            let diff = v - mean_v;
            sum_sq = diff.mul_add(diff, sum_sq);
        }

        let mut total = sum_sq.reduce_add();
        for &x in data.chunks_exact(8).remainder() {
            let diff = x - mean;
            total += diff * diff;
        }
        total / data.len() as f32
    }

    fn layer_norm_scalar(data: &mut [f32], gamma: f32, beta: f32, eps: f32) {
        let n = data.len() as f32;
        let mean: f32 = data.iter().sum::<f32>() / n;
        let var: f32 = data.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / n;
        let inv_std = gamma / (var + eps).sqrt();
        for x in data {
            *x = (*x - mean) * inv_std + beta;
        }
    }
}

// ============================================================================
// REAL ALGORITHM: Cosine Similarity
// ============================================================================

#[cfg(target_arch = "x86_64")]
mod cosine_similarity {
    use super::*;

    pub fn cosine_sim(a: &[f32], b: &[f32]) -> f32 {
        assert_eq!(a.len(), b.len());

        if let Some(token) = Desktop64::summon() {
            cosine_sim_avx2(token, a, b)
        } else {
            cosine_sim_scalar(a, b)
        }
    }

    #[arcane]
    fn cosine_sim_avx2(token: Desktop64, a: &[f32], b: &[f32]) -> f32 {
        let mut dot = f32x8::zero(token);
        let mut norm_a = f32x8::zero(token);
        let mut norm_b = f32x8::zero(token);

        for (a_chunk, b_chunk) in a.chunks_exact(8).zip(b.chunks_exact(8)) {
            let va = f32x8::from_array(token, a_chunk.try_into().unwrap());
            let vb = f32x8::from_array(token, b_chunk.try_into().unwrap());

            dot = va.mul_add(vb, dot);
            norm_a = va.mul_add(va, norm_a);
            norm_b = vb.mul_add(vb, norm_b);
        }

        let mut dot_sum = dot.reduce_add();
        let mut norm_a_sum = norm_a.reduce_add();
        let mut norm_b_sum = norm_b.reduce_add();

        for (&x, &y) in a
            .chunks_exact(8)
            .remainder()
            .iter()
            .zip(b.chunks_exact(8).remainder())
        {
            dot_sum += x * y;
            norm_a_sum += x * x;
            norm_b_sum += y * y;
        }

        dot_sum / (norm_a_sum.sqrt() * norm_b_sum.sqrt())
    }

    fn cosine_sim_scalar(a: &[f32], b: &[f32]) -> f32 {
        let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
        let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        dot / (norm_a * norm_b)
    }
}

// ============================================================================
// REAL ALGORITHM: Softmax (full)
// ============================================================================

#[cfg(target_arch = "x86_64")]
mod softmax {
    use super::*;

    pub fn softmax(data: &mut [f32]) {
        if data.is_empty() {
            return;
        }

        if let Some(token) = Desktop64::summon() {
            softmax_avx2(token, data);
        } else {
            softmax_scalar(data);
        }
    }

    #[arcane]
    fn softmax_avx2(token: Desktop64, data: &mut [f32]) {
        // Find max
        let max_val = find_max(token, data);

        // exp(x - max) and accumulate sum
        let max_v = f32x8::splat(token, max_val);
        let mut sum_vec = f32x8::zero(token);

        for chunk in data.chunks_exact_mut(8) {
            let v = f32x8::from_array(token, chunk.as_ref().try_into().unwrap());
            let e = (v - max_v).exp_lowp();
            e.store(chunk.try_into().unwrap());
            sum_vec += e;
        }

        let mut sum = sum_vec.reduce_add();
        for x in data.chunks_exact_mut(8).into_remainder() {
            *x = (*x - max_val).exp();
            sum += *x;
        }

        // Normalize
        let inv = f32x8::splat(token, 1.0 / sum);
        for chunk in data.chunks_exact_mut(8) {
            let v = f32x8::from_array(token, chunk.as_ref().try_into().unwrap());
            (v * inv).store(chunk.try_into().unwrap());
        }
        for x in data.chunks_exact_mut(8).into_remainder() {
            *x /= sum;
        }
    }

    #[rite]
    fn find_max(token: Desktop64, data: &[f32]) -> f32 {
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
}

// ============================================================================
// MAIN
// ============================================================================

#[cfg(target_arch = "x86_64")]
fn main() {
    println!("Magetypes Showcase\n");

    if let Some(token) = Desktop64::summon() {
        // Comparison: raw vs clean
        let a = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let b = [2.0f32; 8];

        let raw = comparison::dot_product_raw(token, &a, &b);
        let clean = comparison::dot_product_clean(token, &a, &b);
        println!("Dot product (raw intrinsics): {}", raw);
        println!("Dot product (magetypes):      {}", clean);
        println!();

        // Statistics
        let (sum, min, max) = methods::statistics(token, &a);
        println!("Statistics of {:?}:", a);
        println!("  sum={}, min={}, max={}", sum, min, max);
        println!();

        // Transcendentals
        let logits = [1.0f32, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0];
        let (exp_vals, total) = transcendentals::softmax_chunk(token, &logits, 4.0);
        let probs: Vec<f32> = exp_vals.iter().map(|x| x / total).collect();
        println!("Softmax of {:?}:", logits);
        println!("  {:?}", probs);
        println!();

        // Conditionals
        let x = [-2.0f32, -1.0, 0.0, 1.0, 2.0, 3.0, -0.5, 0.5];
        println!("Activations on {:?}:", x);
        println!("  ReLU:       {:?}", conditionals::relu(token, &x));
        println!(
            "  LeakyReLU:  {:?}",
            conditionals::leaky_relu(token, &x, 0.1)
        );
        println!();

        // Layer norm
        let mut data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        println!("Before layer norm: {:?}", data);
        layer_norm::layer_norm(&mut data, 1.0, 0.0, 1e-5);
        println!("After layer norm:  {:?}", data);
        println!();

        // Cosine similarity
        let v1 = vec![1.0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let v2 = vec![1.0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let v3 = vec![0.0f32, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        println!("Cosine similarity:");
        println!(
            "  identical vectors: {}",
            cosine_similarity::cosine_sim(&v1, &v2)
        );
        println!(
            "  orthogonal vectors: {}",
            cosine_similarity::cosine_sim(&v1, &v3)
        );
    } else {
        println!("Desktop64 (AVX2+FMA) not available on this CPU");
    }
}

#[cfg(all(test, target_arch = "x86_64"))]
mod tests {
    use super::*;

    #[test]
    fn test_dot_equivalence() {
        if let Some(token) = Desktop64::summon() {
            let a = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
            let b = [2.0f32; 8];
            let raw = comparison::dot_product_raw(token, &a, &b);
            let clean = comparison::dot_product_clean(token, &a, &b);
            assert!((raw - clean).abs() < 0.001);
        }
    }

    #[test]
    fn test_relu() {
        if let Some(token) = Desktop64::summon() {
            let x = [-2.0f32, -1.0, 0.0, 1.0, 2.0, 3.0, -0.5, 0.5];
            let result = conditionals::relu(token, &x);
            assert_eq!(result, [0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 0.0, 0.5]);
        }
    }

    #[test]
    fn test_cosine_identical() {
        let v = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let sim = cosine_similarity::cosine_sim(&v, &v);
        assert!((sim - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_layer_norm() {
        let mut data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        layer_norm::layer_norm(&mut data, 1.0, 0.0, 1e-5);
        let mean: f32 = data.iter().sum::<f32>() / data.len() as f32;
        assert!(mean.abs() < 0.01, "Mean should be ~0, got {}", mean);
    }

    #[test]
    fn test_softmax_sums_to_one() {
        let mut data = vec![1.0f32, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0];
        softmax::softmax(&mut data);
        let sum: f32 = data.iter().sum();
        assert!((sum - 1.0).abs() < 0.01);
    }
}
