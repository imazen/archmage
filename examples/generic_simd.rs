//! Generic SIMD: Write Once, Run Everywhere
//!
//! This example demonstrates archmage's generic SIMD types — the key feature
//! that lets you write ONE algorithm and run it on any backend (AVX2, NEON,
//! WASM SIMD128, or scalar fallback) without code duplication.
//!
//! The pattern:
//!   1. Write functions generic over `T: F32x8Backend`
//!   2. Call them with any token: `X64V3Token`, `NeonToken`, `ScalarToken`
//!   3. The compiler monomorphizes per backend — zero-cost abstraction
//!
//! Run with:
//!   cargo run --example generic_simd --release
//!
//! Note: magetypes::simd requires x86_64, aarch64, or wasm32.

// magetypes::simd is only available on 64-bit x86, aarch64, and wasm32
#[cfg(not(any(
    target_arch = "x86_64",
    target_arch = "aarch64",
    target_arch = "wasm32"
)))]
fn main() {
    println!("generic_simd example requires x86_64, aarch64, or wasm32");
}

#[cfg(any(
    target_arch = "x86_64",
    target_arch = "aarch64",
    target_arch = "wasm32"
))]
fn main() {
    inner::main();
}

#[cfg(any(
    target_arch = "x86_64",
    target_arch = "aarch64",
    target_arch = "wasm32"
))]
mod inner {

    use std::time::Instant;

    use archmage::{ScalarToken, SimdToken};
    use magetypes::simd::backends::F32x8Backend;
    use magetypes::simd::generic::f32x8;

    // ============================================================================
    // Generic algorithms — ONE implementation, works with ALL backends
    //
    // These functions don't know or care which CPU they'll run on.
    // The token type parameter T carries that information.
    // ============================================================================

    /// Dot product of two slices. Works with AVX2, NEON, WASM, or scalar.
    #[inline(always)]
    fn dot_product<T: F32x8Backend>(token: T, a: &[f32], b: &[f32]) -> f32 {
        let mut acc = f32x8::<T>::zero(token);

        for (ac, bc) in a.chunks_exact(8).zip(b.chunks_exact(8)) {
            let va = f32x8::<T>::load(token, ac.try_into().unwrap());
            let vb = f32x8::<T>::load(token, bc.try_into().unwrap());
            acc = va.mul_add(vb, acc); // a*b + acc — single FMA on AVX2
        }

        let mut total = acc.reduce_add();
        for (&x, &y) in a
            .chunks_exact(8)
            .remainder()
            .iter()
            .zip(b.chunks_exact(8).remainder())
        {
            total += x * y;
        }
        total
    }

    /// Euclidean distance between two vectors.
    fn euclidean_distance<T: F32x8Backend>(token: T, a: &[f32], b: &[f32]) -> f32 {
        let mut acc = f32x8::<T>::zero(token);

        for (ac, bc) in a.chunks_exact(8).zip(b.chunks_exact(8)) {
            let va = f32x8::<T>::load(token, ac.try_into().unwrap());
            let vb = f32x8::<T>::load(token, bc.try_into().unwrap());
            let diff = va - vb;
            acc = diff.mul_add(diff, acc);
        }

        let mut total = acc.reduce_add();
        for (&x, &y) in a
            .chunks_exact(8)
            .remainder()
            .iter()
            .zip(b.chunks_exact(8).remainder())
        {
            let d = x - y;
            total += d * d;
        }
        total.sqrt()
    }

    /// Cosine similarity: dot(a,b) / (|a| * |b|).
    fn cosine_similarity<T: F32x8Backend>(token: T, a: &[f32], b: &[f32]) -> f32 {
        let mut dot = f32x8::<T>::zero(token);
        let mut norm_a = f32x8::<T>::zero(token);
        let mut norm_b = f32x8::<T>::zero(token);

        for (ac, bc) in a.chunks_exact(8).zip(b.chunks_exact(8)) {
            let va = f32x8::<T>::load(token, ac.try_into().unwrap());
            let vb = f32x8::<T>::load(token, bc.try_into().unwrap());
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

    /// Normalize a vector in-place to unit length.
    fn normalize_inplace<T: F32x8Backend>(token: T, data: &mut [f32]) {
        let norm_sq = dot_product(token, data, data);
        if norm_sq == 0.0 {
            return;
        }
        let inv_norm = 1.0 / norm_sq.sqrt();
        let inv_v = f32x8::<T>::splat(token, inv_norm);

        for chunk in data.chunks_exact_mut(8) {
            let v = f32x8::<T>::load(token, chunk.as_ref().try_into().unwrap());
            (v * inv_v).store(chunk.try_into().unwrap());
        }
        for x in data.chunks_exact_mut(8).into_remainder() {
            *x *= inv_norm;
        }
    }

    /// Batch normalization: (x - mean) / sqrt(var + eps).
    fn batch_normalize<T: F32x8Backend>(token: T, data: &mut [f32], eps: f32) {
        // Compute mean
        let n = data.len() as f32;
        let mut sum_v = f32x8::<T>::zero(token);
        for chunk in data.chunks_exact(8) {
            sum_v = sum_v + f32x8::<T>::load(token, chunk.try_into().unwrap());
        }
        let mut sum = sum_v.reduce_add();
        for &x in data.chunks_exact(8).remainder() {
            sum += x;
        }
        let mean = sum / n;

        // Compute variance
        let mean_v = f32x8::<T>::splat(token, mean);
        let mut var_v = f32x8::<T>::zero(token);
        for chunk in data.chunks_exact(8) {
            let v = f32x8::<T>::load(token, chunk.try_into().unwrap());
            let diff = v - mean_v;
            var_v = diff.mul_add(diff, var_v);
        }
        let mut var = var_v.reduce_add();
        for &x in data.chunks_exact(8).remainder() {
            let d = x - mean;
            var += d * d;
        }
        let inv_std = 1.0 / (var / n + eps).sqrt();

        // Normalize
        let inv_std_v = f32x8::<T>::splat(token, inv_std);
        for chunk in data.chunks_exact_mut(8) {
            let v = f32x8::<T>::load(token, chunk.as_ref().try_into().unwrap());
            ((v - mean_v) * inv_std_v).store(chunk.try_into().unwrap());
        }
        for x in data.chunks_exact_mut(8).into_remainder() {
            *x = (*x - mean) * inv_std;
        }
    }

    /// ReLU activation: max(0, x).
    fn relu_inplace<T: F32x8Backend>(token: T, data: &mut [f32]) {
        let zero = f32x8::<T>::zero(token);
        for chunk in data.chunks_exact_mut(8) {
            let v = f32x8::<T>::load(token, chunk.as_ref().try_into().unwrap());
            v.max(zero).store(chunk.try_into().unwrap());
        }
        for x in data.chunks_exact_mut(8).into_remainder() {
            *x = x.max(0.0);
        }
    }

    // ============================================================================
    // Main: demonstrate runtime dispatch + correctness verification
    // ============================================================================

    pub fn main() {
        println!("\n=== Generic SIMD: Write Once, Run Everywhere ===\n");
        println!("The SAME generic functions run with different backends.");
        println!("Zero code duplication. The compiler monomorphizes per token.\n");

        // Test data
        let n = 1024;
        let a: Vec<f32> = (0..n).map(|i| (i as f32 + 1.0) * 0.01).collect();
        let b: Vec<f32> = (0..n).map(|i| ((n - i) as f32) * 0.01).collect();

        // --- ScalarToken: always available, no CPU detection needed ---
        println!("--- ScalarToken (universal fallback) ---");
        run_suite(ScalarToken, &a, &b);

        // --- Platform-specific tokens ---
        #[cfg(target_arch = "x86_64")]
        {
            if let Some(token) = archmage::X64V3Token::summon() {
                println!("--- X64V3Token / Desktop64 (AVX2 + FMA) ---");
                run_suite(token, &a, &b);
                benchmark(token, &a, &b);
            } else {
                println!("--- X64V3Token not available on this CPU ---");
            }
        }

        #[cfg(target_arch = "aarch64")]
        {
            if let Some(token) = archmage::NeonToken::summon() {
                println!("--- NeonToken (ARM NEON) ---");
                run_suite(token, &a, &b);
            }
        }

        #[cfg(target_arch = "wasm32")]
        {
            println!("--- Wasm128Token (WASM SIMD128) ---");
            // WASM detection is compile-time, not runtime
        }
    }

    /// Run the full algorithm suite with any backend.
    fn run_suite<T: F32x8Backend>(token: T, a: &[f32], b: &[f32]) {
        let dot = dot_product(token, a, b);
        let dist = euclidean_distance(token, a, b);
        let cosine = cosine_similarity(token, a, b);

        println!("  dot_product:        {dot:.4}");
        println!("  euclidean_distance: {dist:.4}");
        println!("  cosine_similarity:  {cosine:.6}");

        // Normalize
        let mut v = a.to_vec();
        normalize_inplace(token, &mut v);
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        println!("  normalize |v|:      {norm:.6} (should be ~1.0)");

        // Batch normalize
        let mut data: Vec<f32> = (0..64).map(|i| i as f32).collect();
        batch_normalize(token, &mut data, 1e-5);
        let mean: f32 = data.iter().sum::<f32>() / data.len() as f32;
        println!("  batch_norm mean:    {mean:.6} (should be ~0.0)");

        // ReLU
        let mut relu_data = vec![-2.0, -1.0, 0.0, 1.0, 2.0, 3.0, -0.5, 0.5];
        relu_inplace(token, &mut relu_data);
        println!("  relu([-2..3]):      {relu_data:?}");

        println!();
    }

    /// Benchmark: generic function, same code path, two different backends.
    ///
    /// Two key insights demonstrated here:
    ///
    /// 1. **`#[arcane]` at the boundary**: The benchmark loop is inside an `#[arcane]`
    ///    function. This puts the entire loop in one `#[target_feature]` region, so
    ///    `#[inline(always)]` generic functions get their backend calls inlined.
    ///    Without this, each load/mul_add/reduce_add would cross a target-feature
    ///    boundary, preventing optimization (see docs/PERFORMANCE.md).
    ///
    /// 2. **ScalarToken is competitive for simple ops**: LLVM auto-vectorizes the
    ///    scalar array operations, so the speedup for simple patterns like dot
    ///    product is modest. Explicit SIMD shines for complex operations (shuffles,
    ///    bit manipulation, transcendentals) that auto-vectorization can't handle.
    #[cfg(target_arch = "x86_64")]
    fn benchmark(avx2: archmage::X64V3Token, a: &[f32], b: &[f32]) {
        const ITERS: u32 = 50_000;

        println!(
            "--- Benchmark: dot_product ({ITERS} iters, {} elements) ---",
            a.len()
        );

        // Scalar baseline (LLVM auto-vectorizes these array ops in release mode)
        let scalar_time = bench_dot_scalar(a, b, ITERS);

        // AVX2 — the benchmark loop runs inside #[arcane], so generic SIMD
        // ops (load, mul_add, reduce_add) all inline into one AVX2 region
        let avx2_time = bench_dot_avx2(avx2, a, b, ITERS);

        println!("  Scalar: {:.2} ms", scalar_time.as_secs_f64() * 1000.0);
        println!("  AVX2:   {:.2} ms", avx2_time.as_secs_f64() * 1000.0);
        println!(
            "  Speedup: {:.1}x\n",
            scalar_time.as_secs_f64() / avx2_time.as_secs_f64()
        );

        // Verify correctness: both backends produce same result
        let scalar_dot = dot_product(ScalarToken, a, b);
        let avx2_dot = dot_product(avx2, a, b);
        let rel_err = (scalar_dot - avx2_dot).abs() / scalar_dot.abs();
        println!(
            "  Correctness: scalar={scalar_dot:.6}, avx2={avx2_dot:.6}, rel_err={rel_err:.2e}"
        );
        assert!(rel_err < 1e-4, "Results should match within tolerance");
        println!();
    }

    #[cfg(target_arch = "x86_64")]
    fn bench_dot_scalar(a: &[f32], b: &[f32], iters: u32) -> std::time::Duration {
        let start = Instant::now();
        for _ in 0..iters {
            std::hint::black_box(dot_product(
                ScalarToken,
                std::hint::black_box(a),
                std::hint::black_box(b),
            ));
        }
        start.elapsed()
    }

    /// The benchmark loop runs inside `#[arcane]`, so all generic SIMD operations
    /// (load, mul_add, reduce_add) inline into one AVX2 region — no boundaries.
    #[cfg(target_arch = "x86_64")]
    #[archmage::arcane]
    fn bench_dot_avx2(
        _token: archmage::X64V3Token,
        a: &[f32],
        b: &[f32],
        iters: u32,
    ) -> std::time::Duration {
        let start = Instant::now();
        for _ in 0..iters {
            std::hint::black_box(dot_product(
                _token,
                std::hint::black_box(a),
                std::hint::black_box(b),
            ));
        }
        start.elapsed()
    }

    // ============================================================================
    // Tests — verify all generic algorithms produce correct results
    // ============================================================================

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn dot_product_known_values() {
            let a = [1.0f32; 8];
            let b = [2.0f32; 8];
            assert_eq!(dot_product(ScalarToken, &a, &b), 16.0);
        }

        #[test]
        fn dot_product_with_remainder() {
            // 10 elements: 8 in SIMD loop + 2 remainder
            let a: Vec<f32> = (1..=10).map(|i| i as f32).collect();
            let b = vec![1.0f32; 10];
            assert_eq!(dot_product(ScalarToken, &a, &b), 55.0);
        }

        #[test]
        fn euclidean_distance_orthogonal() {
            let mut a = [0.0f32; 8];
            let mut b = [0.0f32; 8];
            a[0] = 1.0;
            b[1] = 1.0;
            let dist = euclidean_distance(ScalarToken, &a, &b);
            assert!((dist - std::f32::consts::SQRT_2).abs() < 1e-6);
        }

        #[test]
        fn cosine_similarity_identical() {
            let v = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
            let sim = cosine_similarity(ScalarToken, &v, &v);
            assert!((sim - 1.0).abs() < 1e-6);
        }

        #[test]
        fn cosine_similarity_orthogonal() {
            let mut a = [0.0f32; 8];
            let mut b = [0.0f32; 8];
            a[0] = 1.0;
            b[1] = 1.0;
            let sim = cosine_similarity(ScalarToken, &a, &b);
            assert!(sim.abs() < 1e-6);
        }

        #[test]
        fn normalize_produces_unit_vector() {
            let mut v = vec![3.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
            normalize_inplace(ScalarToken, &mut v);
            let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            assert!((norm - 1.0).abs() < 1e-6);
            assert!((v[0] - 0.6).abs() < 1e-6);
            assert!((v[1] - 0.8).abs() < 1e-6);
        }

        #[test]
        fn batch_normalize_centers_data() {
            let mut data: Vec<f32> = (0..16).map(|i| i as f32 * 10.0).collect();
            batch_normalize(ScalarToken, &mut data, 1e-5);
            let mean: f32 = data.iter().sum::<f32>() / data.len() as f32;
            assert!(mean.abs() < 1e-4, "mean should be ~0, got {mean}");
        }

        #[test]
        fn relu_clips_negatives() {
            let mut data = vec![-2.0, -1.0, 0.0, 1.0, 2.0, 3.0, -0.5, 0.5];
            relu_inplace(ScalarToken, &mut data);
            assert_eq!(data, vec![0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 0.0, 0.5]);
        }

        #[cfg(target_arch = "x86_64")]
        #[test]
        fn cross_backend_consistency() {
            use archmage::X64V3Token;

            if let Some(t) = X64V3Token::summon() {
                let a: Vec<f32> = (0..64).map(|i| (i as f32 + 1.0) * 0.1).collect();
                let b: Vec<f32> = (0..64).map(|i| ((64 - i) as f32) * 0.1).collect();

                let scalar_dot = dot_product(ScalarToken, &a, &b);
                let simd_dot = dot_product(t, &a, &b);
                assert!(
                    (scalar_dot - simd_dot).abs() / scalar_dot.abs() < 1e-5,
                    "scalar={scalar_dot}, simd={simd_dot}"
                );

                let scalar_dist = euclidean_distance(ScalarToken, &a, &b);
                let simd_dist = euclidean_distance(t, &a, &b);
                assert!(
                    (scalar_dist - simd_dist).abs() / scalar_dist.abs() < 1e-4,
                    "scalar={scalar_dist}, simd={simd_dist}"
                );

                let scalar_cos = cosine_similarity(ScalarToken, &a, &b);
                let simd_cos = cosine_similarity(t, &a, &b);
                assert!(
                    (scalar_cos - simd_cos).abs() < 1e-5,
                    "scalar={scalar_cos}, simd={simd_cos}"
                );
            }
        }
    }
} // mod inner
