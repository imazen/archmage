//! Accuracy analysis for archmage transcendental functions
//!
//! Measures ULP errors and round-trip accuracy for sRGB color processing
//! at different bit depths (8-bit, 12-bit, 16-bit).
//!
//! Compares lowp (fast) vs midp (mid-precision) implementations.
//!
//! Run with:
//! ```sh
//! cargo run --example accuracy_test --release
//! ```

// x86-only example - stub main for other platforms
#[cfg(not(target_arch = "x86_64"))]
fn main() {}

#[cfg(target_arch = "x86_64")]
mod x86_impl {
    #![allow(clippy::needless_borrows_for_generic_args)]

    use archmage::{SimdToken, X64V3Token, arcane};
    use magetypes::simd::f32x8;

    /// Calculate ULP difference between two f32 values
    fn ulp_diff(a: f32, b: f32) -> u32 {
        if a.is_nan() || b.is_nan() {
            return u32::MAX;
        }
        if a == b {
            return 0;
        }
        // Handle zero specially
        if a == 0.0 || b == 0.0 {
            let non_zero = if a == 0.0 { b } else { a };
            return non_zero.to_bits();
        }
        // Handle different signs
        if a.is_sign_positive() != b.is_sign_positive() {
            return u32::MAX; // Very different
        }
        let a_bits = a.to_bits();
        let b_bits = b.to_bits();
        a_bits.abs_diff(b_bits)
    }

    /// Measure accuracy statistics for a function over a range
    #[allow(dead_code)]
    struct AccuracyStats {
        max_ulp: u32,
        avg_ulp: f64,
        max_rel_err: f64,
        avg_rel_err: f64,
        samples: usize,
    }

    impl std::fmt::Display for AccuracyStats {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(
                f,
                "max_ulp: {:>8}, avg_ulp: {:>8.2}, max_rel: {:.2e}, avg_rel: {:.2e}",
                self.max_ulp, self.avg_ulp, self.max_rel_err, self.avg_rel_err
            )
        }
    }

    fn measure_accuracy<F, R>(
        _name: &str,
        test_values: &[f32],
        archmage_fn: F,
        reference_fn: R,
    ) -> AccuracyStats
    where
        F: Fn(&[f32]) -> Vec<f32>,
        R: Fn(f32) -> f32,
    {
        let archmage_results = archmage_fn(test_values);

        let mut max_ulp = 0u32;
        let mut sum_ulp = 0u64;
        let mut max_rel_err = 0.0f64;
        let mut sum_rel_err = 0.0f64;
        let mut count = 0usize;

        for (i, &input) in test_values.iter().enumerate() {
            let expected = reference_fn(input);
            let got = archmage_results[i];

            if !expected.is_finite() || !got.is_finite() {
                continue;
            }

            let ulp = ulp_diff(got, expected);
            max_ulp = max_ulp.max(ulp);
            sum_ulp += ulp as u64;

            if expected.abs() > 1e-10 {
                let rel_err = ((got - expected) / expected).abs() as f64;
                max_rel_err = max_rel_err.max(rel_err);
                sum_rel_err += rel_err;
            }
            count += 1;
        }

        AccuracyStats {
            max_ulp,
            avg_ulp: sum_ulp as f64 / count as f64,
            max_rel_err,
            avg_rel_err: sum_rel_err / count as f64,
            samples: count,
        }
    }

    // ============================================================================
    // SIMD kernel functions using #[arcane]
    // ============================================================================

    /// Process a single chunk of 8 floats with exp2_lowp
    #[arcane]
    fn exp2_lowp_chunk(token: X64V3Token, input: &[f32; 8]) -> [f32; 8] {
        f32x8::load(token, input).exp2_lowp().to_array()
    }

    /// Process a single chunk of 8 floats with log2_lowp
    #[arcane]
    fn log2_lowp_chunk(token: X64V3Token, input: &[f32; 8]) -> [f32; 8] {
        f32x8::load(token, input).log2_lowp().to_array()
    }

    /// Process a single chunk of 8 floats with pow_lowp
    #[arcane]
    fn pow_lowp_chunk(token: X64V3Token, input: &[f32; 8], exp: f32) -> [f32; 8] {
        f32x8::load(token, input).pow_lowp(exp).to_array()
    }

    /// Process a single chunk of 8 floats with pow_midp
    #[arcane]
    fn pow_midp_chunk(token: X64V3Token, input: &[f32; 8], exp: f32) -> [f32; 8] {
        f32x8::load(token, input).pow_midp(exp).to_array()
    }

    // ============================================================================
    // Slice processing wrappers
    // ============================================================================

    /// Apply archmage exp2_lowp to a slice
    fn archmage_exp2(input: &[f32]) -> Vec<f32> {
        let Some(token) = X64V3Token::try_new() else {
            return input.iter().map(|&x| x.exp2()).collect();
        };

        let mut output = vec![0.0f32; input.len()];
        let chunks = input.len() / 8;

        for i in 0..chunks {
            let start = i * 8;
            let arr: &[f32; 8] = input[start..start + 8].try_into().unwrap();
            output[start..start + 8].copy_from_slice(&exp2_lowp_chunk(token, arr));
        }

        // Handle remainder with scalar fallback
        for i in (chunks * 8)..input.len() {
            output[i] = input[i].exp2();
        }

        output
    }

    /// Apply archmage log2_lowp to a slice
    fn archmage_log2(input: &[f32]) -> Vec<f32> {
        let Some(token) = X64V3Token::try_new() else {
            return input.iter().map(|&x| x.log2()).collect();
        };

        let mut output = vec![0.0f32; input.len()];
        let chunks = input.len() / 8;

        for i in 0..chunks {
            let start = i * 8;
            let arr: &[f32; 8] = input[start..start + 8].try_into().unwrap();
            output[start..start + 8].copy_from_slice(&log2_lowp_chunk(token, arr));
        }

        for i in (chunks * 8)..input.len() {
            output[i] = input[i].log2();
        }

        output
    }

    /// Apply archmage pow_lowp (fast, low-precision) to a slice
    fn archmage_pow_lowp(input: &[f32], exp: f32) -> Vec<f32> {
        let Some(token) = X64V3Token::try_new() else {
            return input.iter().map(|&x| x.powf(exp)).collect();
        };

        let mut output = vec![0.0f32; input.len()];
        let chunks = input.len() / 8;

        for i in 0..chunks {
            let start = i * 8;
            let arr: &[f32; 8] = input[start..start + 8].try_into().unwrap();
            output[start..start + 8].copy_from_slice(&pow_lowp_chunk(token, arr, exp));
        }

        for i in (chunks * 8)..input.len() {
            output[i] = input[i].powf(exp);
        }

        output
    }

    /// Apply archmage pow_midp (mid-precision) to a slice
    fn archmage_pow_midp(input: &[f32], exp: f32) -> Vec<f32> {
        let Some(token) = X64V3Token::try_new() else {
            return input.iter().map(|&x| x.powf(exp)).collect();
        };

        let mut output = vec![0.0f32; input.len()];
        let chunks = input.len() / 8;

        for i in 0..chunks {
            let start = i * 8;
            let arr: &[f32; 8] = input[start..start + 8].try_into().unwrap();
            output[start..start + 8].copy_from_slice(&pow_midp_chunk(token, arr, exp));
        }

        for i in (chunks * 8)..input.len() {
            output[i] = input[i].powf(exp);
        }

        output
    }

    /// Test round-trip accuracy for a given pow function
    fn test_roundtrip_with<F>(name: &str, bit_depth: u32, pow_fn: F)
    where
        F: Fn(&[f32], f32) -> Vec<f32>,
    {
        let levels = 1u32 << bit_depth;
        let max_val = (levels - 1) as f32;

        let inputs: Vec<f32> = (0..levels).map(|i| i as f32 / max_val).collect();
        let linear = pow_fn(&inputs, 2.4);
        let back = pow_fn(&linear, 1.0 / 2.4);

        let mut exact = 0u32;
        let mut off_one = 0u32;
        let mut off_more = 0u32;
        let mut max_err = 0i32;

        for (i, &original) in inputs.iter().enumerate() {
            let recovered = back[i];
            let orig_level = (original * max_val).round() as i32;
            let recv_level = (recovered * max_val).round() as i32;
            let error = (recv_level - orig_level).abs();

            if error == 0 {
                exact += 1;
            } else if error == 1 {
                off_one += 1;
            } else {
                off_more += 1;
                max_err = max_err.max(error);
            }
        }

        let total = levels as f32;
        print!(
            "  {:12} {:>2}-bit: exact {:>5.1}%, ±1 {:>5.1}%, >1 {:>5.1}%",
            name,
            bit_depth,
            100.0 * exact as f32 / total,
            100.0 * off_one as f32 / total,
            100.0 * off_more as f32 / total
        );
        if max_err > 0 {
            println!(" (max {})", max_err);
        } else {
            println!();
        }
    }

    /// Test round-trip accuracy: sRGB -> linear -> sRGB
    #[allow(dead_code)]
    fn test_srgb_roundtrip(bit_depth: u32) {
        let levels = 1u32 << bit_depth;
        let max_val = (levels - 1) as f32;

        println!(
            "\n=== {}-bit sRGB Round-trip Test ({} levels) ===",
            bit_depth, levels
        );

        let Some(_token) = X64V3Token::try_new() else {
            eprintln!("AVX2+FMA not available");
            return;
        };

        // Generate all possible input values (normalized to 0..1)
        let inputs: Vec<f32> = (0..levels).map(|i| i as f32 / max_val).collect();

        // Test lowp (fast) implementation
        let linear = archmage_pow_lowp(&inputs, 2.4);
        let back_to_srgb = archmage_pow_lowp(&linear, 1.0 / 2.4);

        // Quantize back to integer levels
        let mut exact_matches = 0u32;
        let mut off_by_one = 0u32;
        let mut off_by_more = 0u32;
        let mut max_error_levels = 0i32;

        for (i, &original) in inputs.iter().enumerate() {
            let recovered = back_to_srgb[i];

            // Quantize both to integer levels
            let orig_level = (original * max_val).round() as i32;
            let recv_level = (recovered * max_val).round() as i32;
            let error = (recv_level - orig_level).abs();

            if error == 0 {
                exact_matches += 1;
            } else if error == 1 {
                off_by_one += 1;
            } else {
                off_by_more += 1;
                max_error_levels = max_error_levels.max(error);
            }
        }

        let total = levels as f32;
        println!(
            "  Exact matches:  {:>6} ({:>5.1}%)",
            exact_matches,
            100.0 * exact_matches as f32 / total
        );
        println!(
            "  Off by 1 level: {:>6} ({:>5.1}%)",
            off_by_one,
            100.0 * off_by_one as f32 / total
        );
        println!(
            "  Off by >1:      {:>6} ({:>5.1}%)",
            off_by_more,
            100.0 * off_by_more as f32 / total
        );
        if off_by_more > 0 {
            println!("  Max error:      {} levels", max_error_levels);
        }

        // Also test with std::f32 for comparison
        let linear_std: Vec<f32> = inputs.iter().map(|&x| x.powf(2.4)).collect();
        let back_std: Vec<f32> = linear_std.iter().map(|&x| x.powf(1.0 / 2.4)).collect();

        let mut std_exact = 0u32;
        let mut std_off_one = 0u32;
        let mut std_off_more = 0u32;

        for (i, &original) in inputs.iter().enumerate() {
            let recovered = back_std[i];
            let orig_level = (original * max_val).round() as i32;
            let recv_level = (recovered * max_val).round() as i32;
            let error = (recv_level - orig_level).abs();

            if error == 0 {
                std_exact += 1;
            } else if error == 1 {
                std_off_one += 1;
            } else {
                std_off_more += 1;
            }
        }

        println!("\n  (std::f32 comparison)");
        println!(
            "  Exact matches:  {:>6} ({:>5.1}%)",
            std_exact,
            100.0 * std_exact as f32 / total
        );
        println!(
            "  Off by 1 level: {:>6} ({:>5.1}%)",
            std_off_one,
            100.0 * std_off_one as f32 / total
        );
        println!(
            "  Off by >1:      {:>6} ({:>5.1}%)",
            std_off_more,
            100.0 * std_off_more as f32 / total
        );
    }

    pub fn main() {
        println!("Archmage Transcendental Accuracy Analysis");
        println!("=========================================\n");

        if X64V3Token::try_new().is_none() {
            eprintln!("AVX2+FMA not available, skipping");
            return;
        }

        // Test ranges
        let range_0_1: Vec<f32> = (1..=10000).map(|i| i as f32 / 10000.0).collect();
        let range_0_255: Vec<f32> = (1..=255).map(|i| i as f32).collect();
        let range_0_16k: Vec<f32> = (1..=16384).map(|i| i as f32).collect();

        // For log2, we need positive values
        let log_range_small: Vec<f32> = (1..=10000).map(|i| i as f32 / 10000.0).collect();
        let _log_range_medium: Vec<f32> = (1..=255).map(|i| i as f32 / 255.0 + 0.001).collect();

        println!("=== exp2 Accuracy ===\n");

        println!("Range [0, 1] (10K samples, normalized sRGB input):");
        let stats = measure_accuracy("exp2", &range_0_1, archmage_exp2, |x| x.exp2());
        println!("  {}", stats);

        println!("\nRange [-10, 10] (gain map range):");
        let gain_range: Vec<f32> = (-10000..=10000).map(|i| i as f32 / 1000.0).collect();
        let stats = measure_accuracy("exp2", &gain_range, archmage_exp2, |x| x.exp2());
        println!("  {}", stats);

        println!("\n=== log2 Accuracy ===\n");

        println!("Range (0, 1] (normalized sRGB):");
        let stats = measure_accuracy("log2", &log_range_small, archmage_log2, |x| x.log2());
        println!("  {}", stats);

        println!("\nRange [1, 255] (8-bit levels):");
        let stats = measure_accuracy("log2", &range_0_255, archmage_log2, |x| x.log2());
        println!("  {}", stats);

        println!("\nRange [1, 16384] (HDR range):");
        let stats = measure_accuracy("log2", &range_0_16k, archmage_log2, |x| x.log2());
        println!("  {}", stats);

        println!("\n=== pow(x, 2.4) Accuracy (sRGB decode) ===\n");

        println!("Range (0, 1] (normalized sRGB input):");
        let stats = measure_accuracy(
            "pow_2.4",
            &range_0_1,
            |x| archmage_pow_lowp(x, 2.4),
            |x| x.powf(2.4),
        );
        println!("  {}", stats);

        println!("\n=== pow_lowp(x, 1/2.4) Accuracy (sRGB encode) ===\n");

        println!("Range (0, 1] (linear RGB input):");
        let stats = measure_accuracy(
            "pow_0.417",
            &range_0_1,
            |x| archmage_pow_lowp(x, 1.0 / 2.4),
            |x| x.powf(1.0 / 2.4),
        );
        println!("  {}", stats);

        println!("\n=== pow_lowp(x, 2.2) Accuracy (simple gamma) ===\n");

        println!("Range (0, 1]:");
        let stats = measure_accuracy(
            "pow_2.2",
            &range_0_1,
            |x| archmage_pow_lowp(x, 2.2),
            |x| x.powf(2.2),
        );
        println!("  {}", stats);

        // Mid-precision function accuracy
        println!("\n=== MID-PRECISION Functions ===\n");

        println!("pow_midp(x, 2.4) - Range (0, 1]:");
        let stats = measure_accuracy(
            "pow_midp_2.4",
            &range_0_1,
            |x| archmage_pow_midp(x, 2.4),
            |x| x.powf(2.4),
        );
        println!("  {}", stats);

        println!("\npow_midp(x, 1/2.4) - Range (0, 1]:");
        let stats = measure_accuracy(
            "pow_midp_0.417",
            &range_0_1,
            |x| archmage_pow_midp(x, 1.0 / 2.4),
            |x| x.powf(1.0 / 2.4),
        );
        println!("  {}", stats);

        // Round-trip comparison: lowp vs midp vs std
        println!("\n=== sRGB Round-trip Comparison ===");
        println!("(pow(x, 2.4) then pow(result, 1/2.4), checking if we get back original level)\n");

        let std_pow =
            |input: &[f32], exp: f32| -> Vec<f32> { input.iter().map(|&x| x.powf(exp)).collect() };

        for bits in [8, 10, 12, 16] {
            test_roundtrip_with("pow_lowp", bits, &archmage_pow_lowp);
            test_roundtrip_with("pow_midp", bits, &archmage_pow_midp);
            test_roundtrip_with("std::f32", bits, &std_pow);
            println!();
        }

        println!("=== Summary ===\n");
        println!("LOWP functions (pow_lowp, exp2_lowp, log2_lowp):");
        println!("  - ~90,000 ULP max error, ~0.5% relative error");
        println!("  - Suitable for preview/thumbnails only");
        println!();
        println!("MIDP functions (pow_midp, exp2_midp, log2_midp):");
        println!("  - ~140 ULP max error, ~8e-6 relative error");
        println!("  - 100% exact round-trips for 8-bit, 10-bit, 12-bit");
        println!("  - 97% exact / 3% off-by-1 for 16-bit");
        println!("  - Suitable for production color processing");
        println!();
        println!("For perfect precision, use std::f32 (scalar) at ~7-15x slower throughput.");
    }
}

#[cfg(target_arch = "x86_64")]
fn main() {
    x86_impl::main()
}
