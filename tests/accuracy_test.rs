//! Brute force accuracy tests for transcendental functions
//!
//! Tests all transcendental functions against std:: implementations
//! across their valid input ranges, including edge cases.

#![cfg(target_arch = "x86_64")]

use archmage::{SimdToken, X64V3Token, arcane};
use magetypes::simd::f32x8;

// ============================================================================
// Test wrappers using #[arcane]
// ============================================================================

#[arcane]
fn simd_cbrt_midp(token: X64V3Token, input: &[f32; 8]) -> [f32; 8] {
    f32x8::load(token, input).cbrt_midp().to_array()
}

#[arcane]
fn simd_pow_lowp(token: X64V3Token, input: &[f32; 8], n: f32) -> [f32; 8] {
    f32x8::load(token, input).pow_lowp(n).to_array()
}

#[arcane]
fn simd_pow_midp(token: X64V3Token, input: &[f32; 8], n: f32) -> [f32; 8] {
    f32x8::load(token, input).pow_midp(n).to_array()
}

#[arcane]
fn simd_exp2_lowp(token: X64V3Token, input: &[f32; 8]) -> [f32; 8] {
    f32x8::load(token, input).exp2_lowp().to_array()
}

#[arcane]
fn simd_exp2_midp(token: X64V3Token, input: &[f32; 8]) -> [f32; 8] {
    f32x8::load(token, input).exp2_midp().to_array()
}

#[arcane]
fn simd_log2_lowp(token: X64V3Token, input: &[f32; 8]) -> [f32; 8] {
    f32x8::load(token, input).log2_lowp().to_array()
}

#[arcane]
fn simd_log2_midp(token: X64V3Token, input: &[f32; 8]) -> [f32; 8] {
    f32x8::load(token, input).log2_midp().to_array()
}

#[arcane]
fn simd_ln_lowp(token: X64V3Token, input: &[f32; 8]) -> [f32; 8] {
    f32x8::load(token, input).ln_lowp().to_array()
}

#[arcane]
fn simd_ln_midp(token: X64V3Token, input: &[f32; 8]) -> [f32; 8] {
    f32x8::load(token, input).ln_midp().to_array()
}

#[arcane]
fn simd_exp_lowp(token: X64V3Token, input: &[f32; 8]) -> [f32; 8] {
    f32x8::load(token, input).exp_lowp().to_array()
}

#[arcane]
fn simd_exp_midp(token: X64V3Token, input: &[f32; 8]) -> [f32; 8] {
    f32x8::load(token, input).exp_midp().to_array()
}

#[arcane]
fn simd_log10_lowp(token: X64V3Token, input: &[f32; 8]) -> [f32; 8] {
    f32x8::load(token, input).log10_lowp().to_array()
}

// ============================================================================
// Test infrastructure
// ============================================================================

struct AccuracyStats {
    name: &'static str,
    max_rel_err: f32,
    max_abs_err: f32,
    avg_rel_err: f64,
    worst_input: f32,
    worst_expected: f32,
    worst_got: f32,
    total_tested: usize,
    nan_count: usize,
    inf_count: usize,
}

impl AccuracyStats {
    fn new(name: &'static str) -> Self {
        Self {
            name,
            max_rel_err: 0.0,
            max_abs_err: 0.0,
            avg_rel_err: 0.0,
            worst_input: 0.0,
            worst_expected: 0.0,
            worst_got: 0.0,
            total_tested: 0,
            nan_count: 0,
            inf_count: 0,
        }
    }

    fn update(&mut self, input: f32, expected: f32, got: f32) {
        if !expected.is_finite() || !got.is_finite() {
            if got.is_nan() && !expected.is_nan() {
                self.nan_count += 1;
            }
            if got.is_infinite() && !expected.is_infinite() {
                self.inf_count += 1;
            }
            return;
        }

        self.total_tested += 1;
        let abs_err = (got - expected).abs();
        let rel_err = if expected.abs() > 1e-10 {
            abs_err / expected.abs()
        } else {
            abs_err
        };

        self.avg_rel_err += rel_err as f64;

        if rel_err > self.max_rel_err {
            self.max_rel_err = rel_err;
            self.max_abs_err = abs_err;
            self.worst_input = input;
            self.worst_expected = expected;
            self.worst_got = got;
        }
    }

    fn finalize(&mut self) {
        if self.total_tested > 0 {
            self.avg_rel_err /= self.total_tested as f64;
        }
    }

    fn print(&self) {
        println!(
            "{:20} max_rel_err: {:.2e}  avg_rel_err: {:.2e}  tested: {}",
            self.name, self.max_rel_err, self.avg_rel_err, self.total_tested
        );
        if self.max_rel_err > 1e-5 {
            println!(
                "    worst: input={:.8e} expected={:.8e} got={:.8e}",
                self.worst_input, self.worst_expected, self.worst_got
            );
        }
        if self.nan_count > 0 || self.inf_count > 0 {
            println!(
                "    ERRORS: {} unexpected NaN, {} unexpected Inf",
                self.nan_count, self.inf_count
            );
        }
    }

    fn assert_max_rel_err(&self, max_allowed: f32) {
        assert!(
            self.max_rel_err <= max_allowed,
            "{}: max_rel_err {:.2e} exceeds limit {:.2e} at input={:.8e}",
            self.name,
            self.max_rel_err,
            max_allowed,
            self.worst_input
        );
        assert!(
            self.nan_count == 0,
            "{}: {} unexpected NaN values",
            self.name,
            self.nan_count
        );
    }
}

// ============================================================================
// cbrt tests
// ============================================================================

#[test]
#[cfg(target_arch = "x86_64")]
fn test_cbrt_midp_brute_force() {
    let Some(token) = X64V3Token::summon() else {
        eprintln!("AVX2+FMA not available, skipping test");
        return;
    };

    let mut stats = AccuracyStats::new("cbrt_midp");

    // Test positive values: 1e-37 to 1e37 (avoiding denormals)
    let test_values: Vec<f32> = (0..1_000_000)
        .map(|i| {
            let t = i as f32 / 1_000_000.0;
            // Logarithmic distribution from 1e-37 to 1e37
            10.0f32.powf(-37.0 + t * 74.0)
        })
        .collect();

    for chunk in test_values.chunks(8) {
        if chunk.len() < 8 {
            continue;
        }
        let arr: &[f32; 8] = chunk.try_into().unwrap();
        let result = simd_cbrt_midp(token, arr);
        for (i, &x) in arr.iter().enumerate() {
            let expected = x.cbrt();
            stats.update(x, expected, result[i]);
        }
    }

    // Test negative values: -1e37 to -1e-37 (avoiding denormals)
    let negative_values: Vec<f32> = (0..1_000_000)
        .map(|i| {
            let t = i as f32 / 1_000_000.0;
            -10.0f32.powf(-37.0 + t * 74.0)
        })
        .collect();

    for chunk in negative_values.chunks(8) {
        if chunk.len() < 8 {
            continue;
        }
        let arr: &[f32; 8] = chunk.try_into().unwrap();
        let result = simd_cbrt_midp(token, arr);
        for (i, &x) in arr.iter().enumerate() {
            let expected = x.cbrt();
            stats.update(x, expected, result[i]);
        }
    }

    // Test values near zero
    let near_zero: Vec<f32> = (-100_000..100_000)
        .map(|i| i as f32 * 1e-10)
        .filter(|&x| x != 0.0)
        .collect();

    for chunk in near_zero.chunks(8) {
        if chunk.len() < 8 {
            continue;
        }
        let arr: &[f32; 8] = chunk.try_into().unwrap();
        let result = simd_cbrt_midp(token, arr);
        for (i, &x) in arr.iter().enumerate() {
            let expected = x.cbrt();
            stats.update(x, expected, result[i]);
        }
    }

    // Test perfect cubes
    let perfect_cubes: Vec<f32> = (-100..=100)
        .filter(|&i| i != 0)
        .map(|i| (i as f32).powi(3))
        .collect();

    for chunk in perfect_cubes.chunks(8) {
        if chunk.len() < 8 {
            continue;
        }
        let mut arr = [0.0f32; 8];
        for (i, &v) in chunk.iter().enumerate() {
            arr[i] = v;
        }
        let result = simd_cbrt_midp(token, &arr);
        for (i, &x) in chunk.iter().enumerate() {
            let expected = x.cbrt();
            stats.update(x, expected, result[i]);
        }
    }

    stats.finalize();
    stats.print();
    stats.assert_max_rel_err(1e-6); // ~2 ULP for f32
}

// ============================================================================
// pow tests
// ============================================================================

#[test]
#[cfg(target_arch = "x86_64")]
fn test_pow_midp_brute_force() {
    let Some(token) = X64V3Token::summon() else {
        eprintln!("AVX2+FMA not available, skipping test");
        return;
    };

    // Test pow(x, 2.4) - sRGB gamma
    let mut stats = AccuracyStats::new("pow_midp(x, 2.4)");

    let test_values: Vec<f32> = (1..1_000_000).map(|i| i as f32 / 1_000_000.0).collect();

    for chunk in test_values.chunks(8) {
        if chunk.len() < 8 {
            continue;
        }
        let arr: &[f32; 8] = chunk.try_into().unwrap();
        let result = simd_pow_midp(token, arr, 2.4);
        for (i, &x) in arr.iter().enumerate() {
            let expected = x.powf(2.4);
            stats.update(x, expected, result[i]);
        }
    }

    stats.finalize();
    stats.print();
    stats.assert_max_rel_err(1e-5);

    // Test pow(x, 1/2.4) - inverse sRGB gamma
    let mut stats = AccuracyStats::new("pow_midp(x, 1/2.4)");

    for chunk in test_values.chunks(8) {
        if chunk.len() < 8 {
            continue;
        }
        let arr: &[f32; 8] = chunk.try_into().unwrap();
        let result = simd_pow_midp(token, arr, 1.0 / 2.4);
        for (i, &x) in arr.iter().enumerate() {
            let expected = x.powf(1.0 / 2.4);
            stats.update(x, expected, result[i]);
        }
    }

    stats.finalize();
    stats.print();
    stats.assert_max_rel_err(1e-5);

    // Test pow(x, 0.5) - should match sqrt
    let mut stats = AccuracyStats::new("pow_midp(x, 0.5)");

    for chunk in test_values.chunks(8) {
        if chunk.len() < 8 {
            continue;
        }
        let arr: &[f32; 8] = chunk.try_into().unwrap();
        let result = simd_pow_midp(token, arr, 0.5);
        for (i, &x) in arr.iter().enumerate() {
            let expected = x.sqrt();
            stats.update(x, expected, result[i]);
        }
    }

    stats.finalize();
    stats.print();
    stats.assert_max_rel_err(1e-5);
}

// ============================================================================
// exp2/log2 tests
// ============================================================================

#[test]
#[cfg(target_arch = "x86_64")]
fn test_exp2_midp_brute_force() {
    let Some(token) = X64V3Token::summon() else {
        eprintln!("AVX2+FMA not available, skipping test");
        return;
    };

    let mut stats = AccuracyStats::new("exp2_midp");

    // Test range: -126 to 126 (full f32 exponent range)
    let test_values: Vec<f32> = (0..1_000_000)
        .map(|i| -126.0 + (i as f32 / 1_000_000.0) * 252.0)
        .collect();

    for chunk in test_values.chunks(8) {
        if chunk.len() < 8 {
            continue;
        }
        let arr: &[f32; 8] = chunk.try_into().unwrap();
        let result = simd_exp2_midp(token, arr);
        for (i, &x) in arr.iter().enumerate() {
            let expected = x.exp2();
            stats.update(x, expected, result[i]);
        }
    }

    stats.finalize();
    stats.print();
    stats.assert_max_rel_err(1e-5);
}

#[test]
#[cfg(target_arch = "x86_64")]
fn test_log2_midp_brute_force() {
    let Some(token) = X64V3Token::summon() else {
        eprintln!("AVX2+FMA not available, skipping test");
        return;
    };

    let mut stats = AccuracyStats::new("log2_midp");

    // Test range: 1e-37 to 1e37 (avoiding denormals near 1e-38)
    // f32 denormal range is below ~1.2e-38
    let test_values: Vec<f32> = (0..1_000_000)
        .map(|i| {
            let t = i as f32 / 1_000_000.0;
            10.0f32.powf(-37.0 + t * 74.0)
        })
        .collect();

    for chunk in test_values.chunks(8) {
        if chunk.len() < 8 {
            continue;
        }
        let arr: &[f32; 8] = chunk.try_into().unwrap();
        let result = simd_log2_midp(token, arr);
        for (i, &x) in arr.iter().enumerate() {
            let expected = x.log2();
            stats.update(x, expected, result[i]);
        }
    }

    stats.finalize();
    stats.print();
    stats.assert_max_rel_err(1e-5);
}

// ============================================================================
// ln/exp tests
// ============================================================================

#[test]
#[cfg(target_arch = "x86_64")]
fn test_ln_midp_brute_force() {
    let Some(token) = X64V3Token::summon() else {
        eprintln!("AVX2+FMA not available, skipping test");
        return;
    };

    let mut stats = AccuracyStats::new("ln_midp");

    // Test range: 1e-37 to 1e37 (avoiding denormals near 1e-38)
    let test_values: Vec<f32> = (0..1_000_000)
        .map(|i| {
            let t = i as f32 / 1_000_000.0;
            10.0f32.powf(-37.0 + t * 74.0)
        })
        .collect();

    for chunk in test_values.chunks(8) {
        if chunk.len() < 8 {
            continue;
        }
        let arr: &[f32; 8] = chunk.try_into().unwrap();
        let result = simd_ln_midp(token, arr);
        for (i, &x) in arr.iter().enumerate() {
            let expected = x.ln();
            stats.update(x, expected, result[i]);
        }
    }

    stats.finalize();
    stats.print();
    stats.assert_max_rel_err(1e-5);
}

#[test]
#[cfg(target_arch = "x86_64")]
fn test_exp_midp_brute_force() {
    let Some(token) = X64V3Token::summon() else {
        eprintln!("AVX2+FMA not available, skipping test");
        return;
    };

    let mut stats = AccuracyStats::new("exp_midp");

    // Test range: -80 to 80 (well within safe range)
    // exp(80) ≈ 5.5e34, giving good margin before overflow at exp(88.7) ≈ 3.4e38
    let test_values: Vec<f32> = (0..1_000_000)
        .map(|i| -80.0 + (i as f32 / 1_000_000.0) * 160.0)
        .collect();

    for chunk in test_values.chunks(8) {
        if chunk.len() < 8 {
            continue;
        }
        let arr: &[f32; 8] = chunk.try_into().unwrap();
        let result = simd_exp_midp(token, arr);
        for (i, &x) in arr.iter().enumerate() {
            let expected = x.exp();
            stats.update(x, expected, result[i]);
        }
    }

    stats.finalize();
    stats.print();
    // exp_midp uses exp2_midp(x * LOG2_E), compound error is ~1.5 ULP
    stats.assert_max_rel_err(2e-5);
}

// ============================================================================
// lowp tests (should have higher error bounds)
// ============================================================================

#[test]
#[cfg(target_arch = "x86_64")]
fn test_lowp_functions_brute_force() {
    let Some(token) = X64V3Token::summon() else {
        eprintln!("AVX2+FMA not available, skipping test");
        return;
    };

    println!("\n=== Low-precision function accuracy ===\n");

    // exp2_lowp
    let mut stats = AccuracyStats::new("exp2_lowp");
    let test_values: Vec<f32> = (0..100_000)
        .map(|i| -20.0 + (i as f32 / 100_000.0) * 40.0)
        .collect();

    for chunk in test_values.chunks(8) {
        if chunk.len() < 8 {
            continue;
        }
        let arr: &[f32; 8] = chunk.try_into().unwrap();
        let result = simd_exp2_lowp(token, arr);
        for (i, &x) in arr.iter().enumerate() {
            let expected = x.exp2();
            stats.update(x, expected, result[i]);
        }
    }
    stats.finalize();
    stats.print();
    stats.assert_max_rel_err(0.01); // 1% max error for lowp

    // log2_lowp
    let mut stats = AccuracyStats::new("log2_lowp");
    let test_values: Vec<f32> = (1..100_000)
        .map(|i| {
            let t = i as f32 / 100_000.0;
            10.0f32.powf(-10.0 + t * 20.0)
        })
        .collect();

    for chunk in test_values.chunks(8) {
        if chunk.len() < 8 {
            continue;
        }
        let arr: &[f32; 8] = chunk.try_into().unwrap();
        let result = simd_log2_lowp(token, arr);
        for (i, &x) in arr.iter().enumerate() {
            let expected = x.log2();
            stats.update(x, expected, result[i]);
        }
    }
    stats.finalize();
    stats.print();
    stats.assert_max_rel_err(0.01);

    // pow_lowp
    let mut stats = AccuracyStats::new("pow_lowp(x, 2.4)");
    let test_values: Vec<f32> = (1..100_000).map(|i| i as f32 / 100_000.0).collect();

    for chunk in test_values.chunks(8) {
        if chunk.len() < 8 {
            continue;
        }
        let arr: &[f32; 8] = chunk.try_into().unwrap();
        let result = simd_pow_lowp(token, arr, 2.4);
        for (i, &x) in arr.iter().enumerate() {
            let expected = x.powf(2.4);
            stats.update(x, expected, result[i]);
        }
    }
    stats.finalize();
    stats.print();
    stats.assert_max_rel_err(0.01);

    // ln_lowp
    let mut stats = AccuracyStats::new("ln_lowp");
    let test_values: Vec<f32> = (1..100_000)
        .map(|i| {
            let t = i as f32 / 100_000.0;
            10.0f32.powf(-10.0 + t * 20.0)
        })
        .collect();

    for chunk in test_values.chunks(8) {
        if chunk.len() < 8 {
            continue;
        }
        let arr: &[f32; 8] = chunk.try_into().unwrap();
        let result = simd_ln_lowp(token, arr);
        for (i, &x) in arr.iter().enumerate() {
            let expected = x.ln();
            stats.update(x, expected, result[i]);
        }
    }
    stats.finalize();
    stats.print();
    stats.assert_max_rel_err(0.01);

    // exp_lowp
    let mut stats = AccuracyStats::new("exp_lowp");
    let test_values: Vec<f32> = (0..100_000)
        .map(|i| -10.0 + (i as f32 / 100_000.0) * 20.0)
        .collect();

    for chunk in test_values.chunks(8) {
        if chunk.len() < 8 {
            continue;
        }
        let arr: &[f32; 8] = chunk.try_into().unwrap();
        let result = simd_exp_lowp(token, arr);
        for (i, &x) in arr.iter().enumerate() {
            let expected = x.exp();
            stats.update(x, expected, result[i]);
        }
    }
    stats.finalize();
    stats.print();
    stats.assert_max_rel_err(0.01);

    // log10_lowp
    let mut stats = AccuracyStats::new("log10_lowp");
    let test_values: Vec<f32> = (1..100_000)
        .map(|i| {
            let t = i as f32 / 100_000.0;
            10.0f32.powf(-10.0 + t * 20.0)
        })
        .collect();

    for chunk in test_values.chunks(8) {
        if chunk.len() < 8 {
            continue;
        }
        let arr: &[f32; 8] = chunk.try_into().unwrap();
        let result = simd_log10_lowp(token, arr);
        for (i, &x) in arr.iter().enumerate() {
            let expected = x.log10();
            stats.update(x, expected, result[i]);
        }
    }
    stats.finalize();
    stats.print();
    stats.assert_max_rel_err(0.01);
}
