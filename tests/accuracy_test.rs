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
fn simd_cbrt_lowp(token: X64V3Token, input: &[f32; 8]) -> [f32; 8] {
    f32x8::load(token, input).cbrt_lowp().to_array()
}

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
// ULP helpers
// ============================================================================

/// Compute ULP distance between two f32 values.
/// Returns None if either is NaN.
fn ulp_distance(a: f32, b: f32) -> Option<u32> {
    if a.is_nan() || b.is_nan() {
        return None;
    }
    if a == b {
        return Some(0);
    }
    // Convert sign-magnitude to two's complement-like ordering for ULP distance
    let ai = a.to_bits() as i32;
    let bi = b.to_bits() as i32;
    let ai = if ai < 0 { i32::MIN - ai } else { ai };
    let bi = if bi < 0 { i32::MIN - bi } else { bi };
    Some((ai - bi).unsigned_abs())
}

struct UlpStats {
    name: &'static str,
    max_ulp: u32,
    max_ulp_input: f32,
    max_ulp_expected: f32,
    max_ulp_got: f32,
    total_tested: usize,
    nan_count: usize,
    inf_count: usize,
    ulp_histogram: [usize; 8], // [0 ulp, 1 ulp, 2 ulp, 3 ulp, 4-7, 8-15, 16-63, 64+]
    max_rel_err: f64,
    avg_rel_err: f64,
}

impl UlpStats {
    fn new(name: &'static str) -> Self {
        Self {
            name,
            max_ulp: 0,
            max_ulp_input: 0.0,
            max_ulp_expected: 0.0,
            max_ulp_got: 0.0,
            total_tested: 0,
            nan_count: 0,
            inf_count: 0,
            ulp_histogram: [0; 8],
            max_rel_err: 0.0,
            avg_rel_err: 0.0,
        }
    }

    fn update(&mut self, input: f32, expected: f32, got: f32) {
        if got.is_nan() && !expected.is_nan() {
            self.nan_count += 1;
            return;
        }
        if got.is_infinite() && !expected.is_infinite() {
            self.inf_count += 1;
            return;
        }
        if !expected.is_finite() || !got.is_finite() {
            return;
        }

        self.total_tested += 1;

        // Relative error
        let abs_err = (got - expected).abs() as f64;
        let rel_err = if expected.abs() > 1e-38 {
            abs_err / expected.abs() as f64
        } else {
            abs_err
        };
        self.avg_rel_err += rel_err;
        if rel_err > self.max_rel_err {
            self.max_rel_err = rel_err;
        }

        // ULP distance
        if let Some(ulps) = ulp_distance(expected, got) {
            let bucket = match ulps {
                0 => 0,
                1 => 1,
                2 => 2,
                3 => 3,
                4..=7 => 4,
                8..=15 => 5,
                16..=63 => 6,
                _ => 7,
            };
            self.ulp_histogram[bucket] += 1;

            if ulps > self.max_ulp {
                self.max_ulp = ulps;
                self.max_ulp_input = input;
                self.max_ulp_expected = expected;
                self.max_ulp_got = got;
            }
        }
    }

    fn finalize(&mut self) {
        if self.total_tested > 0 {
            self.avg_rel_err /= self.total_tested as f64;
        }
    }

    fn print(&self) {
        println!(
            "{:20} max_ulp: {:4}  max_rel: {:.2e}  avg_rel: {:.2e}  tested: {}",
            self.name, self.max_ulp, self.max_rel_err, self.avg_rel_err, self.total_tested
        );
        println!(
            "    ULP histogram: 0:{} 1:{} 2:{} 3:{} 4-7:{} 8-15:{} 16-63:{} 64+:{}",
            self.ulp_histogram[0],
            self.ulp_histogram[1],
            self.ulp_histogram[2],
            self.ulp_histogram[3],
            self.ulp_histogram[4],
            self.ulp_histogram[5],
            self.ulp_histogram[6],
            self.ulp_histogram[7],
        );
        if self.max_ulp > 3 {
            println!(
                "    worst: input={:e} ({:#010x}) expected={:e} ({:#010x}) got={:e} ({:#010x})",
                self.max_ulp_input,
                self.max_ulp_input.to_bits(),
                self.max_ulp_expected,
                self.max_ulp_expected.to_bits(),
                self.max_ulp_got,
                self.max_ulp_got.to_bits(),
            );
        }
        if self.nan_count > 0 || self.inf_count > 0 {
            println!(
                "    ERRORS: {} unexpected NaN, {} unexpected Inf",
                self.nan_count, self.inf_count
            );
        }
    }

    fn assert_max_ulp(&self, max_allowed: u32) {
        assert!(
            self.max_ulp <= max_allowed,
            "{}: max_ulp {} exceeds limit {} at input={:e} ({:#010x})",
            self.name,
            self.max_ulp,
            max_allowed,
            self.max_ulp_input,
            self.max_ulp_input.to_bits(),
        );
        assert!(
            self.nan_count == 0,
            "{}: {} unexpected NaN values",
            self.name,
            self.nan_count,
        );
    }
}

// ============================================================================
// cbrt tests — comprehensive with ULP measurement
// ============================================================================

/// Generate test vectors covering normal, denormal, edge, and special values.
fn cbrt_test_vectors() -> Vec<f32> {
    let mut vals = Vec::with_capacity(5_000_000);

    // 1. Normal range: logarithmic sweep 1e-37 to 1e37
    for i in 0..1_000_000 {
        let t = i as f32 / 1_000_000.0;
        vals.push(10.0f32.powf(-37.0 + t * 74.0));
    }

    // 2. Negative normal range
    for i in 0..1_000_000 {
        let t = i as f32 / 1_000_000.0;
        vals.push(-10.0f32.powf(-37.0 + t * 74.0));
    }

    // 3. Near-zero normal values
    for i in -100_000..100_000i32 {
        let v = i as f32 * 1e-10;
        if v != 0.0 {
            vals.push(v);
        }
    }

    // 4. Denormals (smallest normal is 1.17549435e-38)
    // Positive denormals
    for i in 1..100_000u32 {
        vals.push(f32::from_bits(i)); // smallest denormals
    }
    for i in 1..100_000u32 {
        vals.push(f32::from_bits(0x0080_0000 - i)); // largest denormals (just below normal)
    }
    // Negative denormals
    for i in 1..100_000u32 {
        vals.push(f32::from_bits(0x8000_0000 | i));
    }
    for i in 1..100_000u32 {
        vals.push(f32::from_bits(0x8000_0000 | (0x0080_0000 - i)));
    }

    // 5. Special values
    vals.push(0.0);
    vals.push(-0.0);
    vals.push(f32::INFINITY);
    vals.push(f32::NEG_INFINITY);
    vals.push(f32::NAN);
    vals.push(f32::MIN_POSITIVE); // smallest normal
    vals.push(-f32::MIN_POSITIVE);
    vals.push(f32::MAX);
    vals.push(f32::MIN);
    vals.push(1.0);
    vals.push(-1.0);
    vals.push(8.0);
    vals.push(27.0);
    vals.push(0.125); // 1/8 = 0.5^3

    // 6. Perfect cubes
    for i in -100..=100i32 {
        if i != 0 {
            vals.push((i as f32).powi(3));
        }
    }

    vals
}

/// Run cbrt variant over test vectors, comparing against std::f32::cbrt().
fn run_cbrt_test(
    name: &'static str,
    token: X64V3Token,
    func: fn(X64V3Token, &[f32; 8]) -> [f32; 8],
    vals: &[f32],
) -> UlpStats {
    let mut stats = UlpStats::new(name);

    for chunk in vals.chunks(8) {
        if chunk.len() < 8 {
            continue;
        }
        let arr: &[f32; 8] = chunk.try_into().unwrap();
        let result = func(token, arr);
        for (i, &x) in arr.iter().enumerate() {
            let expected = x.cbrt();
            stats.update(x, expected, result[i]);
        }
    }

    stats.finalize();
    stats
}

/// Compare cbrt_lowp against cbrt_midp (not just std::cbrt).
fn run_cbrt_vs_midp(
    name: &'static str,
    token: X64V3Token,
    func: fn(X64V3Token, &[f32; 8]) -> [f32; 8],
    vals: &[f32],
) -> UlpStats {
    let mut stats = UlpStats::new(name);

    for chunk in vals.chunks(8) {
        if chunk.len() < 8 {
            continue;
        }
        let arr: &[f32; 8] = chunk.try_into().unwrap();
        let result = func(token, arr);
        let midp_result = simd_cbrt_midp(token, arr);
        for i in 0..8 {
            stats.update(arr[i], midp_result[i], result[i]);
        }
    }

    stats.finalize();
    stats
}

#[test]
#[cfg(target_arch = "x86_64")]
fn test_cbrt_all_variants_comprehensive() {
    let Some(token) = X64V3Token::summon() else {
        eprintln!("AVX2+FMA not available, skipping test");
        return;
    };

    let vals = cbrt_test_vectors();

    // Working range: normal nonzero finite values within 1e-37..1e37.
    // This is where image/color processing operates. Excludes:
    // - Zero (bit hack on 0 bits produces garbage; use _precise variant)
    // - Denormals (< MIN_POSITIVE; use _precise variant)
    // - Inf/NaN (not meaningful for cbrt)
    // - Extreme values near f32::MAX (Newton formulation in midp overflows
    //   because 3*y³ > f32::MAX; Halley formulation avoids this)
    let working_range: Vec<f32> = vals
        .iter()
        .copied()
        .filter(|x| x.is_finite() && x.abs() >= f32::MIN_POSITIVE && x.abs() <= 1e37)
        .collect();

    // Extreme values (>1e37): Newton formulation in midp overflows because
    // 3*y³ > f32::MAX. Halley formulation avoids this since ratio stays ~1.0.
    // Pad to 8 values so chunks work.
    let mut extremes: Vec<f32> = vals
        .iter()
        .copied()
        .filter(|x| x.is_finite() && x.abs() > 1e37)
        .collect();
    while extremes.len() % 8 != 0 {
        extremes.push(1e38); // pad
    }

    println!(
        "\n=== cbrt vs std::f32::cbrt() — working range ({} values, 1e-37..1e37) ===\n",
        working_range.len()
    );

    let midp_stats = run_cbrt_test("cbrt_midp", token, simd_cbrt_midp, &working_range);
    midp_stats.print();
    midp_stats.assert_max_ulp(4);

    let lowp_stats = run_cbrt_test("cbrt_lowp", token, simd_cbrt_lowp, &working_range);
    lowp_stats.print();
    // lowp has ~15 bits precision, so expect up to ~256 ULP
    lowp_stats.assert_max_ulp(512);

    println!("\n=== cbrt_lowp vs cbrt_midp (parity check, working range) ===\n");

    let lowp_vs_midp = run_cbrt_vs_midp("lowp vs midp", token, simd_cbrt_lowp, &working_range);
    lowp_vs_midp.print();

    // Full range including extremes — informational (Newton overflow at f32::MAX)
    println!("\n=== cbrt on extreme values near f32::MAX (informational) ===\n");
    let extremes: Vec<f32> = vals
        .iter()
        .copied()
        .filter(|x| x.is_finite() && x.abs() > 1e37)
        .collect();
    if !extremes.is_empty() {
        let midp_ext = run_cbrt_test("midp (extreme)", token, simd_cbrt_midp, &extremes);
        midp_ext.print();
        let lowp_ext = run_cbrt_test("lowp (extreme)", token, simd_cbrt_lowp, &extremes);
        lowp_ext.print();
    }

    // Denormals — informational (none of the base variants handle these)
    println!("\n=== cbrt on denormals (informational, no assertion) ===\n");
    let denormals_only: Vec<f32> = vals
        .iter()
        .copied()
        .filter(|x| x.is_finite() && *x != 0.0 && x.abs() < f32::MIN_POSITIVE)
        .collect();
    if !denormals_only.is_empty() {
        let midp_denorm = run_cbrt_test("midp (denorm)", token, simd_cbrt_midp, &denormals_only);
        midp_denorm.print();
        let lowp_denorm = run_cbrt_test("lowp (denorm)", token, simd_cbrt_lowp, &denormals_only);
        lowp_denorm.print();
    }

    println!("\n=== Edge case spot checks ===\n");

    // Zero handling
    let zeros = [0.0f32, -0.0, 0.0, -0.0, 0.0, -0.0, 0.0, -0.0];
    let lowp_z = simd_cbrt_lowp(token, &zeros);
    let midp_z = simd_cbrt_midp(token, &zeros);
    println!(
        "cbrt(0.0):  lowp={:e}  midp={:e}  std={:e}",
        lowp_z[0],
        midp_z[0],
        0.0f32.cbrt()
    );
    println!(
        "cbrt(-0.0): lowp={:e}  midp={:e}  std={:e}",
        lowp_z[1],
        midp_z[1],
        (-0.0f32).cbrt()
    );

    // NaN handling
    let nans = [f32::NAN; 8];
    let lowp_n = simd_cbrt_lowp(token, &nans);
    let midp_n = simd_cbrt_midp(token, &nans);
    println!(
        "cbrt(NaN):  lowp={}  midp={}  (should be NaN)",
        lowp_n[0].is_nan(),
        midp_n[0].is_nan()
    );

    // Inf handling
    let infs = [
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::INFINITY,
        f32::NEG_INFINITY,
    ];
    let lowp_i = simd_cbrt_lowp(token, &infs);
    let midp_i = simd_cbrt_midp(token, &infs);
    println!(
        "cbrt(+inf): lowp={:e}  midp={:e}  std={:e}",
        lowp_i[0],
        midp_i[0],
        f32::INFINITY.cbrt()
    );
    println!(
        "cbrt(-inf): lowp={:e}  midp={:e}  std={:e}",
        lowp_i[1],
        midp_i[1],
        f32::NEG_INFINITY.cbrt()
    );

    // Denormal spot check
    let denorms = [
        1e-40f32,
        1e-42,
        1e-44,
        f32::from_bits(1),
        f32::from_bits(100),
        f32::from_bits(10000),
        1e-39,
        1e-41,
    ];
    let midp_d = simd_cbrt_midp(token, &denorms);
    let lowp_d = simd_cbrt_lowp(token, &denorms);
    println!("\nDenormal inputs (no denormal handling in any base variant):");
    for i in 0..4 {
        let expected = denorms[i].cbrt();
        println!(
            "  cbrt({:e}): std={:e}  midp={:e}({}ulp)  lowp={:e}({}ulp)",
            denorms[i],
            expected,
            midp_d[i],
            ulp_distance(expected, midp_d[i]).map_or("NaN".to_string(), |u| u.to_string()),
            lowp_d[i],
            ulp_distance(expected, lowp_d[i]).map_or("NaN".to_string(), |u| u.to_string()),
        );
    }
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
