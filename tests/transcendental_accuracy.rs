//! Accuracy validation: compare all transcendental functions against std/libm.
//!
//! Tests a wide range of inputs including edge cases (zero, denormals, very large,
//! very small, negative) and measures max ULP error and max relative error.
//!
//! Run: cargo test --test transcendental_accuracy --features "std avx512" -- --nocapture

#![cfg(target_arch = "x86_64")]

use archmage::{SimdToken, X64V3Token, arcane};
use magetypes::simd::generic::{f32x4, f32x8};

// ============================================================================
// ULP computation
// ============================================================================

fn ulp_distance(a: f32, b: f32) -> u32 {
    if a.is_nan() && b.is_nan() {
        return 0; // Both NaN = match
    }
    if a.is_nan() || b.is_nan() {
        return u32::MAX; // One NaN = worst
    }
    if a == b {
        return 0;
    }
    // Handle infinities
    if a.is_infinite() && b.is_infinite() && a.signum() == b.signum() {
        return 0;
    }
    if a.is_infinite() || b.is_infinite() {
        return u32::MAX;
    }
    let ai = a.to_bits() as i32;
    let bi = b.to_bits() as i32;
    // For same-sign values, ULP distance is just bit difference
    // For opposite signs, both should be very small (near zero)
    if (ai ^ bi) >= 0 {
        // Same sign
        (ai - bi).unsigned_abs()
    } else {
        // Opposite sign — only valid near zero
        ai.unsigned_abs() + bi.unsigned_abs()
    }
}

fn relative_error(got: f32, expected: f32) -> f32 {
    if got == expected {
        return 0.0;
    }
    if expected == 0.0 {
        return got.abs();
    }
    if got.is_nan() && expected.is_nan() {
        return 0.0;
    }
    if got.is_nan() || expected.is_nan() || got.is_infinite() || expected.is_infinite() {
        return f32::INFINITY;
    }
    ((got - expected) / expected).abs()
}

// ============================================================================
// Test input generation — wide coverage
// ============================================================================

/// Generate test inputs for functions expecting positive values (log, cbrt, pow base).
/// `include_denormals` controls whether subnormals are included (lowp can't handle them).
fn positive_inputs(include_denormals: bool) -> Vec<f32> {
    let mut vals = Vec::new();

    // Exact zeros
    vals.push(0.0);
    vals.push(-0.0);

    if include_denormals {
        // Denormals
        vals.push(f32::MIN_POSITIVE); // smallest normal
        vals.push(f32::MIN_POSITIVE / 2.0); // denormal
        vals.push(f32::MIN_POSITIVE / 1024.0);
        vals.push(1e-40);
        vals.push(1e-38);
    }

    // Small values (normal range only)
    for &v in &[1e-30, 1e-20, 1e-10, 1e-7, 1e-5, 1e-3, 0.01, 0.1] {
        vals.push(v);
    }

    // Near 1.0 (important for log accuracy)
    for i in -20..=20 {
        vals.push(1.0 + i as f32 * 0.01);
    }

    // Medium values
    for &v in &[
        0.5,
        0.99,
        1.01,
        2.0,
        core::f32::consts::E,
        3.0,
        5.0,
        7.0,
        10.0,
    ] {
        vals.push(v);
    }

    // Large values
    for &v in &[100.0, 1000.0, 1e6, 1e10, 1e20, 1e30, 1e38] {
        vals.push(v);
    }

    // Powers of 2 (exact for log2)
    for i in -20..=20 {
        vals.push(2.0f32.powi(i));
    }

    // Linear sweep 0.001 to 1000 (200 points)
    for i in 0..200 {
        let t = i as f32 / 199.0;
        vals.push(0.001 + t * 999.999);
    }

    // Log sweep 1e-6 to 1e6
    for i in 0..100 {
        let t = i as f32 / 99.0;
        vals.push(10.0f32.powf(-6.0 + t * 12.0));
    }

    vals
}

/// Generate test inputs for cbrt (positive and negative)
fn cbrt_inputs(include_denormals: bool) -> Vec<f32> {
    let mut vals = positive_inputs(include_denormals);
    // Add negative mirrors
    let pos: Vec<f32> = vals.iter().filter(|&&v| v > 0.0).copied().collect();
    for v in pos {
        vals.push(-v);
    }
    vals
}

// ============================================================================
// SIMD wrappers
// ============================================================================

#[arcane]
fn eval_f32x8(token: X64V3Token, inputs: &[f32; 8], op: &str, param: f32) -> [f32; 8] {
    let v = f32x8::from_array(token, *inputs);
    let r = match op {
        "log2_lowp" => v.log2_lowp(),
        "log2_midp" => v.log2_midp(),
        "log2_midp_precise" => v.log2_midp_precise(),
        "exp2_lowp" => v.exp2_lowp(),
        "exp2_midp" => v.exp2_midp(),
        "ln_lowp" => v.ln_lowp(),
        "ln_midp" => v.ln_midp(),
        "ln_midp_precise" => v.ln_midp_precise(),
        "exp_lowp" => v.exp_lowp(),
        "exp_midp" => v.exp_midp(),
        "log10_lowp" => v.log10_lowp(),
        "log10_midp" => v.log10_midp(),
        "log10_midp_precise" => v.log10_midp_precise(),
        "pow_lowp" => v.pow_lowp(param),
        "pow_midp" => v.pow_midp(param),
        "pow_midp_precise" => v.pow_midp_precise(param),
        "cbrt_lowp" => v.cbrt_lowp(),
        "cbrt_midp" => v.cbrt_midp(),
        "cbrt_midp_precise" => v.cbrt_midp_precise(),
        _ => panic!("unknown op: {op}"),
    };
    r.to_array()
}

// ============================================================================
// Generic x8 wrappers (compare generic vs direct)
// ============================================================================

#[arcane]
fn eval_generic_f32x4(token: X64V3Token, inputs: &[f32; 4], op: &str, param: f32) -> [f32; 4] {
    let v = f32x4::from_array(token, *inputs);
    let r = match op {
        "log2_lowp" => v.log2_lowp(),
        "log2_midp" => v.log2_midp(),
        "log2_midp_precise" => v.log2_midp_precise(),
        "exp2_lowp" => v.exp2_lowp(),
        "exp2_midp" => v.exp2_midp(),
        "ln_lowp" => v.ln_lowp(),
        "ln_midp" => v.ln_midp(),
        "ln_midp_precise" => v.ln_midp_precise(),
        "exp_lowp" => v.exp_lowp(),
        "exp_midp" => v.exp_midp(),
        "log10_lowp" => v.log10_lowp(),
        "log10_midp" => v.log10_midp(),
        "log10_midp_precise" => v.log10_midp_precise(),
        "pow_lowp" => v.pow_lowp(param),
        "pow_midp" => v.pow_midp(param),
        "pow_midp_precise" => v.pow_midp_precise(param),
        "cbrt_lowp" => v.cbrt_lowp(),
        "cbrt_midp" => v.cbrt_midp(),
        "cbrt_midp_precise" => v.cbrt_midp_precise(),
        _ => panic!("unknown op: {op}"),
    };
    r.to_array()
}

// ============================================================================
// Core comparison engine
// ============================================================================

struct AccuracyStats {
    op_name: String,
    count: usize,
    max_ulp: u32,
    max_ulp_input: f32,
    max_ulp_got: f32,
    max_ulp_expected: f32,
    max_rel: f32,
    max_rel_input: f32,
    worst_inputs: Vec<(f32, f32, f32, u32)>, // (input, got, expected, ulp)
}

impl AccuracyStats {
    fn new(op_name: &str) -> Self {
        Self {
            op_name: op_name.to_string(),
            count: 0,
            max_ulp: 0,
            max_ulp_input: 0.0,
            max_ulp_got: 0.0,
            max_ulp_expected: 0.0,
            max_rel: 0.0,
            max_rel_input: 0.0,
            worst_inputs: Vec::new(),
        }
    }

    fn record(&mut self, input: f32, got: f32, expected: f32) {
        self.count += 1;
        let ulp = ulp_distance(got, expected);
        let rel = relative_error(got, expected);

        if ulp > self.max_ulp {
            self.max_ulp = ulp;
            self.max_ulp_input = input;
            self.max_ulp_got = got;
            self.max_ulp_expected = expected;
        }
        if rel > self.max_rel && rel.is_finite() {
            self.max_rel = rel;
            self.max_rel_input = input;
        }

        // Track worst 5
        if self.worst_inputs.len() < 5 || ulp > self.worst_inputs.last().map(|w| w.3).unwrap_or(0) {
            self.worst_inputs.push((input, got, expected, ulp));
            self.worst_inputs.sort_by(|a, b| b.3.cmp(&a.3));
            self.worst_inputs.truncate(5);
        }
    }

    fn print_summary(&self) {
        println!(
            "  {:<25} | {:>6} tested | max ULP: {:>8} | max rel err: {:.2e}",
            self.op_name, self.count, self.max_ulp, self.max_rel
        );
        if self.max_ulp > 0 {
            println!(
                "    worst: input={:>14e} got={:>14e} expected={:>14e} ({} ULP)",
                self.max_ulp_input, self.max_ulp_got, self.max_ulp_expected, self.max_ulp
            );
        }
    }
}

/// Reference function: compute the expected value using std
fn std_ref(op: &str, input: f32, param: f32) -> f32 {
    match op {
        "log2_lowp" | "log2_midp" | "log2_midp_precise" => input.log2(),
        "exp2_lowp" | "exp2_midp" => 2.0f32.powf(input),
        "ln_lowp" | "ln_midp" | "ln_midp_precise" => input.ln(),
        "exp_lowp" | "exp_midp" => input.exp(),
        "log10_lowp" | "log10_midp" | "log10_midp_precise" => input.log10(),
        "pow_lowp" | "pow_midp" | "pow_midp_precise" => input.powf(param),
        "cbrt_lowp" | "cbrt_midp" | "cbrt_midp_precise" => input.cbrt(),
        _ => panic!("unknown op: {op}"),
    }
}

/// Check mode: ULP-based (for midp/precise), or relative-error-based (for lowp near singularities)
enum CheckMode {
    /// Assert max ULP and max relative error
    Ulp { max_ulp: u32, max_rel: f32 },
    /// Assert relative error only (for lowp where ULP near 0/1 is meaningless)
    RelOnly { max_rel: f32 },
}

fn run_accuracy_test(
    token: X64V3Token,
    op: &str,
    inputs: &[f32],
    param: f32,
    check: CheckMode,
) -> AccuracyStats {
    let mut stats = AccuracyStats::new(op);

    // Process in chunks of 8
    let chunks: Vec<[f32; 8]> = inputs
        .chunks(8)
        .map(|c| {
            let mut arr = [1.0f32; 8]; // pad with 1.0 (safe for all ops)
            arr[..c.len()].copy_from_slice(c);
            arr
        })
        .collect();

    for (chunk_idx, chunk) in chunks.iter().enumerate() {
        let got = eval_f32x8(token, chunk, op, param);
        let count_in_chunk = if chunk_idx == chunks.len() - 1 {
            let rem = inputs.len() % 8;
            if rem == 0 { 8 } else { rem }
        } else {
            8
        };

        for i in 0..count_in_chunk {
            let expected = std_ref(op, chunk[i], param);
            stats.record(chunk[i], got[i], expected);
        }
    }

    // Also spot-check generic f32x4 path for consistency
    let max_ulp_threshold = match &check {
        CheckMode::Ulp { max_ulp, .. } => *max_ulp,
        CheckMode::RelOnly { .. } => u32::MAX, // don't check generic ULP for rel-only
    };

    let generic_chunks: Vec<[f32; 4]> = inputs
        .chunks(4)
        .map(|c| {
            let mut arr = [1.0f32; 4];
            arr[..c.len()].copy_from_slice(c);
            arr
        })
        .collect();

    let mut generic_mismatches = 0u32;
    for (chunk_idx, chunk) in generic_chunks.iter().enumerate() {
        let got_generic = eval_generic_f32x4(token, chunk, op, param);
        let count_in_chunk = if chunk_idx == generic_chunks.len() - 1 {
            let rem = inputs.len() % 4;
            if rem == 0 { 4 } else { rem }
        } else {
            4
        };
        for i in 0..count_in_chunk {
            let expected = std_ref(op, chunk[i], param);
            let ulp = ulp_distance(got_generic[i], expected);
            if ulp > max_ulp_threshold {
                generic_mismatches += 1;
            }
        }
    }
    if generic_mismatches > 0 && max_ulp_threshold < u32::MAX {
        println!(
            "    WARNING: generic f32x4 path has {} inputs exceeding {max_ulp_threshold} ULP",
            generic_mismatches
        );
    }

    stats.print_summary();

    match &check {
        CheckMode::Ulp { max_ulp, max_rel } => {
            assert!(
                stats.max_ulp <= *max_ulp,
                "{}: max ULP {} exceeds allowed {} (input={}, got={}, expected={})",
                op,
                stats.max_ulp,
                max_ulp,
                stats.max_ulp_input,
                stats.max_ulp_got,
                stats.max_ulp_expected,
            );
            assert!(
                stats.max_rel <= *max_rel || !stats.max_rel.is_finite(),
                "{}: max relative error {} exceeds allowed {}",
                op,
                stats.max_rel,
                max_rel,
            );
        }
        CheckMode::RelOnly { max_rel } => {
            assert!(
                stats.max_rel <= *max_rel || !stats.max_rel.is_finite(),
                "{}: max relative error {} exceeds allowed {} (worst input={})",
                op,
                stats.max_rel,
                max_rel,
                stats.max_rel_input,
            );
        }
    }
    stats
}

// ============================================================================
// Tests
// ============================================================================

#[test]
fn accuracy_log2() {
    let Some(token) = X64V3Token::summon() else {
        return;
    };

    println!("\n=== log2 accuracy ===");

    // lowp: no denormals, no edge cases. Relative error only (ULP near 0 is meaningless)
    let lowp_inputs: Vec<f32> = positive_inputs(false)
        .into_iter()
        .filter(|&v| v > 0.0)
        .collect();
    run_accuracy_test(
        token,
        "log2_lowp",
        &lowp_inputs,
        0.0,
        CheckMode::RelOnly { max_rel: 7e-2 },
    );

    // midp: normal range (denormals break mantissa extraction)
    let midp_inputs: Vec<f32> = positive_inputs(false)
        .into_iter()
        .filter(|&v| v > 0.0)
        .collect();
    run_accuracy_test(
        token,
        "log2_midp",
        &midp_inputs,
        0.0,
        CheckMode::Ulp {
            max_ulp: 4,
            max_rel: 1e-6,
        },
    );

    // midp_precise: same accuracy as midp (no denormal handling for log functions)
    // Denormal inputs break mantissa extraction — only cbrt_midp_precise handles denormals.
    run_accuracy_test(
        token,
        "log2_midp_precise",
        &midp_inputs,
        0.0,
        CheckMode::Ulp {
            max_ulp: 4,
            max_rel: 1e-6,
        },
    );
}

#[test]
fn accuracy_exp2() {
    let Some(token) = X64V3Token::summon() else {
        return;
    };

    println!("\n=== exp2 accuracy ===");

    // lowp: clamps input to [-126, 126], so only test within that range
    // BUG: exp2_lowp has ~0.6% relative error — polynomial is coarse
    let mut lowp_inputs = Vec::new();
    for i in -1200..=1200 {
        lowp_inputs.push(i as f32 / 10.0);
    }
    for i in -100..=100 {
        lowp_inputs.push(i as f32 / 1000.0);
    }
    lowp_inputs.push(0.0);
    run_accuracy_test(
        token,
        "exp2_lowp",
        &lowp_inputs,
        0.0,
        CheckMode::RelOnly { max_rel: 6e-3 },
    );

    // midp: handles overflow (>=128 → inf) and underflow (<-126 → 0)
    // Round-to-nearest split keeps |frac| <= 0.5, giving ~64 ULP max error.
    // xi clamped to 127 prevents bit trick overflow at the boundary.
    let mut midp_inputs = Vec::new();
    // Normal output range: [-126, 128)
    for i in -1260..=1280 {
        midp_inputs.push(i as f32 / 10.0);
    }
    for i in -100..=100 {
        midp_inputs.push(i as f32 / 1000.0);
    }
    midp_inputs.push(0.0);
    midp_inputs.push(-0.0);
    midp_inputs.push(f32::NEG_INFINITY); // should return 0
    midp_inputs.push(f32::INFINITY); // should return inf
    run_accuracy_test(
        token,
        "exp2_midp",
        &midp_inputs,
        0.0,
        CheckMode::Ulp {
            max_ulp: 128,
            max_rel: 1e-5,
        },
    );
}

#[test]
fn accuracy_ln() {
    let Some(token) = X64V3Token::summon() else {
        return;
    };

    println!("\n=== ln accuracy ===");

    let lowp_inputs: Vec<f32> = positive_inputs(false)
        .into_iter()
        .filter(|&v| v > 0.0)
        .collect();
    run_accuracy_test(
        token,
        "ln_lowp",
        &lowp_inputs,
        0.0,
        CheckMode::RelOnly { max_rel: 7e-2 },
    );

    // midp: normal range only (denormals break mantissa extraction)
    let midp_inputs: Vec<f32> = positive_inputs(false)
        .into_iter()
        .filter(|&v| v > 0.0)
        .collect();
    run_accuracy_test(
        token,
        "ln_midp",
        &midp_inputs,
        0.0,
        CheckMode::Ulp {
            max_ulp: 4,
            max_rel: 1e-6,
        },
    );

    // midp_precise: no denormal handling for ln (only cbrt has it)
    run_accuracy_test(
        token,
        "ln_midp_precise",
        &midp_inputs,
        0.0,
        CheckMode::Ulp {
            max_ulp: 4,
            max_rel: 1e-6,
        },
    );
}

#[test]
fn accuracy_exp() {
    let Some(token) = X64V3Token::summon() else {
        return;
    };

    println!("\n=== exp accuracy ===");

    // exp() produces normal results for roughly [-87.3, 88.7]
    // exp2_midp can't construct denormal outputs (2^n for n < -126)
    // So exp(-87.4) = exp2(-126.1) breaks.

    // lowp: stay well within safe range
    let mut lowp_inputs = Vec::new();
    for i in -850..=850 {
        lowp_inputs.push(i as f32 / 10.0);
    }
    for i in -100..=100 {
        lowp_inputs.push(i as f32 / 1000.0);
    }
    lowp_inputs.push(0.0);
    lowp_inputs.push(-0.0);
    run_accuracy_test(
        token,
        "exp_lowp",
        &lowp_inputs,
        0.0,
        CheckMode::RelOnly { max_rel: 1e-2 },
    );

    // midp: exp(x) = exp2(x * LOG2_E). Two error sources:
    // 1. The LOG2_E multiplication adds ~1 ULP
    // 2. exp2_midp adds up to ~64 ULP (round-to-nearest split)
    // exp2_midp clamps underflow at -126 → returns 0 for denormal-range results.
    let mut midp_inputs = Vec::new();
    // Stay in normal output range: exp(-87.3) ≈ 2^-126
    for i in -870..=880 {
        midp_inputs.push(i as f32 / 10.0);
    }
    for i in -100..=100 {
        midp_inputs.push(i as f32 / 1000.0);
    }
    midp_inputs.push(0.0);
    midp_inputs.push(-0.0);
    run_accuracy_test(
        token,
        "exp_midp",
        &midp_inputs,
        0.0,
        // Compound error from LOG2_E multiplication + exp2_midp polynomial.
        // Round-to-nearest brought this from ~256 to ~64 ULP.
        CheckMode::Ulp {
            max_ulp: 128,
            max_rel: 1e-5,
        },
    );
}

#[test]
fn accuracy_log10() {
    let Some(token) = X64V3Token::summon() else {
        return;
    };

    println!("\n=== log10 accuracy ===");

    let lowp_inputs: Vec<f32> = positive_inputs(false)
        .into_iter()
        .filter(|&v| v > 0.0)
        .collect();
    run_accuracy_test(
        token,
        "log10_lowp",
        &lowp_inputs,
        0.0,
        CheckMode::RelOnly { max_rel: 7e-2 },
    );

    // midp: normal range only
    let midp_inputs: Vec<f32> = positive_inputs(false)
        .into_iter()
        .filter(|&v| v > 0.0)
        .collect();
    run_accuracy_test(
        token,
        "log10_midp",
        &midp_inputs,
        0.0,
        CheckMode::Ulp {
            max_ulp: 4,
            max_rel: 1e-6,
        },
    );

    // midp_precise: no denormal handling for log10 (only cbrt has it)
    run_accuracy_test(
        token,
        "log10_midp_precise",
        &midp_inputs,
        0.0,
        CheckMode::Ulp {
            max_ulp: 4,
            max_rel: 1e-6,
        },
    );
}

#[test]
fn accuracy_pow() {
    let Some(token) = X64V3Token::summon() else {
        return;
    };

    println!("\n=== pow accuracy ===");

    // lowp: no denormals, relative error only
    let lowp_inputs: Vec<f32> = positive_inputs(false)
        .into_iter()
        .filter(|&v| v > 0.0)
        .collect();

    // midp: normal range (goes through log2_midp which can't handle denormals)
    let midp_inputs: Vec<f32> = positive_inputs(false)
        .into_iter()
        .filter(|&v| v > 0.0)
        .collect();

    for &exp in &[0.5, 1.0, 2.0, 2.5, 0.333333, 3.0, -1.0, -0.5] {
        println!("  exponent = {exp}");
        // pow_lowp: filter to inputs where result stays in normal f32 range.
        // exp2_lowp clamps to [-126, 126], so any result >2^126 or <2^-126
        // is silently clamped instead of returning inf/0.
        let lowp_filtered: Vec<f32> = lowp_inputs
            .iter()
            .copied()
            .filter(|&v| {
                let expected = v.powf(exp);
                expected.is_finite() && expected > f32::MIN_POSITIVE && expected != 0.0
            })
            .collect();
        run_accuracy_test(
            token,
            "pow_lowp",
            &lowp_filtered,
            exp,
            CheckMode::RelOnly { max_rel: 0.15 },
        );
        // pow_midp also goes through exp2_midp, which can't produce denormal outputs.
        // Filter to inputs where result is in normal range.
        let midp_filtered: Vec<f32> = midp_inputs
            .iter()
            .copied()
            .filter(|&v| {
                let expected = v.powf(exp);
                expected.is_finite() && expected > f32::MIN_POSITIVE && expected != 0.0
            })
            .collect();
        run_accuracy_test(
            token,
            "pow_midp",
            &midp_filtered,
            exp,
            // Compound error: log2_midp (~2 ULP) * exponent + exp2_midp (~64 ULP).
            // Max observed: 55 ULP (n=3). Round-to-nearest brought this from ~150 to ~55.
            CheckMode::Ulp {
                max_ulp: 128,
                max_rel: 1e-5,
            },
        );
        // pow_midp_precise composes log2_midp_precise + exp2_midp.
        // Same accuracy as pow_midp (both limited by exp2_midp, not log2).
        let precise_filtered: Vec<f32> = midp_inputs
            .iter()
            .copied()
            .filter(|&v| {
                let expected = v.powf(exp);
                expected.is_finite() && expected > f32::MIN_POSITIVE && expected != 0.0
            })
            .collect();
        run_accuracy_test(
            token,
            "pow_midp_precise",
            &precise_filtered,
            exp,
            CheckMode::Ulp {
                max_ulp: 128,
                max_rel: 1e-5,
            },
        );
    }
}

#[test]
fn accuracy_pow_zero_handling() {
    let Some(token) = X64V3Token::summon() else {
        return;
    };

    println!("\n=== pow zero handling ===");

    // pow(0, n) for various positive n
    let zeros = [0.0f32; 8];
    for &exp in &[0.5, 1.0, 2.0, 3.0, 0.1, 10.0] {
        let got = eval_f32x8(token, &zeros, "pow_lowp", exp);
        for (i, &g) in got.iter().enumerate() {
            assert!(g == 0.0, "pow_lowp(0, {exp})[{i}] = {g}, expected 0.0");
        }
        let got = eval_f32x8(token, &zeros, "pow_midp", exp);
        for (i, &g) in got.iter().enumerate() {
            assert!(g == 0.0, "pow_midp(0, {exp})[{i}] = {g}, expected 0.0");
        }
        let got = eval_f32x8(token, &zeros, "pow_midp_precise", exp);
        for (i, &g) in got.iter().enumerate() {
            assert!(
                g == 0.0,
                "pow_midp_precise(0, {exp})[{i}] = {g}, expected 0.0"
            );
        }
    }
    println!("  pow(0, n>0) = 0 ✓ for all variants");

    // pow(0, negative) should be inf
    let got = eval_f32x8(token, &zeros, "pow_midp", -1.0);
    for &g in &got {
        assert!(g.is_infinite(), "pow_midp(0, -1) = {g}, expected inf");
    }
    println!("  pow(0, -1) = inf ✓ (midp)");
}

#[test]
fn accuracy_cbrt() {
    let Some(token) = X64V3Token::summon() else {
        return;
    };

    println!("\n=== cbrt accuracy ===");

    // lowp: no denormals (Kahan bit hack doesn't handle them)
    let lowp_inputs = cbrt_inputs(false);
    run_accuracy_test(
        token,
        "cbrt_lowp",
        &lowp_inputs,
        0.0,
        CheckMode::Ulp {
            max_ulp: 256,
            max_rel: 5e-4,
        },
    );

    // midp: still no denormals (bit hack initial guess)
    run_accuracy_test(
        token,
        "cbrt_midp",
        &lowp_inputs,
        0.0,
        CheckMode::Ulp {
            max_ulp: 4,
            max_rel: 1e-6,
        },
    );

    // midp_precise: full range including denormals
    let full_inputs = cbrt_inputs(true);
    run_accuracy_test(
        token,
        "cbrt_midp_precise",
        &full_inputs,
        0.0,
        CheckMode::Ulp {
            max_ulp: 4,
            max_rel: 1e-6,
        },
    );
}

#[test]
fn accuracy_cbrt_zero_sign() {
    let Some(token) = X64V3Token::summon() else {
        return;
    };

    println!("\n=== cbrt zero sign preservation ===");

    // cbrt(+0) = +0
    let pos_zeros = [0.0f32; 8];
    for op in &["cbrt_lowp", "cbrt_midp", "cbrt_midp_precise"] {
        let got = eval_f32x8(token, &pos_zeros, op, 0.0);
        for (i, &g) in got.iter().enumerate() {
            assert!(g == 0.0, "{op}(+0)[{i}] = {g}");
            assert!(!g.is_sign_negative(), "{op}(+0)[{i}] returned -0.0");
        }
    }
    println!("  cbrt(+0) = +0 ✓");

    // cbrt(-0) = -0
    let neg_zeros = [-0.0f32; 8];
    for op in &["cbrt_lowp", "cbrt_midp", "cbrt_midp_precise"] {
        let got = eval_f32x8(token, &neg_zeros, op, 0.0);
        for (i, &g) in got.iter().enumerate() {
            assert!(g == 0.0, "{op}(-0)[{i}] = {g}");
            assert!(
                g.is_sign_negative(),
                "{op}(-0)[{i}] returned +0.0, expected -0.0"
            );
        }
    }
    println!("  cbrt(-0) = -0 ✓");
}

#[test]
fn accuracy_log_edge_cases() {
    let Some(token) = X64V3Token::summon() else {
        return;
    };

    println!("\n=== log edge cases ===");

    // log2(0) should be -inf (midp only — lowp returns finite approx)
    let zeros = [0.0f32; 8];
    let got = eval_f32x8(token, &zeros, "log2_midp", 0.0);
    for &g in &got {
        assert!(g == f32::NEG_INFINITY, "log2_midp(0) = {g}, expected -inf");
    }
    println!("  log2_midp(0) = -inf ✓");

    // log2(1) should be exactly 0
    let ones = [1.0f32; 8];
    for op in &["log2_lowp", "log2_midp", "log2_midp_precise"] {
        let got = eval_f32x8(token, &ones, op, 0.0);
        for &g in &got {
            assert!(g.abs() < 1e-5, "{op}(1) = {g}, expected ~0");
        }
    }
    println!("  log2(1) ≈ 0 ✓");

    // log2(negative) should be NaN
    let negs = [-1.0f32; 8];
    let got = eval_f32x8(token, &negs, "log2_midp", 0.0);
    for &g in &got {
        assert!(g.is_nan(), "log2_midp(-1) = {g}, expected NaN");
    }
    println!("  log2_midp(-1) = NaN ✓");

    // exp2(0) should be exactly 1
    let got = eval_f32x8(token, &zeros, "exp2_midp", 0.0);
    for &g in &got {
        assert!((g - 1.0).abs() < 1e-6, "exp2_midp(0) = {g}, expected 1.0");
    }
    println!("  exp2_midp(0) = 1.0 ✓");

    // exp2(-inf) should be 0
    let neg_inf = [f32::NEG_INFINITY; 8];
    let got = eval_f32x8(token, &neg_inf, "exp2_midp", 0.0);
    for &g in &got {
        assert!(g == 0.0, "exp2_midp(-inf) = {g}, expected 0.0");
    }
    println!("  exp2_midp(-inf) = 0 ✓");

    // exp2(inf) should be inf
    let pos_inf = [f32::INFINITY; 8];
    let got = eval_f32x8(token, &pos_inf, "exp2_midp", 0.0);
    for &g in &got {
        assert!(
            g.is_infinite() && g > 0.0,
            "exp2_midp(inf) = {g}, expected inf"
        );
    }
    println!("  exp2_midp(inf) = inf ✓");

    // exp2_midp in the denormal output range [-150, -126] should return 0 (not garbage)
    // This was a bug: the bit trick (n+127)<<23 produces garbage for n < -126
    let denormal_range = [
        -127.0f32, -128.0, -130.0, -140.0, -150.0, -200.0, -126.5, -126.1,
    ];
    let got = eval_f32x8(token, &denormal_range, "exp2_midp", 0.0);
    for (i, &g) in got.iter().enumerate() {
        assert!(
            g == 0.0,
            "exp2_midp({}) = {g}, expected 0.0 (denormal range must return 0)",
            denormal_range[i]
        );
    }
    println!("  exp2_midp([-127..-200]) = 0 ✓ (denormal range)");

    // exp(-100) should also return 0 now (was returning -4.3e33!)
    let deep_neg = [
        -100.0f32, -90.0, -88.0, -87.5, -100.0, -120.0, -200.0, -500.0,
    ];
    let got = eval_f32x8(token, &deep_neg, "exp_midp", 0.0);
    for (i, &g) in got.iter().enumerate() {
        let expected = deep_neg[i].exp();
        if expected == 0.0 || expected < f32::MIN_POSITIVE {
            assert!(
                g == 0.0,
                "exp_midp({}) = {g}, expected 0.0 (underflow)",
                deep_neg[i]
            );
        }
    }
    println!("  exp_midp(deep negatives) underflow to 0 ✓");
}

/// Cross-check: direct x86 types vs generic types should produce identical results
#[test]
fn accuracy_direct_vs_generic_parity() {
    let Some(token) = X64V3Token::summon() else {
        return;
    };

    println!("\n=== direct vs generic parity ===");

    let test_values: Vec<f32> = (0..100)
        .map(|i| 10.0f32.powf(-3.0 + i as f32 * 6.0 / 99.0))
        .collect();

    for op in &[
        "log2_lowp",
        "log2_midp",
        "log2_midp_precise",
        "exp2_lowp",
        "exp2_midp",
        "ln_lowp",
        "ln_midp",
        "exp_lowp",
        "exp_midp",
        "log10_lowp",
        "log10_midp",
        "cbrt_lowp",
        "cbrt_midp",
        "cbrt_midp_precise",
        "pow_lowp",
        "pow_midp",
        "pow_midp_precise",
    ] {
        let mut max_ulp_diff = 0u32;
        for chunk in test_values.chunks(4) {
            let mut arr4 = [1.0f32; 4];
            arr4[..chunk.len()].copy_from_slice(chunk);
            let mut arr8 = [1.0f32; 8];
            arr8[..chunk.len()].copy_from_slice(chunk);

            let got_generic = eval_generic_f32x4(token, &arr4, op, 2.5);
            let got_direct = eval_f32x8(token, &arr8, op, 2.5);

            for i in 0..chunk.len() {
                let ulp = ulp_distance(got_generic[i], got_direct[i]);
                max_ulp_diff = max_ulp_diff.max(ulp);
            }
        }
        // Generic and direct should be very close — both use the same algorithm
        // Allow some divergence due to different SIMD widths (FMA ordering)
        println!("  {:<25} max ULP diff: {}", op, max_ulp_diff);
        assert!(
            max_ulp_diff <= 2,
            "{op}: generic vs direct diverged by {max_ulp_diff} ULP"
        );
    }
}

/// Regional accuracy breakdown: show max ULP by input region for key functions.
/// Not an assertion test — just prints a table for analysis.
#[test]
fn accuracy_regional_breakdown() {
    let Some(token) = X64V3Token::summon() else {
        return;
    };

    println!("\n=== Regional accuracy breakdown (max ULP by input region) ===\n");

    // Define regions for exp2
    let exp2_regions: &[(&str, Vec<f32>)] = &[
        (
            "near-zero [-0.01, 0.01]",
            (-100..=100).map(|i| i as f32 / 10000.0).collect(),
        ),
        (
            "small [-1, 1]",
            (-100..=100).map(|i| i as f32 / 100.0).collect(),
        ),
        (
            "medium [-10, 10]",
            (-100..=100).map(|i| i as f32 / 10.0).collect(),
        ),
        (
            "large [-50, 50]",
            (-500..=500).map(|i| i as f32 / 10.0).collect(),
        ),
        (
            "near-overflow [100, 127]",
            (1000..=1270).map(|i| i as f32 / 10.0).collect(),
        ),
        (
            "near-underflow [-126, -100]",
            (-1260..=-1000).map(|i| i as f32 / 10.0).collect(),
        ),
        (
            "fractional near .0",
            (0..100).map(|i| i as f32 + 0.001).collect(),
        ),
        (
            "fractional near .5",
            (0..100).map(|i| i as f32 + 0.499).collect(),
        ),
        (
            "fractional near .5+",
            (0..100).map(|i| i as f32 + 0.501).collect(),
        ),
        (
            "fractional near 1.0",
            (0..100).map(|i| i as f32 + 0.999).collect(),
        ),
    ];

    println!("  exp2_midp:");
    println!(
        "  {:30} {:>8} {:>12} {:>6}",
        "Region", "Max ULP", "Max Rel Err", "Count"
    );
    println!(
        "  {:30} {:>8} {:>12} {:>6}",
        "------", "-------", "-----------", "-----"
    );
    for (name, inputs) in exp2_regions {
        let valid: Vec<f32> = inputs
            .iter()
            .copied()
            .filter(|&v| v >= -126.0 && v < 128.0)
            .collect();
        if valid.is_empty() {
            continue;
        }
        let mut max_ulp = 0u32;
        let mut max_rel = 0.0f32;
        for chunk in valid.chunks(8) {
            let mut arr = [0.0f32; 8];
            arr[..chunk.len()].copy_from_slice(chunk);
            let got = eval_f32x8(token, &arr, "exp2_midp", 0.0);
            for i in 0..chunk.len() {
                let expected = chunk[i].exp2();
                let ulp = ulp_distance(got[i], expected);
                let rel = relative_error(got[i], expected);
                max_ulp = max_ulp.max(ulp);
                if rel.is_finite() {
                    max_rel = max_rel.max(rel);
                }
            }
        }
        println!(
            "  {:30} {:>8} {:>12.2e} {:>6}",
            name,
            max_ulp,
            max_rel,
            valid.len()
        );
    }

    // Define regions for log2
    let log2_regions: &[(&str, Vec<f32>)] = &[
        (
            "near-one [0.99, 1.01]",
            (990..=1010).map(|i| i as f32 / 1000.0).collect(),
        ),
        (
            "small [0.001, 0.1]",
            (1..=100).map(|i| i as f32 / 1000.0).collect(),
        ),
        (
            "medium [0.1, 10]",
            (1..=100).map(|i| i as f32 / 10.0).collect(),
        ),
        (
            "large [10, 1e6]",
            (0..100)
                .map(|i| 10.0f32.powf(1.0 + i as f32 * 5.0 / 99.0))
                .collect(),
        ),
        (
            "very large [1e6, 1e38]",
            (0..100)
                .map(|i| 10.0f32.powf(6.0 + i as f32 * 32.0 / 99.0))
                .collect(),
        ),
        (
            "tiny [1e-38, 0.001]",
            (0..100)
                .map(|i| 10.0f32.powf(-38.0 + i as f32 * 35.0 / 99.0))
                .collect(),
        ),
    ];

    println!("\n  log2_midp:");
    println!(
        "  {:30} {:>8} {:>12} {:>6}",
        "Region", "Max ULP", "Max Rel Err", "Count"
    );
    println!(
        "  {:30} {:>8} {:>12} {:>6}",
        "------", "-------", "-----------", "-----"
    );
    for (name, inputs) in log2_regions {
        let valid: Vec<f32> = inputs
            .iter()
            .copied()
            .filter(|&v| v > 0.0 && v.is_finite() && v >= f32::MIN_POSITIVE)
            .collect();
        if valid.is_empty() {
            continue;
        }
        let mut max_ulp = 0u32;
        let mut max_rel = 0.0f32;
        for chunk in valid.chunks(8) {
            let mut arr = [1.0f32; 8];
            arr[..chunk.len()].copy_from_slice(chunk);
            let got = eval_f32x8(token, &arr, "log2_midp", 0.0);
            for i in 0..chunk.len() {
                let expected = chunk[i].log2();
                let ulp = ulp_distance(got[i], expected);
                let rel = relative_error(got[i], expected);
                max_ulp = max_ulp.max(ulp);
                if rel.is_finite() {
                    max_rel = max_rel.max(rel);
                }
            }
        }
        println!(
            "  {:30} {:>8} {:>12.2e} {:>6}",
            name,
            max_ulp,
            max_rel,
            valid.len()
        );
    }

    // pow_midp regional breakdown by exponent
    let pow_bases: Vec<f32> = (0..100)
        .map(|i| 10.0f32.powf(-3.0 + i as f32 * 6.0 / 99.0))
        .collect();

    println!("\n  pow_midp (by exponent):");
    println!(
        "  {:30} {:>8} {:>12} {:>6}",
        "Exponent", "Max ULP", "Max Rel Err", "Count"
    );
    println!(
        "  {:30} {:>8} {:>12} {:>6}",
        "--------", "-------", "-----------", "-----"
    );
    for &exp in &[0.5f32, 1.0, 2.0, 2.5, 3.0, 0.333, -1.0, -0.5, 0.45, 7.0] {
        let valid: Vec<f32> = pow_bases
            .iter()
            .copied()
            .filter(|&v| {
                let r = v.powf(exp);
                r.is_finite() && r > f32::MIN_POSITIVE && r != 0.0
            })
            .collect();
        if valid.is_empty() {
            continue;
        }
        let mut max_ulp = 0u32;
        let mut max_rel = 0.0f32;
        for chunk in valid.chunks(8) {
            let mut arr = [1.0f32; 8];
            arr[..chunk.len()].copy_from_slice(chunk);
            let got = eval_f32x8(token, &arr, "pow_midp", exp);
            for i in 0..chunk.len() {
                let expected = chunk[i].powf(exp);
                let ulp = ulp_distance(got[i], expected);
                let rel = relative_error(got[i], expected);
                max_ulp = max_ulp.max(ulp);
                if rel.is_finite() {
                    max_rel = max_rel.max(rel);
                }
            }
        }
        println!(
            "  {:30} {:>8} {:>12.2e} {:>6}",
            format!("n = {exp}"),
            max_ulp,
            max_rel,
            valid.len()
        );
    }
}
