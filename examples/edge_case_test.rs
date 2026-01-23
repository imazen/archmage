//! Edge case testing for transcendental functions
//!
//! Run with: `cargo run --example edge_case_test --release`

#![cfg(target_arch = "x86_64")]

use archmage::simd::f32x8;
use archmage::{Avx2FmaToken, SimdToken, arcane};

#[arcane]
fn test_exp(token: Avx2FmaToken, input: &[f32; 8]) -> [f32; 8] {
    f32x8::load(token, input).exp_midp().to_array()
}

#[arcane]
fn test_exp2(token: Avx2FmaToken, input: &[f32; 8]) -> [f32; 8] {
    f32x8::load(token, input).exp2_midp().to_array()
}

#[arcane]
fn test_log2(token: Avx2FmaToken, input: &[f32; 8]) -> [f32; 8] {
    f32x8::load(token, input).log2_midp().to_array()
}

#[arcane]
fn test_ln(token: Avx2FmaToken, input: &[f32; 8]) -> [f32; 8] {
    f32x8::load(token, input).ln_midp().to_array()
}

#[arcane]
fn test_cbrt(token: Avx2FmaToken, input: &[f32; 8]) -> [f32; 8] {
    f32x8::load(token, input).cbrt_midp().to_array()
}

fn compare(name: &str, input: f32, std_result: f32, our_result: f32) {
    let matches = if std_result.is_nan() && our_result.is_nan() {
        true
    } else if std_result.is_infinite() && our_result.is_infinite() {
        std_result.signum() == our_result.signum()
    } else if std_result == 0.0 && our_result == 0.0 {
        std_result.to_bits() == our_result.to_bits() // Check sign of zero
    } else {
        let rel_err = ((our_result - std_result) / std_result).abs();
        rel_err < 0.01 // 1% tolerance for edge cases
    };

    let status = if matches { "✓" } else { "✗" };
    println!(
        "  {} {}({:>12}) std={:>14} ours={:>14}",
        status,
        name,
        format!("{:e}", input),
        format!("{:e}", std_result),
        format!("{:e}", our_result)
    );
}

fn main() {
    let Some(token) = Avx2FmaToken::try_new() else {
        eprintln!("AVX2+FMA not available");
        return;
    };

    println!("=== Edge Case Comparison: archmage vs std ===\n");

    // exp overflow boundary
    println!("exp() overflow boundary:");
    let inputs = [87.0f32, 88.0, 88.5, 88.72, 88.73, 89.0, 100.0, -100.0];
    let results = test_exp(token, &inputs);
    for (i, &x) in inputs.iter().enumerate() {
        compare("exp", x, x.exp(), results[i]);
    }

    // exp2 overflow boundary
    println!("\nexp2() overflow boundary:");
    let inputs = [126.0f32, 127.0, 127.5, 128.0, 129.0, -126.0, -150.0, 200.0];
    let results = test_exp2(token, &inputs);
    for (i, &x) in inputs.iter().enumerate() {
        compare("exp2", x, x.exp2(), results[i]);
    }

    // log2 for small/edge values
    println!("\nlog2() small values:");
    let inputs = [1e-30f32, 1e-35, 1e-37, 1e-38, 1e-40, 1e-45, 0.0, -0.0];
    let results = test_log2(token, &inputs);
    for (i, &x) in inputs.iter().enumerate() {
        compare("log2", x, x.log2(), results[i]);
    }

    // log2 for negative values
    println!("\nlog2() negative values:");
    let inputs = [-1.0f32, -0.5, -1e-10, -1e-30, 1.0, 2.0, 10.0, 100.0];
    let results = test_log2(token, &inputs);
    for (i, &x) in inputs.iter().enumerate() {
        compare("log2", x, x.log2(), results[i]);
    }

    // ln edge cases
    println!("\nln() edge cases:");
    let inputs = [
        1e-30f32,
        1e-37,
        0.0,
        -0.0,
        -1.0,
        1.0,
        std::f32::consts::E,
        1e30,
    ];
    let results = test_ln(token, &inputs);
    for (i, &x) in inputs.iter().enumerate() {
        compare("ln", x, x.ln(), results[i]);
    }

    // cbrt edge cases
    println!("\ncbrt() edge cases:");
    let inputs = [0.0f32, -0.0, 1e-45, -1e-45, 1e-30, -1e-30, 1e30, -1e30];
    let results = test_cbrt(token, &inputs);
    for (i, &x) in inputs.iter().enumerate() {
        compare("cbrt", x, x.cbrt(), results[i]);
    }

    // cbrt special values
    println!("\ncbrt() special values:");
    let inputs = [
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::NAN,
        1.0,
        -1.0,
        8.0,
        -8.0,
        27.0,
    ];
    let results = test_cbrt(token, &inputs);
    for (i, &x) in inputs.iter().enumerate() {
        compare("cbrt", x, x.cbrt(), results[i]);
    }

    println!("\nDone!");
}
