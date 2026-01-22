//! Comparison benchmark: archmage vs sleef-rs transcendentals
//!
//! **Requires nightly Rust** for `portable_simd` feature (sleef dependency).
//!
//! Run with:
//! ```sh
//! rustup run nightly cargo run --example sleef_comparison --release
//! ```

#![feature(portable_simd)]
#![allow(dead_code)]

use std::simd::f32x8;
use std::time::Instant;

// Test data size
const N: usize = 32 * 1024;
const ITERATIONS: u32 = 1000;

fn black_box<T>(x: T) -> T {
    std::hint::black_box(x)
}

fn bench<F>(name: &str, iterations: u32, mut f: F) -> f64
where
    F: FnMut(),
{
    // Warmup
    for _ in 0..10 {
        f();
    }

    let start = Instant::now();
    for _ in 0..iterations {
        f();
    }
    let elapsed = start.elapsed();
    let ns_per_iter = elapsed.as_nanos() as f64 / iterations as f64;
    let throughput = (N as f64 * 1e9) / (ns_per_iter * 1e6);

    println!(
        "{:35} {:>10.2} ns/iter  {:>8.2} M elem/s",
        name, ns_per_iter, throughput
    );
    ns_per_iter
}

fn measure_accuracy(name: &str, test: &[f32], reference: &[f32]) -> (f32, f32) {
    let mut max_rel_err = 0.0f32;
    let mut sum_rel_err = 0.0f32;
    let mut count = 0;

    for (&t, &r) in test.iter().zip(reference.iter()) {
        if r.abs() > 1e-10 && r.is_finite() && t.is_finite() {
            let rel_err = ((t - r) / r).abs();
            max_rel_err = max_rel_err.max(rel_err);
            sum_rel_err += rel_err;
            count += 1;
        }
    }

    let avg_rel_err = if count > 0 {
        sum_rel_err / count as f32
    } else {
        0.0
    };

    println!(
        "{:35} max_rel_err: {:.2e}  avg_rel_err: {:.2e}",
        name, max_rel_err, avg_rel_err
    );
    (max_rel_err, avg_rel_err)
}

// ============================================================================
// Sleef implementations
// ============================================================================

fn sleef_exp2_f32(input: &[f32], output: &mut [f32]) {
    let chunks = input.len() / 8;
    for i in 0..chunks {
        let x = f32x8::from_slice(&input[i * 8..]);
        let r = sleef::f32x::exp2_u10(x);
        output[i * 8..(i + 1) * 8].copy_from_slice(&r.to_array());
    }
    for i in (chunks * 8)..input.len() {
        output[i] = input[i].exp2();
    }
}

fn sleef_log2_f32(input: &[f32], output: &mut [f32]) {
    let chunks = input.len() / 8;
    for i in 0..chunks {
        let x = f32x8::from_slice(&input[i * 8..]);
        let r = sleef::f32x::log2_u10(x);
        output[i * 8..(i + 1) * 8].copy_from_slice(&r.to_array());
    }
    for i in (chunks * 8)..input.len() {
        output[i] = input[i].log2();
    }
}

fn sleef_ln_f32(input: &[f32], output: &mut [f32]) {
    let chunks = input.len() / 8;
    for i in 0..chunks {
        let x = f32x8::from_slice(&input[i * 8..]);
        let r = sleef::f32x::log_u10(x);
        output[i * 8..(i + 1) * 8].copy_from_slice(&r.to_array());
    }
    for i in (chunks * 8)..input.len() {
        output[i] = input[i].ln();
    }
}

fn sleef_exp_f32(input: &[f32], output: &mut [f32]) {
    let chunks = input.len() / 8;
    for i in 0..chunks {
        let x = f32x8::from_slice(&input[i * 8..]);
        let r = sleef::f32x::exp_u10(x);
        output[i * 8..(i + 1) * 8].copy_from_slice(&r.to_array());
    }
    for i in (chunks * 8)..input.len() {
        output[i] = input[i].exp();
    }
}

fn sleef_pow_f32(input: &[f32], exp: f32, output: &mut [f32]) {
    let chunks = input.len() / 8;
    let exp_vec = f32x8::splat(exp);
    for i in 0..chunks {
        let x = f32x8::from_slice(&input[i * 8..]);
        let r = sleef::f32x::pow_u10(x, exp_vec);
        output[i * 8..(i + 1) * 8].copy_from_slice(&r.to_array());
    }
    for i in (chunks * 8)..input.len() {
        output[i] = input[i].powf(exp);
    }
}

// ============================================================================
// Archmage implementations using #[arcane] macro
// ============================================================================

use archmage::simd::f32x8 as am_f32x8;
use archmage::{Avx2FmaToken, SimdToken, arcane};

#[arcane]
fn exp2_chunk(token: Avx2FmaToken, input: &[f32; 8]) -> [f32; 8] {
    am_f32x8::load(token, input).exp2_lowp().to_array()
}

#[arcane]
fn log2_chunk(token: Avx2FmaToken, input: &[f32; 8]) -> [f32; 8] {
    am_f32x8::load(token, input).log2_lowp().to_array()
}

#[arcane]
fn ln_chunk(token: Avx2FmaToken, input: &[f32; 8]) -> [f32; 8] {
    am_f32x8::load(token, input).ln_lowp().to_array()
}

#[arcane]
fn exp_chunk(token: Avx2FmaToken, input: &[f32; 8]) -> [f32; 8] {
    am_f32x8::load(token, input).exp_lowp().to_array()
}

#[arcane]
fn pow_chunk(token: Avx2FmaToken, input: &[f32; 8], exp: f32) -> [f32; 8] {
    am_f32x8::load(token, input).pow_lowp(exp).to_array()
}

fn archmage_exp2_f32(input: &[f32], output: &mut [f32]) {
    let Some(token) = Avx2FmaToken::try_new() else {
        return;
    };

    let chunks = input.len() / 8;
    for i in 0..chunks {
        let start = i * 8;
        let arr: &[f32; 8] = input[start..start + 8].try_into().unwrap();
        output[start..start + 8].copy_from_slice(&exp2_chunk(token, arr));
    }
    for i in (chunks * 8)..input.len() {
        output[i] = input[i].exp2();
    }
}

fn archmage_log2_f32(input: &[f32], output: &mut [f32]) {
    let Some(token) = Avx2FmaToken::try_new() else {
        return;
    };

    let chunks = input.len() / 8;
    for i in 0..chunks {
        let start = i * 8;
        let arr: &[f32; 8] = input[start..start + 8].try_into().unwrap();
        output[start..start + 8].copy_from_slice(&log2_chunk(token, arr));
    }
    for i in (chunks * 8)..input.len() {
        output[i] = input[i].log2();
    }
}

fn archmage_ln_f32(input: &[f32], output: &mut [f32]) {
    let Some(token) = Avx2FmaToken::try_new() else {
        return;
    };

    let chunks = input.len() / 8;
    for i in 0..chunks {
        let start = i * 8;
        let arr: &[f32; 8] = input[start..start + 8].try_into().unwrap();
        output[start..start + 8].copy_from_slice(&ln_chunk(token, arr));
    }
    for i in (chunks * 8)..input.len() {
        output[i] = input[i].ln();
    }
}

fn archmage_exp_f32(input: &[f32], output: &mut [f32]) {
    let Some(token) = Avx2FmaToken::try_new() else {
        return;
    };

    let chunks = input.len() / 8;
    for i in 0..chunks {
        let start = i * 8;
        let arr: &[f32; 8] = input[start..start + 8].try_into().unwrap();
        output[start..start + 8].copy_from_slice(&exp_chunk(token, arr));
    }
    for i in (chunks * 8)..input.len() {
        output[i] = input[i].exp();
    }
}

fn archmage_pow_f32(input: &[f32], exp: f32, output: &mut [f32]) {
    let Some(token) = Avx2FmaToken::try_new() else {
        return;
    };

    let chunks = input.len() / 8;
    for i in 0..chunks {
        let start = i * 8;
        let arr: &[f32; 8] = input[start..start + 8].try_into().unwrap();
        output[start..start + 8].copy_from_slice(&pow_chunk(token, arr, exp));
    }
    for i in (chunks * 8)..input.len() {
        output[i] = input[i].powf(exp);
    }
}

// ============================================================================
// Scalar baseline
// ============================================================================

fn scalar_exp2_f32(input: &[f32], output: &mut [f32]) {
    for (o, &x) in output.iter_mut().zip(input.iter()) {
        *o = x.exp2();
    }
}

fn scalar_log2_f32(input: &[f32], output: &mut [f32]) {
    for (o, &x) in output.iter_mut().zip(input.iter()) {
        *o = x.log2();
    }
}

fn scalar_pow_f32(input: &[f32], exp: f32, output: &mut [f32]) {
    for (o, &x) in output.iter_mut().zip(input.iter()) {
        *o = x.powf(exp);
    }
}

fn main() {
    println!("Archmage vs Sleef-rs Transcendental Comparison");
    println!("===============================================");
    println!("N = {} elements, {} iterations\n", N, ITERATIONS);

    // Check if AVX2+FMA is available
    if Avx2FmaToken::try_new().is_none() {
        eprintln!("AVX2+FMA not available, skipping archmage benchmarks");
        return;
    }

    // Generate test data
    let exp2_input: Vec<f32> = (0..N)
        .map(|i| -10.0 + (i as f32 / N as f32) * 20.0)
        .collect();
    let log2_input: Vec<f32> = (0..N)
        .map(|i| 0.01 + (i as f32 / N as f32) * 99.99)
        .collect();
    let pow_input: Vec<f32> = (0..N)
        .map(|i| 0.001 + (i as f32 / N as f32) * 0.998)
        .collect();
    let _exp_input: Vec<f32> = (0..N)
        .map(|i| -5.0 + (i as f32 / N as f32) * 10.0)
        .collect();

    let mut output = vec![0.0f32; N];
    let mut sleef_out = vec![0.0f32; N];
    let mut archmage_out = vec![0.0f32; N];

    // ========================================================================
    // exp2 benchmarks
    // ========================================================================
    println!("--- exp2 benchmarks ---");

    bench("scalar_std_exp2", ITERATIONS, || {
        scalar_exp2_f32(black_box(&exp2_input), black_box(&mut output));
    });

    bench("sleef_exp2_u10", ITERATIONS, || {
        sleef_exp2_f32(black_box(&exp2_input), black_box(&mut output));
    });

    bench("archmage_exp2", ITERATIONS, || {
        archmage_exp2_f32(black_box(&exp2_input), black_box(&mut output));
    });

    println!();

    // ========================================================================
    // log2 benchmarks
    // ========================================================================
    println!("--- log2 benchmarks ---");

    bench("scalar_std_log2", ITERATIONS, || {
        scalar_log2_f32(black_box(&log2_input), black_box(&mut output));
    });

    bench("sleef_log2_u10", ITERATIONS, || {
        sleef_log2_f32(black_box(&log2_input), black_box(&mut output));
    });

    bench("archmage_log2", ITERATIONS, || {
        archmage_log2_f32(black_box(&log2_input), black_box(&mut output));
    });

    println!();

    // ========================================================================
    // pow(x, 2.4) benchmarks - sRGB decode
    // ========================================================================
    println!("--- pow(x, 2.4) benchmarks (sRGB decode) ---");

    bench("scalar_std_pow_2.4", ITERATIONS, || {
        scalar_pow_f32(black_box(&pow_input), 2.4, black_box(&mut output));
    });

    bench("sleef_pow_u10_2.4", ITERATIONS, || {
        sleef_pow_f32(black_box(&pow_input), 2.4, black_box(&mut output));
    });

    bench("archmage_pow_2.4", ITERATIONS, || {
        archmage_pow_f32(black_box(&pow_input), 2.4, black_box(&mut output));
    });

    println!();

    // ========================================================================
    // Accuracy measurements
    // ========================================================================
    println!("--- Accuracy measurements (vs scalar std) ---");

    // Generate reference values
    let exp2_ref: Vec<f32> = exp2_input.iter().map(|&x| x.exp2()).collect();
    let log2_ref: Vec<f32> = log2_input.iter().map(|&x| x.log2()).collect();
    let pow_ref: Vec<f32> = pow_input.iter().map(|&x| x.powf(2.4)).collect();

    println!("\nexp2 accuracy:");
    sleef_exp2_f32(&exp2_input, &mut sleef_out);
    archmage_exp2_f32(&exp2_input, &mut archmage_out);
    measure_accuracy("sleef_exp2_u10", &sleef_out, &exp2_ref);
    measure_accuracy("archmage_exp2", &archmage_out, &exp2_ref);

    println!("\nlog2 accuracy:");
    sleef_log2_f32(&log2_input, &mut sleef_out);
    archmage_log2_f32(&log2_input, &mut archmage_out);
    measure_accuracy("sleef_log2_u10", &sleef_out, &log2_ref);
    measure_accuracy("archmage_log2", &archmage_out, &log2_ref);

    println!("\npow(x, 2.4) accuracy:");
    sleef_pow_f32(&pow_input, 2.4, &mut sleef_out);
    archmage_pow_f32(&pow_input, 2.4, &mut archmage_out);
    measure_accuracy("sleef_pow_u10_2.4", &sleef_out, &pow_ref);
    measure_accuracy("archmage_pow_2.4", &archmage_out, &pow_ref);

    println!("\nDone!");
}
