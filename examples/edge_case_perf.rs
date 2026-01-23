//! Performance comparison: edge case handling overhead
//!
//! Run with: `cargo run --example edge_case_perf --release`

#![allow(dead_code)]

use std::arch::x86_64::*;
use std::time::Instant;

const N: usize = 32 * 1024;
const ITERATIONS: u32 = 1000;

fn black_box<T>(x: T) -> T {
    std::hint::black_box(x)
}

fn bench<F>(name: &str, mut f: F) -> f64
where
    F: FnMut(),
{
    // Warmup
    for _ in 0..10 {
        f();
    }

    let start = Instant::now();
    for _ in 0..ITERATIONS {
        f();
    }
    let elapsed = start.elapsed();
    let ns_per_iter = elapsed.as_nanos() as f64 / ITERATIONS as f64;
    let throughput = (N as f64 * 1e9) / (ns_per_iter * 1e6);

    println!(
        "{:40} {:>10.2} ns/iter  {:>8.2} M elem/s",
        name, ns_per_iter, throughput
    );
    ns_per_iter
}

// ============================================================================
// cbrt implementations
// ============================================================================

/// Current cbrt_midp - no edge case handling
#[target_feature(enable = "avx2,fma")]
unsafe fn cbrt_current(x: __m256) -> __m256 {
    const B1: u32 = 709_958_130;

    // Extract for initial guess
    let x_arr: [f32; 8] = std::mem::transmute(x);
    let mut y_arr = [0.0f32; 8];

    for i in 0..8 {
        let xi = x_arr[i];
        let ui = xi.to_bits();
        let hx = ui & 0x7FFF_FFFF;
        let approx = hx / 3 + B1;
        y_arr[i] = f32::from_bits(approx);
    }

    let abs_x = _mm256_andnot_ps(_mm256_set1_ps(-0.0), x);
    let sign_mask = _mm256_and_ps(x, _mm256_set1_ps(-0.0));
    let mut y: __m256 = std::mem::transmute(y_arr);

    // Newton-Raphson iterations
    for _ in 0..2 {
        let y2 = _mm256_mul_ps(y, y);
        let y3 = _mm256_mul_ps(y2, y);
        let num = _mm256_fmadd_ps(_mm256_set1_ps(2.0), abs_x, y3);
        let den = _mm256_fmadd_ps(_mm256_set1_ps(2.0), y3, abs_x);
        y = _mm256_mul_ps(y, _mm256_div_ps(num, den));
    }

    _mm256_or_ps(y, sign_mask)
}

/// cbrt with critical fixes only: handle 0 and inf
#[target_feature(enable = "avx2,fma")]
unsafe fn cbrt_critical_fixes(x: __m256) -> __m256 {
    const B1: u32 = 709_958_130;

    let abs_x = _mm256_andnot_ps(_mm256_set1_ps(-0.0), x);
    let sign_mask = _mm256_and_ps(x, _mm256_set1_ps(-0.0));

    // Check for zero and infinity
    let zero = _mm256_setzero_ps();
    let is_zero = _mm256_cmp_ps(abs_x, zero, _CMP_EQ_OQ);
    let inf = _mm256_set1_ps(f32::INFINITY);
    let is_inf = _mm256_cmp_ps(abs_x, inf, _CMP_EQ_OQ);

    // Extract for initial guess
    let x_arr: [f32; 8] = std::mem::transmute(abs_x);
    let mut y_arr = [0.0f32; 8];

    for i in 0..8 {
        let xi = x_arr[i];
        let ui = xi.to_bits();
        let hx = ui & 0x7FFF_FFFF;
        let approx = hx / 3 + B1;
        y_arr[i] = f32::from_bits(approx);
    }

    let mut y: __m256 = std::mem::transmute(y_arr);

    // Newton-Raphson iterations
    for _ in 0..2 {
        let y2 = _mm256_mul_ps(y, y);
        let y3 = _mm256_mul_ps(y2, y);
        let num = _mm256_fmadd_ps(_mm256_set1_ps(2.0), abs_x, y3);
        let den = _mm256_fmadd_ps(_mm256_set1_ps(2.0), y3, abs_x);
        y = _mm256_mul_ps(y, _mm256_div_ps(num, den));
    }

    // Apply sign
    y = _mm256_or_ps(y, sign_mask);

    // Fix edge cases: 0 -> 0, inf -> inf (with sign)
    y = _mm256_blendv_ps(y, _mm256_or_ps(zero, sign_mask), is_zero);
    y = _mm256_blendv_ps(y, _mm256_or_ps(inf, sign_mask), is_inf);

    y
}

/// cbrt matching std exactly: handle 0, inf, NaN, denormals
#[target_feature(enable = "avx2,fma")]
unsafe fn cbrt_std_match(x: __m256) -> __m256 {
    const B1: u32 = 709_958_130;

    let abs_x = _mm256_andnot_ps(_mm256_set1_ps(-0.0), x);
    let sign_mask = _mm256_and_ps(x, _mm256_set1_ps(-0.0));

    // Check for special cases
    let zero = _mm256_setzero_ps();
    let is_zero = _mm256_cmp_ps(abs_x, zero, _CMP_EQ_OQ);
    let inf = _mm256_set1_ps(f32::INFINITY);
    let is_inf = _mm256_cmp_ps(abs_x, inf, _CMP_EQ_OQ);
    let is_nan = _mm256_cmp_ps(x, x, _CMP_UNORD_Q); // NaN != NaN

    // Handle denormals by scaling up
    let min_normal = _mm256_set1_ps(f32::MIN_POSITIVE);
    let is_denormal = _mm256_cmp_ps(abs_x, min_normal, _CMP_LT_OQ);
    let scale_up = _mm256_set1_ps(16777216.0); // 2^24
    let scale_down = _mm256_set1_ps(1.0 / 256.0); // 2^(-24/3) = 2^-8
    let scaled_x = _mm256_blendv_ps(abs_x, _mm256_mul_ps(abs_x, scale_up), is_denormal);

    // Extract for initial guess
    let x_arr: [f32; 8] = std::mem::transmute(scaled_x);
    let mut y_arr = [0.0f32; 8];

    for i in 0..8 {
        let xi = x_arr[i];
        let ui = xi.to_bits();
        let hx = ui & 0x7FFF_FFFF;
        let approx = hx / 3 + B1;
        y_arr[i] = f32::from_bits(approx);
    }

    let mut y: __m256 = std::mem::transmute(y_arr);

    // Newton-Raphson iterations
    for _ in 0..2 {
        let y2 = _mm256_mul_ps(y, y);
        let y3 = _mm256_mul_ps(y2, y);
        let num = _mm256_fmadd_ps(_mm256_set1_ps(2.0), scaled_x, y3);
        let den = _mm256_fmadd_ps(_mm256_set1_ps(2.0), y3, scaled_x);
        y = _mm256_mul_ps(y, _mm256_div_ps(num, den));
    }

    // Scale back down for denormals
    y = _mm256_blendv_ps(y, _mm256_mul_ps(y, scale_down), is_denormal);

    // Apply sign
    y = _mm256_or_ps(y, sign_mask);

    // Fix edge cases
    y = _mm256_blendv_ps(y, _mm256_or_ps(zero, sign_mask), is_zero); // 0 -> ±0
    y = _mm256_blendv_ps(y, _mm256_or_ps(inf, sign_mask), is_inf); // inf -> ±inf
    y = _mm256_blendv_ps(y, x, is_nan); // NaN -> NaN

    y
}

// ============================================================================
// exp2 implementations
// ============================================================================

/// Current exp2_midp - clamps to avoid overflow
#[target_feature(enable = "avx2,fma")]
unsafe fn exp2_current(x: __m256) -> __m256 {
    let x = _mm256_max_ps(x, _mm256_set1_ps(-126.0));
    let x = _mm256_min_ps(x, _mm256_set1_ps(126.0));

    let xi = _mm256_floor_ps(x);
    let xf = _mm256_sub_ps(x, xi);

    const C0: f32 = 1.0;
    const C1: f32 = 0.693_147_18;
    const C2: f32 = 0.240_226_51;
    const C3: f32 = 0.055_504_11;
    const C4: f32 = 0.009_618_13;
    const C5: f32 = 0.001_333_55;

    let poly = _mm256_fmadd_ps(_mm256_set1_ps(C5), xf, _mm256_set1_ps(C4));
    let poly = _mm256_fmadd_ps(poly, xf, _mm256_set1_ps(C3));
    let poly = _mm256_fmadd_ps(poly, xf, _mm256_set1_ps(C2));
    let poly = _mm256_fmadd_ps(poly, xf, _mm256_set1_ps(C1));
    let poly = _mm256_fmadd_ps(poly, xf, _mm256_set1_ps(C0));

    let xi_i32 = _mm256_cvtps_epi32(xi);
    let scale_bits = _mm256_slli_epi32(_mm256_add_epi32(xi_i32, _mm256_set1_epi32(127)), 23);
    let scale = _mm256_castsi256_ps(scale_bits);

    _mm256_mul_ps(poly, scale)
}

/// exp2 with critical fixes: return inf on overflow, 0 on underflow
#[target_feature(enable = "avx2,fma")]
unsafe fn exp2_critical_fixes(x: __m256) -> __m256 {
    // Check for overflow/underflow before clamping
    let overflow_threshold = _mm256_set1_ps(128.0);
    let underflow_threshold = _mm256_set1_ps(-150.0);
    let is_overflow = _mm256_cmp_ps(x, overflow_threshold, _CMP_GE_OQ);
    let is_underflow = _mm256_cmp_ps(x, underflow_threshold, _CMP_LE_OQ);

    let x = _mm256_max_ps(x, _mm256_set1_ps(-126.0));
    let x = _mm256_min_ps(x, _mm256_set1_ps(126.0));

    let xi = _mm256_floor_ps(x);
    let xf = _mm256_sub_ps(x, xi);

    const C0: f32 = 1.0;
    const C1: f32 = 0.693_147_18;
    const C2: f32 = 0.240_226_51;
    const C3: f32 = 0.055_504_11;
    const C4: f32 = 0.009_618_13;
    const C5: f32 = 0.001_333_55;

    let poly = _mm256_fmadd_ps(_mm256_set1_ps(C5), xf, _mm256_set1_ps(C4));
    let poly = _mm256_fmadd_ps(poly, xf, _mm256_set1_ps(C3));
    let poly = _mm256_fmadd_ps(poly, xf, _mm256_set1_ps(C2));
    let poly = _mm256_fmadd_ps(poly, xf, _mm256_set1_ps(C1));
    let poly = _mm256_fmadd_ps(poly, xf, _mm256_set1_ps(C0));

    let xi_i32 = _mm256_cvtps_epi32(xi);
    let scale_bits = _mm256_slli_epi32(_mm256_add_epi32(xi_i32, _mm256_set1_epi32(127)), 23);
    let scale = _mm256_castsi256_ps(scale_bits);

    let mut result = _mm256_mul_ps(poly, scale);

    // Fix edge cases
    result = _mm256_blendv_ps(result, _mm256_set1_ps(f32::INFINITY), is_overflow);
    result = _mm256_blendv_ps(result, _mm256_setzero_ps(), is_underflow);

    result
}

/// exp2 matching std exactly: handle overflow, underflow, NaN
#[target_feature(enable = "avx2,fma")]
unsafe fn exp2_std_match(x: __m256) -> __m256 {
    // Check for special cases
    let is_nan = _mm256_cmp_ps(x, x, _CMP_UNORD_Q);
    let overflow_threshold = _mm256_set1_ps(128.0);
    let underflow_threshold = _mm256_set1_ps(-150.0);
    let is_overflow = _mm256_cmp_ps(x, overflow_threshold, _CMP_GE_OQ);
    let is_underflow = _mm256_cmp_ps(x, underflow_threshold, _CMP_LE_OQ);

    let x_clamped = _mm256_max_ps(x, _mm256_set1_ps(-126.0));
    let x_clamped = _mm256_min_ps(x_clamped, _mm256_set1_ps(126.0));

    let xi = _mm256_floor_ps(x_clamped);
    let xf = _mm256_sub_ps(x_clamped, xi);

    const C0: f32 = 1.0;
    const C1: f32 = 0.693_147_18;
    const C2: f32 = 0.240_226_51;
    const C3: f32 = 0.055_504_11;
    const C4: f32 = 0.009_618_13;
    const C5: f32 = 0.001_333_55;

    let poly = _mm256_fmadd_ps(_mm256_set1_ps(C5), xf, _mm256_set1_ps(C4));
    let poly = _mm256_fmadd_ps(poly, xf, _mm256_set1_ps(C3));
    let poly = _mm256_fmadd_ps(poly, xf, _mm256_set1_ps(C2));
    let poly = _mm256_fmadd_ps(poly, xf, _mm256_set1_ps(C1));
    let poly = _mm256_fmadd_ps(poly, xf, _mm256_set1_ps(C0));

    let xi_i32 = _mm256_cvtps_epi32(xi);
    let scale_bits = _mm256_slli_epi32(_mm256_add_epi32(xi_i32, _mm256_set1_epi32(127)), 23);
    let scale = _mm256_castsi256_ps(scale_bits);

    let mut result = _mm256_mul_ps(poly, scale);

    // Fix edge cases
    result = _mm256_blendv_ps(result, _mm256_set1_ps(f32::INFINITY), is_overflow);
    result = _mm256_blendv_ps(result, _mm256_setzero_ps(), is_underflow);
    result = _mm256_blendv_ps(result, x, is_nan); // NaN -> NaN

    result
}

// ============================================================================
// log2 implementations
// ============================================================================

/// Current log2_midp - no edge case handling
#[target_feature(enable = "avx2,fma")]
unsafe fn log2_current(x: __m256) -> __m256 {
    const SQRT2_OVER_2: u32 = 0x3f3504f3;
    const ONE: u32 = 0x3f800000;

    let bits = _mm256_castps_si256(x);
    let offset = _mm256_set1_epi32((ONE - SQRT2_OVER_2) as i32);
    let adjusted = _mm256_add_epi32(bits, offset);

    let exp_raw = _mm256_srli_epi32(adjusted, 23);
    let exp_i32 = _mm256_sub_epi32(exp_raw, _mm256_set1_epi32(0x7f));
    let n = _mm256_cvtepi32_ps(exp_i32);

    let mantissa_mask = _mm256_set1_epi32(0x007fffff);
    let mantissa_bits = _mm256_add_epi32(
        _mm256_and_si256(adjusted, mantissa_mask),
        _mm256_set1_epi32(SQRT2_OVER_2 as i32),
    );
    let a = _mm256_castsi256_ps(mantissa_bits);

    let one = _mm256_set1_ps(1.0);
    let y = _mm256_div_ps(_mm256_sub_ps(a, one), _mm256_add_ps(a, one));
    let y2 = _mm256_mul_ps(y, y);

    const C0: f32 = 2.885_390_08;
    const C1: f32 = 0.961_796_7;
    const C2: f32 = 0.577_078_04;
    const C3: f32 = 0.412_198_57;

    let mut u = _mm256_set1_ps(C3);
    u = _mm256_fmadd_ps(u, y2, _mm256_set1_ps(C2));
    u = _mm256_fmadd_ps(u, y2, _mm256_set1_ps(C1));
    u = _mm256_fmadd_ps(u, y2, _mm256_set1_ps(C0));

    _mm256_fmadd_ps(u, y, n)
}

/// log2 with critical fixes: return -inf for 0, NaN for negative
#[target_feature(enable = "avx2,fma")]
unsafe fn log2_critical_fixes(x: __m256) -> __m256 {
    // Check for edge cases
    let zero = _mm256_setzero_ps();
    let is_zero = _mm256_cmp_ps(x, zero, _CMP_EQ_OQ);
    let is_negative = _mm256_cmp_ps(x, zero, _CMP_LT_OQ);

    const SQRT2_OVER_2: u32 = 0x3f3504f3;
    const ONE: u32 = 0x3f800000;

    let bits = _mm256_castps_si256(x);
    let offset = _mm256_set1_epi32((ONE - SQRT2_OVER_2) as i32);
    let adjusted = _mm256_add_epi32(bits, offset);

    let exp_raw = _mm256_srli_epi32(adjusted, 23);
    let exp_i32 = _mm256_sub_epi32(exp_raw, _mm256_set1_epi32(0x7f));
    let n = _mm256_cvtepi32_ps(exp_i32);

    let mantissa_mask = _mm256_set1_epi32(0x007fffff);
    let mantissa_bits = _mm256_add_epi32(
        _mm256_and_si256(adjusted, mantissa_mask),
        _mm256_set1_epi32(SQRT2_OVER_2 as i32),
    );
    let a = _mm256_castsi256_ps(mantissa_bits);

    let one = _mm256_set1_ps(1.0);
    let y = _mm256_div_ps(_mm256_sub_ps(a, one), _mm256_add_ps(a, one));
    let y2 = _mm256_mul_ps(y, y);

    const C0: f32 = 2.885_390_08;
    const C1: f32 = 0.961_796_7;
    const C2: f32 = 0.577_078_04;
    const C3: f32 = 0.412_198_57;

    let mut u = _mm256_set1_ps(C3);
    u = _mm256_fmadd_ps(u, y2, _mm256_set1_ps(C2));
    u = _mm256_fmadd_ps(u, y2, _mm256_set1_ps(C1));
    u = _mm256_fmadd_ps(u, y2, _mm256_set1_ps(C0));

    let mut result = _mm256_fmadd_ps(u, y, n);

    // Fix edge cases
    result = _mm256_blendv_ps(result, _mm256_set1_ps(f32::NEG_INFINITY), is_zero);
    result = _mm256_blendv_ps(result, _mm256_set1_ps(f32::NAN), is_negative);

    result
}

/// log2 matching std exactly: handle 0, negative, inf, NaN
#[target_feature(enable = "avx2,fma")]
unsafe fn log2_std_match(x: __m256) -> __m256 {
    // Check for edge cases
    let zero = _mm256_setzero_ps();
    let is_zero = _mm256_cmp_ps(x, zero, _CMP_EQ_OQ);
    let is_negative = _mm256_cmp_ps(x, zero, _CMP_LT_OQ);
    let is_nan = _mm256_cmp_ps(x, x, _CMP_UNORD_Q);
    let inf = _mm256_set1_ps(f32::INFINITY);
    let is_inf = _mm256_cmp_ps(x, inf, _CMP_EQ_OQ);

    const SQRT2_OVER_2: u32 = 0x3f3504f3;
    const ONE: u32 = 0x3f800000;

    let bits = _mm256_castps_si256(x);
    let offset = _mm256_set1_epi32((ONE - SQRT2_OVER_2) as i32);
    let adjusted = _mm256_add_epi32(bits, offset);

    let exp_raw = _mm256_srli_epi32(adjusted, 23);
    let exp_i32 = _mm256_sub_epi32(exp_raw, _mm256_set1_epi32(0x7f));
    let n = _mm256_cvtepi32_ps(exp_i32);

    let mantissa_mask = _mm256_set1_epi32(0x007fffff);
    let mantissa_bits = _mm256_add_epi32(
        _mm256_and_si256(adjusted, mantissa_mask),
        _mm256_set1_epi32(SQRT2_OVER_2 as i32),
    );
    let a = _mm256_castsi256_ps(mantissa_bits);

    let one = _mm256_set1_ps(1.0);
    let y = _mm256_div_ps(_mm256_sub_ps(a, one), _mm256_add_ps(a, one));
    let y2 = _mm256_mul_ps(y, y);

    const C0: f32 = 2.885_390_08;
    const C1: f32 = 0.961_796_7;
    const C2: f32 = 0.577_078_04;
    const C3: f32 = 0.412_198_57;

    let mut u = _mm256_set1_ps(C3);
    u = _mm256_fmadd_ps(u, y2, _mm256_set1_ps(C2));
    u = _mm256_fmadd_ps(u, y2, _mm256_set1_ps(C1));
    u = _mm256_fmadd_ps(u, y2, _mm256_set1_ps(C0));

    let mut result = _mm256_fmadd_ps(u, y, n);

    // Fix edge cases (order matters - NaN check last)
    result = _mm256_blendv_ps(result, _mm256_set1_ps(f32::NEG_INFINITY), is_zero);
    result = _mm256_blendv_ps(result, _mm256_set1_ps(f32::NAN), is_negative);
    result = _mm256_blendv_ps(result, inf, is_inf); // log2(inf) = inf
    result = _mm256_blendv_ps(result, x, is_nan); // NaN passthrough

    result
}

// ============================================================================
// Benchmark harness
// ============================================================================

fn main() {
    if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
        eprintln!("AVX2+FMA not available");
        return;
    }

    println!("Edge Case Handling Performance Comparison");
    println!("==========================================");
    println!("N = {} elements, {} iterations\n", N, ITERATIONS);

    // Generate test data - normal range (no edge cases in data)
    let cbrt_input: Vec<f32> = (0..N)
        .map(|i| 0.001 + (i as f32 / N as f32) * 999.999)
        .collect();
    let exp2_input: Vec<f32> = (0..N)
        .map(|i| -20.0 + (i as f32 / N as f32) * 40.0)
        .collect();
    let log2_input: Vec<f32> = (0..N)
        .map(|i| 0.001 + (i as f32 / N as f32) * 999.999)
        .collect();

    let mut output = vec![0.0f32; N];

    // ========================================================================
    // cbrt benchmarks
    // ========================================================================
    println!("--- cbrt benchmarks (normal data) ---");

    let current = bench("cbrt_current (no edge handling)", || unsafe {
        for i in 0..(N / 8) {
            let x = _mm256_loadu_ps(cbrt_input.as_ptr().add(i * 8));
            let r = cbrt_current(x);
            _mm256_storeu_ps(output.as_mut_ptr().add(i * 8), r);
        }
        black_box(&output);
    });

    let critical = bench("cbrt_critical_fixes (0, inf)", || unsafe {
        for i in 0..(N / 8) {
            let x = _mm256_loadu_ps(cbrt_input.as_ptr().add(i * 8));
            let r = cbrt_critical_fixes(x);
            _mm256_storeu_ps(output.as_mut_ptr().add(i * 8), r);
        }
        black_box(&output);
    });

    let std_match = bench("cbrt_std_match (full)", || unsafe {
        for i in 0..(N / 8) {
            let x = _mm256_loadu_ps(cbrt_input.as_ptr().add(i * 8));
            let r = cbrt_std_match(x);
            _mm256_storeu_ps(output.as_mut_ptr().add(i * 8), r);
        }
        black_box(&output);
    });

    println!(
        "  → critical fixes overhead: {:.1}%",
        (critical / current - 1.0) * 100.0
    );
    println!(
        "  → std match overhead: {:.1}%",
        (std_match / current - 1.0) * 100.0
    );
    println!();

    // ========================================================================
    // exp2 benchmarks
    // ========================================================================
    println!("--- exp2 benchmarks (normal data) ---");

    let current = bench("exp2_current (clamp only)", || unsafe {
        for i in 0..(N / 8) {
            let x = _mm256_loadu_ps(exp2_input.as_ptr().add(i * 8));
            let r = exp2_current(x);
            _mm256_storeu_ps(output.as_mut_ptr().add(i * 8), r);
        }
        black_box(&output);
    });

    let critical = bench("exp2_critical_fixes (inf, 0)", || unsafe {
        for i in 0..(N / 8) {
            let x = _mm256_loadu_ps(exp2_input.as_ptr().add(i * 8));
            let r = exp2_critical_fixes(x);
            _mm256_storeu_ps(output.as_mut_ptr().add(i * 8), r);
        }
        black_box(&output);
    });

    let std_match = bench("exp2_std_match (full)", || unsafe {
        for i in 0..(N / 8) {
            let x = _mm256_loadu_ps(exp2_input.as_ptr().add(i * 8));
            let r = exp2_std_match(x);
            _mm256_storeu_ps(output.as_mut_ptr().add(i * 8), r);
        }
        black_box(&output);
    });

    println!(
        "  → critical fixes overhead: {:.1}%",
        (critical / current - 1.0) * 100.0
    );
    println!(
        "  → std match overhead: {:.1}%",
        (std_match / current - 1.0) * 100.0
    );
    println!();

    // ========================================================================
    // log2 benchmarks
    // ========================================================================
    println!("--- log2 benchmarks (normal data) ---");

    let current = bench("log2_current (no edge handling)", || unsafe {
        for i in 0..(N / 8) {
            let x = _mm256_loadu_ps(log2_input.as_ptr().add(i * 8));
            let r = log2_current(x);
            _mm256_storeu_ps(output.as_mut_ptr().add(i * 8), r);
        }
        black_box(&output);
    });

    let critical = bench("log2_critical_fixes (-inf, NaN)", || unsafe {
        for i in 0..(N / 8) {
            let x = _mm256_loadu_ps(log2_input.as_ptr().add(i * 8));
            let r = log2_critical_fixes(x);
            _mm256_storeu_ps(output.as_mut_ptr().add(i * 8), r);
        }
        black_box(&output);
    });

    let std_match = bench("log2_std_match (full)", || unsafe {
        for i in 0..(N / 8) {
            let x = _mm256_loadu_ps(log2_input.as_ptr().add(i * 8));
            let r = log2_std_match(x);
            _mm256_storeu_ps(output.as_mut_ptr().add(i * 8), r);
        }
        black_box(&output);
    });

    println!(
        "  → critical fixes overhead: {:.1}%",
        (critical / current - 1.0) * 100.0
    );
    println!(
        "  → std match overhead: {:.1}%",
        (std_match / current - 1.0) * 100.0
    );
    println!();

    println!("Done!");
}
