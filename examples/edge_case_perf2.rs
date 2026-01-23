//! Performance: easy ranges vs hard ranges (with edge cases)
//!
//! Run with: `cargo run --example edge_case_perf2 --release`

#![allow(dead_code)]
#![allow(unsafe_op_in_unsafe_fn)]
#![allow(clippy::excessive_precision)]
#![allow(clippy::approx_constant)]

use std::arch::x86_64::*;
use std::time::Instant;

const N: usize = 32 * 1024;
const ITERATIONS: u32 = 1000;

fn black_box<T>(x: T) -> T {
    std::hint::black_box(x)
}

fn bench<F>(_name: &str, mut f: F) -> f64
where
    F: FnMut(),
{
    for _ in 0..10 {
        f();
    }
    let start = Instant::now();
    for _ in 0..ITERATIONS {
        f();
    }
    let elapsed = start.elapsed();
    elapsed.as_nanos() as f64 / ITERATIONS as f64
}

// ============================================================================
// cbrt implementations
// ============================================================================

#[target_feature(enable = "avx2,fma")]
unsafe fn cbrt_no_checks(x: __m256) -> __m256 {
    const B1: u32 = 709_958_130;
    let x_arr: [f32; 8] = std::mem::transmute(x);
    let mut y_arr = [0.0f32; 8];
    for i in 0..8 {
        let ui = x_arr[i].to_bits() & 0x7FFF_FFFF;
        y_arr[i] = f32::from_bits(ui / 3 + B1);
    }
    let abs_x = _mm256_andnot_ps(_mm256_set1_ps(-0.0), x);
    let sign_mask = _mm256_and_ps(x, _mm256_set1_ps(-0.0));
    let mut y: __m256 = std::mem::transmute(y_arr);
    for _ in 0..2 {
        let y2 = _mm256_mul_ps(y, y);
        let y3 = _mm256_mul_ps(y2, y);
        let num = _mm256_fmadd_ps(_mm256_set1_ps(2.0), abs_x, y3);
        let den = _mm256_fmadd_ps(_mm256_set1_ps(2.0), y3, abs_x);
        y = _mm256_mul_ps(y, _mm256_div_ps(num, den));
    }
    _mm256_or_ps(y, sign_mask)
}

#[target_feature(enable = "avx2,fma")]
unsafe fn cbrt_std_match(x: __m256) -> __m256 {
    const B1: u32 = 709_958_130;
    let abs_x = _mm256_andnot_ps(_mm256_set1_ps(-0.0), x);
    let sign_mask = _mm256_and_ps(x, _mm256_set1_ps(-0.0));

    let zero = _mm256_setzero_ps();
    let is_zero = _mm256_cmp_ps(abs_x, zero, _CMP_EQ_OQ);
    let inf = _mm256_set1_ps(f32::INFINITY);
    let is_inf = _mm256_cmp_ps(abs_x, inf, _CMP_EQ_OQ);
    let is_nan = _mm256_cmp_ps(x, x, _CMP_UNORD_Q);

    let min_normal = _mm256_set1_ps(f32::MIN_POSITIVE);
    let is_denormal = _mm256_andnot_ps(is_zero, _mm256_cmp_ps(abs_x, min_normal, _CMP_LT_OQ));
    let scale_up = _mm256_set1_ps(16777216.0);
    let scale_down = _mm256_set1_ps(1.0 / 256.0);
    let scaled_x = _mm256_blendv_ps(abs_x, _mm256_mul_ps(abs_x, scale_up), is_denormal);

    let x_arr: [f32; 8] = std::mem::transmute(scaled_x);
    let mut y_arr = [0.0f32; 8];
    for i in 0..8 {
        let ui = x_arr[i].to_bits() & 0x7FFF_FFFF;
        y_arr[i] = f32::from_bits(ui / 3 + B1);
    }
    let mut y: __m256 = std::mem::transmute(y_arr);
    for _ in 0..2 {
        let y2 = _mm256_mul_ps(y, y);
        let y3 = _mm256_mul_ps(y2, y);
        let num = _mm256_fmadd_ps(_mm256_set1_ps(2.0), scaled_x, y3);
        let den = _mm256_fmadd_ps(_mm256_set1_ps(2.0), y3, scaled_x);
        y = _mm256_mul_ps(y, _mm256_div_ps(num, den));
    }
    y = _mm256_blendv_ps(y, _mm256_mul_ps(y, scale_down), is_denormal);
    y = _mm256_or_ps(y, sign_mask);
    y = _mm256_blendv_ps(y, _mm256_or_ps(zero, sign_mask), is_zero);
    y = _mm256_blendv_ps(y, _mm256_or_ps(inf, sign_mask), is_inf);
    y = _mm256_blendv_ps(y, x, is_nan);
    y
}

// ============================================================================
// exp2 implementations
// ============================================================================

#[target_feature(enable = "avx2,fma")]
unsafe fn exp2_no_checks(x: __m256) -> __m256 {
    let x = _mm256_max_ps(x, _mm256_set1_ps(-126.0));
    let x = _mm256_min_ps(x, _mm256_set1_ps(126.0));
    let xi = _mm256_floor_ps(x);
    let xf = _mm256_sub_ps(x, xi);
    let poly = _mm256_fmadd_ps(
        _mm256_set1_ps(0.001_333_55),
        xf,
        _mm256_set1_ps(0.009_618_13),
    );
    let poly = _mm256_fmadd_ps(poly, xf, _mm256_set1_ps(0.055_504_11));
    let poly = _mm256_fmadd_ps(poly, xf, _mm256_set1_ps(0.240_226_51));
    let poly = _mm256_fmadd_ps(poly, xf, _mm256_set1_ps(0.693_147_18));
    let poly = _mm256_fmadd_ps(poly, xf, _mm256_set1_ps(1.0));
    let xi_i32 = _mm256_cvtps_epi32(xi);
    let scale = _mm256_castsi256_ps(_mm256_slli_epi32(
        _mm256_add_epi32(xi_i32, _mm256_set1_epi32(127)),
        23,
    ));
    _mm256_mul_ps(poly, scale)
}

#[target_feature(enable = "avx2,fma")]
unsafe fn exp2_std_match(x: __m256) -> __m256 {
    let is_nan = _mm256_cmp_ps(x, x, _CMP_UNORD_Q);
    let is_overflow = _mm256_cmp_ps(x, _mm256_set1_ps(128.0), _CMP_GE_OQ);
    let is_underflow = _mm256_cmp_ps(x, _mm256_set1_ps(-150.0), _CMP_LE_OQ);

    let x_clamped = _mm256_max_ps(x, _mm256_set1_ps(-126.0));
    let x_clamped = _mm256_min_ps(x_clamped, _mm256_set1_ps(126.0));
    let xi = _mm256_floor_ps(x_clamped);
    let xf = _mm256_sub_ps(x_clamped, xi);
    let poly = _mm256_fmadd_ps(
        _mm256_set1_ps(0.001_333_55),
        xf,
        _mm256_set1_ps(0.009_618_13),
    );
    let poly = _mm256_fmadd_ps(poly, xf, _mm256_set1_ps(0.055_504_11));
    let poly = _mm256_fmadd_ps(poly, xf, _mm256_set1_ps(0.240_226_51));
    let poly = _mm256_fmadd_ps(poly, xf, _mm256_set1_ps(0.693_147_18));
    let poly = _mm256_fmadd_ps(poly, xf, _mm256_set1_ps(1.0));
    let xi_i32 = _mm256_cvtps_epi32(xi);
    let scale = _mm256_castsi256_ps(_mm256_slli_epi32(
        _mm256_add_epi32(xi_i32, _mm256_set1_epi32(127)),
        23,
    ));
    let mut result = _mm256_mul_ps(poly, scale);
    result = _mm256_blendv_ps(result, _mm256_set1_ps(f32::INFINITY), is_overflow);
    result = _mm256_blendv_ps(result, _mm256_setzero_ps(), is_underflow);
    result = _mm256_blendv_ps(result, x, is_nan);
    result
}

// ============================================================================
// log2 implementations
// ============================================================================

#[target_feature(enable = "avx2,fma")]
unsafe fn log2_no_checks(x: __m256) -> __m256 {
    const SQRT2_OVER_2: u32 = 0x3f3504f3;
    const ONE: u32 = 0x3f800000;
    let bits = _mm256_castps_si256(x);
    let adjusted = _mm256_add_epi32(bits, _mm256_set1_epi32((ONE - SQRT2_OVER_2) as i32));
    let exp_i32 = _mm256_sub_epi32(_mm256_srli_epi32(adjusted, 23), _mm256_set1_epi32(0x7f));
    let n = _mm256_cvtepi32_ps(exp_i32);
    let mantissa_bits = _mm256_add_epi32(
        _mm256_and_si256(adjusted, _mm256_set1_epi32(0x007fffff)),
        _mm256_set1_epi32(SQRT2_OVER_2 as i32),
    );
    let a = _mm256_castsi256_ps(mantissa_bits);
    let one = _mm256_set1_ps(1.0);
    let y = _mm256_div_ps(_mm256_sub_ps(a, one), _mm256_add_ps(a, one));
    let y2 = _mm256_mul_ps(y, y);
    let mut u = _mm256_fmadd_ps(
        _mm256_set1_ps(0.412_198_57),
        y2,
        _mm256_set1_ps(0.577_078_04),
    );
    u = _mm256_fmadd_ps(u, y2, _mm256_set1_ps(0.961_796_7));
    u = _mm256_fmadd_ps(u, y2, _mm256_set1_ps(2.885_390_08));
    _mm256_fmadd_ps(u, y, n)
}

#[target_feature(enable = "avx2,fma")]
unsafe fn log2_std_match(x: __m256) -> __m256 {
    let zero = _mm256_setzero_ps();
    let is_zero = _mm256_cmp_ps(x, zero, _CMP_EQ_OQ);
    let is_negative = _mm256_cmp_ps(x, zero, _CMP_LT_OQ);
    let is_nan = _mm256_cmp_ps(x, x, _CMP_UNORD_Q);
    let inf = _mm256_set1_ps(f32::INFINITY);
    let is_inf = _mm256_cmp_ps(x, inf, _CMP_EQ_OQ);

    const SQRT2_OVER_2: u32 = 0x3f3504f3;
    const ONE: u32 = 0x3f800000;
    let bits = _mm256_castps_si256(x);
    let adjusted = _mm256_add_epi32(bits, _mm256_set1_epi32((ONE - SQRT2_OVER_2) as i32));
    let exp_i32 = _mm256_sub_epi32(_mm256_srli_epi32(adjusted, 23), _mm256_set1_epi32(0x7f));
    let n = _mm256_cvtepi32_ps(exp_i32);
    let mantissa_bits = _mm256_add_epi32(
        _mm256_and_si256(adjusted, _mm256_set1_epi32(0x007fffff)),
        _mm256_set1_epi32(SQRT2_OVER_2 as i32),
    );
    let a = _mm256_castsi256_ps(mantissa_bits);
    let one = _mm256_set1_ps(1.0);
    let y = _mm256_div_ps(_mm256_sub_ps(a, one), _mm256_add_ps(a, one));
    let y2 = _mm256_mul_ps(y, y);
    let mut u = _mm256_fmadd_ps(
        _mm256_set1_ps(0.412_198_57),
        y2,
        _mm256_set1_ps(0.577_078_04),
    );
    u = _mm256_fmadd_ps(u, y2, _mm256_set1_ps(0.961_796_7));
    u = _mm256_fmadd_ps(u, y2, _mm256_set1_ps(2.885_390_08));
    let mut result = _mm256_fmadd_ps(u, y, n);
    result = _mm256_blendv_ps(result, _mm256_set1_ps(f32::NEG_INFINITY), is_zero);
    result = _mm256_blendv_ps(result, _mm256_set1_ps(f32::NAN), is_negative);
    result = _mm256_blendv_ps(result, inf, is_inf);
    result = _mm256_blendv_ps(result, x, is_nan);
    result
}

// ============================================================================
// pow implementations (uses log2 + exp2)
// ============================================================================

#[target_feature(enable = "avx2,fma")]
unsafe fn pow_no_checks(x: __m256, n: f32) -> __m256 {
    let lg = log2_no_checks(x);
    exp2_no_checks(_mm256_mul_ps(_mm256_set1_ps(n), lg))
}

#[target_feature(enable = "avx2,fma")]
unsafe fn pow_std_match(x: __m256, n: f32) -> __m256 {
    // pow(x, n) edge cases: x=0, x<0, x=inf, x=NaN, and result overflow/underflow
    let zero = _mm256_setzero_ps();
    let is_zero = _mm256_cmp_ps(x, zero, _CMP_EQ_OQ);
    let is_negative = _mm256_cmp_ps(x, zero, _CMP_LT_OQ);
    let is_nan = _mm256_cmp_ps(x, x, _CMP_UNORD_Q);
    let inf = _mm256_set1_ps(f32::INFINITY);
    let is_inf = _mm256_cmp_ps(x, inf, _CMP_EQ_OQ);

    let lg = log2_no_checks(x); // internal, no checks needed
    let scaled = _mm256_mul_ps(_mm256_set1_ps(n), lg);
    let mut result = exp2_std_match(scaled); // handles overflow/underflow

    // pow(0, n) = 0 for n > 0, inf for n < 0, 1 for n = 0
    let pow_zero = if n > 0.0 {
        zero
    } else if n < 0.0 {
        inf
    } else {
        _mm256_set1_ps(1.0)
    };
    result = _mm256_blendv_ps(result, pow_zero, is_zero);

    // pow(negative, n) = NaN for non-integer n
    result = _mm256_blendv_ps(result, _mm256_set1_ps(f32::NAN), is_negative);

    // pow(inf, n) = inf for n > 0, 0 for n < 0
    let pow_inf = if n > 0.0 { inf } else { zero };
    result = _mm256_blendv_ps(result, pow_inf, is_inf);

    // NaN passthrough
    result = _mm256_blendv_ps(result, x, is_nan);

    result
}

// ============================================================================
// exp implementations (exp(x) = exp2(x * log2(e)))
// ============================================================================

#[target_feature(enable = "avx2,fma")]
unsafe fn exp_no_checks(x: __m256) -> __m256 {
    exp2_no_checks(_mm256_mul_ps(x, _mm256_set1_ps(std::f32::consts::LOG2_E)))
}

#[target_feature(enable = "avx2,fma")]
unsafe fn exp_std_match(x: __m256) -> __m256 {
    exp2_std_match(_mm256_mul_ps(x, _mm256_set1_ps(std::f32::consts::LOG2_E)))
}

// ============================================================================
// ln implementations (ln(x) = log2(x) * ln(2))
// ============================================================================

#[target_feature(enable = "avx2,fma")]
unsafe fn ln_no_checks(x: __m256) -> __m256 {
    _mm256_mul_ps(log2_no_checks(x), _mm256_set1_ps(std::f32::consts::LN_2))
}

#[target_feature(enable = "avx2,fma")]
unsafe fn ln_std_match(x: __m256) -> __m256 {
    _mm256_mul_ps(log2_std_match(x), _mm256_set1_ps(std::f32::consts::LN_2))
}

// ============================================================================
// Benchmark
// ============================================================================

fn run_bench<F>(name: &str, input: &[f32], output: &mut [f32], f: F) -> f64
where
    F: Fn(*const f32, *mut f32, usize),
{
    let ns = bench(name, || {
        f(input.as_ptr(), output.as_mut_ptr(), N);
        black_box(&output);
    });
    let throughput = (N as f64 * 1e9) / (ns * 1e6);
    println!("{:45} {:>10.2} ns  {:>8.2} M/s", name, ns, throughput);
    ns
}

fn main() {
    if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
        eprintln!("AVX2+FMA not available");
        return;
    }

    println!("Easy vs Hard Data Performance Comparison");
    println!("=========================================");
    println!("N = {} elements, {} iterations\n", N, ITERATIONS);

    // Easy data: normal range, no edge cases
    let easy_pos: Vec<f32> = (1..=N)
        .map(|i| 0.001 + (i as f32 / N as f32) * 999.0)
        .collect();
    let easy_exp: Vec<f32> = (0..N)
        .map(|i| -20.0 + (i as f32 / N as f32) * 40.0)
        .collect();

    // Hard data: includes edge cases (0, inf, negative, denormals, NaN)
    let mut hard_pos: Vec<f32> = easy_pos.clone();
    let mut hard_exp: Vec<f32> = easy_exp.clone();
    // Sprinkle in edge cases (every 1000th element)
    for i in (0..N).step_by(1000) {
        hard_pos[i] = match i % 5000 {
            0 => 0.0,
            1000 => f32::INFINITY,
            2000 => -1.0,
            3000 => 1e-45, // denormal
            4000 => f32::NAN,
            _ => hard_pos[i],
        };
        hard_exp[i] = match i % 4000 {
            0 => 200.0,     // overflow
            1000 => -200.0, // underflow
            2000 => f32::INFINITY,
            3000 => f32::NAN,
            _ => hard_exp[i],
        };
    }

    let mut output = vec![0.0f32; N];

    // ========================================================================
    println!("=== cbrt ===");
    // ========================================================================

    let easy_no = run_bench(
        "cbrt no_checks (easy data)",
        &easy_pos,
        &mut output,
        |i, o, n| unsafe {
            for j in 0..(n / 8) {
                _mm256_storeu_ps(o.add(j * 8), cbrt_no_checks(_mm256_loadu_ps(i.add(j * 8))));
            }
        },
    );
    let easy_std = run_bench(
        "cbrt std_match (easy data)",
        &easy_pos,
        &mut output,
        |i, o, n| unsafe {
            for j in 0..(n / 8) {
                _mm256_storeu_ps(o.add(j * 8), cbrt_std_match(_mm256_loadu_ps(i.add(j * 8))));
            }
        },
    );
    let hard_no = run_bench(
        "cbrt no_checks (hard data)",
        &hard_pos,
        &mut output,
        |i, o, n| unsafe {
            for j in 0..(n / 8) {
                _mm256_storeu_ps(o.add(j * 8), cbrt_no_checks(_mm256_loadu_ps(i.add(j * 8))));
            }
        },
    );
    let hard_std = run_bench(
        "cbrt std_match (hard data)",
        &hard_pos,
        &mut output,
        |i, o, n| unsafe {
            for j in 0..(n / 8) {
                _mm256_storeu_ps(o.add(j * 8), cbrt_std_match(_mm256_loadu_ps(i.add(j * 8))));
            }
        },
    );
    println!(
        "  easy: std_match overhead = {:.1}%",
        (easy_std / easy_no - 1.0) * 100.0
    );
    println!(
        "  hard: std_match overhead = {:.1}%",
        (hard_std / hard_no - 1.0) * 100.0
    );
    println!();

    // ========================================================================
    println!("=== exp2 ===");
    // ========================================================================

    let easy_no = run_bench(
        "exp2 no_checks (easy data)",
        &easy_exp,
        &mut output,
        |i, o, n| unsafe {
            for j in 0..(n / 8) {
                _mm256_storeu_ps(o.add(j * 8), exp2_no_checks(_mm256_loadu_ps(i.add(j * 8))));
            }
        },
    );
    let easy_std = run_bench(
        "exp2 std_match (easy data)",
        &easy_exp,
        &mut output,
        |i, o, n| unsafe {
            for j in 0..(n / 8) {
                _mm256_storeu_ps(o.add(j * 8), exp2_std_match(_mm256_loadu_ps(i.add(j * 8))));
            }
        },
    );
    let hard_no = run_bench(
        "exp2 no_checks (hard data)",
        &hard_exp,
        &mut output,
        |i, o, n| unsafe {
            for j in 0..(n / 8) {
                _mm256_storeu_ps(o.add(j * 8), exp2_no_checks(_mm256_loadu_ps(i.add(j * 8))));
            }
        },
    );
    let hard_std = run_bench(
        "exp2 std_match (hard data)",
        &hard_exp,
        &mut output,
        |i, o, n| unsafe {
            for j in 0..(n / 8) {
                _mm256_storeu_ps(o.add(j * 8), exp2_std_match(_mm256_loadu_ps(i.add(j * 8))));
            }
        },
    );
    println!(
        "  easy: std_match overhead = {:.1}%",
        (easy_std / easy_no - 1.0) * 100.0
    );
    println!(
        "  hard: std_match overhead = {:.1}%",
        (hard_std / hard_no - 1.0) * 100.0
    );
    println!();

    // ========================================================================
    println!("=== log2 ===");
    // ========================================================================

    let easy_no = run_bench(
        "log2 no_checks (easy data)",
        &easy_pos,
        &mut output,
        |i, o, n| unsafe {
            for j in 0..(n / 8) {
                _mm256_storeu_ps(o.add(j * 8), log2_no_checks(_mm256_loadu_ps(i.add(j * 8))));
            }
        },
    );
    let easy_std = run_bench(
        "log2 std_match (easy data)",
        &easy_pos,
        &mut output,
        |i, o, n| unsafe {
            for j in 0..(n / 8) {
                _mm256_storeu_ps(o.add(j * 8), log2_std_match(_mm256_loadu_ps(i.add(j * 8))));
            }
        },
    );
    let hard_no = run_bench(
        "log2 no_checks (hard data)",
        &hard_pos,
        &mut output,
        |i, o, n| unsafe {
            for j in 0..(n / 8) {
                _mm256_storeu_ps(o.add(j * 8), log2_no_checks(_mm256_loadu_ps(i.add(j * 8))));
            }
        },
    );
    let hard_std = run_bench(
        "log2 std_match (hard data)",
        &hard_pos,
        &mut output,
        |i, o, n| unsafe {
            for j in 0..(n / 8) {
                _mm256_storeu_ps(o.add(j * 8), log2_std_match(_mm256_loadu_ps(i.add(j * 8))));
            }
        },
    );
    println!(
        "  easy: std_match overhead = {:.1}%",
        (easy_std / easy_no - 1.0) * 100.0
    );
    println!(
        "  hard: std_match overhead = {:.1}%",
        (hard_std / hard_no - 1.0) * 100.0
    );
    println!();

    // ========================================================================
    println!("=== pow(x, 2.4) ===");
    // ========================================================================

    let easy_no = run_bench(
        "pow no_checks (easy data)",
        &easy_pos,
        &mut output,
        |i, o, n| unsafe {
            for j in 0..(n / 8) {
                _mm256_storeu_ps(
                    o.add(j * 8),
                    pow_no_checks(_mm256_loadu_ps(i.add(j * 8)), 2.4),
                );
            }
        },
    );
    let easy_std = run_bench(
        "pow std_match (easy data)",
        &easy_pos,
        &mut output,
        |i, o, n| unsafe {
            for j in 0..(n / 8) {
                _mm256_storeu_ps(
                    o.add(j * 8),
                    pow_std_match(_mm256_loadu_ps(i.add(j * 8)), 2.4),
                );
            }
        },
    );
    let hard_no = run_bench(
        "pow no_checks (hard data)",
        &hard_pos,
        &mut output,
        |i, o, n| unsafe {
            for j in 0..(n / 8) {
                _mm256_storeu_ps(
                    o.add(j * 8),
                    pow_no_checks(_mm256_loadu_ps(i.add(j * 8)), 2.4),
                );
            }
        },
    );
    let hard_std = run_bench(
        "pow std_match (hard data)",
        &hard_pos,
        &mut output,
        |i, o, n| unsafe {
            for j in 0..(n / 8) {
                _mm256_storeu_ps(
                    o.add(j * 8),
                    pow_std_match(_mm256_loadu_ps(i.add(j * 8)), 2.4),
                );
            }
        },
    );
    println!(
        "  easy: std_match overhead = {:.1}%",
        (easy_std / easy_no - 1.0) * 100.0
    );
    println!(
        "  hard: std_match overhead = {:.1}%",
        (hard_std / hard_no - 1.0) * 100.0
    );
    println!();

    // ========================================================================
    println!("=== exp ===");
    // ========================================================================

    let easy_no = run_bench(
        "exp no_checks (easy data)",
        &easy_exp,
        &mut output,
        |i, o, n| unsafe {
            for j in 0..(n / 8) {
                _mm256_storeu_ps(o.add(j * 8), exp_no_checks(_mm256_loadu_ps(i.add(j * 8))));
            }
        },
    );
    let easy_std = run_bench(
        "exp std_match (easy data)",
        &easy_exp,
        &mut output,
        |i, o, n| unsafe {
            for j in 0..(n / 8) {
                _mm256_storeu_ps(o.add(j * 8), exp_std_match(_mm256_loadu_ps(i.add(j * 8))));
            }
        },
    );
    let hard_no = run_bench(
        "exp no_checks (hard data)",
        &hard_exp,
        &mut output,
        |i, o, n| unsafe {
            for j in 0..(n / 8) {
                _mm256_storeu_ps(o.add(j * 8), exp_no_checks(_mm256_loadu_ps(i.add(j * 8))));
            }
        },
    );
    let hard_std = run_bench(
        "exp std_match (hard data)",
        &hard_exp,
        &mut output,
        |i, o, n| unsafe {
            for j in 0..(n / 8) {
                _mm256_storeu_ps(o.add(j * 8), exp_std_match(_mm256_loadu_ps(i.add(j * 8))));
            }
        },
    );
    println!(
        "  easy: std_match overhead = {:.1}%",
        (easy_std / easy_no - 1.0) * 100.0
    );
    println!(
        "  hard: std_match overhead = {:.1}%",
        (hard_std / hard_no - 1.0) * 100.0
    );
    println!();

    // ========================================================================
    println!("=== ln ===");
    // ========================================================================

    let easy_no = run_bench(
        "ln no_checks (easy data)",
        &easy_pos,
        &mut output,
        |i, o, n| unsafe {
            for j in 0..(n / 8) {
                _mm256_storeu_ps(o.add(j * 8), ln_no_checks(_mm256_loadu_ps(i.add(j * 8))));
            }
        },
    );
    let easy_std = run_bench(
        "ln std_match (easy data)",
        &easy_pos,
        &mut output,
        |i, o, n| unsafe {
            for j in 0..(n / 8) {
                _mm256_storeu_ps(o.add(j * 8), ln_std_match(_mm256_loadu_ps(i.add(j * 8))));
            }
        },
    );
    let hard_no = run_bench(
        "ln no_checks (hard data)",
        &hard_pos,
        &mut output,
        |i, o, n| unsafe {
            for j in 0..(n / 8) {
                _mm256_storeu_ps(o.add(j * 8), ln_no_checks(_mm256_loadu_ps(i.add(j * 8))));
            }
        },
    );
    let hard_std = run_bench(
        "ln std_match (hard data)",
        &hard_pos,
        &mut output,
        |i, o, n| unsafe {
            for j in 0..(n / 8) {
                _mm256_storeu_ps(o.add(j * 8), ln_std_match(_mm256_loadu_ps(i.add(j * 8))));
            }
        },
    );
    println!(
        "  easy: std_match overhead = {:.1}%",
        (easy_std / easy_no - 1.0) * 100.0
    );
    println!(
        "  hard: std_match overhead = {:.1}%",
        (hard_std / hard_no - 1.0) * 100.0
    );
    println!();

    println!("Done!");
}
