//! Transcendental function benchmark - comparing scalar vs SIMD implementations.
//!
//! Run with: `cargo run --example transcendental_bench --release`
//!
//! This example iterates on different implementations of exp2, log2, and pow
//! to find the best approach for archmage's SIMD types.

#![allow(dead_code)]
#![allow(unsafe_op_in_unsafe_fn)]

use std::time::Instant;

// Test data size - large enough to amortize overhead, small enough to fit in L2
const N: usize = 32 * 1024;
const ITERATIONS: u32 = 1000;

// ============================================================================
// Scalar implementations
// ============================================================================

/// Baseline: std library
fn scalar_std_exp2(input: &[f32], output: &mut [f32]) {
    for (o, &x) in output.iter_mut().zip(input.iter()) {
        *o = x.exp2();
    }
}

fn scalar_std_log2(input: &[f32], output: &mut [f32]) {
    for (o, &x) in output.iter_mut().zip(input.iter()) {
        *o = x.log2();
    }
}

fn scalar_std_pow(input: &[f32], exp: f32, output: &mut [f32]) {
    for (o, &x) in output.iter_mut().zip(input.iter()) {
        *o = x.powf(exp);
    }
}

/// Fast scalar log2 using bit manipulation + polynomial
/// From butteraugli/jpegli
fn fast_log2f_scalar(x: f32) -> f32 {
    // Rational polynomial coefficients
    const P0: f32 = -1.850_383_34e-6;
    const P1: f32 = 1.428_716_05;
    const P2: f32 = 0.742_458_73;
    const Q0: f32 = 0.990_328_14;
    const Q1: f32 = 1.009_671_86;
    const Q2: f32 = 0.174_093_43;

    let x_bits = x.to_bits() as i32;
    let exp_bits = x_bits.wrapping_sub(0x3f2aaaab_u32 as i32);
    let exp_shifted = exp_bits >> 23;
    let mantissa_bits = (x_bits - (exp_shifted << 23)) as u32;
    let mantissa = f32::from_bits(mantissa_bits);
    let exp_val = exp_shifted as f32;

    let m = mantissa - 1.0;
    let yp = P2.mul_add(m, P1).mul_add(m, P0);
    let yq = Q2.mul_add(m, Q1).mul_add(m, Q0);

    yp / yq + exp_val
}

/// Fast scalar exp2 using bit manipulation + polynomial
fn fast_exp2f_scalar(x: f32) -> f32 {
    // Clamp to avoid overflow/underflow
    let x = x.clamp(-126.0, 126.0);

    // Split into integer and fractional parts
    let xi = x.floor();
    let xf = x - xi;

    // Polynomial for 2^frac, frac in [0, 1)
    // Minimax polynomial coefficients for 2^x on [0, 1]
    const C0: f32 = 1.0;
    const C1: f32 = 0.693_147_18; // ln(2)
    const C2: f32 = 0.240_226_5;
    const C3: f32 = 0.055_504_11;

    let poly = C3.mul_add(xf, C2).mul_add(xf, C1).mul_add(xf, C0);

    // Scale by 2^integer using bit manipulation
    let scale_bits = ((xi as i32 + 127) << 23) as u32;
    let scale = f32::from_bits(scale_bits);

    poly * scale
}

/// Fast scalar pow using log2 + exp2
fn fast_powf_scalar(x: f32, n: f32) -> f32 {
    fast_exp2f_scalar(n * fast_log2f_scalar(x))
}

fn scalar_fast_exp2(input: &[f32], output: &mut [f32]) {
    for (o, &x) in output.iter_mut().zip(input.iter()) {
        *o = fast_exp2f_scalar(x);
    }
}

fn scalar_fast_log2(input: &[f32], output: &mut [f32]) {
    for (o, &x) in output.iter_mut().zip(input.iter()) {
        *o = fast_log2f_scalar(x);
    }
}

fn scalar_fast_pow(input: &[f32], exp: f32, output: &mut [f32]) {
    for (o, &x) in output.iter_mut().zip(input.iter()) {
        *o = fast_powf_scalar(x, exp);
    }
}

// ============================================================================
// SIMD implementations using raw intrinsics
// ============================================================================

#[cfg(target_arch = "x86_64")]
mod simd_avx2 {
    use std::arch::x86_64::*;

    /// Check if AVX2+FMA available
    pub fn available() -> bool {
        is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma")
    }

    /// Fast log2 for 8 f32s using AVX2
    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn log2_x8(x: __m256) -> __m256 {
        // Constants
        const P0: f32 = -1.850_383_34e-6;
        const P1: f32 = 1.428_716_05;
        const P2: f32 = 0.742_458_73;
        const Q0: f32 = 0.990_328_14;
        const Q1: f32 = 1.009_671_86;
        const Q2: f32 = 0.174_093_43;

        let x_bits = _mm256_castps_si256(x);
        let offset = _mm256_set1_epi32(0x3f2aaaab_u32 as i32);
        let exp_bits = _mm256_sub_epi32(x_bits, offset);
        let exp_shifted = _mm256_srai_epi32(exp_bits, 23);

        let mantissa_bits = _mm256_sub_epi32(x_bits, _mm256_slli_epi32(exp_shifted, 23));
        let mantissa = _mm256_castsi256_ps(mantissa_bits);
        let exp_val = _mm256_cvtepi32_ps(exp_shifted);

        let one = _mm256_set1_ps(1.0);
        let m = _mm256_sub_ps(mantissa, one);

        // Horner's for numerator
        let yp = _mm256_fmadd_ps(_mm256_set1_ps(P2), m, _mm256_set1_ps(P1));
        let yp = _mm256_fmadd_ps(yp, m, _mm256_set1_ps(P0));

        // Horner's for denominator
        let yq = _mm256_fmadd_ps(_mm256_set1_ps(Q2), m, _mm256_set1_ps(Q1));
        let yq = _mm256_fmadd_ps(yq, m, _mm256_set1_ps(Q0));

        _mm256_add_ps(_mm256_div_ps(yp, yq), exp_val)
    }

    /// Fast exp2 for 8 f32s using AVX2
    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn exp2_x8(x: __m256) -> __m256 {
        // Clamp to safe range
        let x = _mm256_max_ps(x, _mm256_set1_ps(-126.0));
        let x = _mm256_min_ps(x, _mm256_set1_ps(126.0));

        // Split into integer and fractional
        let xi = _mm256_floor_ps(x);
        let xf = _mm256_sub_ps(x, xi);

        // Polynomial for 2^frac
        const C0: f32 = 1.0;
        const C1: f32 = 0.693_147_18;
        const C2: f32 = 0.240_226_5;
        const C3: f32 = 0.055_504_11;

        let poly = _mm256_fmadd_ps(_mm256_set1_ps(C3), xf, _mm256_set1_ps(C2));
        let poly = _mm256_fmadd_ps(poly, xf, _mm256_set1_ps(C1));
        let poly = _mm256_fmadd_ps(poly, xf, _mm256_set1_ps(C0));

        // Scale by 2^integer
        let xi_i32 = _mm256_cvtps_epi32(xi);
        let bias = _mm256_set1_epi32(127);
        let scale_bits = _mm256_slli_epi32(_mm256_add_epi32(xi_i32, bias), 23);
        let scale = _mm256_castsi256_ps(scale_bits);

        _mm256_mul_ps(poly, scale)
    }

    /// Fast pow for 8 f32s: x^n = exp2(n * log2(x))
    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn pow_x8(x: __m256, n: f32) -> __m256 {
        let lg = log2_x8(x);
        exp2_x8(_mm256_mul_ps(_mm256_set1_ps(n), lg))
    }

    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn simd_exp2(input: &[f32], output: &mut [f32]) {
        let chunks = input.len() / 8;
        for i in 0..chunks {
            let x = _mm256_loadu_ps(input.as_ptr().add(i * 8));
            let r = exp2_x8(x);
            _mm256_storeu_ps(output.as_mut_ptr().add(i * 8), r);
        }
        // Handle remainder with scalar
        for i in (chunks * 8)..input.len() {
            output[i] = super::fast_exp2f_scalar(input[i]);
        }
    }

    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn simd_log2(input: &[f32], output: &mut [f32]) {
        let chunks = input.len() / 8;
        for i in 0..chunks {
            let x = _mm256_loadu_ps(input.as_ptr().add(i * 8));
            let r = log2_x8(x);
            _mm256_storeu_ps(output.as_mut_ptr().add(i * 8), r);
        }
        for i in (chunks * 8)..input.len() {
            output[i] = super::fast_log2f_scalar(input[i]);
        }
    }

    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn simd_pow(input: &[f32], exp: f32, output: &mut [f32]) {
        let chunks = input.len() / 8;
        for i in 0..chunks {
            let x = _mm256_loadu_ps(input.as_ptr().add(i * 8));
            let r = pow_x8(x, exp);
            _mm256_storeu_ps(output.as_mut_ptr().add(i * 8), r);
        }
        for i in (chunks * 8)..input.len() {
            output[i] = super::fast_powf_scalar(input[i], exp);
        }
    }
}

// ============================================================================
// Higher-precision SIMD implementations
// ============================================================================

#[cfg(target_arch = "x86_64")]
mod simd_avx2_hp {
    //! Higher precision variants with more polynomial terms

    use std::arch::x86_64::*;

    /// Higher-precision log2 with degree-4 polynomial
    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn log2_x8_hp(x: __m256) -> __m256 {
        // Use a degree-4 polynomial for better precision
        // Coefficients from minimax fitting on [sqrt(2)/2, sqrt(2)]
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
        let x = _mm256_div_ps(_mm256_sub_ps(a, one), _mm256_add_ps(a, one));
        let x2 = _mm256_mul_ps(x, x);

        // Polynomial: log2(1+y)/(1-y) where y = (a-1)/(a+1)
        // p(x^2) = c0 + c1*x^2 + c2*x^4
        const C0: f32 = 0.961_796_7;
        const C1: f32 = 0.577_078_04;
        const C2: f32 = 0.412_198_57;

        let mut u = _mm256_set1_ps(C2);
        u = _mm256_fmadd_ps(u, x2, _mm256_set1_ps(C1));
        u = _mm256_fmadd_ps(u, x2, _mm256_set1_ps(C0));

        // result = x * u + 2/ln(2) * x + n
        let log2_scale = _mm256_set1_ps(2.885_39); // 2/ln(2)
        let x3 = _mm256_mul_ps(x2, x);
        _mm256_add_ps(_mm256_fmadd_ps(x3, u, _mm256_mul_ps(x, log2_scale)), n)
    }

    /// Higher-precision exp2 with degree-5 polynomial
    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn exp2_x8_hp(x: __m256) -> __m256 {
        let x = _mm256_max_ps(x, _mm256_set1_ps(-126.0));
        let x = _mm256_min_ps(x, _mm256_set1_ps(126.0));

        let xi = _mm256_floor_ps(x);
        let xf = _mm256_sub_ps(x, xi);

        // Degree-5 minimax polynomial for 2^x on [0, 1]
        const C0: f32 = 1.0;
        const C1: f32 = 0.693_147_18; // ln(2)
        const C2: f32 = 0.240_226_51; // ln(2)^2 / 2
        const C3: f32 = 0.055_504_11; // ln(2)^3 / 6
        const C4: f32 = 0.009_618_13; // ln(2)^4 / 24
        const C5: f32 = 0.001_333_55; // ln(2)^5 / 120

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

    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn simd_exp2_hp(input: &[f32], output: &mut [f32]) {
        let chunks = input.len() / 8;
        for i in 0..chunks {
            let x = _mm256_loadu_ps(input.as_ptr().add(i * 8));
            let r = exp2_x8_hp(x);
            _mm256_storeu_ps(output.as_mut_ptr().add(i * 8), r);
        }
        for i in (chunks * 8)..input.len() {
            output[i] = input[i].exp2();
        }
    }

    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn simd_log2_hp(input: &[f32], output: &mut [f32]) {
        let chunks = input.len() / 8;
        for i in 0..chunks {
            let x = _mm256_loadu_ps(input.as_ptr().add(i * 8));
            let r = log2_x8_hp(x);
            _mm256_storeu_ps(output.as_mut_ptr().add(i * 8), r);
        }
        for i in (chunks * 8)..input.len() {
            output[i] = input[i].log2();
        }
    }

    /// High-precision pow using HP log2 and HP exp2
    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn pow_x8_hp(x: __m256, n: f32) -> __m256 {
        let lg = log2_x8_hp(x);
        exp2_x8_hp(_mm256_mul_ps(_mm256_set1_ps(n), lg))
    }

    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn simd_pow_hp(input: &[f32], exp: f32, output: &mut [f32]) {
        let chunks = input.len() / 8;
        for i in 0..chunks {
            let x = _mm256_loadu_ps(input.as_ptr().add(i * 8));
            let r = pow_x8_hp(x, exp);
            _mm256_storeu_ps(output.as_mut_ptr().add(i * 8), r);
        }
        for i in (chunks * 8)..input.len() {
            output[i] = input[i].powf(exp);
        }
    }
}

// ============================================================================
// LUT-based SIMD implementation (from linear-srgb)
// ============================================================================

#[cfg(target_arch = "x86_64")]
mod simd_avx2_lut {
    //! LUT-based exp2 from linear-srgb crate - proven implementation

    use std::arch::x86_64::*;

    // 64-entry exp2 lookup table from linear-srgb
    // Each entry is 2^(i/64) scaled to fit in the mantissa range
    #[rustfmt::skip]
    static EXP2_TABLE: [u32; 64] = [
        0x3F3504F3, 0x3F36FD92, 0x3F38FBAF, 0x3F3AFF5B, 0x3F3D08A4, 0x3F3F179A, 0x3F412C4D, 0x3F4346CD,
        0x3F45672A, 0x3F478D75, 0x3F49B9BE, 0x3F4BEC15, 0x3F4E248C, 0x3F506334, 0x3F52A81E, 0x3F54F35B,
        0x3F5744FD, 0x3F599D16, 0x3F5BFBB8, 0x3F5E60F5, 0x3F60CCDF, 0x3F633F89, 0x3F65B907, 0x3F68396A,
        0x3F6AC0C7, 0x3F6D4F30, 0x3F6FE4BA, 0x3F728177, 0x3F75257D, 0x3F77D0DF, 0x3F7A83B3, 0x3F7D3E0C,
        0x3F800000, 0x3F8164D2, 0x3F82CD87, 0x3F843A29, 0x3F85AAC3, 0x3F871F62, 0x3F88980F, 0x3F8A14D5,
        0x3F8B95C2, 0x3F8D1ADF, 0x3F8EA43A, 0x3F9031DC, 0x3F91C3D3, 0x3F935A2B, 0x3F94F4F0, 0x3F96942D,
        0x3F9837F0, 0x3F99E046, 0x3F9B8D3A, 0x3F9D3EDA, 0x3F9EF532, 0x3FA0B051, 0x3FA27043, 0x3FA43516,
        0x3FA5FED7, 0x3FA7CD94, 0x3FA9A15B, 0x3FAB7A3A, 0x3FAD583F, 0x3FAF3B79, 0x3FB123F6, 0x3FB311C4,
    ];

    const TBLSIZE: usize = 64;
    const EXP2_C0: f32 = 0.240_226_5;
    #[allow(clippy::approx_constant)]
    const EXP2_C1: f32 = 0.693_147_2;

    /// LUT-based exp2 from linear-srgb - well-tested implementation
    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn exp2_x8_lut(x: __m256) -> __m256 {
        // Magic constant for rounding to nearest 1/64
        let redux = _mm256_set1_ps(f32::from_bits(0x4b400000) / TBLSIZE as f32);
        let sum = _mm256_add_ps(x, redux);
        let ui = _mm256_castps_si256(sum);

        // Get table index (low 6 bits, with rounding adjustment)
        let half_tbl = _mm256_set1_epi32((TBLSIZE / 2) as i32);
        let tbl_mask = _mm256_set1_epi32((TBLSIZE - 1) as i32);
        let i0 = _mm256_and_si256(_mm256_add_epi32(ui, half_tbl), tbl_mask);

        // Get exponent adjustment (high bits)
        let k = _mm256_srli_epi32(_mm256_add_epi32(ui, half_tbl), 6);

        // Get fractional part
        let uf = _mm256_sub_ps(sum, redux);
        let f = _mm256_sub_ps(x, uf);

        // Gather from table - scalar fallback for portability
        let i0_arr: [i32; 8] = core::mem::transmute(i0);
        let z0 = _mm256_set_ps(
            f32::from_bits(EXP2_TABLE[i0_arr[7] as usize]),
            f32::from_bits(EXP2_TABLE[i0_arr[6] as usize]),
            f32::from_bits(EXP2_TABLE[i0_arr[5] as usize]),
            f32::from_bits(EXP2_TABLE[i0_arr[4] as usize]),
            f32::from_bits(EXP2_TABLE[i0_arr[3] as usize]),
            f32::from_bits(EXP2_TABLE[i0_arr[2] as usize]),
            f32::from_bits(EXP2_TABLE[i0_arr[1] as usize]),
            f32::from_bits(EXP2_TABLE[i0_arr[0] as usize]),
        );

        // Polynomial refinement: z0 * (1 + C1*f + C0*f^2) ≈ z0 * (1 + C1*f*(1 + C0/C1*f))
        let mut u = _mm256_set1_ps(EXP2_C0);
        u = _mm256_fmadd_ps(u, f, _mm256_set1_ps(EXP2_C1));
        u = _mm256_mul_ps(u, f);

        let result_unscaled = _mm256_fmadd_ps(u, z0, z0);

        // Scale by 2^k using exponent manipulation
        let bias = _mm256_set1_epi32(0x7f);
        let scale_bits = _mm256_slli_epi32(_mm256_add_epi32(k, bias), 23);
        let scale = _mm256_castsi256_ps(scale_bits);

        _mm256_mul_ps(result_unscaled, scale)
    }

    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn simd_exp2_lut(input: &[f32], output: &mut [f32]) {
        let chunks = input.len() / 8;
        for i in 0..chunks {
            let x = _mm256_loadu_ps(input.as_ptr().add(i * 8));
            let r = exp2_x8_lut(x);
            _mm256_storeu_ps(output.as_mut_ptr().add(i * 8), r);
        }
        for i in (chunks * 8)..input.len() {
            output[i] = input[i].exp2();
        }
    }

    /// Highest precision pow using LUT exp2 and HP log2
    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn pow_x8_lut(x: __m256, n: f32) -> __m256 {
        let lg = super::simd_avx2_hp::log2_x8_hp(x);
        exp2_x8_lut(_mm256_mul_ps(_mm256_set1_ps(n), lg))
    }

    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn simd_pow_lut(input: &[f32], exp: f32, output: &mut [f32]) {
        let chunks = input.len() / 8;
        for i in 0..chunks {
            let x = _mm256_loadu_ps(input.as_ptr().add(i * 8));
            let r = pow_x8_lut(x, exp);
            _mm256_storeu_ps(output.as_mut_ptr().add(i * 8), r);
        }
        for i in (chunks * 8)..input.len() {
            output[i] = input[i].powf(exp);
        }
    }
}

// ============================================================================
// Benchmark harness
// ============================================================================

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
    let throughput = (N as f64 * 1e9) / (ns_per_iter * 1e6); // M elements/sec

    println!("{:30} {:>10.2} ns/iter  {:>8.2} M elem/s", name, ns_per_iter, throughput);
    ns_per_iter
}

fn measure_accuracy<F>(name: &str, reference: &[f32], test_fn: F) -> (f32, f32)
where
    F: Fn(&[f32], &mut [f32]),
{
    let mut output = vec![0.0f32; reference.len()];
    let input: Vec<f32> = (0..reference.len())
        .map(|i| 0.01 + (i as f32 / reference.len() as f32) * 10.0)
        .collect();

    test_fn(&input, &mut output);

    let mut max_rel_err = 0.0f32;
    let mut sum_rel_err = 0.0f32;
    let mut count = 0;

    for (i, (&test, _)) in output.iter().zip(input.iter()).enumerate() {
        let expected = reference[i];
        if expected.abs() > 1e-10 && expected.is_finite() && test.is_finite() {
            let rel_err = ((test - expected) / expected).abs();
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
        "{:30} max_rel_err: {:.2e}  avg_rel_err: {:.2e}",
        name, max_rel_err, avg_rel_err
    );
    (max_rel_err, avg_rel_err)
}

fn main() {
    println!("Transcendental Function Benchmark");
    println!("==================================");
    println!("N = {} elements, {} iterations\n", N, ITERATIONS);

    // Generate test data
    // For exp2: range [-10, 10] covers practical use cases
    let exp2_input: Vec<f32> = (0..N).map(|i| -10.0 + (i as f32 / N as f32) * 20.0).collect();

    // For log2: range [0.01, 100] covers practical use cases
    let log2_input: Vec<f32> = (0..N)
        .map(|i| 0.01 + (i as f32 / N as f32) * 99.99)
        .collect();

    // For pow(x, 2.4): sRGB gamma, range [0, 1]
    let pow_input: Vec<f32> = (0..N)
        .map(|i| 0.001 + (i as f32 / N as f32) * 0.998)
        .collect();

    let mut output = vec![0.0f32; N];

    // ========================================================================
    // exp2 benchmarks
    // ========================================================================
    println!("--- exp2 benchmarks ---");

    bench("scalar_std_exp2", ITERATIONS, || {
        scalar_std_exp2(black_box(&exp2_input), black_box(&mut output));
    });

    bench("scalar_fast_exp2", ITERATIONS, || {
        scalar_fast_exp2(black_box(&exp2_input), black_box(&mut output));
    });

    #[cfg(target_arch = "x86_64")]
    if simd_avx2::available() {
        bench("simd_avx2_exp2", ITERATIONS, || {
            unsafe { simd_avx2::simd_exp2(black_box(&exp2_input), black_box(&mut output)) };
        });

        bench("simd_avx2_exp2_hp", ITERATIONS, || {
            unsafe { simd_avx2_hp::simd_exp2_hp(black_box(&exp2_input), black_box(&mut output)) };
        });

        bench("simd_avx2_exp2_lut", ITERATIONS, || {
            unsafe { simd_avx2_lut::simd_exp2_lut(black_box(&exp2_input), black_box(&mut output)) };
        });
    }

    println!();

    // ========================================================================
    // log2 benchmarks
    // ========================================================================
    println!("--- log2 benchmarks ---");

    bench("scalar_std_log2", ITERATIONS, || {
        scalar_std_log2(black_box(&log2_input), black_box(&mut output));
    });

    bench("scalar_fast_log2", ITERATIONS, || {
        scalar_fast_log2(black_box(&log2_input), black_box(&mut output));
    });

    #[cfg(target_arch = "x86_64")]
    if simd_avx2::available() {
        bench("simd_avx2_log2", ITERATIONS, || {
            unsafe { simd_avx2::simd_log2(black_box(&log2_input), black_box(&mut output)) };
        });

        bench("simd_avx2_log2_hp", ITERATIONS, || {
            unsafe { simd_avx2_hp::simd_log2_hp(black_box(&log2_input), black_box(&mut output)) };
        });
    }

    println!();

    // ========================================================================
    // pow(x, 2.4) benchmarks - sRGB decode
    // ========================================================================
    println!("--- pow(x, 2.4) benchmarks (sRGB decode) ---");

    bench("scalar_std_pow_2.4", ITERATIONS, || {
        scalar_std_pow(black_box(&pow_input), 2.4, black_box(&mut output));
    });

    bench("scalar_fast_pow_2.4", ITERATIONS, || {
        scalar_fast_pow(black_box(&pow_input), 2.4, black_box(&mut output));
    });

    #[cfg(target_arch = "x86_64")]
    if simd_avx2::available() {
        bench("simd_avx2_pow_2.4", ITERATIONS, || {
            unsafe { simd_avx2::simd_pow(black_box(&pow_input), 2.4, black_box(&mut output)) };
        });

        bench("simd_avx2_pow_2.4_hp", ITERATIONS, || {
            unsafe { simd_avx2_hp::simd_pow_hp(black_box(&pow_input), 2.4, black_box(&mut output)) };
        });

        bench("simd_avx2_pow_2.4_lut", ITERATIONS, || {
            unsafe { simd_avx2_lut::simd_pow_lut(black_box(&pow_input), 2.4, black_box(&mut output)) };
        });
    }

    println!();

    // ========================================================================
    // pow(x, 1/2.4) benchmarks - sRGB encode
    // ========================================================================
    println!("--- pow(x, 1/2.4) benchmarks (sRGB encode) ---");

    bench("scalar_std_pow_inv2.4", ITERATIONS, || {
        scalar_std_pow(black_box(&pow_input), 1.0 / 2.4, black_box(&mut output));
    });

    bench("scalar_fast_pow_inv2.4", ITERATIONS, || {
        scalar_fast_pow(black_box(&pow_input), 1.0 / 2.4, black_box(&mut output));
    });

    #[cfg(target_arch = "x86_64")]
    if simd_avx2::available() {
        bench("simd_avx2_pow_inv2.4", ITERATIONS, || {
            unsafe {
                simd_avx2::simd_pow(black_box(&pow_input), 1.0 / 2.4, black_box(&mut output))
            };
        });

        bench("simd_avx2_pow_inv2.4_hp", ITERATIONS, || {
            unsafe {
                simd_avx2_hp::simd_pow_hp(black_box(&pow_input), 1.0 / 2.4, black_box(&mut output))
            };
        });

        bench("simd_avx2_pow_inv2.4_lut", ITERATIONS, || {
            unsafe {
                simd_avx2_lut::simd_pow_lut(black_box(&pow_input), 1.0 / 2.4, black_box(&mut output))
            };
        });
    }

    println!();

    // ========================================================================
    // Accuracy measurements
    // ========================================================================
    println!("--- Accuracy measurements (vs std) ---");

    // Generate reference values
    let exp2_ref: Vec<f32> = (0..1024)
        .map(|i| (-10.0 + (i as f32 / 1024.0) * 20.0).exp2())
        .collect();
    let log2_ref: Vec<f32> = (0..1024)
        .map(|i| (0.01 + (i as f32 / 1024.0) * 99.99).log2())
        .collect();
    let pow_ref: Vec<f32> = (0..1024)
        .map(|i| (0.001 + (i as f32 / 1024.0) * 0.998).powf(2.4))
        .collect();

    let exp2_test_input: Vec<f32> = (0..1024)
        .map(|i| -10.0 + (i as f32 / 1024.0) * 20.0)
        .collect();
    let log2_test_input: Vec<f32> = (0..1024)
        .map(|i| 0.01 + (i as f32 / 1024.0) * 99.99)
        .collect();
    let pow_test_input: Vec<f32> = (0..1024)
        .map(|i| 0.001 + (i as f32 / 1024.0) * 0.998)
        .collect();

    println!("\nexp2 accuracy:");
    measure_accuracy("scalar_fast_exp2", &exp2_ref, |inp, out| {
        let _ = inp;
        scalar_fast_exp2(&exp2_test_input, out);
    });

    #[cfg(target_arch = "x86_64")]
    if simd_avx2::available() {
        measure_accuracy("simd_avx2_exp2", &exp2_ref, |inp, out| {
            let _ = inp;
            unsafe { simd_avx2::simd_exp2(&exp2_test_input, out) };
        });
        measure_accuracy("simd_avx2_exp2_hp", &exp2_ref, |inp, out| {
            let _ = inp;
            unsafe { simd_avx2_hp::simd_exp2_hp(&exp2_test_input, out) };
        });
        measure_accuracy("simd_avx2_exp2_lut", &exp2_ref, |inp, out| {
            let _ = inp;
            unsafe { simd_avx2_lut::simd_exp2_lut(&exp2_test_input, out) };
        });
    }

    println!("\nlog2 accuracy:");
    measure_accuracy("scalar_fast_log2", &log2_ref, |inp, out| {
        let _ = inp;
        scalar_fast_log2(&log2_test_input, out);
    });

    #[cfg(target_arch = "x86_64")]
    if simd_avx2::available() {
        measure_accuracy("simd_avx2_log2", &log2_ref, |inp, out| {
            let _ = inp;
            unsafe { simd_avx2::simd_log2(&log2_test_input, out) };
        });
        measure_accuracy("simd_avx2_log2_hp", &log2_ref, |inp, out| {
            let _ = inp;
            unsafe { simd_avx2_hp::simd_log2_hp(&log2_test_input, out) };
        });
    }

    println!("\npow(x, 2.4) accuracy:");
    measure_accuracy("scalar_fast_pow_2.4", &pow_ref, |inp, out| {
        let _ = inp;
        scalar_fast_pow(&pow_test_input, 2.4, out);
    });

    #[cfg(target_arch = "x86_64")]
    if simd_avx2::available() {
        measure_accuracy("simd_avx2_pow_2.4", &pow_ref, |inp, out| {
            let _ = inp;
            unsafe { simd_avx2::simd_pow(&pow_test_input, 2.4, out) };
        });
        measure_accuracy("simd_avx2_pow_2.4_hp", &pow_ref, |inp, out| {
            let _ = inp;
            unsafe { simd_avx2_hp::simd_pow_hp(&pow_test_input, 2.4, out) };
        });
        measure_accuracy("simd_avx2_pow_2.4_lut", &pow_ref, |inp, out| {
            let _ = inp;
            unsafe { simd_avx2_lut::simd_pow_lut(&pow_test_input, 2.4, out) };
        });
    }

    println!("\nDone!");
}
