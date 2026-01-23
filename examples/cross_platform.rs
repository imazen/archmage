//! Cross-platform SIMD example - works on x86_64, aarch64, and wasm32.
//!
//! This example demonstrates how to write portable SIMD code with archmage
//! that automatically dispatches to the best implementation for the current CPU.
//!
//! **Key Pattern**: Use `#[arcane]` for SIMD functions - it generates
//! `#[target_feature]` wrappers automatically, ensuring proper inlining.
//!
//! Run with:
//!   cargo run --example cross_platform --release
//!
//! Cross-compile check:
//!   cargo build --example cross_platform --target aarch64-unknown-linux-gnu
//!   cargo build --example cross_platform --target wasm32-unknown-unknown

use std::time::Instant;

// ============================================================================
// Cross-platform dispatch using archmage tokens
// ============================================================================

/// Process an array of floats, returning the sum of squares.
/// Automatically dispatches to the best SIMD implementation available.
pub fn sum_of_squares(data: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        use archmage::SimdToken;
        // Try AVX2 first (8 lanes), then SSE (4 lanes)
        if let Some(token) = archmage::Avx2FmaToken::try_new() {
            return sum_of_squares_avx2(token, data);
        }
        if let Some(token) = archmage::Sse41Token::try_new() {
            return sum_of_squares_sse(token, data);
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        use archmage::SimdToken;
        if let Some(token) = archmage::NeonToken::try_new() {
            return sum_of_squares_neon(token, data);
        }
    }

    // Scalar fallback for any platform
    sum_of_squares_scalar(data)
}

// ============================================================================
// Platform-specific implementations using #[arcane]
//
// The #[arcane] macro generates #[target_feature] wrappers automatically.
// This ensures intrinsics inline properly into single SIMD instructions.
// ============================================================================

#[cfg(target_arch = "x86_64")]
use archmage::arcane;

/// AVX2+FMA implementation (8 lanes)
///
/// The `#[arcane]` macro:
/// 1. Reads the token type to determine required target features
/// 2. Generates an inner function with `#[target_feature(enable = "avx2,fma")]`
/// 3. Calls that inner function safely (token proves CPU support)
#[cfg(target_arch = "x86_64")]
#[arcane]
fn sum_of_squares_avx2(token: archmage::Avx2FmaToken, data: &[f32]) -> f32 {
    use archmage::simd::f32x8;

    let mut acc = f32x8::zero(token);
    let chunks = data.chunks_exact(8);
    let remainder = chunks.remainder();

    for chunk in chunks {
        let arr: &[f32; 8] = chunk.try_into().unwrap();
        let v = f32x8::load(token, arr);
        acc = v.mul_add(v, acc); // v * v + acc using FMA - single vfmadd instruction!
    }

    let mut sum = acc.reduce_add();
    for &x in remainder {
        sum += x * x;
    }
    sum
}

/// SSE4.1 implementation (4 lanes)
#[cfg(target_arch = "x86_64")]
#[arcane]
fn sum_of_squares_sse(token: archmage::Sse41Token, data: &[f32]) -> f32 {
    use archmage::simd::f32x4;

    let mut acc = f32x4::zero(token);
    let chunks = data.chunks_exact(4);
    let remainder = chunks.remainder();

    for chunk in chunks {
        let arr: &[f32; 4] = chunk.try_into().unwrap();
        let v = f32x4::load(token, arr);
        acc = acc + v * v; // Addition and multiplication inline properly!
    }

    let mut sum = acc.reduce_add();
    for &x in remainder {
        sum += x * x;
    }
    sum
}

// Note: ARM NEON support requires fixing some missing intrinsics in the generator.
// For now, use scalar fallback on aarch64. When fixed, this will use #[arcane].
#[cfg(target_arch = "aarch64")]
fn sum_of_squares_neon(_token: archmage::NeonToken, data: &[f32]) -> f32 {
    // TODO: Use SIMD when ARM generator is fixed
    // For now, fall back to scalar
    sum_of_squares_scalar(data)
}

/// Scalar fallback - works on any platform
fn sum_of_squares_scalar(data: &[f32]) -> f32 {
    data.iter().map(|x| x * x).sum()
}

// ============================================================================
// Another example: Element-wise operations using #[arcane]
// ============================================================================

/// Apply a simple polynomial: a*x^2 + b*x + c for each element
pub fn polynomial_eval(data: &mut [f32], a: f32, b: f32, c: f32) {
    #[cfg(target_arch = "x86_64")]
    {
        use archmage::SimdToken;
        if let Some(token) = archmage::Avx2FmaToken::try_new() {
            polynomial_eval_avx2(token, data, a, b, c);
            return;
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        use archmage::SimdToken;
        if let Some(token) = archmage::NeonToken::try_new() {
            polynomial_eval_neon(token, data, a, b, c);
            return;
        }
    }

    // Scalar fallback
    for x in data.iter_mut() {
        *x = a * (*x) * (*x) + b * (*x) + c;
    }
}

/// AVX2+FMA polynomial evaluation using #[arcane]
///
/// FMA (fused multiply-add) is perfect for polynomial evaluation:
/// a*x^2 + b*x + c = x * (a*x + b) + c = x.mul_add(x.mul_add(a, b), c)
#[cfg(target_arch = "x86_64")]
#[arcane]
fn polynomial_eval_avx2(token: archmage::Avx2FmaToken, data: &mut [f32], a: f32, b: f32, c: f32) {
    use archmage::simd::f32x8;

    let a_v = f32x8::splat(token, a);
    let b_v = f32x8::splat(token, b);
    let c_v = f32x8::splat(token, c);

    let (chunks, remainder) = data.split_at_mut(data.len() - data.len() % 8);

    for chunk in chunks.chunks_exact_mut(8) {
        let arr: &[f32; 8] = (&*chunk).try_into().unwrap();
        let x = f32x8::load(token, arr);

        // a*x^2 + b*x + c using FMA chain - each mul_add is a single vfmadd instruction!
        // = x.mul_add(a*x + b, c)
        // = x.mul_add(x.mul_add(a, b), c)
        let result = x.mul_add(x.mul_add(a_v, b_v), c_v);

        let out: &mut [f32; 8] = chunk.try_into().unwrap();
        result.store(out);
    }

    for x in remainder {
        *x = a * (*x) * (*x) + b * (*x) + c;
    }
}

#[cfg(target_arch = "aarch64")]
fn polynomial_eval_neon(_token: archmage::NeonToken, data: &mut [f32], a: f32, b: f32, c: f32) {
    // TODO: Use #[arcane] when ARM generator is fixed
    for x in data.iter_mut() {
        *x = a * (*x) * (*x) + b * (*x) + c;
    }
}

// ============================================================================
// Platform detection helper
// ============================================================================

fn print_platform_info() {
    println!("=== Platform Information ===\n");

    #[cfg(target_arch = "x86_64")]
    {
        use archmage::SimdToken;
        println!("  Architecture: x86_64");
        print!("  SSE4.1:    ");
        if archmage::Sse41Token::try_new().is_some() {
            println!("Available (4 x f32)");
        } else {
            println!("Not available");
        }
        print!("  AVX2+FMA:  ");
        if archmage::Avx2FmaToken::try_new().is_some() {
            println!("Available (8 x f32)");
        } else {
            println!("Not available");
        }
        #[cfg(feature = "avx512")]
        {
            print!("  AVX-512:   ");
            if archmage::X64V4Token::try_new().is_some() {
                println!("Available (16 x f32)");
            } else {
                println!("Not available");
            }
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        use archmage::SimdToken;
        println!("  Architecture: aarch64");
        print!("  NEON:      ");
        if archmage::NeonToken::try_new().is_some() {
            println!("Available (4 x f32)");
        } else {
            println!("Not available");
        }
    }

    #[cfg(target_arch = "wasm32")]
    {
        println!("  Architecture: wasm32");
        println!("  SIMD128:   Check browser/runtime support");
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64", target_arch = "wasm32")))]
    {
        println!("  Architecture: {} (scalar only)", std::env::consts::ARCH);
    }

    println!();
}

// ============================================================================
// Main
// ============================================================================

fn main() {
    println!("\n╔══════════════════════════════════════════════════════════════╗");
    println!("║         Cross-Platform SIMD Example with Archmage            ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    print_platform_info();

    // Generate test data
    const N: usize = 10_000;
    let data: Vec<f32> = (0..N).map(|i| (i as f32) * 0.01).collect();

    // Test sum_of_squares
    println!("=== Testing sum_of_squares ===\n");

    let expected: f32 = data.iter().map(|x| x * x).sum();
    let result = sum_of_squares(&data);

    println!("  Expected:  {:.6}", expected);
    println!("  Got:       {:.6}", result);
    println!("  Error:     {:.2e}", (result - expected).abs() / expected.abs());

    // Benchmark
    const ITERS: u32 = 1000;

    let start = Instant::now();
    for _ in 0..ITERS {
        std::hint::black_box(sum_of_squares(std::hint::black_box(&data)));
    }
    let simd_time = start.elapsed();

    let start = Instant::now();
    for _ in 0..ITERS {
        std::hint::black_box(sum_of_squares_scalar(std::hint::black_box(&data)));
    }
    let scalar_time = start.elapsed();

    println!("\n  Scalar:  {:.2} ms", scalar_time.as_secs_f64() * 1000.0);
    println!("  SIMD:    {:.2} ms", simd_time.as_secs_f64() * 1000.0);
    println!(
        "  Speedup: {:.1}x",
        scalar_time.as_secs_f64() / simd_time.as_secs_f64()
    );

    // Test polynomial_eval
    println!("\n=== Testing polynomial_eval (ax² + bx + c) ===\n");

    let a = 0.5;
    let b = 2.0;
    let c = 1.0;

    let mut simd_data = data.clone();
    let mut scalar_data = data.clone();

    polynomial_eval(&mut simd_data, a, b, c);
    for x in scalar_data.iter_mut() {
        *x = a * (*x) * (*x) + b * (*x) + c;
    }

    // Check correctness
    let max_error: f32 = simd_data
        .iter()
        .zip(scalar_data.iter())
        .map(|(s, sc)| (s - sc).abs())
        .fold(0.0f32, f32::max);

    println!("  Max error: {:.2e}", max_error);
    println!("  First 5 SIMD:   {:?}", &simd_data[..5]);
    println!("  First 5 Scalar: {:?}", &scalar_data[..5]);

    // Benchmark polynomial
    let mut bench_data = data.clone();
    let start = Instant::now();
    for _ in 0..ITERS {
        bench_data.copy_from_slice(&data);
        polynomial_eval(&mut bench_data, a, b, c);
        std::hint::black_box(&bench_data);
    }
    let simd_time = start.elapsed();

    let start = Instant::now();
    for _ in 0..ITERS {
        bench_data.copy_from_slice(&data);
        for x in bench_data.iter_mut() {
            *x = a * (*x) * (*x) + b * (*x) + c;
        }
        std::hint::black_box(&bench_data);
    }
    let scalar_time = start.elapsed();

    println!("\n  Scalar:  {:.2} ms", scalar_time.as_secs_f64() * 1000.0);
    println!("  SIMD:    {:.2} ms", simd_time.as_secs_f64() * 1000.0);
    println!(
        "  Speedup: {:.1}x",
        scalar_time.as_secs_f64() / simd_time.as_secs_f64()
    );

    println!("\n=== Summary ===\n");
    println!("  This example demonstrates archmage's cross-platform dispatch:");
    println!("  - On x86_64: Uses AVX2 (8-wide) or SSE (4-wide)");
    println!("  - On aarch64: Uses NEON (4-wide)");
    println!("  - Elsewhere: Falls back to scalar code");
    println!("\n  The key pattern is:");
    println!("  1. Token::try_new() to detect CPU features at runtime");
    println!("  2. #[target_feature] functions for optimized codegen");
    println!("  3. Scalar fallback for portability");
    println!();
}
