//! Benchmark summon() overhead to see if caching helps.
//!
//! std::arch::is_x86_feature_detected! already caches internally,
//! but we call it multiple times (once per feature). Let's see if
//! adding our own single-bit cache helps.

#![cfg(target_arch = "x86_64")]

use archmage::{Desktop64, SimdToken, X64V3Token};
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use std::sync::atomic::{AtomicU8, Ordering};

// ============================================================================
// Current implementation (multiple is_x86_feature_detected! calls)
// ============================================================================

fn summon_current() -> Option<Desktop64> {
    Desktop64::summon()
}

// ============================================================================
// Cached implementation using atomic
// ============================================================================

static V3_CACHED: AtomicU8 = AtomicU8::new(0); // 0 = unknown, 1 = no, 2 = yes

fn summon_cached_atomic() -> Option<X64V3Token> {
    match V3_CACHED.load(Ordering::Relaxed) {
        2 => Some(unsafe { forge_v3() }),
        1 => None,
        _ => {
            // First call - do the actual detection
            let result = X64V3Token::summon();
            V3_CACHED.store(if result.is_some() { 2 } else { 1 }, Ordering::Relaxed);
            result
        }
    }
}

#[inline(always)]
unsafe fn forge_v3() -> X64V3Token {
    // SAFETY: Only called after we've verified the features
    unsafe { std::mem::transmute(()) }
}

// ============================================================================
// Cached implementation using thread_local
// ============================================================================

thread_local! {
    static V3_CACHED_TLS: std::cell::Cell<Option<Option<X64V3Token>>> = const { std::cell::Cell::new(None) };
}

fn summon_cached_tls() -> Option<X64V3Token> {
    V3_CACHED_TLS.with(|cache| {
        if let Some(result) = cache.get() {
            result
        } else {
            let result = X64V3Token::summon();
            cache.set(Some(result));
            result
        }
    })
}

// ============================================================================
// Just the std detection for comparison
// ============================================================================

fn just_std_detection() -> bool {
    std::arch::is_x86_feature_detected!("avx2")
        && std::arch::is_x86_feature_detected!("fma")
}

fn just_one_std_detection() -> bool {
    std::arch::is_x86_feature_detected!("avx2")
}

// Test with more features (like AVX-512 modern would need)
fn std_many_features() -> bool {
    std::arch::is_x86_feature_detected!("avx512f")
        && std::arch::is_x86_feature_detected!("avx512bw")
        && std::arch::is_x86_feature_detected!("avx512cd")
        && std::arch::is_x86_feature_detected!("avx512dq")
        && std::arch::is_x86_feature_detected!("avx512vl")
        && std::arch::is_x86_feature_detected!("avx512vpopcntdq")
        && std::arch::is_x86_feature_detected!("avx512ifma")
        && std::arch::is_x86_feature_detected!("avx512vbmi")
        && std::arch::is_x86_feature_detected!("avx512vnni")
}

// Test with Avx512ModernToken-like features (27 features)
fn std_avx512_modern_features() -> bool {
    std::arch::is_x86_feature_detected!("sse3")
        && std::arch::is_x86_feature_detected!("ssse3")
        && std::arch::is_x86_feature_detected!("sse4.1")
        && std::arch::is_x86_feature_detected!("sse4.2")
        && std::arch::is_x86_feature_detected!("popcnt")
        && std::arch::is_x86_feature_detected!("avx")
        && std::arch::is_x86_feature_detected!("avx2")
        && std::arch::is_x86_feature_detected!("fma")
        && std::arch::is_x86_feature_detected!("bmi1")
        && std::arch::is_x86_feature_detected!("bmi2")
        && std::arch::is_x86_feature_detected!("f16c")
        && std::arch::is_x86_feature_detected!("lzcnt")
        && std::arch::is_x86_feature_detected!("avx512f")
        && std::arch::is_x86_feature_detected!("avx512bw")
        && std::arch::is_x86_feature_detected!("avx512cd")
        && std::arch::is_x86_feature_detected!("avx512dq")
        && std::arch::is_x86_feature_detected!("avx512vl")
        && std::arch::is_x86_feature_detected!("avx512vpopcntdq")
        && std::arch::is_x86_feature_detected!("avx512ifma")
        && std::arch::is_x86_feature_detected!("avx512vbmi")
        && std::arch::is_x86_feature_detected!("avx512vbmi2")
        && std::arch::is_x86_feature_detected!("avx512bitalg")
        && std::arch::is_x86_feature_detected!("avx512vnni")
        && std::arch::is_x86_feature_detected!("avx512bf16")
        && std::arch::is_x86_feature_detected!("vpclmulqdq")
        && std::arch::is_x86_feature_detected!("gfni")
        && std::arch::is_x86_feature_detected!("vaes")
}

// ============================================================================
// Benchmarks
// ============================================================================

fn bench_summon(c: &mut Criterion) {
    let mut group = c.benchmark_group("summon_overhead");

    group.bench_function("current_summon", |b| {
        b.iter(|| black_box(summon_current()))
    });

    group.bench_function("cached_atomic", |b| {
        b.iter(|| black_box(summon_cached_atomic()))
    });

    group.bench_function("cached_tls", |b| {
        b.iter(|| black_box(summon_cached_tls()))
    });

    group.bench_function("std_two_features", |b| {
        b.iter(|| black_box(just_std_detection()))
    });

    group.bench_function("std_one_feature", |b| {
        b.iter(|| black_box(just_one_std_detection()))
    });

    group.bench_function("std_nine_features", |b| {
        b.iter(|| black_box(std_many_features()))
    });

    group.bench_function("std_27_features", |b| {
        b.iter(|| black_box(std_avx512_modern_features()))
    });

    group.finish();
}

// Also benchmark in a loop context
fn bench_summon_in_loop(c: &mut Criterion) {
    let mut group = c.benchmark_group("summon_in_loop");

    group.bench_function("current_1000x", |b| {
        b.iter(|| {
            let mut count = 0u32;
            for _ in 0..1000 {
                if summon_current().is_some() {
                    count += 1;
                }
            }
            black_box(count)
        })
    });

    group.bench_function("cached_atomic_1000x", |b| {
        b.iter(|| {
            let mut count = 0u32;
            for _ in 0..1000 {
                if summon_cached_atomic().is_some() {
                    count += 1;
                }
            }
            black_box(count)
        })
    });

    group.bench_function("cached_tls_1000x", |b| {
        b.iter(|| {
            let mut count = 0u32;
            for _ in 0..1000 {
                if summon_cached_tls().is_some() {
                    count += 1;
                }
            }
            black_box(count)
        })
    });

    group.finish();
}

criterion_group!(benches, bench_summon, bench_summon_in_loop);
criterion_main!(benches);
