//! ASM comparison: generic magetypes f32x8<T> vs concrete f32x8<x64v3>
//!
//! Tests whether `#[inline(always)]` on backend trait methods is sufficient for
//! LLVM to produce identical assembly when:
//!   1. Generic function called without #[target_feature] on caller
//!   2. Generic function called from inside #[arcane] (has target features)
//!   3. Concrete f32x8<x64v3> inside #[arcane] (the "ideal" case)
//!   5. Generic #[inline(never)] called from inside #[arcane] — forced no-inline
//!   6. Generic #[inline(always)] called from inside #[arcane] — guaranteed inline
//!
//! Run:
//!   cargo bench --bench generic_vs_concrete --features "std macros"
//!
//! Inspect assembly:
//!   cargo asm -p archmage --bench generic_vs_concrete --features "std macros" generic_no_target_feature
//!   cargo asm -p archmage --bench generic_vs_concrete --features "std macros" generic_inside_arcane
//!   cargo asm -p archmage --bench generic_vs_concrete --features "std macros" concrete_v3_in_arcane
//!   cargo asm -p archmage --bench generic_vs_concrete --features "std macros" generic_noinline_inside_arcane
//!   cargo asm -p archmage --bench generic_vs_concrete --features "std macros" generic_inline_always_inside_arcane

#![cfg(target_arch = "x86_64")]
#![allow(dead_code)]

use archmage::{arcane, rite, SimdToken, X64V3Token};

use magetypes::simd::backends::{F32x8Backend, x64v3};
use magetypes::simd::generic::f32x8;

// ============================================================================
// Shared generic functions — these are the core of the test.
// They have NO #[target_feature] — just #[inline(always)] backend methods.
// ============================================================================

fn generic_sum<T: F32x8Backend>(token: T, data: &[f32; 8]) -> f32 {
    let v = f32x8::<T>::from_array(token, *data);
    let doubled = v + v;
    doubled.reduce_add()
}

#[inline(never)]
fn generic_sum_noinline<T: F32x8Backend>(token: T, data: &[f32; 8]) -> f32 {
    let v = f32x8::<T>::from_array(token, *data);
    let doubled = v + v;
    doubled.reduce_add()
}

#[inline(always)]
fn generic_sum_inline_always<T: F32x8Backend>(token: T, data: &[f32; 8]) -> f32 {
    let v = f32x8::<T>::from_array(token, *data);
    let doubled = v + v;
    doubled.reduce_add()
}

fn generic_dot<T: F32x8Backend>(token: T, a: &[f32; 8], b: &[f32; 8]) -> f32 {
    let va = f32x8::<T>::from_array(token, *a);
    let vb = f32x8::<T>::from_array(token, *b);
    let product = va * vb;
    product.reduce_add()
}

// ============================================================================
// Pattern 1: Generic, NO #[target_feature] on caller
// Backend methods use unsafe { intrinsic } but caller has baseline features.
// ============================================================================

#[unsafe(no_mangle)]
#[inline(never)]
fn generic_no_target_feature(data: &[f32; 8]) -> f32 {
    if let Some(token) = X64V3Token::summon() {
        generic_sum(token, data)
    } else {
        0.0
    }
}

// ============================================================================
// Pattern 2: Generic called from inside #[arcane]
// Caller has #[target_feature(enable = "avx2,fma,...")].
// Does the generic callee inline into the target-featured region?
// ============================================================================

#[unsafe(no_mangle)]
#[inline(never)]
fn generic_inside_arcane(token: X64V3Token, data: &[f32; 8]) -> f32 {
    generic_inside_arcane_entry(token, data)
}

#[arcane]
fn generic_inside_arcane_entry(token: X64V3Token, data: &[f32; 8]) -> f32 {
    generic_sum(token, data)
}

// ============================================================================
// Pattern 3: Concrete f32x8<x64v3> inside #[arcane] — the "ideal" case
// No generics, no trait dispatch, everything directly typed.
// ============================================================================

#[unsafe(no_mangle)]
#[inline(never)]
fn concrete_v3_in_arcane(token: X64V3Token, data: &[f32; 8]) -> f32 {
    concrete_v3_entry(token, data)
}

#[arcane]
fn concrete_v3_entry(token: X64V3Token, data: &[f32; 8]) -> f32 {
    let v = f32x8::<x64v3>::from_array(token, *data);
    let doubled = v + v;
    doubled.reduce_add()
}

// ============================================================================
// Pattern 4: Concrete inside #[rite] called from #[arcane]
// ============================================================================

#[unsafe(no_mangle)]
#[inline(never)]
fn concrete_v3_via_rite(token: X64V3Token, data: &[f32; 8]) -> f32 {
    concrete_v3_rite_outer(token, data)
}

#[arcane]
fn concrete_v3_rite_outer(token: X64V3Token, data: &[f32; 8]) -> f32 {
    concrete_v3_rite_inner(token, data)
}

#[rite]
fn concrete_v3_rite_inner(token: X64V3Token, data: &[f32; 8]) -> f32 {
    let v = f32x8::<x64v3>::from_array(token, *data);
    let doubled = v + v;
    doubled.reduce_add()
}

// ============================================================================
// Pattern 5: Generic #[inline(never)] called from inside #[arcane]
// Forces the generic function NOT to inline — proves the inline requirement.
// ============================================================================

#[unsafe(no_mangle)]
#[inline(never)]
fn generic_noinline_inside_arcane(token: X64V3Token, data: &[f32; 8]) -> f32 {
    generic_noinline_arcane_entry(token, data)
}

#[arcane]
fn generic_noinline_arcane_entry(token: X64V3Token, data: &[f32; 8]) -> f32 {
    generic_sum_noinline(token, data)
}

// ============================================================================
// Pattern 6: Generic #[inline(always)] called from inside #[arcane]
// Guaranteed to inline — the correct pattern for cross-crate use.
// ============================================================================

#[unsafe(no_mangle)]
#[inline(never)]
fn generic_inline_always_inside_arcane(token: X64V3Token, data: &[f32; 8]) -> f32 {
    generic_inline_always_arcane_entry(token, data)
}

#[arcane]
fn generic_inline_always_arcane_entry(token: X64V3Token, data: &[f32; 8]) -> f32 {
    generic_sum_inline_always(token, data)
}

// ============================================================================
// Dot product variants — more complex operation
// ============================================================================

#[unsafe(no_mangle)]
#[inline(never)]
fn dot_generic_no_target(data_a: &[f32; 8], data_b: &[f32; 8]) -> f32 {
    if let Some(token) = X64V3Token::summon() {
        generic_dot(token, data_a, data_b)
    } else {
        0.0
    }
}

#[unsafe(no_mangle)]
#[inline(never)]
fn dot_generic_in_arcane(token: X64V3Token, data_a: &[f32; 8], data_b: &[f32; 8]) -> f32 {
    dot_generic_arcane_entry(token, data_a, data_b)
}

#[arcane]
fn dot_generic_arcane_entry(token: X64V3Token, data_a: &[f32; 8], data_b: &[f32; 8]) -> f32 {
    generic_dot(token, data_a, data_b)
}

#[unsafe(no_mangle)]
#[inline(never)]
fn dot_concrete_in_arcane(token: X64V3Token, data_a: &[f32; 8], data_b: &[f32; 8]) -> f32 {
    dot_concrete_arcane_entry(token, data_a, data_b)
}

#[arcane]
fn dot_concrete_arcane_entry(token: X64V3Token, data_a: &[f32; 8], data_b: &[f32; 8]) -> f32 {
    let va = f32x8::<x64v3>::from_array(token, *data_a);
    let vb = f32x8::<x64v3>::from_array(token, *data_b);
    let product = va * vb;
    product.reduce_add()
}

// ============================================================================
// Criterion benchmarks
// ============================================================================

use criterion::{Criterion, black_box, criterion_group, criterion_main};

fn bench_patterns(c: &mut Criterion) {
    let data = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let data2 = [8.0f32, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0];

    c.bench_function("1_generic_no_target_feature", |b| {
        b.iter(|| generic_no_target_feature(black_box(&data)))
    });

    if let Some(token) = X64V3Token::summon() {
        c.bench_function("2_generic_inside_arcane", |b| {
            b.iter(|| generic_inside_arcane(token, black_box(&data)))
        });
        c.bench_function("3_concrete_v3_in_arcane", |b| {
            b.iter(|| concrete_v3_in_arcane(token, black_box(&data)))
        });
        c.bench_function("4_concrete_v3_via_rite", |b| {
            b.iter(|| concrete_v3_via_rite(token, black_box(&data)))
        });
        c.bench_function("5_generic_noinline_inside_arcane", |b| {
            b.iter(|| generic_noinline_inside_arcane(token, black_box(&data)))
        });
        c.bench_function("6_generic_inline_always_inside_arcane", |b| {
            b.iter(|| generic_inline_always_inside_arcane(token, black_box(&data)))
        });
        c.bench_function("7_dot_generic_no_target", |b| {
            b.iter(|| dot_generic_no_target(black_box(&data), black_box(&data2)))
        });
        c.bench_function("8_dot_generic_in_arcane", |b| {
            b.iter(|| dot_generic_in_arcane(token, black_box(&data), black_box(&data2)))
        });
        c.bench_function("9_dot_concrete_in_arcane", |b| {
            b.iter(|| dot_concrete_in_arcane(token, black_box(&data), black_box(&data2)))
        });
    }
}

criterion_group!(benches, bench_patterns);
criterion_main!(benches);
