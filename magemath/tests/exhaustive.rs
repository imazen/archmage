//! Every one of the 2^32 f32 inputs, every op, every tier this CPU has, against
//! the mirrors. Too slow for a default run: `just magemath-exhaustive`.

mod common;

use archmage::prelude::*;
use common::*;
use std::sync::atomic::{AtomicU64, Ordering};

const BLOCK: u64 = 1 << 16;

fn threads() -> usize {
    let available = std::thread::available_parallelism().map_or(1, usize::from);
    let limit = std::env::var("RAYON_NUM_THREADS")
        .map(|value| {
            value
                .parse::<usize>()
                .expect("RAYON_NUM_THREADS must be a positive integer")
        })
        .unwrap_or_else(|_| available.saturating_sub(4).max(1));
    assert!(limit > 0, "RAYON_NUM_THREADS must be positive");
    available.min(limit)
}

fn check_tier<T: IntoConcreteToken + Copy + Send + Sync>(tier: &str, token: T) {
    let rounding = relaxed_rounding(tier, token);
    check_surface(tier, "portable", token, fused::slice_on::<T>, fused::OPS);
    check_surface(tier, "unfused", token, unfused::slice_on::<T>, unfused::OPS);
    check_surface(
        tier,
        "relaxed",
        token,
        relaxed::slice_on::<T>,
        rounding.ops(),
    );
}

fn check_surface<T: Copy + Send + Sync>(
    tier: &str,
    surface: &str,
    token: T,
    slice: fn(T, &str, &mut [f32]),
    mirrors: &[Op],
) {
    let threads = threads();
    for op in mirrors {
        let next = AtomicU64::new(0);
        std::thread::scope(|s| {
            for _ in 0..threads {
                s.spawn(|| {
                    let mut input = vec![0.0f32; BLOCK as usize];
                    let mut got = vec![0.0f32; BLOCK as usize];
                    loop {
                        let start = next.fetch_add(BLOCK, Ordering::Relaxed);
                        if start >= 1 << 32 {
                            return;
                        }
                        for (i, x) in input.iter_mut().enumerate() {
                            *x = f32::from_bits((start + i as u64) as u32);
                        }
                        got.copy_from_slice(&input);
                        slice(token, op.name, &mut got);
                        for (&g, &x) in got.iter().zip(&input) {
                            assert_same(tier, op.name, x, g, (op.mirror)(x));
                        }
                    }
                });
            }
        });
        println!("{tier} {surface} {}: all 2^32 inputs match", op.name);
    }
}

#[test]
#[ignore = "takes minutes on many cores; run through just magemath-exhaustive"]
fn all_inputs_all_tiers() {
    check_tier("scalar", ScalarToken);
    #[cfg(target_arch = "x86_64")]
    if let Some(token) = require::<X64V3Token>("v3") {
        check_tier("v3", token);
    }
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    if let Some(token) = require::<X64V4xToken>("v4x") {
        check_tier("v4x", token);
    }
    #[cfg(target_arch = "aarch64")]
    if let Some(token) = require::<NeonToken>("neon") {
        check_tier("neon", token);
    }
}

/// Distance between two results in representable values; NaN against non-NaN
/// is `u64::MAX`, since no number of ULPs covers it.
fn ulp_distance(a: f32, b: f32) -> u64 {
    if a.is_nan() || b.is_nan() {
        return if a.is_nan() == b.is_nan() {
            0
        } else {
            u64::MAX
        };
    }
    let line = |x: f32| -> i64 {
        let magnitude = i64::from(x.to_bits() & 0x7fff_ffff);
        if x.is_sign_negative() {
            -magnitude
        } else {
            magnitude
        }
    };
    line(a).abs_diff(line(b))
}

/// What choosing one rounding over the other changes: over all 2^32 inputs, how
/// many results differ and by how much. This prints a table and asserts only that
/// NaN-ness agrees, because specials must not depend on the rounding.
#[test]
#[ignore = "takes minutes on many cores; run through just magemath-exhaustive"]
fn rounding_gap() {
    println!("op, inputs differing of 2^32, max ULP apart");
    for (once, twice) in fused::OPS.iter().zip(unfused::OPS) {
        let next = AtomicU64::new(0);
        let differing = AtomicU64::new(0);
        let widest = AtomicU64::new(0);
        std::thread::scope(|s| {
            for _ in 0..threads() {
                s.spawn(|| {
                    loop {
                        let start = next.fetch_add(BLOCK, Ordering::Relaxed);
                        if start >= 1 << 32 {
                            return;
                        }
                        let (mut count, mut most) = (0, 0);
                        for i in 0..BLOCK {
                            let x = f32::from_bits((start + i) as u32);
                            let (a, b) = ((once.mirror)(x), (twice.mirror)(x));
                            let gap = ulp_distance(a, b);
                            assert_ne!(gap, u64::MAX, "{}({x:e}): {a:e} vs {b:e}", once.name);
                            count += u64::from(gap != 0);
                            most = most.max(gap);
                        }
                        differing.fetch_add(count, Ordering::Relaxed);
                        widest.fetch_max(most, Ordering::Relaxed);
                    }
                });
            }
        });
        println!(
            "{}, {}, {}",
            once.name,
            differing.load(Ordering::Relaxed),
            widest.load(Ordering::Relaxed)
        );
    }
}
