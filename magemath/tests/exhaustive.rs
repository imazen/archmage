//! Every one of the 2^32 f32 inputs, every op, every tier this CPU has, against
//! the mirrors. Too slow for a default run: `just magemath-exhaustive`.

mod common;

use archmage::prelude::*;
use common::*;
use std::sync::atomic::{AtomicU64, Ordering};

const BLOCK: u64 = 1 << 16;

fn check_tier<T: IntoConcreteToken + Copy + Send + Sync>(tier: &str, token: T) {
    let threads = std::thread::available_parallelism().map_or(1, usize::from);
    for op in OPS {
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
                        slice_on(token, op.name, &mut got);
                        for (&g, &x) in got.iter().zip(&input) {
                            assert_same(tier, op.name, x, g, (op.mirror)(x));
                        }
                    }
                });
            }
        });
        println!("{tier} {}: all 2^32 inputs match", op.name);
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
