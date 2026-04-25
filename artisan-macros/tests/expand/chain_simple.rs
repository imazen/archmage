//! Minimal `#[chain]`: one tier on x86_64, scalar default. Exercises:
//! - entry fn with compile-time arch switch + compile-time feature elision
//! - one trampoline with AtomicU8 cache + thread-local forced-check
//! - per-arch feature-string assertion const block
//! - test-hook surface: enum, scope, force fn, thread-local

use artisan_macros::{chain, cpu_tier};

fn run_scalar(data: &[f32]) -> f32 {
    data.iter().sum()
}

#[cpu_tier(enable = "avx2")]
fn run_v3(data: &[f32]) -> f32 {
    data.iter().sum()
}

#[chain(
    x86_64 = [run_v3 = "avx2"],
    default = run_scalar,
)]
pub fn run(data: &[f32]) -> f32 {}
