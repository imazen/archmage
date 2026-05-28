// === INPUT ===
// //! Minimal `#[chain]`: one tier on x86_64, scalar default. Exercises:
// //! - entry fn with compile-time arch switch + compile-time feature elision
// //! - one trampoline with AtomicU8 cache + thread-local forced-check
// //! - per-arch feature-string assertion const block
// //! - test-hook surface: enum, scope, force fn, thread-local
//
// use artisan_macros::{chain, cpu_tier};
//
// fn run_scalar(data: &[f32]) -> f32 {
//     data.iter().sum()
// }
//
// #[cpu_tier(enable = "avx2")]
// fn run_v3(data: &[f32]) -> f32 {
//     data.iter().sum()
// }
//
// #[chain(
//     x86_64 = [run_v3 = "avx2"],
//     default = run_scalar,
// )]
// pub fn run(data: &[f32]) -> f32 {}
// === END INPUT ===

//! Minimal `#[chain]`: one tier on x86_64, scalar default. Exercises:
//! - entry fn with compile-time arch switch + compile-time feature elision
//! - one trampoline with AtomicU8 cache + thread-local forced-check
//! - per-arch feature-string assertion const block
//! - test-hook surface: enum, scope, force fn, thread-local
use artisan_macros::{chain, cpu_tier};
fn run_scalar(data: &[f32]) -> f32 {
    data.iter().sum()
}
#[target_feature(enable = "avx2")]
#[inline]
fn run_v3(data: &[f32]) -> f32 {
    data.iter().sum()
}
#[doc(hidden)]
#[allow(non_upper_case_globals)]
const __ARTISAN_CPU_TIER_FEATS_run_v3: &str = "avx2";
pub fn run(data: &[f32]) -> f32 {
    { { __artisan_run__x86_64__run_v3__chain(data) } }
}
#[inline]
#[allow(non_snake_case)]
fn __artisan_run__x86_64__run_v3__chain(data: &[f32]) -> f32 {
    use ::core::sync::atomic::{AtomicU8, Ordering};
    static CACHE: AtomicU8 = AtomicU8::new(0);
    match CACHE.load(Ordering::Relaxed) {
        2u8 => unsafe { run_v3(data) }
        1u8 => run_scalar(data),
        _ => {
            let ok = false || ::std_detect::detect::__is_feature_detected::avx2();
            CACHE.store(if ok { 2u8 } else { 1u8 }, Ordering::Relaxed);
            if ok { unsafe { run_v3(data) } } else { run_scalar(data) }
        }
    }
}
const _: () = {
    {
        const CHAIN_SITE: &str = "avx2";
        const fn __artisan_str_eq(a: &str, b: &str) -> bool {
            let a = a.as_bytes();
            let b = b.as_bytes();
            if a.len() != b.len() {
                return false;
            }
            let mut i = 0usize;
            while i < a.len() {
                if a[i] != b[i] {
                    return false;
                }
                i += 1;
            }
            true
        }
        if !__artisan_str_eq(__ARTISAN_CPU_TIER_FEATS_run_v3, CHAIN_SITE) {
            {
                ::core::panicking::panic_fmt(
                    format_args!(
                        "artisan-macros feature-string mismatch for tier `run_v3` on arch `x86_64`:\n  chain site declares (normalized): avx2\n  cpu_tier declares (normalized):   (see const __ARTISAN_CPU_TIER_FEATS_run_v3)\n  Both strings must be equal after normalization (sort+dedupe+trim).\n  Fix: make the feature strings in #[cpu_tier(enable=\"...\")] and #[chain(... = \"...\")] list the same features.",
                    ),
                );
            }
        }
    }
};
