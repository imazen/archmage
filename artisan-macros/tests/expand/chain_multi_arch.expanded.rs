// === INPUT ===
// //! `#[chain]` with multiple arches and multiple tiers per arch. Exercises:
// //! - two x86_64 tiers (v3 and v2) with chained trampolines
// //! - one aarch64 tier
// //! - not_any_cfg fallback path (for riscv64 / s390x / etc.)
// //! - four tier variants in the test-hook enum, plus Default = 4
//
// use artisan_macros::{chain, cpu_tier};
//
// fn pixel_scalar(r: u8, g: u8, b: u8) -> u32 {
//     (r as u32) << 16 | (g as u32) << 8 | (b as u32)
// }
//
// #[cpu_tier(enable = "avx2,fma")]
// fn pixel_v3(r: u8, g: u8, b: u8) -> u32 {
//     pixel_scalar(r, g, b)
// }
//
// #[cpu_tier(enable = "sse4.2")]
// fn pixel_v2(r: u8, g: u8, b: u8) -> u32 {
//     pixel_scalar(r, g, b)
// }
//
// #[cpu_tier(enable = "neon")]
// fn pixel_neon(r: u8, g: u8, b: u8) -> u32 {
//     pixel_scalar(r, g, b)
// }
//
// #[chain(
//     x86_64 = [
//         pixel_v3 = "avx2,fma",
//         pixel_v2 = "sse4.2",
//     ],
//     aarch64 = [
//         pixel_neon = "neon",
//     ],
//     default = pixel_scalar,
// )]
// pub fn pixel(r: u8, g: u8, b: u8) -> u32 {}
// === END INPUT ===

//! `#[chain]` with multiple arches and multiple tiers per arch. Exercises:
//! - two x86_64 tiers (v3 and v2) with chained trampolines
//! - one aarch64 tier
//! - not_any_cfg fallback path (for riscv64 / s390x / etc.)
//! - four tier variants in the test-hook enum, plus Default = 4
use artisan_macros::{chain, cpu_tier};
fn pixel_scalar(r: u8, g: u8, b: u8) -> u32 {
    (r as u32) << 16 | (g as u32) << 8 | (b as u32)
}
#[target_feature(enable = "avx2,fma")]
#[inline]
fn pixel_v3(r: u8, g: u8, b: u8) -> u32 {
    pixel_scalar(r, g, b)
}
#[doc(hidden)]
#[allow(non_upper_case_globals)]
const __ARTISAN_CPU_TIER_FEATS_pixel_v3: &str = "avx2,fma";
#[target_feature(enable = "sse4.2")]
#[inline]
fn pixel_v2(r: u8, g: u8, b: u8) -> u32 {
    pixel_scalar(r, g, b)
}
#[doc(hidden)]
#[allow(non_upper_case_globals)]
const __ARTISAN_CPU_TIER_FEATS_pixel_v2: &str = "sse4.2";
pub fn pixel(r: u8, g: u8, b: u8) -> u32 {
    { { __artisan_pixel__x86_64__pixel_v3__chain(r, g, b) } }
}
#[inline]
#[allow(non_snake_case)]
fn __artisan_pixel__x86_64__pixel_v3__chain(r: u8, g: u8, b: u8) -> u32 {
    use ::core::sync::atomic::{AtomicU8, Ordering};
    static CACHE: AtomicU8 = AtomicU8::new(0);
    match CACHE.load(Ordering::Relaxed) {
        2u8 => unsafe { pixel_v3(r, g, b) }
        1u8 => __artisan_pixel__x86_64__pixel_v2__chain(r, g, b),
        _ => {
            let ok = (false || ::std_detect::detect::__is_feature_detected::avx2())
                && (false || ::std_detect::detect::__is_feature_detected::fma());
            CACHE.store(if ok { 2u8 } else { 1u8 }, Ordering::Relaxed);
            if ok {
                unsafe { pixel_v3(r, g, b) }
            } else {
                __artisan_pixel__x86_64__pixel_v2__chain(r, g, b)
            }
        }
    }
}
#[inline]
#[allow(non_snake_case)]
fn __artisan_pixel__x86_64__pixel_v2__chain(r: u8, g: u8, b: u8) -> u32 {
    use ::core::sync::atomic::{AtomicU8, Ordering};
    static CACHE: AtomicU8 = AtomicU8::new(0);
    match CACHE.load(Ordering::Relaxed) {
        2u8 => unsafe { pixel_v2(r, g, b) }
        1u8 => pixel_scalar(r, g, b),
        _ => {
            let ok = false || ::std_detect::detect::__is_feature_detected::sse4_2();
            CACHE.store(if ok { 2u8 } else { 1u8 }, Ordering::Relaxed);
            if ok { unsafe { pixel_v2(r, g, b) } } else { pixel_scalar(r, g, b) }
        }
    }
}
const _: () = {
    {
        const CHAIN_SITE: &str = "avx2,fma";
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
        if !__artisan_str_eq(__ARTISAN_CPU_TIER_FEATS_pixel_v3, CHAIN_SITE) {
            {
                ::core::panicking::panic_fmt(
                    format_args!(
                        "artisan-macros feature-string mismatch for tier `pixel_v3` on arch `x86_64`:\n  chain site declares (normalized): avx2,fma\n  cpu_tier declares (normalized):   (see const __ARTISAN_CPU_TIER_FEATS_pixel_v3)\n  Both strings must be equal after normalization (sort+dedupe+trim).\n  Fix: make the feature strings in #[cpu_tier(enable=\"...\")] and #[chain(... = \"...\")] list the same features.",
                    ),
                );
            }
        }
    }
    {
        const CHAIN_SITE: &str = "sse4.2";
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
        if !__artisan_str_eq(__ARTISAN_CPU_TIER_FEATS_pixel_v2, CHAIN_SITE) {
            {
                ::core::panicking::panic_fmt(
                    format_args!(
                        "artisan-macros feature-string mismatch for tier `pixel_v2` on arch `x86_64`:\n  chain site declares (normalized): sse4.2\n  cpu_tier declares (normalized):   (see const __ARTISAN_CPU_TIER_FEATS_pixel_v2)\n  Both strings must be equal after normalization (sort+dedupe+trim).\n  Fix: make the feature strings in #[cpu_tier(enable=\"...\")] and #[chain(... = \"...\")] list the same features.",
                    ),
                );
            }
        }
    }
};
