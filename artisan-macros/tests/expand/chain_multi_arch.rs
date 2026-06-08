//! `#[chain]` with multiple arches and multiple tiers per arch. Exercises:
//! - two x86_64 tiers (v3 and v2) with chained trampolines
//! - one aarch64 tier
//! - not_any_cfg fallback path (for riscv64 / s390x / etc.)
//! - four tier variants in the test-hook enum, plus Default = 4

use artisan_macros::{chain, cpu_tier};

fn pixel_scalar(r: u8, g: u8, b: u8) -> u32 {
    (r as u32) << 16 | (g as u32) << 8 | (b as u32)
}

#[cpu_tier(enable = "avx2,fma")]
fn pixel_v3(r: u8, g: u8, b: u8) -> u32 {
    pixel_scalar(r, g, b)
}

#[cpu_tier(enable = "sse4.2")]
fn pixel_v2(r: u8, g: u8, b: u8) -> u32 {
    pixel_scalar(r, g, b)
}

#[cpu_tier(enable = "neon")]
fn pixel_neon(r: u8, g: u8, b: u8) -> u32 {
    pixel_scalar(r, g, b)
}

#[chain(
    x86_64 = [
        pixel_v3 = "avx2,fma",
        pixel_v2 = "sse4.2",
    ],
    aarch64 = [
        pixel_neon = "neon",
    ],
    default = pixel_scalar,
)]
pub fn pixel(r: u8, g: u8, b: u8) -> u32 {}
