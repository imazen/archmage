// scalar and default are mutually exclusive fallback tiers
use archmage::prelude::*;

fn add_scalar(_: ScalarToken, x: f32) -> f32 { x }
fn add_default(x: f32) -> f32 { x }

#[cfg(target_arch = "x86_64")]
#[archmage::arcane]
fn add_v3(_: archmage::X64V3Token, x: f32) -> f32 { x }

fn main() {
    let _ = incant!(add(1.0), [v3, scalar, default]);
}
