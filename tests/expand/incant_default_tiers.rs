// incant! with default tier list (no explicit tiers)
// Note: default includes v4(cfg(avx512)) — must provide work_v4 when avx512 feature is active,
// or it's cfg'd out. We provide all default-tier variants to be safe.
use archmage::{arcane, incant, X64V3Token, X64V4Token, NeonToken, ScalarToken};

#[arcane]
fn work_v4(_token: X64V4Token, x: f32) -> f32 { x * 2.0 }

#[arcane]
fn work_v3(_token: X64V3Token, x: f32) -> f32 { x * 2.0 }

#[arcane]
fn work_neon(_token: NeonToken, x: f32) -> f32 { x * 2.0 }

fn work_scalar(_token: ScalarToken, x: f32) -> f32 { x * 2.0 }

fn dispatch(x: f32) -> f32 {
    incant!(work(x))
}

fn main() {}
