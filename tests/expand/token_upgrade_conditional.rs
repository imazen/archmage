// Token upgrade: V3 context checking for V4 at runtime (conditional)
use archmage::{arcane, SimdToken, X64V3Token, X64V4Token};

#[arcane]
fn v4_fast_path(_token: X64V4Token, data: &[f32; 4]) -> f32 {
    data.iter().sum()
}

#[arcane]
fn v3_with_upgrade(_token: X64V3Token, data: &[f32; 4]) -> f32 {
    // Try V4 fast path, fall back to V3
    if let Some(v4) = X64V4Token::summon() {
        v4_fast_path(v4, data)
    } else {
        data.iter().sum()
    }
}

fn main() {}
