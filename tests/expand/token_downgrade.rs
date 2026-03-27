// Token downgrade: V4 function calling V3 function (safe — superset features)
use archmage::{arcane, X64V3Token, X64V4Token};

#[arcane]
fn v3_helper(_token: X64V3Token, data: &[f32; 4]) -> f32 {
    data.iter().sum()
}

#[arcane]
fn v4_caller(token: X64V4Token, data: &[f32; 4]) -> f32 {
    // V4 can call V3 — token downcasts via .v3()
    v3_helper(token.v3(), data) + 1.0
}

fn main() {}
