// incant! with tier modifiers — remove neon and wasm128 from defaults
use archmage::{arcane, incant, X64V3Token, X64V4Token, ScalarToken};

#[arcane]
fn work_v4(_token: X64V4Token, x: f32) -> f32 { x * 2.0 }

#[arcane]
fn work_v3(_token: X64V3Token, x: f32) -> f32 { x * 2.0 }

fn work_scalar(_token: ScalarToken, x: f32) -> f32 { x * 2.0 }

fn dispatch(x: f32) -> f32 {
    incant!(work(x), [-neon, -wasm128])
}

fn main() {}
