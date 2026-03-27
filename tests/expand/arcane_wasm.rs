// #[arcane] with Wasm128Token — WASM-safe mode (no unsafe wrapper)
use archmage::{arcane, Wasm128Token};

#[arcane]
fn process_wasm(token: Wasm128Token, a: f32, b: f32) -> f32 {
    let _ = token;
    a + b
}

fn main() {}
