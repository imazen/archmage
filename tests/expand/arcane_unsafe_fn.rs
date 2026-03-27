// #[arcane] on an unsafe fn — the generated wrapper must preserve unsafe fn
use archmage::{arcane, X64V3Token};

#[arcane]
unsafe fn unsafe_process(token: X64V3Token, ptr: *const f32, len: usize) -> f32 {
    let mut sum = 0.0f32;
    for i in 0..len {
        sum += unsafe { *ptr.add(i) };
    }
    sum
}

fn main() {}
