// #[arcane] with unsafe operations in the body — compiler must still catch them
use archmage::{arcane, X64V3Token};

#[arcane]
fn with_raw_ptr(token: X64V3Token, ptr: *const f32, len: usize) -> f32 {
    let mut sum = 0.0f32;
    for i in 0..len {
        sum += unsafe { *ptr.add(i) };
    }
    sum
}

fn main() {}
