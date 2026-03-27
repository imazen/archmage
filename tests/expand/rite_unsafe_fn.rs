// #[rite] on an unsafe fn
use archmage::{rite, X64V3Token};

#[rite]
unsafe fn unsafe_helper(token: X64V3Token, ptr: *const f32) -> f32 {
    unsafe { *ptr + *ptr.add(1) }
}

fn main() {}
