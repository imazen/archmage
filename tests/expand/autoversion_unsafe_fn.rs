// #[autoversion] on an unsafe fn
use archmage::autoversion;

#[autoversion]
unsafe fn unsafe_sum(ptr: *const f32, len: usize) -> f32 {
    let mut sum = 0.0f32;
    for i in 0..len {
        sum += unsafe { *ptr.add(i) };
    }
    sum
}

fn main() {}
