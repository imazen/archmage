#![forbid(unsafe_code)]
#[archmage::rite(use(f32xN))]
fn generic<T: archmage::HasNeon>(_: T) { let _ = f32xN::splat(1.0); }
fn main() {}
