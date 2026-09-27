#![forbid(unsafe_code)]
#[archmage::rite(v3, use(f32xN))]
fn helper() { let _ = f32xN::splat(1.0); }
#[archmage::rite(v2)]
fn weaker() { helper(); }
fn main() {}
