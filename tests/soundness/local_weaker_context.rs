#![forbid(unsafe_code)]
use archmage::X64V3Token;
use magetypes::simd::generic::local::f32x8;
type V = f32x8<X64V3Token>;
#[target_feature(enable="avx2")]
fn attempt() { let _ = V::splat(1.0); }
fn main() {}
