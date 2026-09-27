#![forbid(unsafe_code)]
use archmage::X64V3Token;
use magetypes::simd::generic::local::f32x8;
type V = f32x8<X64V3Token>;

fn attempt() { let _ = V::zero(); }
fn main() {}
