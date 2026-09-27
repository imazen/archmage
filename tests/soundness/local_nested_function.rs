#![forbid(unsafe_code)]
use archmage::X64V3Token;
use magetypes::simd::generic::local::f32x8;
type V = f32x8<X64V3Token>;
#[archmage::rite(v3)]
fn attempt() { fn nested() { let _ = V::zero(); } nested(); }
fn main() {}
