#![forbid(unsafe_code)]
use archmage::X64V3Token;
use magetypes::simd::generic::local::f32x8;
type V = f32x8<X64V3Token>;
trait Make { fn make() -> Self; }
impl Make for V { fn make() -> Self { V::zero() } }
fn main() {}
