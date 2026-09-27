use magetypes::simd::generic::f32x8;
use archmage::X64V3Token;
fn main() {
    let _: fn(core::arch::x86_64::__m256) -> f32x8<X64V3Token> = f32x8::<X64V3Token>::from_raw;
}
