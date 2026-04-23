use archmage::magetypes;
use magetypes::simd::generic::f32x8 as GenericF32x8;
#[target_feature(
    enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe"
)]
#[inline]
fn kernel_v3(token: archmage::X64V3Token, data: &[f32; 8]) -> f32 {
    #[allow(unused_imports)]
    use archmage::intrinsics::x86_64::*;
    type Vec8 = GenericF32x8<archmage::X64V3Token>;
    Vec8::load(token, data).reduce_add()
}
fn kernel_scalar(token: archmage::ScalarToken, data: &[f32; 8]) -> f32 {
    type Vec8 = GenericF32x8<archmage::ScalarToken>;
    Vec8::load(token, data).reduce_add()
}
fn main() {}
