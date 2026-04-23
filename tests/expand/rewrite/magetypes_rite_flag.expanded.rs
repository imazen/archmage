use archmage::magetypes;
#[target_feature(
    enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe"
)]
#[inline]
fn helper_v3(_t: archmage::X64V3Token, x: f32) -> f32 {
    #[allow(unused_imports)]
    use archmage::intrinsics::x86_64::*;
    x * x
}
fn helper_scalar(_t: archmage::ScalarToken, x: f32) -> f32 {
    x * x
}
fn main() {}
