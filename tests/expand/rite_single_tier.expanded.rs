use archmage::rite;
#[target_feature(
    enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe"
)]
#[inline]
fn helper_v3(a: f32, b: f32) -> f32 {
    a * b + a
}
fn main() {}
