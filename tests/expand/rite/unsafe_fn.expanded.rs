use archmage::{rite, X64V3Token};
#[target_feature(
    enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe"
)]
#[inline]
unsafe fn helper(token: X64V3Token, ptr: *const f32) -> f32 {
    unsafe { *ptr }
}
fn main() {}
