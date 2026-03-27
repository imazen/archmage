use archmage::{arcane, X64V3Token};
struct Processor {
    threshold: f32,
}
impl Processor {
    #[doc(hidden)]
    #[target_feature(
        enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe"
    )]
    #[inline]
    fn __arcane_compute(&self, token: X64V3Token, value: f32) -> f32 {
        value + self.threshold
    }
    #[inline(always)]
    fn compute(&self, token: X64V3Token, value: f32) -> f32 {
        unsafe { self.__arcane_compute(token, value) }
    }
}
fn main() {}
