use archmage::{arcane, X64V3Token};
struct Processor {
    threshold: f32,
}
impl Processor {
    #[doc(hidden)]
    #[target_feature(enable = "sse")]
    #[target_feature(enable = "sse2")]
    #[target_feature(enable = "sse3")]
    #[target_feature(enable = "ssse3")]
    #[target_feature(enable = "sse4.1")]
    #[target_feature(enable = "sse4.2")]
    #[target_feature(enable = "popcnt")]
    #[target_feature(enable = "cmpxchg16b")]
    #[target_feature(enable = "avx")]
    #[target_feature(enable = "avx2")]
    #[target_feature(enable = "fma")]
    #[target_feature(enable = "bmi1")]
    #[target_feature(enable = "bmi2")]
    #[target_feature(enable = "f16c")]
    #[target_feature(enable = "lzcnt")]
    #[target_feature(enable = "movbe")]
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
