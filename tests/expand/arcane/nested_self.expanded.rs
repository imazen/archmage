use archmage::{arcane, X64V3Token};
struct Processor {
    val: f32,
}
impl Processor {
    #[inline(always)]
    fn process(&self, token: X64V3Token, a: f32) -> f32 {
        #[target_feature(
            enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe"
        )]
        #[inline]
        fn __simd_inner_process(_self: &Processor, token: X64V3Token, a: f32) -> f32 {
            _self.val + a
        }
        unsafe { __simd_inner_process(self, token, a) }
    }
}
fn main() {}
