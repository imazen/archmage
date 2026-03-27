use archmage::{arcane, X64V3Token};
struct Buffer {
    data: [f32; 4],
}
impl Buffer {
    #[doc(hidden)]
    #[target_feature(
        enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe"
    )]
    #[inline]
    fn __arcane_scale(&mut self, token: X64V3Token, factor: f32) {
        for x in &mut self.data {
            *x *= factor;
        }
    }
    #[inline(always)]
    fn scale(&mut self, token: X64V3Token, factor: f32) {
        unsafe { self.__arcane_scale(token, factor) }
    }
}
fn main() {}
