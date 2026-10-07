use archmage::prelude::*;
struct S {
    k: f32,
}
impl S {
    #[target_feature(
        enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe"
    )]
    #[inline]
    fn probe(&mut self, token: X64V3Token, x: f32) -> f32 {
        let _ = token;
        {
            self.k += x;
            self.k
        }
    }
}
fn main() {}
