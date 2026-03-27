use archmage::{rite, X64V3Token};
struct Engine {
    factor: f32,
}
trait Compute {
    fn compute(&self, token: X64V3Token, x: f32) -> f32;
}
impl Compute for Engine {
    #[target_feature(
        enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe"
    )]
    #[inline]
    fn compute(&self, token: X64V3Token, x: f32) -> f32 {
        x * self.factor
    }
}
fn main() {}
