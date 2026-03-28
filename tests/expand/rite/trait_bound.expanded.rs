use archmage::{rite, HasX64V2};
#[target_feature(enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b")]
#[inline]
fn helper(token: impl HasX64V2, a: f32) -> f32 {
    let _ = token;
    a
}
fn main() {}
