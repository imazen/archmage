use archmage::{ScalarToken, rite};
#[inline]
fn compute(_t: ScalarToken, data: &[f32; 4]) -> f32 {
    data.iter().sum()
}
fn main() {}
