use archmage::{ScalarToken, arcane};
struct Processor {
    val: f32,
}
impl Processor {
    #[inline]
    fn process(&self, token: ScalarToken, a: f32) -> Self {
        let _self: &Processor = self;
        let _: () = <ScalarToken>::__ARCHMAGE_ASSERT_TIER_D141EACA;
        let _ = token;
        Self { val: _self.val + a }
    }
}
fn main() {}
