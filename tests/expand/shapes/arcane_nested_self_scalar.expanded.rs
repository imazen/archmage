use archmage::{ScalarToken, arcane};
struct Processor {
    val: f32,
}
impl Processor {
    #[inline(always)]
    fn process(&self, token: ScalarToken, a: f32) -> Self {
        #[inline]
        fn __simd_inner_process(
            _self: &Processor,
            token: ScalarToken,
            a: f32,
        ) -> Processor {
            let _ = token;
            Processor { val: _self.val + a }
        }
        let _: () = <ScalarToken>::__ARCHMAGE_ASSERT_TIER_D141EACA;
        __simd_inner_process(self, token, a)
    }
}
fn main() {}
