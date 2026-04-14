use archmage::autoversion;
struct P {
    f: f32,
}
impl P {
    fn apply(&self, x: f32) -> f32 {
        use archmage::SimdToken;
        {
            if let Some(__t) = archmage::X64V4Token::summon() {
                return self.apply_v4(__t, x);
            }
            if let Some(__t) = archmage::X64V3Token::summon() {
                return self.apply_v3(__t, x);
            }
        }
        self.apply_scalar(archmage::ScalarToken, x)
    }
    #[allow(dead_code)]
    #[inline(always)]
    fn apply_v4(&self, _token: archmage::X64V4Token, x: f32) -> f32 {
        #[target_feature(
            enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe,pclmulqdq,aes,avx512f,avx512bw,avx512cd,avx512dq,avx512vl"
        )]
        #[inline]
        #[allow(dead_code)]
        fn __simd_inner_apply_v4(
            _self: &P,
            _token: archmage::X64V4Token,
            x: f32,
        ) -> f32 {
            x * _self.f
        }
        const _: () = [
            (),
        ][!(<archmage::X64V4Token>::__ARCHMAGE_TIER_TAG == 4263219212u32) as usize];
        unsafe { __simd_inner_apply_v4(self, _token, x) }
    }
    #[allow(dead_code)]
    #[inline(always)]
    fn apply_v3(&self, _token: archmage::X64V3Token, x: f32) -> f32 {
        #[target_feature(
            enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe"
        )]
        #[inline]
        #[allow(dead_code)]
        fn __simd_inner_apply_v3(
            _self: &P,
            _token: archmage::X64V3Token,
            x: f32,
        ) -> f32 {
            x * _self.f
        }
        const _: () = [
            (),
        ][!(<archmage::X64V3Token>::__ARCHMAGE_TIER_TAG == 4085983307u32) as usize];
        unsafe { __simd_inner_apply_v3(self, _token, x) }
    }
    #[allow(dead_code)]
    fn apply_scalar(&self, _token: archmage::ScalarToken, x: f32) -> f32 {
        let _self = self;
        x * _self.f
    }
}
fn main() {}
