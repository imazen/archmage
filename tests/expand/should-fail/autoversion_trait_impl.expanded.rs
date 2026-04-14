use archmage::autoversion;
struct Filter {
    cutoff: f32,
}
trait Process {
    fn process(&self, data: &[f32; 4]) -> f32;
}
impl Process for Filter {
    fn process(&self, data: &[f32; 4]) -> f32 {
        use archmage::SimdToken;
        {
            if let Some(__t) = archmage::X64V4Token::summon() {
                return self.process_v4(__t, data);
            }
            if let Some(__t) = archmage::X64V3Token::summon() {
                return self.process_v3(__t, data);
            }
        }
        self.process_scalar(archmage::ScalarToken, data)
    }
    #[allow(dead_code)]
    #[inline(always)]
    fn process_v4(&self, _token: archmage::X64V4Token, data: &[f32; 4]) -> f32 {
        #[target_feature(
            enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe,pclmulqdq,aes,avx512f,avx512bw,avx512cd,avx512dq,avx512vl"
        )]
        #[inline]
        #[allow(dead_code)]
        fn __simd_inner_process_v4(
            _self: &Filter,
            _token: archmage::X64V4Token,
            data: &[f32; 4],
        ) -> f32 {
            let mut sum = 0.0f32;
            for &x in data {
                if x > _self.cutoff {
                    sum += x;
                }
            }
            sum
        }
        const _ARCHMAGE_TOKEN_MISMATCH: () = [
            (),
        ][!(<archmage::X64V4Token>::__ARCHMAGE_TIER_TAG == 4263219212u32) as usize];
        unsafe { __simd_inner_process_v4(self, _token, data) }
    }
    #[allow(dead_code)]
    #[inline(always)]
    fn process_v3(&self, _token: archmage::X64V3Token, data: &[f32; 4]) -> f32 {
        #[target_feature(
            enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe"
        )]
        #[inline]
        #[allow(dead_code)]
        fn __simd_inner_process_v3(
            _self: &Filter,
            _token: archmage::X64V3Token,
            data: &[f32; 4],
        ) -> f32 {
            let mut sum = 0.0f32;
            for &x in data {
                if x > _self.cutoff {
                    sum += x;
                }
            }
            sum
        }
        const _ARCHMAGE_TOKEN_MISMATCH: () = [
            (),
        ][!(<archmage::X64V3Token>::__ARCHMAGE_TIER_TAG == 4085983307u32) as usize];
        unsafe { __simd_inner_process_v3(self, _token, data) }
    }
    #[allow(dead_code)]
    fn process_scalar(&self, _token: archmage::ScalarToken, data: &[f32; 4]) -> f32 {
        let _self = self;
        let mut sum = 0.0f32;
        for &x in data {
            if x > _self.cutoff {
                sum += x;
            }
        }
        sum
    }
}
fn main() {}
