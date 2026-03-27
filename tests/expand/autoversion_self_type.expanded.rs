use archmage::autoversion;
struct Filter {
    threshold: f32,
}
impl Filter {
    fn apply(&self, data: &[f32; 4]) -> f32 {
        use archmage::SimdToken;
        {
            if let Some(__t) = archmage::X64V4Token::summon() {
                return self.apply_v4(__t, data);
            }
            if let Some(__t) = archmage::X64V3Token::summon() {
                return self.apply_v3(__t, data);
            }
        }
        self.apply_scalar(archmage::ScalarToken, data)
    }
    #[allow(dead_code)]
    #[inline(always)]
    fn apply_v4(&self, _token: archmage::X64V4Token, data: &[f32; 4]) -> f32 {
        #[target_feature(
            enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe,pclmulqdq,aes,avx512f,avx512bw,avx512cd,avx512dq,avx512vl"
        )]
        #[inline]
        #[allow(dead_code)]
        fn __simd_inner_apply_v4(
            _self: &Filter,
            _token: archmage::X64V4Token,
            data: &[f32; 4],
        ) -> f32 {
            let mut sum = 0.0f32;
            for &x in data {
                if x > _self.threshold {
                    sum += x;
                }
            }
            sum
        }
        unsafe { __simd_inner_apply_v4(self, _token, data) }
    }
    #[allow(dead_code)]
    #[inline(always)]
    fn apply_v3(&self, _token: archmage::X64V3Token, data: &[f32; 4]) -> f32 {
        #[target_feature(
            enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe"
        )]
        #[inline]
        #[allow(dead_code)]
        fn __simd_inner_apply_v3(
            _self: &Filter,
            _token: archmage::X64V3Token,
            data: &[f32; 4],
        ) -> f32 {
            let mut sum = 0.0f32;
            for &x in data {
                if x > _self.threshold {
                    sum += x;
                }
            }
            sum
        }
        unsafe { __simd_inner_apply_v3(self, _token, data) }
    }
    #[allow(dead_code)]
    fn apply_scalar(&self, _token: archmage::ScalarToken, data: &[f32; 4]) -> f32 {
        let _self = self;
        let mut sum = 0.0f32;
        for &x in data {
            if x > _self.threshold {
                sum += x;
            }
        }
        sum
    }
}
fn main() {}
