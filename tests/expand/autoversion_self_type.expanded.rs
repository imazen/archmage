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
        #[target_feature(enable = "sse")]
        #[target_feature(enable = "sse2")]
        #[target_feature(enable = "sse3")]
        #[target_feature(enable = "ssse3")]
        #[target_feature(enable = "sse4.1")]
        #[target_feature(enable = "sse4.2")]
        #[target_feature(enable = "popcnt")]
        #[target_feature(enable = "cmpxchg16b")]
        #[target_feature(enable = "avx")]
        #[target_feature(enable = "avx2")]
        #[target_feature(enable = "fma")]
        #[target_feature(enable = "bmi1")]
        #[target_feature(enable = "bmi2")]
        #[target_feature(enable = "f16c")]
        #[target_feature(enable = "lzcnt")]
        #[target_feature(enable = "movbe")]
        #[target_feature(enable = "pclmulqdq")]
        #[target_feature(enable = "aes")]
        #[target_feature(enable = "avx512f")]
        #[target_feature(enable = "avx512bw")]
        #[target_feature(enable = "avx512cd")]
        #[target_feature(enable = "avx512dq")]
        #[target_feature(enable = "avx512vl")]
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
        #[target_feature(enable = "sse")]
        #[target_feature(enable = "sse2")]
        #[target_feature(enable = "sse3")]
        #[target_feature(enable = "ssse3")]
        #[target_feature(enable = "sse4.1")]
        #[target_feature(enable = "sse4.2")]
        #[target_feature(enable = "popcnt")]
        #[target_feature(enable = "cmpxchg16b")]
        #[target_feature(enable = "avx")]
        #[target_feature(enable = "avx2")]
        #[target_feature(enable = "fma")]
        #[target_feature(enable = "bmi1")]
        #[target_feature(enable = "bmi2")]
        #[target_feature(enable = "f16c")]
        #[target_feature(enable = "lzcnt")]
        #[target_feature(enable = "movbe")]
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
