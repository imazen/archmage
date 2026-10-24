// Rust 1.99.0; cargo-expand 1.0.126; x86_64-unknown-linux-gnu; features: avx512
// example: direct
pub mod direct {
    use archmage::prelude::*;
    #[target_feature(
        enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe"
    )]
    #[inline]
    pub fn add_one(x: u32) -> u32 {
        x + 1
    }
}

// example: inferred
pub mod inferred {
    use archmage::prelude::*;
    #[target_feature(
        enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe"
    )]
    #[inline]
    pub fn add_one_v3(x: u32) -> u32 {
        x + 1
    }
}

// example: wrap
pub mod wrap {
    use archmage::prelude::*;
    #[doc(hidden)]
    #[target_feature(
        enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe"
    )]
    #[inline]
    fn __arcane_add_one(_token: X64V3Token, x: u32) -> u32 {
        x + 1
    }
    #[inline(always)]
    pub fn add_one(__archmage_token: X64V3Token, x: u32) -> u32 {
        let _: () = <X64V3Token>::__ARCHMAGE_ASSERT_TIER_F38B284B;
        unsafe { __arcane_add_one(__archmage_token, x) }
    }
}

// example: inferred_wrap
pub mod inferred_wrap {
    use archmage::prelude::*;
    #[doc(hidden)]
    #[target_feature(
        enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe"
    )]
    #[inline]
    fn __arcane_add_one_v3_t(__attune_token: ::archmage::X64V3Token, x: u32) -> u32 {
        x + 1
    }
    #[inline(always)]
    pub fn add_one_v3_t(__attune_token: ::archmage::X64V3Token, x: u32) -> u32 {
        let _: () = <::archmage::X64V3Token>::__ARCHMAGE_ASSERT_TIER_F38B284B;
        unsafe { __arcane_add_one_v3_t(__attune_token, x) }
    }
}

// example: direct_family
pub mod direct_family {
    use archmage::prelude::*;
    #[target_feature(
        enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe"
    )]
    #[inline]
    pub fn add_one_v3(x: u32) -> u32 {
        x + 1
    }
    #[inline]
    pub fn add_one_scalar(x: u32) -> u32 {
        x + 1
    }
}

// example: proof_family
pub mod proof_family {
    use archmage::prelude::*;
    #[target_feature(
        enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe"
    )]
    #[inline]
    fn __attune_add_one_v3(x: u32) -> u32 {
        x + 1
    }
    #[inline(always)]
    #[allow(clippy::inline_always, dead_code)]
    pub fn add_one_v3_t(__attune_token: ::archmage::X64V3Token, x: u32) -> u32 {
        unsafe { __attune_add_one_v3(x) }
    }
    #[inline]
    fn __attune_add_one_scalar(x: u32) -> u32 {
        x + 1
    }
    #[inline(always)]
    #[allow(clippy::inline_always, dead_code)]
    pub fn add_one_scalar_t(__attune_token: ::archmage::ScalarToken, x: u32) -> u32 {
        __attune_add_one_scalar(x)
    }
}

// example: dispatcher
pub mod dispatcher {
    use archmage::prelude::*;
    #[target_feature(
        enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe"
    )]
    #[inline]
    fn __attune_scale_v3(values: &mut [f32], factor: f32) {
        for value in values {
            *value *= factor;
        }
    }
    #[inline(always)]
    #[allow(clippy::inline_always)]
    fn __attune_scale_v3_t(
        __attune_token: ::archmage::X64V3Token,
        values: &mut [f32],
        factor: f32,
    ) {
        unsafe { __attune_scale_v3(values, factor) }
    }
    #[inline]
    fn __attune_scale_scalar(values: &mut [f32], factor: f32) {
        for value in values {
            *value *= factor;
        }
    }
    #[inline(always)]
    #[allow(clippy::inline_always)]
    fn __attune_scale_scalar_t(
        __attune_token: ::archmage::ScalarToken,
        values: &mut [f32],
        factor: f32,
    ) {
        __attune_scale_scalar(values, factor)
    }
    pub fn scale(values: &mut [f32], factor: f32) {
        '__attune_dispatch: {
            {
                if let Some(__attune_proof) = <::archmage::X64V3Token as ::archmage::SimdToken>::summon() {
                    break '__attune_dispatch __attune_scale_v3_t(
                        __attune_proof,
                        values,
                        factor,
                    );
                }
            }
            {
                break '__attune_dispatch __attune_scale_scalar_t(
                    ::archmage::ScalarToken,
                    values,
                    factor,
                );
            }
        }
    }
}

// example: all_outputs
pub mod all_outputs {
    use archmage::prelude::*;
    #[target_feature(
        enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe"
    )]
    #[inline]
    pub fn add_one_v3(x: u32) -> u32 {
        x + 1
    }
    #[inline(always)]
    #[allow(clippy::inline_always, dead_code)]
    pub fn add_one_v3_t(__attune_token: ::archmage::X64V3Token, x: u32) -> u32 {
        unsafe { add_one_v3(x) }
    }
    #[inline]
    pub fn add_one_scalar(x: u32) -> u32 {
        x + 1
    }
    #[inline(always)]
    #[allow(clippy::inline_always, dead_code)]
    pub fn add_one_scalar_t(__attune_token: ::archmage::ScalarToken, x: u32) -> u32 {
        add_one_scalar(x)
    }
    pub fn add_one(x: u32) -> u32 {
        '__attune_dispatch: {
            {
                if let Some(__attune_proof) = <::archmage::X64V3Token as ::archmage::SimdToken>::summon() {
                    break '__attune_dispatch add_one_v3_t(__attune_proof, x);
                }
            }
            {
                break '__attune_dispatch add_one_scalar_t(::archmage::ScalarToken, x);
            }
        }
    }
}

// example: gates
pub mod gates {
    use archmage::prelude::*;
    #[target_feature(
        enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe,pclmulqdq,aes,avx512f,avx512bw,avx512cd,avx512dq,avx512vl,avx512vpopcntdq,avx512ifma,avx512vbmi,avx512vbmi2,avx512bitalg,avx512vnni,vpclmulqdq,gfni,vaes"
    )]
    #[inline]
    pub fn add_one_v4x(x: u32) -> u32 {
        x + 1
    }
    #[inline(always)]
    #[allow(clippy::inline_always)]
    fn __attune_add_one_v4x_t(__attune_token: ::archmage::X64V4xToken, x: u32) -> u32 {
        unsafe { add_one_v4x(x) }
    }
    #[target_feature(
        enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe"
    )]
    #[inline]
    pub fn add_one_v3(x: u32) -> u32 {
        x + 1
    }
    #[inline(always)]
    #[allow(clippy::inline_always)]
    fn __attune_add_one_v3_t(__attune_token: ::archmage::X64V3Token, x: u32) -> u32 {
        unsafe { add_one_v3(x) }
    }
    #[inline]
    pub fn add_one_scalar(x: u32) -> u32 {
        x + 1
    }
    #[inline(always)]
    #[allow(clippy::inline_always)]
    fn __attune_add_one_scalar_t(
        __attune_token: ::archmage::ScalarToken,
        x: u32,
    ) -> u32 {
        add_one_scalar(x)
    }
    pub fn add_one(x: u32) -> u32 {
        '__attune_dispatch: {
            {
                if let Some(__attune_proof) = <::archmage::X64V4xToken as ::archmage::SimdToken>::summon() {
                    break '__attune_dispatch __attune_add_one_v4x_t(__attune_proof, x);
                }
            }
            {
                if let Some(__attune_proof) = <::archmage::X64V3Token as ::archmage::SimdToken>::summon() {
                    break '__attune_dispatch __attune_add_one_v3_t(__attune_proof, x);
                }
            }
            {
                break '__attune_dispatch __attune_add_one_scalar_t(
                    ::archmage::ScalarToken,
                    x,
                );
            }
        }
    }
}

// example: output_options
pub mod output_options {
    use archmage::prelude::*;
    #[target_feature(
        enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe"
    )]
    #[inline(never)]
    pub(crate) fn add_one_v3(x: u32) -> u32 {
        x + 1
    }
    #[inline(always)]
    #[allow(clippy::inline_always)]
    fn __attune_add_one_v3_t(__attune_token: ::archmage::X64V3Token, x: u32) -> u32 {
        unsafe { add_one_v3(x) }
    }
    #[inline]
    fn add_one_scalar(x: u32) -> u32 {
        x + 1
    }
    #[inline(always)]
    #[allow(clippy::inline_always)]
    fn __attune_add_one_scalar_t(
        __attune_token: ::archmage::ScalarToken,
        x: u32,
    ) -> u32 {
        add_one_scalar(x)
    }
    #[inline(always)]
    pub fn add_one(x: u32) -> u32 {
        '__attune_dispatch: {
            {
                if let Some(__attune_proof) = <::archmage::X64V3Token as ::archmage::SimdToken>::summon() {
                    break '__attune_dispatch __attune_add_one_v3_t(__attune_proof, x);
                }
            }
            {
                break '__attune_dispatch __attune_add_one_scalar_t(
                    ::archmage::ScalarToken,
                    x,
                );
            }
        }
    }
}

// example: vectors
pub mod vectors {
    use archmage::prelude::*;
    #[target_feature(
        enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe"
    )]
    #[inline]
    fn __attune_scale_v3(values: &mut [f32], factor: f32) {
        #[allow(non_camel_case_types, dead_code)]
        type f32x8 = ::magetypes::simd::generic::f32x8<::archmage::X64V3Token>;
        let token = ::archmage::X64V3Token::from_context();
        let multiplier = f32x8::splat_t(token, factor);
        let (chunks, tail) = f32x8::partition_slice_mut_t(token, values);
        for chunk in chunks {
            (f32x8::load_t(token, chunk) * multiplier).store(chunk);
        }
        for value in tail {
            *value *= factor;
        }
    }
    #[inline(always)]
    #[allow(clippy::inline_always)]
    fn __attune_scale_v3_t(
        __attune_token: ::archmage::X64V3Token,
        values: &mut [f32],
        factor: f32,
    ) {
        unsafe { __attune_scale_v3(values, factor) }
    }
    #[inline]
    fn __attune_scale_scalar(values: &mut [f32], factor: f32) {
        #[allow(non_camel_case_types, dead_code)]
        type f32x8 = ::magetypes::simd::generic::f32x8<::archmage::ScalarToken>;
        let token = ::archmage::ScalarToken::from_context();
        let multiplier = f32x8::splat_t(token, factor);
        let (chunks, tail) = f32x8::partition_slice_mut_t(token, values);
        for chunk in chunks {
            (f32x8::load_t(token, chunk) * multiplier).store(chunk);
        }
        for value in tail {
            *value *= factor;
        }
    }
    #[inline(always)]
    #[allow(clippy::inline_always)]
    fn __attune_scale_scalar_t(
        __attune_token: ::archmage::ScalarToken,
        values: &mut [f32],
        factor: f32,
    ) {
        __attune_scale_scalar(values, factor)
    }
    pub fn scale(values: &mut [f32], factor: f32) {
        '__attune_dispatch: {
            {
                if let Some(__attune_proof) = <::archmage::X64V3Token as ::archmage::SimdToken>::summon() {
                    break '__attune_dispatch __attune_scale_v3_t(
                        __attune_proof,
                        values,
                        factor,
                    );
                }
            }
            {
                break '__attune_dispatch __attune_scale_scalar_t(
                    ::archmage::ScalarToken,
                    values,
                    factor,
                );
            }
        }
    }
}

// example: covered_calls
pub mod covered_calls {
    use archmage::prelude::*;
    #[target_feature(
        enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe"
    )]
    #[inline]
    fn scale_one_v3(value: f32, factor: f32) -> f32 {
        value * factor
    }
    #[inline]
    fn scale_one_scalar(value: f32, factor: f32) -> f32 {
        value * factor
    }
    #[target_feature(
        enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe"
    )]
    #[inline]
    fn __attune_scale_v3(values: &mut [f32], factor: f32) {
        for value in values {
            *value = { '__attune_call_0: { { scale_one_v3(*value, factor) } } };
        }
    }
    #[inline(always)]
    #[allow(clippy::inline_always)]
    fn __attune_scale_v3_t(
        __attune_token: ::archmage::X64V3Token,
        values: &mut [f32],
        factor: f32,
    ) {
        unsafe { __attune_scale_v3(values, factor) }
    }
    #[inline]
    fn __attune_scale_scalar(values: &mut [f32], factor: f32) {
        for value in values {
            *value = { '__attune_call_0: { { scale_one_scalar(*value, factor) } } };
        }
    }
    #[inline(always)]
    #[allow(clippy::inline_always)]
    fn __attune_scale_scalar_t(
        __attune_token: ::archmage::ScalarToken,
        values: &mut [f32],
        factor: f32,
    ) {
        __attune_scale_scalar(values, factor)
    }
    pub fn scale(values: &mut [f32], factor: f32) {
        '__attune_dispatch: {
            {
                if let Some(__attune_proof) = <::archmage::X64V3Token as ::archmage::SimdToken>::summon() {
                    break '__attune_dispatch __attune_scale_v3_t(
                        __attune_proof,
                        values,
                        factor,
                    );
                }
            }
            {
                break '__attune_dispatch __attune_scale_scalar_t(
                    ::archmage::ScalarToken,
                    values,
                    factor,
                );
            }
        }
    }
}

// example: raw_register
pub mod raw_register {
    use archmage::prelude::*;
    #[target_feature(
        enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe"
    )]
    #[inline]
    pub fn broadcast(value: f32) -> [f32; 8] {
        #[allow(unused_imports)]
        use archmage::intrinsics::x86_64::*;
        #[allow(non_camel_case_types, dead_code)]
        type f32x8 = ::magetypes::simd::generic::f32x8<::archmage::X64V3Token>;
        f32x8::from_raw(_mm256_set1_ps(value)).to_array()
    }
}

// example: generic
pub mod generic {
    use archmage::prelude::*;
    #[target_feature(
        enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe"
    )]
    #[inline]
    fn __attune_map_v3<F: Fn(f32) -> f32 + Copy>(values: &mut [f32], f: F) {
        for value in values {
            *value = f(*value);
        }
    }
    #[inline(always)]
    #[allow(clippy::inline_always)]
    fn __attune_map_v3_t<F: Fn(f32) -> f32 + Copy>(
        __attune_token: ::archmage::X64V3Token,
        values: &mut [f32],
        f: F,
    ) {
        unsafe { __attune_map_v3::<F>(values, f) }
    }
    #[inline]
    fn __attune_map_scalar<F: Fn(f32) -> f32 + Copy>(values: &mut [f32], f: F) {
        for value in values {
            *value = f(*value);
        }
    }
    #[inline(always)]
    #[allow(clippy::inline_always)]
    fn __attune_map_scalar_t<F: Fn(f32) -> f32 + Copy>(
        __attune_token: ::archmage::ScalarToken,
        values: &mut [f32],
        f: F,
    ) {
        __attune_map_scalar::<F>(values, f)
    }
    pub fn map<F: Fn(f32) -> f32 + Copy>(values: &mut [f32], f: F) {
        '__attune_dispatch: {
            {
                if let Some(__attune_proof) = <::archmage::X64V3Token as ::archmage::SimdToken>::summon() {
                    break '__attune_dispatch __attune_map_v3_t::<
                        F,
                    >(__attune_proof, values, f);
                }
            }
            {
                break '__attune_dispatch __attune_map_scalar_t::<
                    F,
                >(::archmage::ScalarToken, values, f);
            }
        }
    }
}

// example: method
pub mod method {
    use archmage::prelude::*;
    pub struct Gain(pub f32);
    impl Gain {
        #[target_feature(
            enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,cmpxchg16b,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,movbe"
        )]
        #[inline]
        pub fn scale_v3(&self, values: &mut [f32]) {
            for value in values {
                *value *= self.0;
            }
        }
        #[inline(always)]
        #[allow(clippy::inline_always)]
        fn __attune_scale_v3_t(
            &self,
            __attune_token: ::archmage::X64V3Token,
            values: &mut [f32],
        ) {
            unsafe { self.scale_v3(values) }
        }
        #[inline]
        pub fn scale_scalar(&self, values: &mut [f32]) {
            for value in values {
                *value *= self.0;
            }
        }
        #[inline(always)]
        #[allow(clippy::inline_always)]
        fn __attune_scale_scalar_t(
            &self,
            __attune_token: ::archmage::ScalarToken,
            values: &mut [f32],
        ) {
            self.scale_scalar(values)
        }
        pub fn scale(&self, values: &mut [f32]) {
            '__attune_dispatch: {
                {
                    if let Some(__attune_proof) = <::archmage::X64V3Token as ::archmage::SimdToken>::summon() {
                        break '__attune_dispatch self
                            .__attune_scale_v3_t(__attune_proof, values);
                    }
                }
                {
                    break '__attune_dispatch self
                        .__attune_scale_scalar_t(::archmage::ScalarToken, values);
                }
            }
        }
    }
}
