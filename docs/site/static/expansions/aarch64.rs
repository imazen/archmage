// Rust 1.99.0; cargo-expand 1.0.126; aarch64-unknown-linux-gnu; features: default
// example: direct
pub mod direct {
    use archmage::prelude::*;
}

// example: inferred
pub mod inferred {
    use archmage::prelude::*;
}

// example: wrap
pub mod wrap {
    use archmage::prelude::*;
}

// example: inferred_wrap
pub mod inferred_wrap {
    use archmage::prelude::*;
}

// example: direct_family
pub mod direct_family {
    use archmage::prelude::*;
    #[target_feature(enable = "neon")]
    #[inline]
    #[allow(dead_code)]
    pub fn add_one_neon(x: u32) -> u32 {
        x + 1
    }
    #[inline]
    #[allow(dead_code)]
    pub fn add_one_scalar(x: u32) -> u32 {
        x + 1
    }
}

// example: proof_family
pub mod proof_family {
    use archmage::prelude::*;
    #[target_feature(enable = "neon")]
    #[inline]
    #[allow(dead_code)]
    fn __attune_add_one_neon(x: u32) -> u32 {
        x + 1
    }
    #[inline(always)]
    #[allow(clippy::inline_always, clippy::used_underscore_binding, dead_code)]
    pub fn add_one_neon_t(__attune_token: ::archmage::NeonToken, x: u32) -> u32 {
        unsafe { __attune_add_one_neon(x) }
    }
    #[inline]
    #[allow(dead_code)]
    fn __attune_add_one_scalar(x: u32) -> u32 {
        x + 1
    }
    #[inline(always)]
    #[allow(clippy::inline_always, clippy::used_underscore_binding, dead_code)]
    pub fn add_one_scalar_t(__attune_token: ::archmage::ScalarToken, x: u32) -> u32 {
        __attune_add_one_scalar(x)
    }
}

// example: dispatcher
pub mod dispatcher {
    use archmage::prelude::*;
    #[target_feature(enable = "neon")]
    #[inline]
    fn __attune_scale_neon(values: &mut [f32], factor: f32) {
        for value in values {
            *value *= factor;
        }
    }
    #[inline(always)]
    #[allow(clippy::inline_always, clippy::used_underscore_binding)]
    fn __attune_scale_neon_t(
        __attune_token: ::archmage::NeonToken,
        values: &mut [f32],
        factor: f32,
    ) {
        unsafe { __attune_scale_neon(values, factor) }
    }
    #[inline]
    fn __attune_scale_scalar(values: &mut [f32], factor: f32) {
        for value in values {
            *value *= factor;
        }
    }
    #[inline(always)]
    #[allow(clippy::inline_always, clippy::used_underscore_binding)]
    fn __attune_scale_scalar_t(
        __attune_token: ::archmage::ScalarToken,
        values: &mut [f32],
        factor: f32,
    ) {
        __attune_scale_scalar(values, factor)
    }
    #[allow(clippy::used_underscore_binding)]
    pub fn scale(values: &mut [f32], factor: f32) {
        '__attune_dispatch: {
            {
                if let Some(__attune_proof) = <::archmage::NeonToken as ::archmage::SimdToken>::summon() {
                    break '__attune_dispatch __attune_scale_neon_t(
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
    #[target_feature(enable = "neon")]
    #[inline]
    #[allow(dead_code)]
    pub fn add_one_neon(x: u32) -> u32 {
        x + 1
    }
    #[inline(always)]
    #[allow(clippy::inline_always, clippy::used_underscore_binding, dead_code)]
    pub fn add_one_neon_t(__attune_token: ::archmage::NeonToken, x: u32) -> u32 {
        unsafe { add_one_neon(x) }
    }
    #[inline]
    #[allow(dead_code)]
    pub fn add_one_scalar(x: u32) -> u32 {
        x + 1
    }
    #[inline(always)]
    #[allow(clippy::inline_always, clippy::used_underscore_binding, dead_code)]
    pub fn add_one_scalar_t(__attune_token: ::archmage::ScalarToken, x: u32) -> u32 {
        add_one_scalar(x)
    }
    #[allow(clippy::used_underscore_binding)]
    pub fn add_one(x: u32) -> u32 {
        '__attune_dispatch: {
            {
                if let Some(__attune_proof) = <::archmage::NeonToken as ::archmage::SimdToken>::summon() {
                    break '__attune_dispatch add_one_neon_t(__attune_proof, x);
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
    #[inline]
    #[allow(dead_code)]
    pub fn add_one_scalar(x: u32) -> u32 {
        x + 1
    }
    #[inline(always)]
    #[allow(clippy::inline_always, clippy::used_underscore_binding)]
    fn __attune_add_one_scalar_t(
        __attune_token: ::archmage::ScalarToken,
        x: u32,
    ) -> u32 {
        add_one_scalar(x)
    }
    #[allow(clippy::used_underscore_binding)]
    pub fn add_one(x: u32) -> u32 {
        '__attune_dispatch: {
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
    #[inline]
    fn add_one_scalar(x: u32) -> u32 {
        x + 1
    }
    #[inline(always)]
    #[allow(clippy::inline_always, clippy::used_underscore_binding)]
    fn __attune_add_one_scalar_t(
        __attune_token: ::archmage::ScalarToken,
        x: u32,
    ) -> u32 {
        add_one_scalar(x)
    }
    #[inline(always)]
    #[allow(clippy::used_underscore_binding)]
    pub fn add_one(x: u32) -> u32 {
        '__attune_dispatch: {
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
    #[target_feature(enable = "neon")]
    #[inline]
    fn __attune_scale_neon(values: &mut [f32], factor: f32) {
        #[allow(non_camel_case_types, dead_code)]
        type f32x8 = ::magetypes::simd::generic::f32x8<::archmage::NeonToken>;
        let token = ::archmage::NeonToken::from_context();
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
    #[allow(clippy::inline_always, clippy::used_underscore_binding)]
    fn __attune_scale_neon_t(
        __attune_token: ::archmage::NeonToken,
        values: &mut [f32],
        factor: f32,
    ) {
        unsafe { __attune_scale_neon(values, factor) }
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
    #[allow(clippy::inline_always, clippy::used_underscore_binding)]
    fn __attune_scale_scalar_t(
        __attune_token: ::archmage::ScalarToken,
        values: &mut [f32],
        factor: f32,
    ) {
        __attune_scale_scalar(values, factor)
    }
    #[allow(clippy::used_underscore_binding)]
    pub fn scale(values: &mut [f32], factor: f32) {
        '__attune_dispatch: {
            {
                if let Some(__attune_proof) = <::archmage::NeonToken as ::archmage::SimdToken>::summon() {
                    break '__attune_dispatch __attune_scale_neon_t(
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
    #[target_feature(enable = "neon")]
    #[inline]
    #[allow(dead_code)]
    fn scale_one_neon(value: f32, factor: f32) -> f32 {
        value * factor
    }
    #[inline]
    #[allow(dead_code)]
    fn scale_one_scalar(value: f32, factor: f32) -> f32 {
        value * factor
    }
    #[target_feature(enable = "neon")]
    #[inline]
    fn __attune_scale_neon(values: &mut [f32], factor: f32) {
        for value in values {
            *value = { '__attune_call_0: { { scale_one_neon(*value, factor) } } };
        }
    }
    #[inline(always)]
    #[allow(clippy::inline_always, clippy::used_underscore_binding)]
    fn __attune_scale_neon_t(
        __attune_token: ::archmage::NeonToken,
        values: &mut [f32],
        factor: f32,
    ) {
        unsafe { __attune_scale_neon(values, factor) }
    }
    #[inline]
    fn __attune_scale_scalar(values: &mut [f32], factor: f32) {
        for value in values {
            *value = { '__attune_call_0: { { scale_one_scalar(*value, factor) } } };
        }
    }
    #[inline(always)]
    #[allow(clippy::inline_always, clippy::used_underscore_binding)]
    fn __attune_scale_scalar_t(
        __attune_token: ::archmage::ScalarToken,
        values: &mut [f32],
        factor: f32,
    ) {
        __attune_scale_scalar(values, factor)
    }
    #[allow(clippy::used_underscore_binding)]
    pub fn scale(values: &mut [f32], factor: f32) {
        '__attune_dispatch: {
            {
                if let Some(__attune_proof) = <::archmage::NeonToken as ::archmage::SimdToken>::summon() {
                    break '__attune_dispatch __attune_scale_neon_t(
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
}

// example: generic
pub mod generic {
    use archmage::prelude::*;
    #[target_feature(enable = "neon")]
    #[inline]
    fn __attune_map_neon<F: Fn(f32) -> f32 + Copy>(values: &mut [f32], f: F) {
        for value in values {
            *value = f(*value);
        }
    }
    #[inline(always)]
    #[allow(clippy::inline_always, clippy::used_underscore_binding)]
    fn __attune_map_neon_t<F: Fn(f32) -> f32 + Copy>(
        __attune_token: ::archmage::NeonToken,
        values: &mut [f32],
        f: F,
    ) {
        unsafe { __attune_map_neon::<F>(values, f) }
    }
    #[inline]
    fn __attune_map_scalar<F: Fn(f32) -> f32 + Copy>(values: &mut [f32], f: F) {
        for value in values {
            *value = f(*value);
        }
    }
    #[inline(always)]
    #[allow(clippy::inline_always, clippy::used_underscore_binding)]
    fn __attune_map_scalar_t<F: Fn(f32) -> f32 + Copy>(
        __attune_token: ::archmage::ScalarToken,
        values: &mut [f32],
        f: F,
    ) {
        __attune_map_scalar::<F>(values, f)
    }
    #[allow(clippy::used_underscore_binding)]
    pub fn map<F: Fn(f32) -> f32 + Copy>(values: &mut [f32], f: F) {
        '__attune_dispatch: {
            {
                if let Some(__attune_proof) = <::archmage::NeonToken as ::archmage::SimdToken>::summon() {
                    break '__attune_dispatch __attune_map_neon_t::<
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
        #[target_feature(enable = "neon")]
        #[inline]
        #[allow(dead_code)]
        pub fn scale_neon(&self, values: &mut [f32]) {
            for value in values {
                *value *= self.0;
            }
        }
        #[inline(always)]
        #[allow(clippy::inline_always, clippy::used_underscore_binding)]
        fn __attune_scale_neon_t(
            &self,
            __attune_token: ::archmage::NeonToken,
            values: &mut [f32],
        ) {
            unsafe { self.scale_neon(values) }
        }
        #[inline]
        #[allow(dead_code)]
        pub fn scale_scalar(&self, values: &mut [f32]) {
            for value in values {
                *value *= self.0;
            }
        }
        #[inline(always)]
        #[allow(clippy::inline_always, clippy::used_underscore_binding)]
        fn __attune_scale_scalar_t(
            &self,
            __attune_token: ::archmage::ScalarToken,
            values: &mut [f32],
        ) {
            self.scale_scalar(values)
        }
        #[allow(clippy::used_underscore_binding)]
        pub fn scale(&self, values: &mut [f32]) {
            '__attune_dispatch: {
                {
                    if let Some(__attune_proof) = <::archmage::NeonToken as ::archmage::SimdToken>::summon() {
                        break '__attune_dispatch self
                            .__attune_scale_neon_t(__attune_proof, values);
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
