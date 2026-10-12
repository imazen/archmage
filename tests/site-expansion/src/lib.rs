#![forbid(unsafe_code)]
#![allow(dead_code, unused_imports)]

// example: direct
pub mod direct {
    use archmage::prelude::*;
    #[attune(v3)]
    pub fn add_one(x: u32) -> u32 {
        x + 1
    }
}

// example: inferred
pub mod inferred {
    use archmage::prelude::*;
    #[attune]
    pub fn add_one_v3(x: u32) -> u32 {
        x + 1
    }
}

// example: wrap
pub mod wrap {
    use archmage::prelude::*;
    #[attune(wrap)]
    pub fn add_one(_token: X64V3Token, x: u32) -> u32 {
        x + 1
    }
}

// example: inferred_wrap
pub mod inferred_wrap {
    use archmage::prelude::*;
    #[attune]
    pub fn add_one_v3_t(x: u32) -> u32 {
        x + 1
    }
}

// example: direct_family
pub mod direct_family {
    use archmage::prelude::*;
    #[attune(_*)]
    pub fn add_one(x: u32) -> u32 {
        x + 1
    }
}

// example: proof_family
pub mod proof_family {
    use archmage::prelude::*;
    #[attune(_*_t)]
    pub fn add_one(x: u32) -> u32 {
        x + 1
    }
}

// example: dispatcher
pub mod dispatcher {
    use archmage::prelude::*;
    #[attune(dispatch)]
    pub fn scale(values: &mut [f32], factor: f32) {
        for value in values {
            *value *= factor;
        }
    }
}

// example: all_outputs
pub mod all_outputs {
    use archmage::prelude::*;
    #[attune(all)]
    pub fn add_one(x: u32) -> u32 {
        x + 1
    }
}

// example: gates
pub mod gates {
    use archmage::prelude::*;
    #[attune(_*, dispatch, +v4x(cfg(avx512)), -_neon)]
    pub fn add_one(x: u32) -> u32 {
        x + 1
    }
}

// example: output_options
pub mod output_options {
    use archmage::prelude::*;
    #[attune(_v3(pub(crate), inline(never)), _scalar, dispatch(pub, inline(always)))]
    fn add_one(x: u32) -> u32 {
        x + 1
    }
}

// example: vectors
pub mod vectors {
    use archmage::prelude::*;
    #[attune(dispatch, define(f32x8))]
    pub fn scale(values: &mut [f32], factor: f32) {
        let token = Token::from_context();
        let multiplier = f32x8::splat_t(token, factor);
        let (chunks, tail) = f32x8::partition_slice_mut_t(token, values);
        for chunk in chunks {
            (f32x8::load_t(token, chunk) * multiplier).store(chunk);
        }
        for value in tail {
            *value *= factor;
        }
    }
}

// example: covered_calls
pub mod covered_calls {
    use archmage::prelude::*;
    #[attune(_*)]
    fn scale_one(value: f32, factor: f32) -> f32 {
        value * factor
    }
    #[attune(dispatch)]
    pub fn scale(values: &mut [f32], factor: f32) {
        for value in values {
            *value = attuned!(scale_one(*value, factor));
        }
    }
}

// example: raw_register
pub mod raw_register {
    use archmage::prelude::*;
    #[attune(v3, import_intrinsics, define(f32x8))]
    pub fn broadcast(value: f32) -> [f32; 8] {
        f32x8::from_raw(_mm256_set1_ps(value)).to_array()
    }
}

// example: generic
pub mod generic {
    use archmage::prelude::*;
    #[attune(dispatch)]
    pub fn map<F: Fn(f32) -> f32 + Copy>(values: &mut [f32], f: F) {
        for value in values {
            *value = f(*value);
        }
    }
}

// example: method
pub mod method {
    use archmage::prelude::*;
    pub struct Gain(pub f32);
    impl Gain {
        #[attune(_*, dispatch)]
        pub fn scale(&self, values: &mut [f32]) {
            for value in values {
                *value *= self.0;
            }
        }
    }
}
