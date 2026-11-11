//! Compile-only public wrapper probes. This package is unpublished.
#![no_std]
#![forbid(unsafe_code)]
use magemath::{minmax::*, recip::*};
mod rcp14;
mod seed_graphs;
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
#[archmage::arcane]
pub fn native_rcp14_hardware(t: archmage::X64V4Token, x: [f32; 8]) -> [f32; 8] {
    rcp14::native::native8(t, x)
}
#[cfg(target_arch = "x86_64")]
#[archmage::arcane]
pub fn native_rcp14_emulated(t: archmage::X64V3Token, x: [f32; 8]) -> [f32; 8] {
    rcp14::emulate8(t, magetypes::simd::generic::f32x8::from_array_t(t, x)).to_array()
}
#[cfg(target_arch = "aarch64")]
#[archmage::arcane]
pub fn native_rcp14_emulated(t: archmage::NeonToken, x: [f32; 8]) -> [f32; 8] {
    rcp14::emulate8(t, magetypes::simd::generic::f32x8::from_array_t(t, x)).to_array()
}
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
#[inline(never)]
pub fn probe_rcp14_hardware(t: archmage::X64V4Token, x: [f32; 8]) -> [f32; 8] {
    native_rcp14_hardware(t, x)
}
#[cfg(target_arch = "x86_64")]
#[inline(never)]
pub fn probe_rcp14_emulated(t: archmage::X64V3Token, x: [f32; 8]) -> [f32; 8] {
    native_rcp14_emulated(t, x)
}
#[cfg(target_arch = "aarch64")]
#[inline(never)]
pub fn probe_rcp14_emulated(t: archmage::NeonToken, x: [f32; 8]) -> [f32; 8] {
    native_rcp14_emulated(t, x)
}
#[inline(never)]
pub fn scalar_highp_recip(x: f32) -> f32 {
    try_recip_with::<{ RecipOptions::HIGHP.bits() }>(x).unwrap()
}
#[inline(never)]
pub fn scalar_highp_rsqrt(x: f32) -> f32 {
    try_rsqrt_with::<{ RecipOptions::HIGHP.bits() }>(x).unwrap()
}
#[inline(never)]
pub fn scalar_normal_min(a: f32, b: f32) -> f32 {
    try_min_with::<{ MinMaxOptions::UNCHECKED_NORMAL.bits() }>(a, b).unwrap()
}
macro_rules! probes {
    ($token:ident) => {
        #[archmage::arcane]
        pub fn native_recip_parallel(t: archmage::$token, x: [f32; 8]) -> [f32; 8] {
            seed_graphs::recip8(t, magetypes::simd::generic::f32x8::from_array_t(t, x)).to_array()
        }
        #[archmage::arcane]
        pub fn native_rsqrt_quadratic(t: archmage::$token, x: [f32; 8]) -> [f32; 8] {
            seed_graphs::rsqrt8(t, magetypes::simd::generic::f32x8::from_array_t(t, x)).to_array()
        }
        #[inline(never)]
        pub fn probe_recip_parallel(t: archmage::$token, x: [f32; 8]) -> [f32; 8] {
            native_recip_parallel(t, x)
        }
        #[inline(never)]
        pub fn probe_rsqrt_quadratic(t: archmage::$token, x: [f32; 8]) -> [f32; 8] {
            native_rsqrt_quadratic(t, x)
        }
        #[archmage::arcane]
        pub fn native_recip_portable(t: archmage::$token, a: [f32; 8]) -> [f32; 8] {
            use magetypes::simd::generic::f32x8 as V;
            try_recip_f32x8_with::<
                {
                    RecipOptions::PORTABLE_MIDP
                        .with_validation(RecipValidation::AssumeValid)
                        .bits()
                },
                _,
            >(t, V::from_array_t(t, a))
            .unwrap()
            .to_array()
        }
        #[inline(never)]
        pub fn probe_recip_portable(t: archmage::$token, a: [f32; 8]) -> [f32; 8] {
            native_recip_portable(t, a)
        }
        #[inline(never)]
        pub fn probe_min_normal(t: archmage::$token, a: [f32; 8], b: [f32; 8]) -> [f32; 8] {
            native_min_normal(t, a, b)
        }
        #[inline(never)]
        pub fn probe_min_legacy(t: archmage::$token, a: [f32; 8], b: [f32; 8]) -> [f32; 8] {
            native_min_legacy(t, a, b)
        }
        #[inline(never)]
        pub fn probe_min_number(t: archmage::$token, a: [f32; 8], b: [f32; 8]) -> [f32; 8] {
            native_min_number(t, a, b)
        }
        #[inline(never)]
        pub fn probe_min_propagate(t: archmage::$token, a: [f32; 8], b: [f32; 8]) -> [f32; 8] {
            native_min_propagate(t, a, b)
        }
        #[inline(never)]
        pub fn probe_recip_midp(t: archmage::$token, a: [f32; 8]) -> [f32; 8] {
            native_recip_midp(t, a)
        }
        #[inline(never)]
        pub fn probe_recip_legacy(t: archmage::$token, a: [f32; 8]) -> [f32; 8] {
            native_recip_legacy(t, a)
        }
        #[inline(never)]
        pub fn probe_rsqrt_midp(t: archmage::$token, a: [f32; 8]) -> [f32; 8] {
            native_rsqrt_midp(t, a)
        }
        #[inline(never)]
        pub fn probe_rsqrt_legacy(t: archmage::$token, a: [f32; 8]) -> [f32; 8] {
            native_rsqrt_legacy(t, a)
        }
        #[archmage::arcane]
        pub fn native_min_legacy(t: archmage::$token, a: [f32; 8], b: [f32; 8]) -> [f32; 8] {
            use magetypes::simd::generic::f32x8 as V;
            V::from_array_t(t, a).min(V::from_array_t(t, b)).to_array()
        }
        #[archmage::arcane]
        pub fn native_min_normal(t: archmage::$token, a: [f32; 8], b: [f32; 8]) -> [f32; 8] {
            use magetypes::simd::generic::f32x8 as V;
            try_min_f32x8_with::<{ MinMaxOptions::UNCHECKED_NORMAL.bits() }, _>(
                t,
                V::from_array_t(t, a),
                V::from_array_t(t, b),
            )
            .unwrap()
            .to_array()
        }
        #[archmage::arcane]
        pub fn native_min_number(t: archmage::$token, a: [f32; 8], b: [f32; 8]) -> [f32; 8] {
            use magetypes::simd::generic::f32x8 as V;
            try_min_f32x8_with::<{ MinMaxOptions::PORTABLE_NUMBER.bits() }, _>(
                t,
                V::from_array_t(t, a),
                V::from_array_t(t, b),
            )
            .unwrap()
            .to_array()
        }
        #[archmage::arcane]
        pub fn native_min_propagate(t: archmage::$token, a: [f32; 8], b: [f32; 8]) -> [f32; 8] {
            use magetypes::simd::generic::f32x8 as V;
            try_min_f32x8_with::<{ MinMaxOptions::PORTABLE_PROPAGATE.bits() }, _>(
                t,
                V::from_array_t(t, a),
                V::from_array_t(t, b),
            )
            .unwrap()
            .to_array()
        }
        #[archmage::arcane]
        pub fn native_recip_midp(t: archmage::$token, a: [f32; 8]) -> [f32; 8] {
            use magetypes::simd::generic::f32x8 as V;
            try_recip_f32x8_with::<
                {
                    RecipOptions::BACKEND_MIDP
                        .with_validation(RecipValidation::AssumeValid)
                        .bits()
                },
                _,
            >(t, V::from_array_t(t, a))
            .unwrap()
            .to_array()
        }
        #[archmage::arcane]
        pub fn native_recip_legacy(t: archmage::$token, a: [f32; 8]) -> [f32; 8] {
            use magetypes::simd::generic::f32x8 as V;
            V::from_array_t(t, a).recip().to_array()
        }
        #[archmage::arcane]
        pub fn native_rsqrt_midp(t: archmage::$token, a: [f32; 8]) -> [f32; 8] {
            use magetypes::simd::generic::f32x8 as V;
            try_rsqrt_f32x8_with::<
                {
                    RecipOptions::BACKEND_MIDP
                        .with_validation(RecipValidation::AssumeValid)
                        .bits()
                },
                _,
            >(t, V::from_array_t(t, a))
            .unwrap()
            .to_array()
        }
        #[archmage::arcane]
        pub fn native_rsqrt_legacy(t: archmage::$token, a: [f32; 8]) -> [f32; 8] {
            use magetypes::simd::generic::f32x8 as V;
            V::from_array_t(t, a).rsqrt().to_array()
        }
    };
}
#[cfg(target_arch = "x86_64")]
probes!(X64V3Token);
#[cfg(target_arch = "aarch64")]
probes!(NeonToken);
#[cfg(target_arch = "wasm32")]
probes!(Wasm128Token);

#[inline(never)]
pub fn scalar_portable_midp_caller(x: f32) -> f32 {
    try_recip_with::<
        {
            RecipOptions::PORTABLE_MIDP
                .with_validation(RecipValidation::AssumeValid)
                .with_execution(RecipExecution::Caller)
                .bits()
        },
    >(x)
    .unwrap()
}
#[inline(never)]
pub fn scalar_portable_midp_automatic(x: f32) -> f32 {
    try_recip_with::<
        {
            RecipOptions::PORTABLE_MIDP
                .with_validation(RecipValidation::AssumeValid)
                .bits()
        },
    >(x)
    .unwrap()
}
#[inline(never)]
pub fn scalar_representation_midp_automatic(x: [f32; 4]) -> [f32; 4] {
    use magetypes::simd::generic::f32x4 as V;
    try_recip_f32x4_with::<
        {
            RecipOptions::PORTABLE_MIDP
                .with_validation(RecipValidation::AssumeValid)
                .bits()
        },
        _,
    >(
        archmage::ScalarToken,
        V::from_array_t(archmage::ScalarToken, x),
    )
    .unwrap()
    .to_array()
}
