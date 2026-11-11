//! Shared fixed kernels, used unchanged by native and hosted WASM timings.
#![allow(dead_code)]
use magemath::recip::{RecipExecution, RecipOptions, RecipValidation};
const MID: u32 = RecipOptions::PORTABLE_MIDP
    .with_execution(RecipExecution::Caller)
    .with_validation(RecipValidation::AssumeValid)
    .bits();
const HIGH: u32 = RecipOptions::HIGHP
    .with_execution(RecipExecution::Caller)
    .bits();
macro_rules! kernels {
    ($op:ident,$stream:ident,$chain:ident,$v:ident,$bound:ident,$w:literal,$parallel:ident,$emulate:ident,$native14:ident,$current:ident,$bounded:ident,$canonical:ident) => {
        #[inline(always)]
        pub(crate) fn $op<T: magetypes::simd::backends::$bound, const MODE: u8>(
            t: T,
            x: magetypes::simd::generic::$v<T>,
        ) -> magetypes::simd::generic::$v<T> {
            if MODE == 0 {
                return x.recip();
            }
            if MODE == 1 {
                return crate::seed_graphs::$parallel(t, x);
            }
            if MODE == 2 {
                return magemath::recip::$current::<MID, _>(t, x).unwrap();
            }
            if MODE == 3 {
                return magemath::recip::$current::<HIGH, _>(t, x).unwrap();
            }
            if MODE == 10 {
                return crate::recip_research::$bounded::<2, _>(t, x);
            }
            if MODE == 11 {
                return crate::recip_research::$canonical::<14, 0, _>(t, x);
            }
            if MODE == 12 {
                return crate::recip_research::$canonical::<14, 1, _>(t, x);
            }
            if MODE == 13 {
                return crate::recip_research::$canonical::<14, 2, _>(t, x);
            }
            #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
            {
                let any = &t as &dyn core::any::Any;
                if let Some(proof) = any.downcast_ref::<archmage::X64V4Token>() {
                    return magetypes::simd::generic::$v::from_array_t(
                        t,
                        crate::rcp14::native::$native14(*proof, x.to_array()),
                    );
                }
                if let Some(proof) = any.downcast_ref::<archmage::X64V4xToken>() {
                    return magetypes::simd::generic::$v::from_array_t(
                        t,
                        crate::rcp14::native::$native14(proof.v4(), x.to_array()),
                    );
                }
            }
            crate::rcp14::$emulate(t, x)
        }
        #[archmage::magetypes(v4x(cfg(avx512)), v4(cfg(avx512)), v3, neon, wasm128, scalar)]
        pub(crate) fn $stream<const MODE: u8, const N: usize>(
            t: Token,
            input: &[f32; N],
            out: &mut [f32; N],
        ) {
            use magetypes::simd::generic::$v as V;
            for (a, b) in input
                .as_chunks::<$w>()
                .0
                .iter()
                .zip(out.as_chunks_mut::<$w>().0.iter_mut())
            {
                *b = $op::<_, MODE>(t, V::from_array_t(t, *a)).to_array();
            }
        }
        #[archmage::magetypes(v4x(cfg(avx512)), v4(cfg(avx512)), v3, neon, wasm128, scalar)]
        pub(crate) fn $chain<const MODE: u8>(t: Token, initial: f32) -> [f32; $w] {
            use magetypes::simd::generic::$v as V;
            let mut x = V::splat_t(t, initial);
            for _ in 0..64 {
                x = $op::<_, MODE>(t, x) + V::splat_t(t, 0.5);
            }
            x.to_array()
        }
    };
}
kernels!(
    op4,
    stream4,
    chain4,
    f32x4,
    F32x4Convert,
    4,
    recip4,
    emulate4,
    native4,
    try_recip_f32x4_with,
    bounded4,
    canonical4
);
kernels!(
    op8,
    stream8,
    chain8,
    f32x8,
    F32x8Convert,
    8,
    recip8,
    emulate8,
    native8,
    try_recip_f32x8_with,
    bounded8,
    canonical8
);
#[cfg(feature = "w512")]
kernels!(
    op16,
    stream16,
    chain16,
    f32x16,
    F32x16Convert,
    16,
    recip16,
    emulate16,
    native16,
    try_recip_f32x16_with,
    bounded16,
    canonical16
);
pub(crate) fn input(i: usize, distribution: u8) -> f32 {
    let x = match distribution {
        0 => 0.25 + ((i % 64 + 1) as f32) * 0.03125,
        1 => {
            if i.is_multiple_of(257) {
                f32::from_bits(252 << 23)
            } else {
                f32::from_bits(
                    ((2 + (i as u32 * 73) % 250) << 23)
                        | ((i as u32).wrapping_mul(0x9e3779b9) & 0x7fffff),
                )
            }
        }
        2 => {
            let k = 8192 + (i as u64 % 8191);
            let den = 2 * k + 1;
            let m = ((1_u64 << 38) + den / 2) / den;
            f32::from_bits(0x3f800000 + (m as u32 - 0x800000))
        }
        _ => panic!("distribution"),
    };
    if i % 3 == 1 { -x } else { x }
}
