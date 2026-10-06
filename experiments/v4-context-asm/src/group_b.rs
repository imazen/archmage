//! Group B: plain scalar code, vectorized by LLVM, one body per tier.
use archmage::prelude::*;

#[cfg_attr(feature = "avx512", rite(v3, v4))]
#[cfg_attr(not(feature = "avx512"), rite(v3, neon))]
pub fn b1_body(plane: &mut [f32], gain: f32) {
    for value in plane {
        *value *= gain;
    }
}

#[cfg_attr(feature = "avx512", rite(v3, v4))]
#[cfg_attr(not(feature = "avx512"), rite(v3, neon))]
pub fn b2_body(x: &[f32]) -> f32 {
    let mut sum = 0.0f32;
    for &v in x {
        sum += v * v;
    }
    sum
}

macro_rules! bentry {
    ($n3:ident, $n4:ident, $b3:ident, $b4:ident, ($($p:ident: $t:ty),*), $ret:ty) => {
        #[inline(never)]
        pub fn $n3(token: X64V3Token, $($p: $t),*) -> $ret {
            #[arcane]
            fn inner(_token: X64V3Token, $($p: $t),*) -> $ret {
                $b3($($p),*)
            }
            inner(token, $($p),*)
        }
        #[cfg(feature = "avx512")]
        #[inline(never)]
        pub fn $n4(token: X64V4Token, $($p: $t),*) -> $ret {
            #[arcane]
            fn inner(_token: X64V4Token, $($p: $t),*) -> $ret {
                $b4($($p),*)
            }
            inner(token, $($p),*)
        }
    };
}

bentry!(b1_v3, b1_v4, b1_body_v3, b1_body_v4, (plane: &mut [f32], gain: f32), ());
bentry!(b2_v3, b2_v4, b2_body_v3, b2_body_v4, (x: &[f32]), f32);
