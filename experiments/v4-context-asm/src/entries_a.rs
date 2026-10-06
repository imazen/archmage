//! `#[inline(never)]` entries taking the token, one per kernel and tier.
use archmage::prelude::*;

use crate::group_a::*;

/// `entry!(v3_name, v4_name, impl_v3, impl_v4, (params), (args), ret)`
macro_rules! entry {
    ($n3:ident, $n4:ident, $i3:ident, $i4:ident, ($($p:ident: $t:ty),*), $ret:ty) => {
        #[inline(never)]
        pub fn $n3(token: X64V3Token, $($p: $t),*) -> $ret {
            $i3(token, $($p),*)
        }
        #[cfg(feature = "avx512")]
        #[inline(never)]
        pub fn $n4(token: X64V4Token, $($p: $t),*) -> $ret {
            $i4(token, $($p),*)
        }
    };
}

entry!(a1_v3, a1_v4, a1_impl_v3, a1_impl_v4, (plane: &mut [f32], gain: f32), ());
entry!(a2_v3, a2_v4, a2_impl_v3, a2_impl_v4, (plane: &mut [f32], gain: f32), ());
entry!(a3_v3, a3_v4, a3_impl_v3, a3_impl_v4, (plane: &mut [f32], gain: f32), ());
entry!(a4_v3, a4_v4, a4_impl_v3, a4_impl_v4, (x: &mut [f32], t: f32, scale: f32, fill: f32), ());
entry!(a5_v3, a5_v4, a5_impl_v3, a5_impl_v4, (x: &[f32], m: &[f32], out: &mut [f32]), ());
entry!(a6_v3, a6_v4, a6_impl_v3, a6_impl_v4, (x: &mut [f32]), ());
entry!(a7_v3, a7_v4, a7_impl_v3, a7_impl_v4, (input: &[f32], coef: &[f32; A7_TAPS], out: &mut [f32]), ());
entry!(a8_round_v3, a8_round_v4, a8_round_impl_v3, a8_round_impl_v4, (x: &mut [f32]), ());
entry!(a8_u8_v3, a8_u8_v4, a8_u8_impl_v3, a8_u8_impl_v4, (x: &[f32], out: &mut [u8]), ());
entry!(a9_sum_v3, a9_sum_v4, a9_sum_impl_v3, a9_sum_impl_v4, (x: &[f32]), f32);
entry!(a9_max_v3, a9_max_v4, a9_max_impl_v3, a9_max_impl_v4, (x: &[f32]), f32);
entry!(a10_recip_v3, a10_recip_v4, a10_recip_impl_v3, a10_recip_impl_v4, (x: &mut [f32]), ());
entry!(a10_rsqrt_v3, a10_rsqrt_v4, a10_rsqrt_impl_v3, a10_rsqrt_impl_v4, (x: &mut [f32]), ());
entry!(a12_v3, a12_v4, a12_impl_v3, a12_impl_v4, (plane: &mut [f32], gain: f32), ());

// A8-sat and A11 have no v4 tier (see group_a.rs); the v3 entries wrap the tier-v3 kernel.
#[inline(never)]
pub fn a8_sat_v3(token: X64V3Token, x: &[f32], out: &mut [i32]) {
    a8_sat_impl_v3(token, x, out)
}
#[inline(never)]
pub fn a11_exp2_v3(token: X64V3Token, x: &mut [f32]) {
    a11_exp2_impl_v3(token, x)
}
#[inline(never)]
pub fn a11_ln_v3(token: X64V3Token, x: &mut [f32]) {
    a11_ln_impl_v3(token, x)
}
