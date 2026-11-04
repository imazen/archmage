#![allow(clippy::all)]
use archmage::prelude::*;
use magemath::portable::kernels;

macro_rules! probe {
    ($tier:ident, $token:ident, $($name:ident = |$t:ident, $v:ident| $body:expr;)*) => {$(
        #[attune($tier, define(f32x8))]
        #[inline(never)]
        pub fn $name(a: &[f32; 8], out: &mut [f32; 8]) {
            let $t = archmage::$token::from_context();
            let $v = f32x8::load_t($t, a);
            ($body).store(out);
        }
    )*};
}

#[cfg(target_arch = "x86_64")]
pub mod x86 {
    use super::*;
    probe!(v3, X64V3Token,
        exp2_portable = |t, v| kernels::exp2_midp(t, v);
        exp2_relaxed = |t, v| v.exp2_midp();
        exp2_unchecked = |t, v| kernels::exp2_midp_unchecked(t, v);
        log2_portable = |t, v| kernels::log2_midp(t, v);
        log2_relaxed = |t, v| v.log2_midp();
        log2_unchecked = |t, v| kernels::log2_midp_unchecked(t, v);
        chain_exp2_log2 = |t, v| kernels::log2_midp(t, kernels::exp2_midp(t, v));
        chain_gamma = |t, v| kernels::pow_midp(t, kernels::pow_midp(t, v, 2.4), 1.0 / 2.4);
    );
    #[attune(v3)]
    #[inline(never)]
    pub fn array_chain_v3(a: [f32; 8]) -> [f32; 8] {
        magemath::portable::array::log2_midp_f32x8_v3(magemath::portable::array::exp2_midp_f32x8_v3(a))
    }
}

#[cfg(target_arch = "aarch64")]
pub mod arm {
    use super::*;
    probe!(neon, NeonToken,
        exp2_portable = |t, v| kernels::exp2_midp(t, v);
        exp2_relaxed = |t, v| v.exp2_midp();
        log2_portable = |t, v| kernels::log2_midp(t, v);
        chain_exp2_log2 = |t, v| kernels::log2_midp(t, kernels::exp2_midp(t, v));
    );
}
