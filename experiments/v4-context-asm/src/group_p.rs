//! Polyfill probe: one gain kernel and one sum kernel at three logical
//! widths, compiled for every tier from one body each. On AVX2 an `f32x16` is
//! two 256-bit halves; on NEON and WASM an `f32x8` is two 128-bit halves and
//! an `f32x16` is four.
use archmage::prelude::*;

#[magetypes(define(f32x4), v3, neon, wasm128, scalar)]
pub fn p4_gain_impl(token: Token, plane: &mut [f32], gain: f32) {
    let factor = f32x4::splat_t(token, gain);
    let (chunks, tail) = f32x4::partition_slice_mut_t(token, plane);
    for chunk in chunks {
        (f32x4::load_t(token, chunk) * factor).store(chunk);
    }
    for value in tail {
        *value *= gain;
    }
}

#[magetypes(define(f32x4), v3, neon, wasm128, scalar)]
pub fn p4_sum_impl(token: Token, x: &[f32]) -> f32 {
    let (chunks, tail) = f32x4::partition_slice_t(token, x);
    let mut acc = f32x4::zero_t(token);
    for chunk in chunks {
        acc += f32x4::load_t(token, chunk);
    }
    let mut s = acc.reduce_add();
    for v in tail {
        s += *v;
    }
    s
}

#[magetypes(define(f32x8), v3, neon, wasm128, scalar)]
pub fn p8_gain_impl(token: Token, plane: &mut [f32], gain: f32) {
    let factor = f32x8::splat_t(token, gain);
    let (chunks, tail) = f32x8::partition_slice_mut_t(token, plane);
    for chunk in chunks {
        (f32x8::load_t(token, chunk) * factor).store(chunk);
    }
    for value in tail {
        *value *= gain;
    }
}

#[magetypes(define(f32x8), v3, neon, wasm128, scalar)]
pub fn p8_sum_impl(token: Token, x: &[f32]) -> f32 {
    let (chunks, tail) = f32x8::partition_slice_t(token, x);
    let mut acc = f32x8::zero_t(token);
    for chunk in chunks {
        acc += f32x8::load_t(token, chunk);
    }
    let mut s = acc.reduce_add();
    for v in tail {
        s += *v;
    }
    s
}

#[magetypes(define(f32x16), v3, neon, wasm128, scalar)]
pub fn p16_gain_impl(token: Token, plane: &mut [f32], gain: f32) {
    let factor = f32x16::splat_t(token, gain);
    let (chunks, tail) = f32x16::partition_slice_mut_t(token, plane);
    for chunk in chunks {
        (f32x16::load_t(token, chunk) * factor).store(chunk);
    }
    for value in tail {
        *value *= gain;
    }
}

#[magetypes(define(f32x16), v3, neon, wasm128, scalar)]
pub fn p16_sum_impl(token: Token, x: &[f32]) -> f32 {
    let (chunks, tail) = f32x16::partition_slice_t(token, x);
    let mut acc = f32x16::zero_t(token);
    for chunk in chunks {
        acc += f32x16::load_t(token, chunk);
    }
    let mut s = acc.reduce_add();
    for v in tail {
        s += *v;
    }
    s
}

/// One `#[inline(never)]` entry per kernel and tier, so each keeps its symbol.
macro_rules! entries {
    ($arch:literal, $tok:ty, $gain:ident => $gain_impl:ident, $sum:ident => $sum_impl:ident) => {
        #[cfg(target_arch = $arch)]
        #[inline(never)]
        pub fn $gain(token: $tok, plane: &mut [f32], gain: f32) {
            $gain_impl(token, plane, gain)
        }
        #[cfg(target_arch = $arch)]
        #[inline(never)]
        pub fn $sum(token: $tok, x: &[f32]) -> f32 {
            $sum_impl(token, x)
        }
    };
}
entries!("x86_64", X64V3Token, p4_gain_v3 => p4_gain_impl_v3, p4_sum_v3 => p4_sum_impl_v3);
entries!("aarch64", NeonToken, p4_gain_neon => p4_gain_impl_neon, p4_sum_neon => p4_sum_impl_neon);
entries!("wasm32", Wasm128Token, p4_gain_wasm128 => p4_gain_impl_wasm128, p4_sum_wasm128 => p4_sum_impl_wasm128);
entries!("x86_64", X64V3Token, p8_gain_v3 => p8_gain_impl_v3, p8_sum_v3 => p8_sum_impl_v3);
entries!("aarch64", NeonToken, p8_gain_neon => p8_gain_impl_neon, p8_sum_neon => p8_sum_impl_neon);
entries!("wasm32", Wasm128Token, p8_gain_wasm128 => p8_gain_impl_wasm128, p8_sum_wasm128 => p8_sum_impl_wasm128);
entries!("x86_64", X64V3Token, p16_gain_v3 => p16_gain_impl_v3, p16_sum_v3 => p16_sum_impl_v3);
entries!("aarch64", NeonToken, p16_gain_neon => p16_gain_impl_neon, p16_sum_neon => p16_sum_impl_neon);
entries!("wasm32", Wasm128Token, p16_gain_wasm128 => p16_gain_impl_wasm128, p16_sum_wasm128 => p16_sum_impl_wasm128);
