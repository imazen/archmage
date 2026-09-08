//! Adapted from zenpipe/zenfilters scale_plane_simd at 12b468e3e0a8ef46e2eeb1c92d7b1a859e53036a.
//! Consolidates the production per-ISA forwarding functions into one macro.
//! See docs/site/content/magetypes/examples/generic-kernels.md for the complete source chain.
#![forbid(unsafe_code)]
use archmage::prelude::*;

#[magetypes(define(f32x8), v3, neon, wasm128, scalar)]
fn gain_impl(token: Token, plane: &mut [f32], gain: f32) {
    let factor = f32x8::splat(token, gain);
    let (chunks, tail) = f32x8::partition_slice_mut(token, plane);
    for chunk in chunks {
        (f32x8::load(token, chunk) * factor).store(chunk);
    }
    for value in tail {
        *value *= gain;
    }
}

pub fn apply_gain(plane: &mut [f32], gain: f32) {
    incant!(gain_impl(plane, gain), [v3, neon, wasm128, scalar])
}

fn main() {
    let mut plane = vec![0.25; 19];
    apply_gain(&mut plane, 2.0);
    assert_eq!(plane, vec![0.5; 19]);
}
