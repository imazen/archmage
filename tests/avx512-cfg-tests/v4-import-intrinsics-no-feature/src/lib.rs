//! Test: V4 + import_intrinsics WITHOUT avx512 feature on archmage.
//!
//! This crate is expected to FAIL compilation.
//! Without avx512, `_mm512_loadu_ps` resolves to the core::arch version
//! which takes `*const f32`, not `&[f32; 16]`.
//!
//! The purpose is to document what error message the user gets.
#![deny(warnings)]

use archmage::prelude::*;

#[cfg(target_arch = "x86_64")]
#[arcane(import_intrinsics)]
pub fn v4_load_add(token: X64V4Token, a: &[f32; 16], b: &[f32; 16]) -> [f32; 16] {
    // _mm512_loadu_ps: if safe version available, takes &[f32; 16]
    // if only core::arch version, takes *const f32 — type error here
    let va = _mm512_loadu_ps(a);
    let vb = _mm512_loadu_ps(b);
    let vc = _mm512_add_ps(va, vb);
    let mut out = [0.0f32; 16];
    _mm512_storeu_ps(&mut out, vc);
    out
}
