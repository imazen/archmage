//! Test: calling crate defines `avx512 = ["archmage/avx512"]` and builds with it.
//!
//! When built with `--features avx512`:
//! - archmage has avx512 → v4 in default tier list, safe memory ops available
//! - incant! dispatches v4 → v3 → scalar
//!
//! When built WITHOUT --features avx512:
//! - archmage lacks avx512 → v4 not in default tier list
//! - incant! dispatches v3 → scalar (v4 function doesn't exist)
#![deny(warnings)]

use archmage::prelude::*;

// v4 variant — only exists when avx512 feature is enabled
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
#[arcane(import_intrinsics)]
fn add_v4(_token: X64V4Token, a: &[f32; 16], b: &[f32; 16]) -> [f32; 16] {
    let va = _mm512_loadu_ps(a);
    let vb = _mm512_loadu_ps(b);
    let vc = _mm512_add_ps(va, vb);
    let mut out = [0.0f32; 16];
    _mm512_storeu_ps(&mut out, vc);
    out
}

#[cfg(target_arch = "x86_64")]
#[arcane(import_intrinsics)]
fn add_v3(_token: X64V3Token, a: &[f32; 16], b: &[f32; 16]) -> [f32; 16] {
    let mut out = [0.0f32; 16];
    for i in 0..16 {
        out[i] = a[i] + b[i];
    }
    out
}

fn add_scalar(_token: ScalarToken, a: &[f32; 16], b: &[f32; 16]) -> [f32; 16] {
    let mut out = [0.0f32; 16];
    for i in 0..16 {
        out[i] = a[i] + b[i];
    }
    out
}

/// Uses default tiers — v4 is included when avx512 feature is on archmage,
/// excluded when it's off. No explicit tier list needed.
pub fn add_dispatched(a: &[f32; 16], b: &[f32; 16]) -> [f32; 16] {
    incant!(add(a, b))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dispatch_works() {
        let a = [1.0f32; 16];
        let b = [2.0f32; 16];
        let out = add_dispatched(&a, &b);
        assert!((out[0] - 3.0).abs() < 1e-6);
    }
}
