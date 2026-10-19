//! Integration test crate: archmage with NO feature flags.
//!
//! This crate depends on archmage with zero features enabled — no "avx512",
//! no "std". It exercises all dispatch macros to verify they generate valid,
//! warning-free code without any cargo feature gates.
//!
//! If a macro silently wraps variants in `#[cfg(feature = "avx512")]`, those
//! variants become dead code here (the feature doesn't exist in this crate),
//! and `#[deny(dead_code)]` will catch it.
#![deny(warnings)]
#![allow(deprecated)] // Legacy SimdToken usage in autoversion — will migrate to tokenless
#![no_std]

extern crate alloc;
use alloc::vec::Vec;

use archmage::prelude::*;

// ============================================================================
// #[autoversion] — default tiers (includes v4)
// ============================================================================

#[autoversion]
pub fn sum_squares(_token: SimdToken, data: &[f32]) -> f32 {
    let mut sum = 0.0f32;
    for &x in data {
        sum += x * x;
    }
    sum
}

#[autoversion]
pub fn scale_vec(_token: SimdToken, data: &[f32], factor: f32) -> Vec<f32> {
    data.iter().map(|&x| x * factor).collect()
}

// ============================================================================
// #[autoversion] — explicit tiers including v4
// ============================================================================

#[autoversion(v3, v4, neon)]
pub fn dot_product(_token: SimdToken, a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    let mut sum = 0.0f32;
    for i in 0..n {
        sum += a[i] * b[i];
    }
    sum
}

// ============================================================================
// #[autoversion] — explicit non-x86 tiers (issue #34 repro)
// ============================================================================
//
// On 32-bit x86 (target_arch = "x86"), none of `v3` / `neon` / `wasm128`
// are dispatchable — every variant is cfg'd out. The dispatcher's
// `use archmage::SimdToken;` then has nothing to refer to and triggers
// `unused_imports` warnings unless the macro suppresses them.
//
// The downstream symptom (per issue #34) is a warning like:
//   warning: unused import: `entropy_score`
// pointing at the user's fn declaration via the dispatcher's quote_spanned!.
//
// This crate compiles with `#![deny(warnings)]`, so any regression breaks
// the build on i686.

#[autoversion(v3, neon, wasm128)]
pub fn entropy_score(_token: SimdToken, data: &[u8]) -> u32 {
    let mut score = 0u32;
    for &b in data {
        score = score.wrapping_add(b as u32);
    }
    score
}

#[autoversion(v3, neon, wasm128)]
pub fn premul_u8_impl(_token: SimdToken, buf: &mut [u8]) {
    for chunk in buf.as_chunks_mut::<4>().0 {
        let a = chunk[3] as u16;
        chunk[0] = ((chunk[0] as u16 * a + 127) / 255) as u8;
        chunk[1] = ((chunk[1] as u16 * a + 127) / 255) as u8;
        chunk[2] = ((chunk[2] as u16 * a + 127) / 255) as u8;
    }
}

// ============================================================================
// #[autoversion] with self receiver
// ============================================================================

pub struct Buffer {
    data: Vec<f32>,
}

impl Buffer {
    #[autoversion]
    pub fn total(&self, _token: SimdToken) -> f32 {
        self.data.iter().sum()
    }
}

// ============================================================================
// attuned! in value position (clippy -D warnings runs on this crate)
// ============================================================================
//
// A call that resolves to a single guaranteed arm (every covered call in an
// #[attune] body, or a scalar-only list) used to expand to
// `{ 'l: { { break 'l f(x); } } }`; in a `let`, argument or `return` clippy
// reported diverging_sub_expression. A statement-start continuation such as
// `attuned!(f(x)) + 1` in an #[attune] body did not parse.

#[attune(make(_v3, _v3_t, _scalar, _scalar_t, _))]
pub fn bump(x: u32) -> u32 {
    x + 1
}

fn identity(x: u32) -> u32 {
    x
}

pub fn scalar_let(x: u32) -> u32 {
    let y = attuned!(bump(x), [_scalar]);
    y * 2
}

pub fn scalar_arg(x: u32) -> u32 {
    identity(attuned!(bump(x), [_scalar]))
}

#[attune(make(_v3, _v3_t, _scalar, _scalar_t, _))]
pub fn covered_let(x: u32) -> u32 {
    let y = attuned!(bump(x));
    y * 2
}

#[attune(make(_v3, _v3_t, _scalar, _scalar_t, _))]
pub fn covered_arg(x: u32) -> u32 {
    let y = identity(attuned!(bump(x)));
    y * 2
}

#[attune(make(_v3, _v3_t, _scalar, _scalar_t, _))]
pub fn covered_return(x: u32) -> u32 {
    if x == 0 {
        return attuned!(bump(x));
    }
    x
}

#[attune(make(_v3, _v3_t, _scalar, _scalar_t, _))]
pub fn covered_statement_start(x: u32) -> u32 {
    attuned!(bump(x)) + attuned!(bump(x)).wrapping_mul(2)
}

// ============================================================================
// Smoke tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    #[test]
    fn autoversion_default_tiers() {
        let data = [1.0f32, 2.0, 3.0, 4.0];
        let result = sum_squares(&data);
        assert!((result - 30.0).abs() < 1e-6);
    }

    #[test]
    fn autoversion_explicit_v4_tier() {
        let a = [1.0f32, 2.0, 3.0];
        let b = [4.0f32, 5.0, 6.0];
        let result = dot_product(&a, &b);
        assert!((result - 32.0).abs() < 1e-6);
    }

    #[test]
    fn autoversion_self_receiver() {
        let buf = Buffer {
            data: vec![1.0, 2.0, 3.0, 4.0],
        };
        assert!((buf.total() - 10.0).abs() < 1e-6);
    }

    #[test]
    fn autoversion_allocating_return() {
        let data = [1.0f32, 2.0, 3.0];
        let result = scale_vec(&data, 2.0);
        assert_eq!(result, vec![2.0, 4.0, 6.0]);
    }

    /// Verify v4 variant is callable on x86_64 (if CPU supports it).
    /// The point is that it COMPILES — the cfg guard doesn't eliminate it.
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn v4_variant_reachable() {
        if let Some(token) = X64V4Token::summon() {
            let data = [1.0f32, 2.0, 3.0, 4.0];
            let result = sum_squares_v4(token, &data);
            assert!((result - 30.0).abs() < 1e-6);
        }
    }

    /// Verify v3 variant is callable on x86_64.
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn v3_variant_reachable() {
        if let Some(token) = X64V3Token::summon() {
            let data = [1.0f32, 2.0, 3.0, 4.0];
            let result = sum_squares_v3(token, &data);
            assert!((result - 30.0).abs() < 1e-6);
        }
    }

    /// Issue #34 repro: explicit non-x86 tier list dispatcher must compile
    /// cleanly on every arch including i686 (where every cfg arm is excluded).
    #[test]
    fn issue_34_explicit_non_x86_tiers_compiles() {
        let data = [1u8, 2, 3, 4];
        let _ = entropy_score(&data);
        let mut buf = [10u8, 20, 30, 200];
        premul_u8_impl(&mut buf);
    }

    #[test]
    fn attuned_value_positions() {
        assert_eq!(scalar_let(1), 4);
        assert_eq!(scalar_arg(1), 2);
        assert_eq!(covered_let(1), 4);
        assert_eq!(covered_arg(1), 4);
        assert_eq!(covered_return(0), 1);
        assert_eq!(covered_return(5), 5);
        assert_eq!(covered_statement_start(1), 6);
    }

    #[test]
    fn scalar_variant_reachable() {
        let data = [1.0f32, 2.0, 3.0, 4.0];
        let result = sum_squares_scalar(ScalarToken, &data);
        assert!((result - 30.0).abs() < 1e-6);
    }
}
