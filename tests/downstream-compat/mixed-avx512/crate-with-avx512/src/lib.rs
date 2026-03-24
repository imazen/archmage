//! Crate that enables avx512 on archmage. Has _v4 functions.
#![deny(warnings)]

use archmage::prelude::*;

#[cfg(target_arch = "x86_64")]
#[arcane]
fn add_v4(_token: X64V4Token, a: f32, b: f32) -> f32 { a + b }

#[cfg(target_arch = "x86_64")]
#[arcane]
fn add_v3(_token: X64V3Token, a: f32, b: f32) -> f32 { a + b }

fn add_scalar(_token: ScalarToken, a: f32, b: f32) -> f32 { a + b }

pub fn add(a: f32, b: f32) -> f32 {
    incant!(add(a, b), [v4, v3, scalar])
}
