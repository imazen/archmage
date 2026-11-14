#![forbid(unsafe_code)]
//! Raw-identifier function names (`fn r#move`) emit `__arcane_move` /
//! `__simd_inner_move` (with `r#` stripped by `format_ident!`). The parameter
//! shadowing check must reject a parameter named `__arcane_move` or
//! `__simd_inner_move`, not compare against `__arcane_r#move`.

use archmage::{X64V3Token, arcane};

#[arcane]
fn r#move(_token: X64V3Token, __arcane_move: u32) -> u32 {
    __arcane_move
}

fn main() {}
