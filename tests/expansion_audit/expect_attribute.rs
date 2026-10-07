#![deny(unfulfilled_lint_expectations)]
#![allow(dead_code)]
use archmage::{arcane, X64V3Token};
#[arcane]
#[expect(unused_variables)]
fn unused(token: X64V3Token) {}
fn main() {}
