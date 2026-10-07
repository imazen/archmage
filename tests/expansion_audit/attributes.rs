#![allow(dead_code, unused_variables)]
use archmage::{arcane, SimdToken, X64V3Token};
#[arcane]
#[track_caller]
fn caller(token: X64V3Token) -> u32 { core::panic::Location::caller().line() }
#[arcane]
#[cfg(any())]
fn removed(token: X64V3Token, x: MissingType) { missing_function(x); }
fn main() {
    let t = X64V3Token::summon().expect("audit host needs V3");
    let expected = line!() + 1;
    let observed = caller(t);
    println!("expected={expected} observed={observed}");
    assert_eq!(observed, expected);
}
