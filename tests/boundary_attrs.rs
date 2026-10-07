//! Attributes that must follow the body across a boundary expansion.
//!
//! `#[arcane]` (sibling and nested), `#[autoversion]` and `#[magetypes]` move
//! the user's body into a `#[target_feature]` function and leave a forwarding
//! wrapper or dispatcher under the user's name. Two attributes only work when
//! placed on the right half:
//!
//! - `#[track_caller]` has to sit on every function between a panic and the
//!   caller it should name, so it goes on both halves; on the wrapper alone,
//!   `Location::caller()` in the body named the body's own line.
//! - `#[expect(..)]` is fulfilled by the body, so the body half keeps it and
//!   the forwarding half gets `#[allow(..)]`; copied onto a wrapper that uses
//!   every parameter, the expectation was unfulfilled (a warning, or an error
//!   under the `deny` below).
//!
//! Both were reported by an external review before 0.9.30 (2026-10-07).
#![deny(unfulfilled_lint_expectations)]
#![cfg(target_arch = "x86_64")]

use archmage::{SimdToken, X64V3Token, arcane, autoversion, magetypes, rite};
use std::panic::Location;

#[arcane]
#[track_caller]
fn sibling_caller_line(_t: X64V3Token) -> u32 {
    Location::caller().line()
}

#[arcane(nested)]
#[track_caller]
fn nested_caller_line(_t: X64V3Token) -> u32 {
    Location::caller().line()
}

#[autoversion(v3, scalar)]
#[track_caller]
fn autoversion_caller_line() -> u32 {
    Location::caller().line()
}

struct Holder;
impl Holder {
    #[arcane]
    #[track_caller]
    fn method_caller_line(&self, _t: X64V3Token) -> u32 {
        Location::caller().line()
    }
}

#[test]
fn track_caller_names_the_call_site() {
    let Some(t) = X64V3Token::summon() else {
        return;
    };
    assert_eq!(sibling_caller_line(t), line!());
    assert_eq!(nested_caller_line(t), line!());
    assert_eq!(Holder.method_caller_line(t), line!());
    assert_eq!(autoversion_caller_line(), line!());
}

#[arcane]
#[expect(unused_variables)]
fn sibling_unused(_t: X64V3Token, unused: f32) -> f32 {
    1.0
}

#[arcane(nested)]
#[expect(unused_variables)]
fn nested_unused(_t: X64V3Token, unused: f32) -> f32 {
    2.0
}

impl Holder {
    #[arcane]
    #[expect(unused_variables)]
    fn method_unused(&self, _t: X64V3Token, unused: f32) -> f32 {
        3.0
    }
}

#[autoversion(v3, scalar)]
#[expect(unused_variables)]
fn autoversion_unused(unused: f32) -> f32 {
    4.0
}

#[rite(v3, scalar)]
#[expect(unused_variables)]
fn rite_unused(unused: f32) -> f32 {
    5.0
}

#[magetypes(v3, scalar)]
#[expect(unused_variables)]
fn magetypes_unused(_t: Token, unused: f32) -> f32 {
    6.0
}

#[arcane]
#[expect(unused_variables, reason = "the reason is kept on the allow too")]
fn sibling_unused_with_reason(_t: X64V3Token, unused: f32) -> f32 {
    7.0
}

#[test]
fn expect_is_fulfilled_by_the_body() {
    let Some(t) = X64V3Token::summon() else {
        return;
    };
    assert_eq!(sibling_unused(t, 0.0), 1.0);
    assert_eq!(nested_unused(t, 0.0), 2.0);
    assert_eq!(Holder.method_unused(t, 0.0), 3.0);
    assert_eq!(autoversion_unused(0.0), 4.0);
    assert_eq!(rite_unused_scalar(0.0), 5.0);
    assert_eq!(magetypes_unused_v3(t, 0.0), 6.0);
    assert_eq!(magetypes_unused_scalar(archmage::ScalarToken, 0.0), 6.0);
    assert_eq!(sibling_unused_with_reason(t, 0.0), 7.0);
}
