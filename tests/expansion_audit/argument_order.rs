#![forbid(unsafe_code)]
use archmage::{arcane, autoversion, incant, ScalarToken, X64V3Token};
#[arcane]
fn pair_v3(_: X64V3Token, a: String, b: String) -> String { a + &b }
fn pair_scalar(_: ScalarToken, a: String, b: String) -> String { a + &b }
#[autoversion(v3, scalar)]
fn destructured((a, b): (String, String), _: String) -> String { a + &b }
fn main() {
    let mut order = Vec::new();
    let result = incant!(pair({ order.push(1); String::from("a") }, { order.push(2); String::from("b") }), [v3, scalar]);
    assert_eq!(order, [1, 2]);
    assert_eq!(result, "ab");
    assert_eq!(destructured(("c".into(), "d".into()), "ignored".into()), "cd");
}
