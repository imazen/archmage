#![forbid(unsafe_code)]
#![deny(warnings)]
use dependency_renamed::{macro_alias as m, type_alias as t};

fn main() {
    let expected = if cfg!(expected_fast) { 14 } else { 13 };
    assert_eq!(m(10), expected);
    assert_eq!(m!(10), expected);
    assert_eq!(t(10), expected);
    assert_eq!(t::selected(10), expected);
    assert_eq!(t::generic::<u8, 3>([1, 2, 3]), 6);
    println!("provider-selected={expected}; macro/function/type re-exports passed");
}
