// A wildcard family's body is the author's code: its nested items keep the
// author's lint levels (no macro-emitted allow may cover them).
#![deny(dead_code)]
use archmage::prelude::*;

#[attune(_*, dispatch)]
pub fn scale(x: u32) -> u32 {
    fn nested_unused() {}
    x * 2
}

fn main() {
    let _ = scale(1);
}
