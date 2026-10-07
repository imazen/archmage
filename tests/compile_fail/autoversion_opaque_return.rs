//! Every tier variant of an `impl Trait` return would be its own opaque type,
//! so the dispatcher cannot return it. The macro says so (issue #122).
#![forbid(unsafe_code)]
use archmage::autoversion;

#[autoversion(v3, scalar)]
fn numbers(n: u32) -> impl Iterator<Item = u32> {
    0..n
}

fn main() {
    let _ = numbers(4);
}
