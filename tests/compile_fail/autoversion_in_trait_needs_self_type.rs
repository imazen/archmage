//! `in_trait` nests the variants inside the dispatcher, so a method's receiver
//! must become a `_self` parameter of a named type.
#![forbid(unsafe_code)]
use archmage::autoversion;

struct S {
    k: f32,
}

trait Work {
    fn run(&self, x: f32) -> f32;
}

impl Work for S {
    #[autoversion(v3, scalar, in_trait)]
    fn run(&self, x: f32) -> f32 {
        self.k + x
    }
}

fn main() {}
