// #[arcane] nested mode with _self = Type (for trait impls)
use archmage::{arcane, X64V3Token};

struct Processor {
    threshold: f32,
}

trait SimdOps {
    fn process(&self, token: X64V3Token, value: f32) -> f32;
}

impl SimdOps for Processor {
    #[arcane(_self = Processor)]
    fn process(&self, token: X64V3Token, value: f32) -> f32 {
        value + _self.threshold
    }
}

fn main() {}
