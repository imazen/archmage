// #[arcane] sibling mode with self receiver (inherent method)
use archmage::{arcane, X64V3Token};

struct Processor {
    threshold: f32,
}

impl Processor {
    #[arcane]
    fn compute(&self, token: X64V3Token, value: f32) -> f32 {
        value + self.threshold
    }
}

fn main() {}
