//! `in_impl` belongs to `#[arcane]`; `#[rite]` has no wrapper to qualify.
#![forbid(unsafe_code)]
use archmage::prelude::*;

struct Engine;

impl Engine {
    #[rite(in_impl)]
    fn probe(_token: X64V3Token, x: f32) -> f32 {
        x
    }
}

fn main() {}
