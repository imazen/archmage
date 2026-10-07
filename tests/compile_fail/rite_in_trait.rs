//! A trait method cannot carry `#[target_feature]`, so `#[rite(in_trait)]`
//! names the fix instead of leaving rustc's error on generated code.
#![forbid(unsafe_code)]
use archmage::prelude::*;

struct Engine {
    factor: f32,
}

trait Compute {
    fn compute(&self, token: X64V3Token, x: f32) -> f32;
}

impl Compute for Engine {
    #[rite(in_trait)]
    fn compute(&self, _token: X64V3Token, x: f32) -> f32 {
        x * self.factor
    }
}

fn main() {}
