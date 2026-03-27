// #[rite] on a trait impl method — single-tier with token
use archmage::{rite, X64V3Token};

struct Engine {
    factor: f32,
}

trait Compute {
    fn compute(&self, token: X64V3Token, x: f32) -> f32;
}

impl Compute for Engine {
    #[rite]
    fn compute(&self, token: X64V3Token, x: f32) -> f32 {
        x * self.factor
    }
}

fn main() {}
