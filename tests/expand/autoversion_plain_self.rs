// #[autoversion] with &self but no _self = Type — sibling mode on inherent method
use archmage::autoversion;

struct Processor {
    factor: f32,
}

impl Processor {
    #[autoversion]
    fn apply(&self, data: &[f32; 4]) -> f32 {
        let mut sum = 0.0f32;
        for &x in data {
            sum += x * self.factor;
        }
        sum
    }
}

fn main() {}
