// #[autoversion] with _self = Type on an inherent method
use archmage::autoversion;

struct Filter {
    threshold: f32,
}

impl Filter {
    #[autoversion(_self = Filter)]
    fn apply(&self, data: &[f32; 4]) -> f32 {
        let mut sum = 0.0f32;
        for &x in data {
            if x > _self.threshold {
                sum += x;
            }
        }
        sum
    }
}

fn main() {}
