// #[autoversion] on a trait impl method with _self = Type
use archmage::autoversion;

struct Filter {
    cutoff: f32,
}

trait Process {
    fn process(&self, data: &[f32; 4]) -> f32;
}

impl Process for Filter {
    #[autoversion(_self = Filter)]
    fn process(&self, data: &[f32; 4]) -> f32 {
        let mut sum = 0.0f32;
        for &x in data {
            if x > _self.cutoff {
                sum += x;
            }
        }
        sum
    }
}

fn main() {}
