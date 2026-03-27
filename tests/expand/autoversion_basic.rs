// #[autoversion] basic — tokenless, default tier list
use archmage::autoversion;

#[autoversion]
fn sum_squares(data: &[f32; 4]) -> f32 {
    let mut sum = 0.0f32;
    for &x in data {
        sum += x * x;
    }
    sum
}

fn main() {}
