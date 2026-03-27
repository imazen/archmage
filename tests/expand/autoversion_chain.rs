// Two #[autoversion] functions — outer calls inner (re-dispatches today)
use archmage::autoversion;

#[autoversion]
fn inner_work(data: &[f32; 4]) -> f32 {
    data.iter().sum()
}

#[autoversion]
fn outer_work(data: &[f32; 4], scale: f32) -> f32 {
    inner_work(data) * scale
}

fn main() {}
