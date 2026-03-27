// #[autoversion] with ScalarToken — kept in dispatcher for incant! nesting
use archmage::{autoversion, ScalarToken};

#[autoversion]
fn process(token: ScalarToken, data: &[f32; 4]) -> f32 {
    data.iter().sum()
}

fn main() {}
