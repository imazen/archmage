// An unused #[autoversion] method reports exactly once, on the author's name;
// variants must not join (and so suppress) the dispatcher's diagnostic.
#![deny(dead_code)]
use archmage::prelude::*;

pub struct Buffer;

impl Buffer {
    #[autoversion]
    fn total(&self, data: &[f32]) -> f32 {
        data.iter().sum()
    }
}

fn main() {
    let _ = Buffer;
}
