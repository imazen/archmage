// #[arcane] with &mut self receiver
use archmage::{arcane, X64V3Token};

struct Buffer {
    data: [f32; 4],
}

impl Buffer {
    #[arcane]
    fn scale(&mut self, token: X64V3Token, factor: f32) {
        for x in &mut self.data {
            *x *= factor;
        }
    }
}

fn main() {}
