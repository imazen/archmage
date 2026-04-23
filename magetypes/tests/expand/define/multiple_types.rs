// Tests: multiple types in define(...) all get aliases injected.
use archmage::magetypes;

#[magetypes(define(f32x8, f32x4, u8x16), v3, scalar)]
fn kernel(
    token: Token,
    data_8: &[f32; 8],
    data_4: &[f32; 4],
    bytes: &[u8; 16],
) -> f32 {
    let a = f32x8::load(token, data_8).reduce_add();
    let b = f32x4::load(token, data_4).reduce_add();
    let _c = u8x16::load(token, bytes);
    a + b
}

fn main() {}
