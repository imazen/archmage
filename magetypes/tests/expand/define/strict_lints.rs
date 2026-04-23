// Tests: generated code survives strict lint settings.
// Users with `deny(warnings)` or `deny(non_camel_case_types)` shouldn't
// get failures from the macro's emitted type aliases.
#![deny(non_camel_case_types, non_snake_case, warnings)]

use archmage::magetypes;

#[magetypes(define(f32x8, u8x16), v3, scalar)]
fn kernel(token: Token, data: &[f32; 8], bytes: &[u8; 16]) -> f32 {
    let _ = u8x16::load(token, bytes);
    f32x8::load(token, data).reduce_add()
}

fn main() {}
