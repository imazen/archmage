// Tests: generated code survives strict naming-lint settings.
// Users with `deny(non_camel_case_types)` or `deny(non_snake_case)` shouldn't
// get failures from the macro's emitted type aliases — the emitted `type f32x8`
// carries `#[allow(non_camel_case_types)]`, and all generated fn names are
// snake_case.
//
// (Not testing `deny(warnings)` — `use archmage::magetypes` shows as unused
// in the post-expansion view since the attribute was consumed; that's
// cosmetic and unrelated to the naming-lint story this file tests.)
#![deny(non_camel_case_types, non_snake_case)]

use archmage::magetypes;

#[magetypes(define(f32x8, u8x16), v3, scalar)]
fn kernel(token: Token, data: &[f32; 8], bytes: &[u8; 16]) -> f32 {
    let _ = u8x16::load(token, bytes);
    f32x8::load(token, data).reduce_add()
}

fn main() {}
