#![no_std]
#![forbid(unsafe_code)]
pub fn add_f32(a: f32, b: f32) -> f32 {
    a.algebraic_add(b)
}
pub fn sub_f32(a: f32, b: f32) -> f32 {
    a.algebraic_sub(b)
}
pub fn mul_f32(a: f32, b: f32) -> f32 {
    a.algebraic_mul(b)
}
pub fn div_f32(a: f32, b: f32) -> f32 {
    a.algebraic_div(b)
}
pub fn rem_f32(a: f32, b: f32) -> f32 {
    a.algebraic_rem(b)
}
pub fn add_f64(a: f64, b: f64) -> f64 {
    a.algebraic_add(b)
}
pub fn sub_f64(a: f64, b: f64) -> f64 {
    a.algebraic_sub(b)
}
pub fn mul_f64(a: f64, b: f64) -> f64 {
    a.algebraic_mul(b)
}
pub fn div_f64(a: f64, b: f64) -> f64 {
    a.algebraic_div(b)
}
pub fn rem_f64(a: f64, b: f64) -> f64 {
    a.algebraic_rem(b)
}
