#![forbid(unsafe_code)]
use archmage::X64V3Token;
use magetypes::simd::generic::{self, local::f32x8};
type V = f32x8<X64V3Token>;
#[archmage::rite(v3)]
fn matching() -> V {
    let constructor: fn() -> V = V::zero;
    let closure = || V::splat(2.0);
    constructor() + closure()
}
#[archmage::rite(v4)]
fn superset() -> V { V::zero() }
fn old_generic<T: magetypes::simd::backends::F32x8Backend>(token: T) {
    let _: generic::f32x8<T> = generic::f32x8::zero(token);
}
fn main() {}
