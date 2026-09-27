//! Constructor policies. Both modes retain the same capability token.
mod sealed {
    pub trait Sealed {}
}
/// Sealed policy selecting a vector's construction API.
pub trait ConstructorMode: sealed::Sealed + Copy + 'static {}
/// Construction with an explicit CPU capability token.
#[derive(Clone, Copy, Debug)]
pub struct Explicit;
/// Construction in a compiler-checked target-feature context.
#[derive(Clone, Copy, Debug)]
pub struct Context;
impl sealed::Sealed for Explicit {}
impl sealed::Sealed for Context {}
impl ConstructorMode for Explicit {}
impl ConstructorMode for Context {}
