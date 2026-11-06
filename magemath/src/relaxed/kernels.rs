//! Token-generic vector kernels, one per op, chainable inside one `#[attune]` body.
//! See the [module docs](super) for what `relaxed` promises.
//!
//! In-domain results equal magetypes' `f32x8::exp2_midp`-style methods bit for bit
//! on every tier, since both use `mul_add_relaxed` with the same polynomials. They
//! differ where those methods do not define a result: NaN, for one.
//!
//! The `*_unchecked` kernels skip edge handling. In their stated domain they return
//! the checked kernel's bits on the same tier. Outside it the result is unspecified
//! and differs by tier; it is never undefined behavior.

crate::surface::kernels_surface!(crate::bodies::Relaxed);
