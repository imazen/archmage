//! Backend-dependent rounding: same polynomials and special-value rules as
//! [`portable`](crate::portable), but the bits may differ between tiers.
//!
//! Each polynomial step uses the backend semantics of `mul_add_relaxed`:
//!
//! | tier | a polynomial step | results equal |
//! |---|---|---|
//! | x86 v3, v4x, NEON | FMA, one rounding | `zenmath::scalar` |
//! | scalar tier, strict WASM | multiply, then add: two roundings | `zenmath::scalar::unfused` |
//! | relaxed WASM | the engine's madd: its choice | one of those two |
//!
//! Special-value rules match the table in
//! [`portable`](crate::portable): NaN in gives NaN out, and so on. Polynomial
//! rounding can change finite results, including outside the accuracy domain. The accuracy
//! bounds in `docs/transcendentals.md` hold for both roundings.
//!
//! - [`kernels`]: token-generic `f32x8` kernels, to compose inside your own
//!   `#[attune]` bodies.
//! - [`slice`](mod@slice): `&mut [f32]` bulk ops.
//! - [`array`](mod@array): `[f32; 8]` ops.

pub mod array;
pub mod kernels;
pub mod slice;
