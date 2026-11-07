//! ISA-consistent math kernels: identical bits on every tier.
//!
//! Every result here is the same on x86 v3 and v4x, NEON, WASM (strict and
//! relaxed) and the scalar tier, and equals the matching `zenmath::scalar`
//! mirror. NaN results agree in being NaN; their payload and sign are
//! unspecified. `tests/parity.rs` and `tests/golden.rs` check it per tier with
//! no tolerance.
//!
//! The lowp/midp polynomial steps use fused multiply-add. On x86 V3/V4x and NEON that is a
//! hardware instruction; strict WASM and baseline x86 scalar use software.
//! [`crate::relaxed`] keeps the same special-value handling and uses each
//! backend's multiply-add rounding. Its finite results may differ across tiers;
//! the size of the difference depends on the operation and input band.
//! `docs/DESIGN.md` records the contracts and measured costs.
//!
//! - [`kernels`]: token-generic `f32x8` kernels, to compose inside your own
//!   `#[attune]` bodies.
//! - [`slice`](mod@slice): `&mut [f32]` bulk ops.
//! - [`array`](mod@array): `[f32; 8]` ops.
//!
//! The unsuffixed `exp_f32`, `ln_f32`, `log2_f32`, and `pow_f32` slice
//! functions (and their array counterparts) are correctly rounded CORE-MATH
//! ports. They preserve the source algorithm's individual fused or separate
//! steps and support subnormals. They assume nearest-even and gradual underflow.
//!
//! ## Approximation specials
//!
//! | ops | NaN | other inputs |
//! |---|---|---|
//! | `sqrt`, `floor`, `ceil`, `roundeven` | NaN | IEEE-754 |
//! | `log2`, `ln`, `log10` (both tiers) | NaN | 0 → −inf, negative → NaN, +inf → +inf |
//! | `exp2_midp`, `exp_midp` | NaN | x < -126 → 0, x ≥ 128 → +inf (`exp`: after scaling by log2 e) |
//! | `exp2_lowp`, `exp_lowp` | NaN | input clamps to [-126, 127.99], so results stay finite and nonzero |
//! | `pow_midp` | NaN | negative x → NaN; 0 · inf in the exponent → NaN |
//! | `pow_lowp` | NaN | negative x → NaN; ±0 → +0 at every n; +inf with n = 0 → NaN; otherwise clamps like `exp2_lowp` |
//! | `cbrt_lowp`, `cbrt_midp` | NaN | ±0 → ±0; subnormals inaccurate; magnitudes above `f32::MAX` / 3 and ±inf → NaN or ±inf |
//! | `cbrt_midp_precise` | NaN | ±0 and ±inf return themselves; every other input is in domain |
//! | `sigmoid_midp`, `silu_midp` | NaN | saturate exactly: `sigmoid` to 0 or 1 |
//!
//! Subnormal inputs to the lowp/midp log family are outside the accuracy domain: finite,
//! and the same on every tier, but about -127 rather than the true value.
//!
//! Per-ISA kernels that do *not* share this contract belong in [`crate::native`].

pub mod array;
pub mod kernels;
pub mod slice;
