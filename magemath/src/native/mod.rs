//! Reserved for per-ISA kernels whose bits differ by tier. Empty on purpose.
//!
//! Nothing lands here without a caller whose measured need [`crate::portable`]
//! cannot meet. Faster-but-less-consistent results do not need this module yet:
//! magetypes' `f32x8::exp2_midp`-style methods are already that, and are one
//! method call away. This is the home for what they are not, such as a kernel
//! with its own polynomial for a narrow input range, or per-ISA table lookups.
//!
//! The rules for anything added, and the reasoning, are in `docs/DESIGN.md`:
//! same names as `portable` under this module, so the import path states the
//! contract; never wired into a `portable` dispatcher; goldens per tier; and the
//! accuracy bounds of `tests/reference.rs` still apply.
