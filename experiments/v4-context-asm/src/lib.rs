//! Assembly probe for the v3 vs v4 target-feature context. Kernels are
//! compiled once per tier from one body; each tier gets an `#[inline(never)]`
//! entry taking the token so the kernel keeps its own symbol.
#![forbid(unsafe_code)]
#![allow(clippy::too_many_arguments)]


mod entries_a;
mod group_a;
mod group_b;
mod group_c;
pub use entries_a::*;
pub use group_a::*;
pub use group_b::*;
pub use group_c::*;
