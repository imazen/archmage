//! Assembly probe for the v3 vs v4 target-feature context. Kernels are
//! compiled once per tier from one body; each tier gets an `#[inline(never)]`
//! entry taking the token so the kernel keeps its own symbol.
#![forbid(unsafe_code)]
#![allow(clippy::too_many_arguments)]


#[cfg(target_arch = "x86_64")]
mod entries_a;
#[cfg(target_arch = "x86_64")]
mod group_a;
#[cfg(target_arch = "x86_64")]
mod group_b;
#[cfg(target_arch = "x86_64")]
mod group_c;
mod group_p;
#[cfg(target_arch = "x86_64")]
pub use entries_a::*;
#[cfg(target_arch = "x86_64")]
pub use group_a::*;
#[cfg(target_arch = "x86_64")]
pub use group_b::*;
#[cfg(target_arch = "x86_64")]
pub use group_c::*;
pub use group_p::*;
