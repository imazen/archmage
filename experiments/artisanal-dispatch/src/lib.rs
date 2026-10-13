//! Small, explicit SIMD boundaries. Experimental; see README for limitations.
#![no_std]
#![forbid(unsafe_code)]
#[cfg(feature = "std")]
extern crate std;
pub use artisanal_dispatch_macros::{bridge, dispatch, tier};
artisanal_dispatch_macros::tokens!();
