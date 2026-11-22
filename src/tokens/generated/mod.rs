//! Generated from token-registry.toml — DO NOT EDIT.
//!
//! cfg-gated module imports and re-exports for all token types.

mod into_concrete;
pub use into_concrete::IntoConcreteToken;

mod traits;
pub use traits::*;

// x86_64: real tokens require baseline SSE2; other targets use stubs.
#[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
mod x86;
#[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
pub use x86::*;

#[cfg(not(all(target_arch = "x86_64", target_feature = "sse2")))]
mod x86_stubs;
#[cfg(not(all(target_arch = "x86_64", target_feature = "sse2")))]
pub use x86_stubs::*;

// aarch64: real implementations on aarch64, stubs elsewhere
#[cfg(target_arch = "aarch64")]
mod arm;
#[cfg(target_arch = "aarch64")]
pub use arm::*;

#[cfg(not(target_arch = "aarch64"))]
mod arm_stubs;
#[cfg(not(target_arch = "aarch64"))]
pub use arm_stubs::*;

// wasm32: real implementations on wasm32, stubs elsewhere
#[cfg(target_arch = "wasm32")]
mod wasm;
#[cfg(target_arch = "wasm32")]
pub use wasm::*;

#[cfg(not(target_arch = "wasm32"))]
mod wasm_stubs;
#[cfg(not(target_arch = "wasm32"))]
pub use wasm_stubs::*;
