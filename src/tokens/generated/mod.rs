//! Generated from token-registry.toml — DO NOT EDIT.
//!
//! cfg-gated module imports and re-exports for all token types.

mod traits;
pub use traits::*;

// x86: real implementations on x86_64, stubs elsewhere
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
mod x86;
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
pub use x86::*;

#[cfg(not(any(target_arch = "x86_64", target_arch = "x86")))]
mod x86_stubs;
#[cfg(not(any(target_arch = "x86_64", target_arch = "x86")))]
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
