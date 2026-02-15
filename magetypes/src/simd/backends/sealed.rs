//! Sealed trait to prevent external implementations of backend traits.
//!
//! **Auto-generated** by `cargo xtask generate` - do not edit manually.

/// Sealed trait — only archmage token types can implement backend traits.
pub trait Sealed {}

impl Sealed for archmage::X64V1Token {}
impl Sealed for archmage::X64V2Token {}
impl Sealed for archmage::X64V3Token {}
impl Sealed for archmage::X64V4Token {}
impl Sealed for archmage::Avx512ModernToken {}
impl Sealed for archmage::Avx512Fp16Token {}
impl Sealed for archmage::NeonToken {}
impl Sealed for archmage::NeonAesToken {}
impl Sealed for archmage::NeonSha3Token {}
impl Sealed for archmage::NeonCrcToken {}
impl Sealed for archmage::Wasm128Token {}
impl Sealed for archmage::ScalarToken {}
