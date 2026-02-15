//! Sealed trait preventing external implementations of backend traits.
//!
//! Only archmage token types can implement `F32x8Backend`, etc.
//! This is enforced at the type level via this sealed supertrait.

/// Sealed trait — cannot be implemented outside magetypes.
pub trait Sealed {}

// Implement for all token types that can have SIMD backends.
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
