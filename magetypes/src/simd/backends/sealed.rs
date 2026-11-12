//! Sealed trait to prevent external implementations of backend traits.
//!
//! **Auto-generated** by `cargo xtask generate` - do not edit manually.

/// Sealed trait — only archmage token types can implement backend traits.
pub trait Sealed {}

impl Sealed for archmage::X64V1Token {}
impl Sealed for archmage::X64V2Token {}
impl Sealed for archmage::X64CryptoToken {}
impl Sealed for archmage::X64V3Token {}
impl Sealed for archmage::X64V3CryptoToken {}
impl Sealed for archmage::X64V3GfniCryptoToken {}
impl Sealed for archmage::X64V3VnniToken {}
impl Sealed for archmage::X64V4Token {}
impl Sealed for archmage::X64V4xToken {}
impl Sealed for archmage::Avx512Fp16Token {}
impl Sealed for archmage::NeonToken {}
impl Sealed for archmage::NeonAesToken {}
impl Sealed for archmage::NeonSha3Token {}
impl Sealed for archmage::NeonCrcToken {}
impl Sealed for archmage::Arm64V2Token {}
impl Sealed for archmage::Arm64V3Token {}
impl Sealed for archmage::Wasm128Token {}
impl Sealed for archmage::Wasm128RelaxedToken {}
impl Sealed for archmage::ScalarToken {}

/// Sealed marker trait for tokens whose 128-bit SIMD `Repr` is a native
/// single-register platform SIMD type (`__m128*`, NEON `*x*_t`, `v128`).
pub trait NativeW128Backend: Sealed {}

/// Sealed marker trait for tokens whose 256-bit SIMD `Repr` is a native
/// single-register platform SIMD type (`__m256*`).
pub trait NativeW256Backend: Sealed {}

/// Sealed marker trait for tokens whose 512-bit SIMD `Repr` is a native
/// single-register platform SIMD type (`__m512*`).
pub trait NativeW512Backend: Sealed {}

#[cfg(target_arch = "x86_64")]
impl NativeW128Backend for archmage::X64V3Token {}
#[cfg(target_arch = "x86_64")]
impl NativeW128Backend for archmage::X64V3CryptoToken {}
#[cfg(target_arch = "x86_64")]
impl NativeW128Backend for archmage::X64V3GfniCryptoToken {}
#[cfg(target_arch = "x86_64")]
impl NativeW128Backend for archmage::X64V3VnniToken {}
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
impl NativeW128Backend for archmage::X64V4Token {}
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
impl NativeW128Backend for archmage::X64V4xToken {}

#[cfg(target_arch = "aarch64")]
impl NativeW128Backend for archmage::NeonToken {}
#[cfg(target_arch = "aarch64")]
impl NativeW128Backend for archmage::Arm64V2Token {}
#[cfg(target_arch = "aarch64")]
impl NativeW128Backend for archmage::Arm64V3Token {}

#[cfg(target_arch = "wasm32")]
impl NativeW128Backend for archmage::Wasm128Token {}
#[cfg(target_arch = "wasm32")]
impl NativeW128Backend for archmage::Wasm128RelaxedToken {}

#[cfg(target_arch = "x86_64")]
impl NativeW256Backend for archmage::X64V3Token {}
#[cfg(target_arch = "x86_64")]
impl NativeW256Backend for archmage::X64V3CryptoToken {}
#[cfg(target_arch = "x86_64")]
impl NativeW256Backend for archmage::X64V3GfniCryptoToken {}
#[cfg(target_arch = "x86_64")]
impl NativeW256Backend for archmage::X64V3VnniToken {}
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
impl NativeW256Backend for archmage::X64V4Token {}
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
impl NativeW256Backend for archmage::X64V4xToken {}

#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
impl NativeW512Backend for archmage::X64V4Token {}
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
impl NativeW512Backend for archmage::X64V4xToken {}
