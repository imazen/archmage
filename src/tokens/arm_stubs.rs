//! ARM token stubs for non-ARM architectures.
//!
//! These types exist so cross-platform code can reference them without cfg guards.
//! `summon()` always returns `None` on non-ARM.

use super::SimdToken;
use super::{Has128BitSimd, HasFma, HasScalableVectors};
use super::{HasNeon, HasSve, HasSve2};

macro_rules! define_arm_stub {
    ($name:ident, $display:literal) => {
        #[doc = concat!("Stub for ", $display, " token (not available on this architecture).")]
        #[derive(Clone, Copy, Debug)]
        pub struct $name {
            _private: (),
        }

        impl SimdToken for $name {
            const NAME: &'static str = $display;

            #[inline]
            fn try_new() -> Option<Self> {
                None // Not available on this architecture
            }

            #[inline(always)]
            unsafe fn forge_token_dangerously() -> Self {
                Self { _private: () }
            }
        }
    };
}

// Define all ARM token stubs
define_arm_stub!(NeonToken, "NEON");
define_arm_stub!(ArmCryptoToken, "ARM Crypto");
define_arm_stub!(ArmCrypto3Token, "ARM Crypto3");
define_arm_stub!(SveToken, "SVE");
define_arm_stub!(Sve2Token, "SVE2");

/// The baseline for AArch64 (NEON) - stub on non-ARM architectures.
pub type Arm64 = NeonToken;

// Implement marker traits for stubs
impl Has128BitSimd for NeonToken {}
impl HasFma for NeonToken {}

impl Has128BitSimd for ArmCryptoToken {}
impl HasFma for ArmCryptoToken {}

impl Has128BitSimd for ArmCrypto3Token {}
impl HasFma for ArmCrypto3Token {}

impl Has128BitSimd for SveToken {}
impl HasFma for SveToken {}
impl HasScalableVectors for SveToken {}

impl Has128BitSimd for Sve2Token {}
impl HasFma for Sve2Token {}
impl HasScalableVectors for Sve2Token {}

impl HasNeon for NeonToken {}
impl HasNeon for ArmCryptoToken {}
impl HasNeon for ArmCrypto3Token {}
impl HasNeon for SveToken {}
impl HasNeon for Sve2Token {}

impl HasSve for SveToken {}
impl HasSve for Sve2Token {}

impl HasSve2 for Sve2Token {}
