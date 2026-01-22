//! ARM token stubs for non-ARM architectures.
//!
//! These types exist so cross-platform code can reference them without cfg guards.
//! `summon()` always returns `None` on non-ARM.

use super::SimdToken;
use super::Has128BitSimd;
use super::{HasNeon, HasNeonAes, HasNeonSha3};

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
define_arm_stub!(NeonAesToken, "NEON+AES");
define_arm_stub!(NeonSha3Token, "NEON+SHA3");
define_arm_stub!(ArmCryptoToken, "ARM Crypto");
define_arm_stub!(ArmCrypto3Token, "ARM Crypto3");

/// The baseline for AArch64 (NEON) - stub on non-ARM architectures.
pub type Arm64 = NeonToken;

// Width traits
impl Has128BitSimd for NeonToken {}
impl Has128BitSimd for NeonAesToken {}
impl Has128BitSimd for NeonSha3Token {}
impl Has128BitSimd for ArmCryptoToken {}
impl Has128BitSimd for ArmCrypto3Token {}

// Tier traits
impl HasNeon for NeonToken {}
impl HasNeon for NeonAesToken {}
impl HasNeon for NeonSha3Token {}
impl HasNeon for ArmCryptoToken {}
impl HasNeon for ArmCrypto3Token {}

impl HasNeonAes for NeonAesToken {}
impl HasNeonAes for ArmCryptoToken {}
impl HasNeonAes for ArmCrypto3Token {}

impl HasNeonSha3 for NeonSha3Token {}
impl HasNeonSha3 for ArmCrypto3Token {}
