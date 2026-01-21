//! ARM token stubs for non-ARM architectures.
//!
//! These types exist so cross-platform code can reference them without cfg guards.
//! `summon()` always returns `None` on non-ARM.

use super::sealed::Sealed;
use super::SimdToken;
use super::{Has128BitSimd, HasFma};
use super::{HasArmAes, HasArmFp16, HasArmSha3, HasNeon};

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

        impl Sealed for $name {}
    };
}

// Define all ARM token stubs
define_arm_stub!(NeonToken, "NEON");
define_arm_stub!(NeonAesToken, "NEON+AES");
define_arm_stub!(NeonSha3Token, "NEON+SHA3");
define_arm_stub!(NeonFp16Token, "NEON+FP16");

/// Backward compatibility alias for [`NeonAesToken`].
#[deprecated(since = "0.2.0", note = "Use NeonAesToken instead")]
pub type ArmCryptoToken = NeonAesToken;

/// Backward compatibility alias for [`NeonSha3Token`].
#[deprecated(since = "0.2.0", note = "Use NeonSha3Token instead")]
pub type ArmCrypto3Token = NeonSha3Token;

/// The recommended starting point for AArch64 - stub on non-ARM architectures.
pub type Arm64 = NeonFp16Token;

// Implement marker traits for stubs
impl Has128BitSimd for NeonToken {}
impl HasFma for NeonToken {}

impl Has128BitSimd for NeonAesToken {}
impl HasFma for NeonAesToken {}

impl Has128BitSimd for NeonSha3Token {}
impl HasFma for NeonSha3Token {}

impl Has128BitSimd for NeonFp16Token {}
impl HasFma for NeonFp16Token {}

impl HasNeon for NeonToken {}
impl HasNeon for NeonAesToken {}
impl HasNeon for NeonSha3Token {}
impl HasNeon for NeonFp16Token {}

impl HasArmAes for NeonAesToken {}
impl HasArmAes for NeonSha3Token {}

impl HasArmSha3 for NeonSha3Token {}

impl HasArmFp16 for NeonFp16Token {}
