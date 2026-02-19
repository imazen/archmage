//! Generated from token-registry.toml — DO NOT EDIT.
//!
//! Regenerate with: cargo xtask generate

use crate::tokens::SimdToken;
use crate::tokens::{Has128BitSimd, HasArm64V2, HasArm64V3, HasNeon, HasNeonAes, HasNeonSha3};
use core::sync::atomic::{AtomicBool, AtomicU8, Ordering};

// Cache statics: 0 = unknown, 1 = unavailable, 2 = available
pub(super) static NEON_CACHE: AtomicU8 = AtomicU8::new(0);
pub(super) static NEON_DISABLED: AtomicBool = AtomicBool::new(false);
pub(super) static NEON_AES_CACHE: AtomicU8 = AtomicU8::new(0);
pub(super) static NEON_AES_DISABLED: AtomicBool = AtomicBool::new(false);
pub(super) static NEON_SHA3_CACHE: AtomicU8 = AtomicU8::new(0);
pub(super) static NEON_SHA3_DISABLED: AtomicBool = AtomicBool::new(false);
pub(super) static NEON_CRC_CACHE: AtomicU8 = AtomicU8::new(0);
pub(super) static NEON_CRC_DISABLED: AtomicBool = AtomicBool::new(false);
pub(super) static ARM64_V2_CACHE: AtomicU8 = AtomicU8::new(0);
pub(super) static ARM64_V2_DISABLED: AtomicBool = AtomicBool::new(false);
pub(super) static ARM64_V3_CACHE: AtomicU8 = AtomicU8::new(0);
pub(super) static ARM64_V3_DISABLED: AtomicBool = AtomicBool::new(false);

/// Proof that NEON is available.
///
/// NEON is available on virtually all AArch64 processors, but requires
/// runtime detection via `summon()` unless compiled with `-Ctarget-feature=+neon`.
#[derive(Clone, Copy, Debug)]
pub struct NeonToken {
    _private: (),
}

impl crate::tokens::Sealed for NeonToken {}

impl SimdToken for NeonToken {
    const NAME: &'static str = "NEON";
    const TARGET_FEATURES: &'static str = "neon";
    const ENABLE_TARGET_FEATURES: &'static str = "-Ctarget-feature=+neon";
    const DISABLE_TARGET_FEATURES: &'static str = "-Ctarget-feature=-neon";

    #[inline]
    fn compiled_with() -> Option<bool> {
        #[cfg(all(target_feature = "neon", not(feature = "testable_dispatch")))]
        {
            Some(true)
        }
        #[cfg(not(all(target_feature = "neon", not(feature = "testable_dispatch"))))]
        {
            None
        }
    }

    #[allow(deprecated)]
    #[inline(always)]
    fn summon() -> Option<Self> {
        // Compile-time fast path (suppressed by testable_dispatch)
        #[cfg(all(target_feature = "neon", not(feature = "testable_dispatch")))]
        {
            return Some(unsafe { Self::forge_token_dangerously() });
        }

        // Runtime path with caching
        #[cfg(not(all(target_feature = "neon", not(feature = "testable_dispatch"))))]
        {
            match NEON_CACHE.load(Ordering::Relaxed) {
                2 => Some(unsafe { Self::forge_token_dangerously() }),
                1 => None,
                _ => {
                    let available = crate::is_aarch64_feature_available!("neon");
                    NEON_CACHE.store(if available { 2 } else { 1 }, Ordering::Relaxed);
                    if available {
                        Some(unsafe { Self::forge_token_dangerously() })
                    } else {
                        None
                    }
                }
            }
        }
    }

    #[inline(always)]
    #[allow(deprecated)]
    unsafe fn forge_token_dangerously() -> Self {
        Self { _private: () }
    }
}

impl NeonToken {
    /// Disable this token process-wide for testing and benchmarking.
    ///
    /// When disabled, `summon()` will return `None` even if the CPU supports
    /// the required features.
    ///
    /// Returns `Err` when all required features are compile-time enabled
    /// (e.g., via `-Ctarget-cpu=native`), since the compiler has already
    /// elided the runtime checks.
    ///
    /// **Cascading:** Also affects descendants:
    /// - `NeonAesToken`
    /// - `NeonSha3Token`
    /// - `NeonCrcToken`
    /// - `Arm64V2Token`
    /// - `Arm64V3Token`
    #[allow(clippy::needless_return)]
    pub fn dangerously_disable_token_process_wide(
        disabled: bool,
    ) -> Result<(), crate::tokens::CompileTimeGuaranteedError> {
        #[cfg(all(target_feature = "neon", not(feature = "testable_dispatch")))]
        {
            let _ = disabled;
            return Err(crate::tokens::CompileTimeGuaranteedError {
                token_name: Self::NAME,
                target_features: Self::TARGET_FEATURES,
                disable_flags: Self::DISABLE_TARGET_FEATURES,
            });
        }
        #[cfg(not(all(target_feature = "neon", not(feature = "testable_dispatch"))))]
        {
            NEON_DISABLED.store(disabled, Ordering::Relaxed);
            let v = if disabled { 1 } else { 0 };
            NEON_CACHE.store(v, Ordering::Relaxed);
            NEON_AES_DISABLED.store(disabled, Ordering::Relaxed);
            NEON_AES_CACHE.store(v, Ordering::Relaxed);
            NEON_SHA3_DISABLED.store(disabled, Ordering::Relaxed);
            NEON_SHA3_CACHE.store(v, Ordering::Relaxed);
            NEON_CRC_DISABLED.store(disabled, Ordering::Relaxed);
            NEON_CRC_CACHE.store(v, Ordering::Relaxed);
            ARM64_V2_DISABLED.store(disabled, Ordering::Relaxed);
            ARM64_V2_CACHE.store(v, Ordering::Relaxed);
            ARM64_V3_DISABLED.store(disabled, Ordering::Relaxed);
            ARM64_V3_CACHE.store(v, Ordering::Relaxed);
            Ok(())
        }
    }

    /// Check if this token has been manually disabled process-wide.
    ///
    /// Returns `Err` when all required features are compile-time enabled.
    #[allow(clippy::needless_return)]
    pub fn manually_disabled() -> Result<bool, crate::tokens::CompileTimeGuaranteedError> {
        #[cfg(all(target_feature = "neon", not(feature = "testable_dispatch")))]
        {
            return Err(crate::tokens::CompileTimeGuaranteedError {
                token_name: Self::NAME,
                target_features: Self::TARGET_FEATURES,
                disable_flags: Self::DISABLE_TARGET_FEATURES,
            });
        }
        #[cfg(not(all(target_feature = "neon", not(feature = "testable_dispatch"))))]
        {
            Ok(NEON_DISABLED.load(Ordering::Relaxed))
        }
    }
}

/// Proof that NEON + AES is available.
///
/// AES extension is common on modern ARM64 devices with crypto extensions.
#[derive(Clone, Copy, Debug)]
pub struct NeonAesToken {
    _private: (),
}

impl crate::tokens::Sealed for NeonAesToken {}

impl SimdToken for NeonAesToken {
    const NAME: &'static str = "NEON+AES";
    const TARGET_FEATURES: &'static str = "neon,aes";
    const ENABLE_TARGET_FEATURES: &'static str = "-Ctarget-feature=+neon,+aes";
    const DISABLE_TARGET_FEATURES: &'static str = "-Ctarget-feature=-neon,-aes";

    #[inline]
    fn compiled_with() -> Option<bool> {
        #[cfg(all(
            target_feature = "neon",
            target_feature = "aes",
            not(feature = "testable_dispatch")
        ))]
        {
            Some(true)
        }
        #[cfg(not(all(
            target_feature = "neon",
            target_feature = "aes",
            not(feature = "testable_dispatch")
        )))]
        {
            None
        }
    }

    #[allow(deprecated)]
    #[inline(always)]
    fn summon() -> Option<Self> {
        // Compile-time fast path (suppressed by testable_dispatch)
        #[cfg(all(
            target_feature = "neon",
            target_feature = "aes",
            not(feature = "testable_dispatch")
        ))]
        {
            return Some(unsafe { Self::forge_token_dangerously() });
        }

        // Runtime path with caching
        #[cfg(not(all(
            target_feature = "neon",
            target_feature = "aes",
            not(feature = "testable_dispatch")
        )))]
        {
            match NEON_AES_CACHE.load(Ordering::Relaxed) {
                2 => Some(unsafe { Self::forge_token_dangerously() }),
                1 => None,
                _ => {
                    let available = crate::is_aarch64_feature_available!("neon")
                        && crate::is_aarch64_feature_available!("aes");
                    NEON_AES_CACHE.store(if available { 2 } else { 1 }, Ordering::Relaxed);
                    if available {
                        Some(unsafe { Self::forge_token_dangerously() })
                    } else {
                        None
                    }
                }
            }
        }
    }

    #[inline(always)]
    #[allow(deprecated)]
    unsafe fn forge_token_dangerously() -> Self {
        Self { _private: () }
    }
}

impl NeonAesToken {
    /// Get a NeonToken (NEON+AES implies NEON)
    #[allow(deprecated)]
    #[inline(always)]
    pub fn neon(self) -> NeonToken {
        unsafe { NeonToken::forge_token_dangerously() }
    }
}

impl NeonAesToken {
    /// Disable this token process-wide for testing and benchmarking.
    ///
    /// When disabled, `summon()` will return `None` even if the CPU supports
    /// the required features.
    ///
    /// Returns `Err` when all required features are compile-time enabled
    /// (e.g., via `-Ctarget-cpu=native`), since the compiler has already
    /// elided the runtime checks.
    #[allow(clippy::needless_return)]
    pub fn dangerously_disable_token_process_wide(
        disabled: bool,
    ) -> Result<(), crate::tokens::CompileTimeGuaranteedError> {
        #[cfg(all(
            target_feature = "neon",
            target_feature = "aes",
            not(feature = "testable_dispatch")
        ))]
        {
            let _ = disabled;
            return Err(crate::tokens::CompileTimeGuaranteedError {
                token_name: Self::NAME,
                target_features: Self::TARGET_FEATURES,
                disable_flags: Self::DISABLE_TARGET_FEATURES,
            });
        }
        #[cfg(not(all(
            target_feature = "neon",
            target_feature = "aes",
            not(feature = "testable_dispatch")
        )))]
        {
            NEON_AES_DISABLED.store(disabled, Ordering::Relaxed);
            let v = if disabled { 1 } else { 0 };
            NEON_AES_CACHE.store(v, Ordering::Relaxed);
            Ok(())
        }
    }

    /// Check if this token has been manually disabled process-wide.
    ///
    /// Returns `Err` when all required features are compile-time enabled.
    #[allow(clippy::needless_return)]
    pub fn manually_disabled() -> Result<bool, crate::tokens::CompileTimeGuaranteedError> {
        #[cfg(all(
            target_feature = "neon",
            target_feature = "aes",
            not(feature = "testable_dispatch")
        ))]
        {
            return Err(crate::tokens::CompileTimeGuaranteedError {
                token_name: Self::NAME,
                target_features: Self::TARGET_FEATURES,
                disable_flags: Self::DISABLE_TARGET_FEATURES,
            });
        }
        #[cfg(not(all(
            target_feature = "neon",
            target_feature = "aes",
            not(feature = "testable_dispatch")
        )))]
        {
            Ok(NEON_AES_DISABLED.load(Ordering::Relaxed))
        }
    }
}

/// Proof that NEON + SHA3 is available.
///
/// SHA3 extension is available on ARMv8.2-A and later.
#[derive(Clone, Copy, Debug)]
pub struct NeonSha3Token {
    _private: (),
}

impl crate::tokens::Sealed for NeonSha3Token {}

impl SimdToken for NeonSha3Token {
    const NAME: &'static str = "NEON+SHA3";
    const TARGET_FEATURES: &'static str = "neon,sha3";
    const ENABLE_TARGET_FEATURES: &'static str = "-Ctarget-feature=+neon,+sha3";
    const DISABLE_TARGET_FEATURES: &'static str = "-Ctarget-feature=-neon,-sha3";

    #[inline]
    fn compiled_with() -> Option<bool> {
        #[cfg(all(
            target_feature = "neon",
            target_feature = "sha3",
            not(feature = "testable_dispatch")
        ))]
        {
            Some(true)
        }
        #[cfg(not(all(
            target_feature = "neon",
            target_feature = "sha3",
            not(feature = "testable_dispatch")
        )))]
        {
            None
        }
    }

    #[allow(deprecated)]
    #[inline(always)]
    fn summon() -> Option<Self> {
        // Compile-time fast path (suppressed by testable_dispatch)
        #[cfg(all(
            target_feature = "neon",
            target_feature = "sha3",
            not(feature = "testable_dispatch")
        ))]
        {
            return Some(unsafe { Self::forge_token_dangerously() });
        }

        // Runtime path with caching
        #[cfg(not(all(
            target_feature = "neon",
            target_feature = "sha3",
            not(feature = "testable_dispatch")
        )))]
        {
            match NEON_SHA3_CACHE.load(Ordering::Relaxed) {
                2 => Some(unsafe { Self::forge_token_dangerously() }),
                1 => None,
                _ => {
                    let available = crate::is_aarch64_feature_available!("neon")
                        && crate::is_aarch64_feature_available!("sha3");
                    NEON_SHA3_CACHE.store(if available { 2 } else { 1 }, Ordering::Relaxed);
                    if available {
                        Some(unsafe { Self::forge_token_dangerously() })
                    } else {
                        None
                    }
                }
            }
        }
    }

    #[inline(always)]
    #[allow(deprecated)]
    unsafe fn forge_token_dangerously() -> Self {
        Self { _private: () }
    }
}

impl NeonSha3Token {
    /// Get a NeonToken (NEON+SHA3 implies NEON)
    #[allow(deprecated)]
    #[inline(always)]
    pub fn neon(self) -> NeonToken {
        unsafe { NeonToken::forge_token_dangerously() }
    }
}

impl NeonSha3Token {
    /// Disable this token process-wide for testing and benchmarking.
    ///
    /// When disabled, `summon()` will return `None` even if the CPU supports
    /// the required features.
    ///
    /// Returns `Err` when all required features are compile-time enabled
    /// (e.g., via `-Ctarget-cpu=native`), since the compiler has already
    /// elided the runtime checks.
    #[allow(clippy::needless_return)]
    pub fn dangerously_disable_token_process_wide(
        disabled: bool,
    ) -> Result<(), crate::tokens::CompileTimeGuaranteedError> {
        #[cfg(all(
            target_feature = "neon",
            target_feature = "sha3",
            not(feature = "testable_dispatch")
        ))]
        {
            let _ = disabled;
            return Err(crate::tokens::CompileTimeGuaranteedError {
                token_name: Self::NAME,
                target_features: Self::TARGET_FEATURES,
                disable_flags: Self::DISABLE_TARGET_FEATURES,
            });
        }
        #[cfg(not(all(
            target_feature = "neon",
            target_feature = "sha3",
            not(feature = "testable_dispatch")
        )))]
        {
            NEON_SHA3_DISABLED.store(disabled, Ordering::Relaxed);
            let v = if disabled { 1 } else { 0 };
            NEON_SHA3_CACHE.store(v, Ordering::Relaxed);
            Ok(())
        }
    }

    /// Check if this token has been manually disabled process-wide.
    ///
    /// Returns `Err` when all required features are compile-time enabled.
    #[allow(clippy::needless_return)]
    pub fn manually_disabled() -> Result<bool, crate::tokens::CompileTimeGuaranteedError> {
        #[cfg(all(
            target_feature = "neon",
            target_feature = "sha3",
            not(feature = "testable_dispatch")
        ))]
        {
            return Err(crate::tokens::CompileTimeGuaranteedError {
                token_name: Self::NAME,
                target_features: Self::TARGET_FEATURES,
                disable_flags: Self::DISABLE_TARGET_FEATURES,
            });
        }
        #[cfg(not(all(
            target_feature = "neon",
            target_feature = "sha3",
            not(feature = "testable_dispatch")
        )))]
        {
            Ok(NEON_SHA3_DISABLED.load(Ordering::Relaxed))
        }
    }
}

/// Proof that NEON + CRC is available.
///
/// CRC32 extension is common on most AArch64 CPUs (part of ARMv8.1-A baseline).
/// Available on: Cortex-A53+, Apple M1+, Graviton 1+, Snapdragon 8xx.
#[derive(Clone, Copy, Debug)]
pub struct NeonCrcToken {
    _private: (),
}

impl crate::tokens::Sealed for NeonCrcToken {}

impl SimdToken for NeonCrcToken {
    const NAME: &'static str = "NEON+CRC";
    const TARGET_FEATURES: &'static str = "neon,crc";
    const ENABLE_TARGET_FEATURES: &'static str = "-Ctarget-feature=+neon,+crc";
    const DISABLE_TARGET_FEATURES: &'static str = "-Ctarget-feature=-neon,-crc";

    #[inline]
    fn compiled_with() -> Option<bool> {
        #[cfg(all(
            target_feature = "neon",
            target_feature = "crc",
            not(feature = "testable_dispatch")
        ))]
        {
            Some(true)
        }
        #[cfg(not(all(
            target_feature = "neon",
            target_feature = "crc",
            not(feature = "testable_dispatch")
        )))]
        {
            None
        }
    }

    #[allow(deprecated)]
    #[inline(always)]
    fn summon() -> Option<Self> {
        // Compile-time fast path (suppressed by testable_dispatch)
        #[cfg(all(
            target_feature = "neon",
            target_feature = "crc",
            not(feature = "testable_dispatch")
        ))]
        {
            return Some(unsafe { Self::forge_token_dangerously() });
        }

        // Runtime path with caching
        #[cfg(not(all(
            target_feature = "neon",
            target_feature = "crc",
            not(feature = "testable_dispatch")
        )))]
        {
            match NEON_CRC_CACHE.load(Ordering::Relaxed) {
                2 => Some(unsafe { Self::forge_token_dangerously() }),
                1 => None,
                _ => {
                    let available = crate::is_aarch64_feature_available!("neon")
                        && crate::is_aarch64_feature_available!("crc");
                    NEON_CRC_CACHE.store(if available { 2 } else { 1 }, Ordering::Relaxed);
                    if available {
                        Some(unsafe { Self::forge_token_dangerously() })
                    } else {
                        None
                    }
                }
            }
        }
    }

    #[inline(always)]
    #[allow(deprecated)]
    unsafe fn forge_token_dangerously() -> Self {
        Self { _private: () }
    }
}

impl NeonCrcToken {
    /// Get a NeonToken (NEON+CRC implies NEON)
    #[allow(deprecated)]
    #[inline(always)]
    pub fn neon(self) -> NeonToken {
        unsafe { NeonToken::forge_token_dangerously() }
    }
}

impl NeonCrcToken {
    /// Disable this token process-wide for testing and benchmarking.
    ///
    /// When disabled, `summon()` will return `None` even if the CPU supports
    /// the required features.
    ///
    /// Returns `Err` when all required features are compile-time enabled
    /// (e.g., via `-Ctarget-cpu=native`), since the compiler has already
    /// elided the runtime checks.
    #[allow(clippy::needless_return)]
    pub fn dangerously_disable_token_process_wide(
        disabled: bool,
    ) -> Result<(), crate::tokens::CompileTimeGuaranteedError> {
        #[cfg(all(
            target_feature = "neon",
            target_feature = "crc",
            not(feature = "testable_dispatch")
        ))]
        {
            let _ = disabled;
            return Err(crate::tokens::CompileTimeGuaranteedError {
                token_name: Self::NAME,
                target_features: Self::TARGET_FEATURES,
                disable_flags: Self::DISABLE_TARGET_FEATURES,
            });
        }
        #[cfg(not(all(
            target_feature = "neon",
            target_feature = "crc",
            not(feature = "testable_dispatch")
        )))]
        {
            NEON_CRC_DISABLED.store(disabled, Ordering::Relaxed);
            let v = if disabled { 1 } else { 0 };
            NEON_CRC_CACHE.store(v, Ordering::Relaxed);
            Ok(())
        }
    }

    /// Check if this token has been manually disabled process-wide.
    ///
    /// Returns `Err` when all required features are compile-time enabled.
    #[allow(clippy::needless_return)]
    pub fn manually_disabled() -> Result<bool, crate::tokens::CompileTimeGuaranteedError> {
        #[cfg(all(
            target_feature = "neon",
            target_feature = "crc",
            not(feature = "testable_dispatch")
        ))]
        {
            return Err(crate::tokens::CompileTimeGuaranteedError {
                token_name: Self::NAME,
                target_features: Self::TARGET_FEATURES,
                disable_flags: Self::DISABLE_TARGET_FEATURES,
            });
        }
        #[cfg(not(all(
            target_feature = "neon",
            target_feature = "crc",
            not(feature = "testable_dispatch")
        )))]
        {
            Ok(NEON_CRC_DISABLED.load(Ordering::Relaxed))
        }
    }
}

/// Proof that the Arm64-v2 feature set is available.
///
/// Arm64-v2 is archmage's second ARM tier, covering: NEON, CRC, RDM, DotProd,
/// FP16, AES, SHA2. This targets the broadest modern ARM baseline.
///
/// Available on: Cortex-A55+, Apple M1+, Graviton 2+, all post-2017 ARM chips.
#[derive(Clone, Copy, Debug)]
pub struct Arm64V2Token {
    _private: (),
}

impl crate::tokens::Sealed for Arm64V2Token {}

impl SimdToken for Arm64V2Token {
    const NAME: &'static str = "Arm64-v2";
    const TARGET_FEATURES: &'static str = "neon,crc,rdm,dotprod,fp16,aes,sha2";
    const ENABLE_TARGET_FEATURES: &'static str =
        "-Ctarget-feature=+neon,+crc,+rdm,+dotprod,+fp16,+aes,+sha2";
    const DISABLE_TARGET_FEATURES: &'static str =
        "-Ctarget-feature=-neon,-crc,-rdm,-dotprod,-fp16,-aes,-sha2";

    #[inline]
    fn compiled_with() -> Option<bool> {
        #[cfg(all(
            target_feature = "neon",
            target_feature = "crc",
            target_feature = "rdm",
            target_feature = "dotprod",
            target_feature = "fp16",
            target_feature = "aes",
            target_feature = "sha2",
            not(feature = "testable_dispatch")
        ))]
        {
            Some(true)
        }
        #[cfg(not(all(
            target_feature = "neon",
            target_feature = "crc",
            target_feature = "rdm",
            target_feature = "dotprod",
            target_feature = "fp16",
            target_feature = "aes",
            target_feature = "sha2",
            not(feature = "testable_dispatch")
        )))]
        {
            None
        }
    }

    #[allow(deprecated)]
    #[inline(always)]
    fn summon() -> Option<Self> {
        // Compile-time fast path (suppressed by testable_dispatch)
        #[cfg(all(
            target_feature = "neon",
            target_feature = "crc",
            target_feature = "rdm",
            target_feature = "dotprod",
            target_feature = "fp16",
            target_feature = "aes",
            target_feature = "sha2",
            not(feature = "testable_dispatch")
        ))]
        {
            return Some(unsafe { Self::forge_token_dangerously() });
        }

        // Runtime path with caching
        #[cfg(not(all(
            target_feature = "neon",
            target_feature = "crc",
            target_feature = "rdm",
            target_feature = "dotprod",
            target_feature = "fp16",
            target_feature = "aes",
            target_feature = "sha2",
            not(feature = "testable_dispatch")
        )))]
        {
            match ARM64_V2_CACHE.load(Ordering::Relaxed) {
                2 => Some(unsafe { Self::forge_token_dangerously() }),
                1 => None,
                _ => {
                    let available = crate::is_aarch64_feature_available!("neon")
                        && crate::is_aarch64_feature_available!("crc")
                        && crate::is_aarch64_feature_available!("rdm")
                        && crate::is_aarch64_feature_available!("dotprod")
                        && crate::is_aarch64_feature_available!("fp16")
                        && crate::is_aarch64_feature_available!("aes")
                        && crate::is_aarch64_feature_available!("sha2");
                    ARM64_V2_CACHE.store(if available { 2 } else { 1 }, Ordering::Relaxed);
                    if available {
                        Some(unsafe { Self::forge_token_dangerously() })
                    } else {
                        None
                    }
                }
            }
        }
    }

    #[inline(always)]
    #[allow(deprecated)]
    unsafe fn forge_token_dangerously() -> Self {
        Self { _private: () }
    }
}

impl Arm64V2Token {
    /// Get a NeonToken (Arm64-v2 implies NEON)
    #[allow(deprecated)]
    #[inline(always)]
    pub fn neon(self) -> NeonToken {
        unsafe { NeonToken::forge_token_dangerously() }
    }
}

impl Arm64V2Token {
    /// Disable this token process-wide for testing and benchmarking.
    ///
    /// When disabled, `summon()` will return `None` even if the CPU supports
    /// the required features.
    ///
    /// Returns `Err` when all required features are compile-time enabled
    /// (e.g., via `-Ctarget-cpu=native`), since the compiler has already
    /// elided the runtime checks.
    ///
    /// **Cascading:** Also affects descendants:
    /// - `Arm64V3Token`
    #[allow(clippy::needless_return)]
    pub fn dangerously_disable_token_process_wide(
        disabled: bool,
    ) -> Result<(), crate::tokens::CompileTimeGuaranteedError> {
        #[cfg(all(
            target_feature = "neon",
            target_feature = "crc",
            target_feature = "rdm",
            target_feature = "dotprod",
            target_feature = "fp16",
            target_feature = "aes",
            target_feature = "sha2",
            not(feature = "testable_dispatch")
        ))]
        {
            let _ = disabled;
            return Err(crate::tokens::CompileTimeGuaranteedError {
                token_name: Self::NAME,
                target_features: Self::TARGET_FEATURES,
                disable_flags: Self::DISABLE_TARGET_FEATURES,
            });
        }
        #[cfg(not(all(
            target_feature = "neon",
            target_feature = "crc",
            target_feature = "rdm",
            target_feature = "dotprod",
            target_feature = "fp16",
            target_feature = "aes",
            target_feature = "sha2",
            not(feature = "testable_dispatch")
        )))]
        {
            ARM64_V2_DISABLED.store(disabled, Ordering::Relaxed);
            let v = if disabled { 1 } else { 0 };
            ARM64_V2_CACHE.store(v, Ordering::Relaxed);
            ARM64_V3_DISABLED.store(disabled, Ordering::Relaxed);
            ARM64_V3_CACHE.store(v, Ordering::Relaxed);
            Ok(())
        }
    }

    /// Check if this token has been manually disabled process-wide.
    ///
    /// Returns `Err` when all required features are compile-time enabled.
    #[allow(clippy::needless_return)]
    pub fn manually_disabled() -> Result<bool, crate::tokens::CompileTimeGuaranteedError> {
        #[cfg(all(
            target_feature = "neon",
            target_feature = "crc",
            target_feature = "rdm",
            target_feature = "dotprod",
            target_feature = "fp16",
            target_feature = "aes",
            target_feature = "sha2",
            not(feature = "testable_dispatch")
        ))]
        {
            return Err(crate::tokens::CompileTimeGuaranteedError {
                token_name: Self::NAME,
                target_features: Self::TARGET_FEATURES,
                disable_flags: Self::DISABLE_TARGET_FEATURES,
            });
        }
        #[cfg(not(all(
            target_feature = "neon",
            target_feature = "crc",
            target_feature = "rdm",
            target_feature = "dotprod",
            target_feature = "fp16",
            target_feature = "aes",
            target_feature = "sha2",
            not(feature = "testable_dispatch")
        )))]
        {
            Ok(ARM64_V2_DISABLED.load(Ordering::Relaxed))
        }
    }
}

/// Proof that the full modern ARM SIMD feature set is available (Arm64-v3).
///
/// Arm64-v3 adds FHM, FCMA, SHA3, I8MM, and BF16 over Arm64-v2.
/// Available on: Cortex-A510+, Apple M2+, Snapdragon X, Graviton 3+, Cobalt 100.
#[derive(Clone, Copy, Debug)]
pub struct Arm64V3Token {
    _private: (),
}

impl crate::tokens::Sealed for Arm64V3Token {}

impl SimdToken for Arm64V3Token {
    const NAME: &'static str = "Arm64-v3";
    const TARGET_FEATURES: &'static str =
        "neon,crc,rdm,dotprod,fp16,aes,sha2,fhm,fcma,sha3,i8mm,bf16";
    const ENABLE_TARGET_FEATURES: &'static str =
        "-Ctarget-feature=+neon,+crc,+rdm,+dotprod,+fp16,+aes,+sha2,+fhm,+fcma,+sha3,+i8mm,+bf16";
    const DISABLE_TARGET_FEATURES: &'static str =
        "-Ctarget-feature=-neon,-crc,-rdm,-dotprod,-fp16,-aes,-sha2,-fhm,-fcma,-sha3,-i8mm,-bf16";

    #[inline]
    fn compiled_with() -> Option<bool> {
        #[cfg(all(
            target_feature = "neon",
            target_feature = "crc",
            target_feature = "rdm",
            target_feature = "dotprod",
            target_feature = "fp16",
            target_feature = "aes",
            target_feature = "sha2",
            target_feature = "fhm",
            target_feature = "fcma",
            target_feature = "sha3",
            target_feature = "i8mm",
            target_feature = "bf16",
            not(feature = "testable_dispatch")
        ))]
        {
            Some(true)
        }
        #[cfg(not(all(
            target_feature = "neon",
            target_feature = "crc",
            target_feature = "rdm",
            target_feature = "dotprod",
            target_feature = "fp16",
            target_feature = "aes",
            target_feature = "sha2",
            target_feature = "fhm",
            target_feature = "fcma",
            target_feature = "sha3",
            target_feature = "i8mm",
            target_feature = "bf16",
            not(feature = "testable_dispatch")
        )))]
        {
            None
        }
    }

    #[allow(deprecated)]
    #[inline(always)]
    fn summon() -> Option<Self> {
        // Compile-time fast path (suppressed by testable_dispatch)
        #[cfg(all(
            target_feature = "neon",
            target_feature = "crc",
            target_feature = "rdm",
            target_feature = "dotprod",
            target_feature = "fp16",
            target_feature = "aes",
            target_feature = "sha2",
            target_feature = "fhm",
            target_feature = "fcma",
            target_feature = "sha3",
            target_feature = "i8mm",
            target_feature = "bf16",
            not(feature = "testable_dispatch")
        ))]
        {
            return Some(unsafe { Self::forge_token_dangerously() });
        }

        // Runtime path with caching
        #[cfg(not(all(
            target_feature = "neon",
            target_feature = "crc",
            target_feature = "rdm",
            target_feature = "dotprod",
            target_feature = "fp16",
            target_feature = "aes",
            target_feature = "sha2",
            target_feature = "fhm",
            target_feature = "fcma",
            target_feature = "sha3",
            target_feature = "i8mm",
            target_feature = "bf16",
            not(feature = "testable_dispatch")
        )))]
        {
            match ARM64_V3_CACHE.load(Ordering::Relaxed) {
                2 => Some(unsafe { Self::forge_token_dangerously() }),
                1 => None,
                _ => {
                    let available = crate::is_aarch64_feature_available!("neon")
                        && crate::is_aarch64_feature_available!("crc")
                        && crate::is_aarch64_feature_available!("rdm")
                        && crate::is_aarch64_feature_available!("dotprod")
                        && crate::is_aarch64_feature_available!("fp16")
                        && crate::is_aarch64_feature_available!("aes")
                        && crate::is_aarch64_feature_available!("sha2")
                        && crate::is_aarch64_feature_available!("fhm")
                        && crate::is_aarch64_feature_available!("fcma")
                        && crate::is_aarch64_feature_available!("sha3")
                        && crate::is_aarch64_feature_available!("i8mm")
                        && crate::is_aarch64_feature_available!("bf16");
                    ARM64_V3_CACHE.store(if available { 2 } else { 1 }, Ordering::Relaxed);
                    if available {
                        Some(unsafe { Self::forge_token_dangerously() })
                    } else {
                        None
                    }
                }
            }
        }
    }

    #[inline(always)]
    #[allow(deprecated)]
    unsafe fn forge_token_dangerously() -> Self {
        Self { _private: () }
    }
}

impl Arm64V3Token {
    /// Get a Arm64V2Token (Arm64-v3 implies Arm64-v2)
    #[allow(deprecated)]
    #[inline(always)]
    pub fn arm_v2(self) -> Arm64V2Token {
        unsafe { Arm64V2Token::forge_token_dangerously() }
    }
    /// Get a NeonToken (Arm64-v3 implies NEON)
    #[allow(deprecated)]
    #[inline(always)]
    pub fn neon(self) -> NeonToken {
        unsafe { NeonToken::forge_token_dangerously() }
    }
}

impl Arm64V3Token {
    /// Disable this token process-wide for testing and benchmarking.
    ///
    /// When disabled, `summon()` will return `None` even if the CPU supports
    /// the required features.
    ///
    /// Returns `Err` when all required features are compile-time enabled
    /// (e.g., via `-Ctarget-cpu=native`), since the compiler has already
    /// elided the runtime checks.
    #[allow(clippy::needless_return)]
    pub fn dangerously_disable_token_process_wide(
        disabled: bool,
    ) -> Result<(), crate::tokens::CompileTimeGuaranteedError> {
        #[cfg(all(
            target_feature = "neon",
            target_feature = "crc",
            target_feature = "rdm",
            target_feature = "dotprod",
            target_feature = "fp16",
            target_feature = "aes",
            target_feature = "sha2",
            target_feature = "fhm",
            target_feature = "fcma",
            target_feature = "sha3",
            target_feature = "i8mm",
            target_feature = "bf16",
            not(feature = "testable_dispatch")
        ))]
        {
            let _ = disabled;
            return Err(crate::tokens::CompileTimeGuaranteedError {
                token_name: Self::NAME,
                target_features: Self::TARGET_FEATURES,
                disable_flags: Self::DISABLE_TARGET_FEATURES,
            });
        }
        #[cfg(not(all(
            target_feature = "neon",
            target_feature = "crc",
            target_feature = "rdm",
            target_feature = "dotprod",
            target_feature = "fp16",
            target_feature = "aes",
            target_feature = "sha2",
            target_feature = "fhm",
            target_feature = "fcma",
            target_feature = "sha3",
            target_feature = "i8mm",
            target_feature = "bf16",
            not(feature = "testable_dispatch")
        )))]
        {
            ARM64_V3_DISABLED.store(disabled, Ordering::Relaxed);
            let v = if disabled { 1 } else { 0 };
            ARM64_V3_CACHE.store(v, Ordering::Relaxed);
            Ok(())
        }
    }

    /// Check if this token has been manually disabled process-wide.
    ///
    /// Returns `Err` when all required features are compile-time enabled.
    #[allow(clippy::needless_return)]
    pub fn manually_disabled() -> Result<bool, crate::tokens::CompileTimeGuaranteedError> {
        #[cfg(all(
            target_feature = "neon",
            target_feature = "crc",
            target_feature = "rdm",
            target_feature = "dotprod",
            target_feature = "fp16",
            target_feature = "aes",
            target_feature = "sha2",
            target_feature = "fhm",
            target_feature = "fcma",
            target_feature = "sha3",
            target_feature = "i8mm",
            target_feature = "bf16",
            not(feature = "testable_dispatch")
        ))]
        {
            return Err(crate::tokens::CompileTimeGuaranteedError {
                token_name: Self::NAME,
                target_features: Self::TARGET_FEATURES,
                disable_flags: Self::DISABLE_TARGET_FEATURES,
            });
        }
        #[cfg(not(all(
            target_feature = "neon",
            target_feature = "crc",
            target_feature = "rdm",
            target_feature = "dotprod",
            target_feature = "fp16",
            target_feature = "aes",
            target_feature = "sha2",
            target_feature = "fhm",
            target_feature = "fcma",
            target_feature = "sha3",
            target_feature = "i8mm",
            target_feature = "bf16",
            not(feature = "testable_dispatch")
        )))]
        {
            Ok(ARM64_V3_DISABLED.load(Ordering::Relaxed))
        }
    }
}

/// Type alias for [`NeonToken`].
pub type Arm64 = NeonToken;

impl Has128BitSimd for NeonToken {}
impl Has128BitSimd for NeonAesToken {}
impl Has128BitSimd for NeonSha3Token {}
impl Has128BitSimd for NeonCrcToken {}
impl Has128BitSimd for Arm64V2Token {}
impl Has128BitSimd for Arm64V3Token {}
impl HasArm64V2 for Arm64V2Token {}
impl HasArm64V2 for Arm64V3Token {}
impl HasArm64V3 for Arm64V3Token {}
impl HasNeon for NeonToken {}
impl HasNeon for NeonAesToken {}
impl HasNeon for NeonSha3Token {}
impl HasNeon for NeonCrcToken {}
impl HasNeon for Arm64V2Token {}
impl HasNeon for Arm64V3Token {}
impl HasNeonAes for NeonAesToken {}
impl HasNeonAes for Arm64V2Token {}
impl HasNeonAes for Arm64V3Token {}
impl HasNeonSha3 for NeonSha3Token {}
impl HasNeonSha3 for Arm64V3Token {}
