//! Generated from token-registry.toml — DO NOT EDIT.
//!
//! Regenerate with: cargo xtask generate

use crate::tokens::SimdToken;
use crate::tokens::{Has128BitSimd, HasNeon, HasNeonAes, HasNeonSha3};
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

    #[inline]
    fn compiled_with() -> Option<bool> {
        #[cfg(all(target_feature = "neon"))]
        {
            Some(true)
        }
        #[cfg(not(all(target_feature = "neon")))]
        {
            None
        }
    }

    #[allow(deprecated)]
    #[inline(always)]
    fn summon() -> Option<Self> {
        // Compile-time fast path
        #[cfg(all(target_feature = "neon"))]
        {
            return Some(unsafe { Self::forge_token_dangerously() });
        }

        // Runtime path with caching
        #[cfg(not(all(target_feature = "neon")))]
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
    pub fn dangerously_disable_token_process_wide(
        disabled: bool,
    ) -> Result<(), crate::tokens::CompileTimeGuaranteedError> {
        #[cfg(all(target_feature = "neon"))]
        {
            let _ = disabled;
            return Err(crate::tokens::CompileTimeGuaranteedError {
                token_name: Self::NAME,
            });
        }
        #[cfg(not(all(target_feature = "neon")))]
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
            Ok(())
        }
    }

    /// Check if this token has been manually disabled process-wide.
    ///
    /// Returns `Err` when all required features are compile-time enabled.
    pub fn manually_disabled() -> Result<bool, crate::tokens::CompileTimeGuaranteedError> {
        #[cfg(all(target_feature = "neon"))]
        {
            return Err(crate::tokens::CompileTimeGuaranteedError {
                token_name: Self::NAME,
            });
        }
        #[cfg(not(all(target_feature = "neon")))]
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

    #[inline]
    fn compiled_with() -> Option<bool> {
        #[cfg(all(target_feature = "neon", target_feature = "aes"))]
        {
            Some(true)
        }
        #[cfg(not(all(target_feature = "neon", target_feature = "aes")))]
        {
            None
        }
    }

    #[allow(deprecated)]
    #[inline(always)]
    fn summon() -> Option<Self> {
        // Compile-time fast path
        #[cfg(all(target_feature = "neon", target_feature = "aes"))]
        {
            return Some(unsafe { Self::forge_token_dangerously() });
        }

        // Runtime path with caching
        #[cfg(not(all(target_feature = "neon", target_feature = "aes")))]
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
    pub fn dangerously_disable_token_process_wide(
        disabled: bool,
    ) -> Result<(), crate::tokens::CompileTimeGuaranteedError> {
        #[cfg(all(target_feature = "neon", target_feature = "aes"))]
        {
            let _ = disabled;
            return Err(crate::tokens::CompileTimeGuaranteedError {
                token_name: Self::NAME,
            });
        }
        #[cfg(not(all(target_feature = "neon", target_feature = "aes")))]
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
    pub fn manually_disabled() -> Result<bool, crate::tokens::CompileTimeGuaranteedError> {
        #[cfg(all(target_feature = "neon", target_feature = "aes"))]
        {
            return Err(crate::tokens::CompileTimeGuaranteedError {
                token_name: Self::NAME,
            });
        }
        #[cfg(not(all(target_feature = "neon", target_feature = "aes")))]
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

    #[inline]
    fn compiled_with() -> Option<bool> {
        #[cfg(all(target_feature = "neon", target_feature = "sha3"))]
        {
            Some(true)
        }
        #[cfg(not(all(target_feature = "neon", target_feature = "sha3")))]
        {
            None
        }
    }

    #[allow(deprecated)]
    #[inline(always)]
    fn summon() -> Option<Self> {
        // Compile-time fast path
        #[cfg(all(target_feature = "neon", target_feature = "sha3"))]
        {
            return Some(unsafe { Self::forge_token_dangerously() });
        }

        // Runtime path with caching
        #[cfg(not(all(target_feature = "neon", target_feature = "sha3")))]
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
    pub fn dangerously_disable_token_process_wide(
        disabled: bool,
    ) -> Result<(), crate::tokens::CompileTimeGuaranteedError> {
        #[cfg(all(target_feature = "neon", target_feature = "sha3"))]
        {
            let _ = disabled;
            return Err(crate::tokens::CompileTimeGuaranteedError {
                token_name: Self::NAME,
            });
        }
        #[cfg(not(all(target_feature = "neon", target_feature = "sha3")))]
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
    pub fn manually_disabled() -> Result<bool, crate::tokens::CompileTimeGuaranteedError> {
        #[cfg(all(target_feature = "neon", target_feature = "sha3"))]
        {
            return Err(crate::tokens::CompileTimeGuaranteedError {
                token_name: Self::NAME,
            });
        }
        #[cfg(not(all(target_feature = "neon", target_feature = "sha3")))]
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

    #[inline]
    fn compiled_with() -> Option<bool> {
        #[cfg(all(target_feature = "neon", target_feature = "crc"))]
        {
            Some(true)
        }
        #[cfg(not(all(target_feature = "neon", target_feature = "crc")))]
        {
            None
        }
    }

    #[allow(deprecated)]
    #[inline(always)]
    fn summon() -> Option<Self> {
        // Compile-time fast path
        #[cfg(all(target_feature = "neon", target_feature = "crc"))]
        {
            return Some(unsafe { Self::forge_token_dangerously() });
        }

        // Runtime path with caching
        #[cfg(not(all(target_feature = "neon", target_feature = "crc")))]
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
    pub fn dangerously_disable_token_process_wide(
        disabled: bool,
    ) -> Result<(), crate::tokens::CompileTimeGuaranteedError> {
        #[cfg(all(target_feature = "neon", target_feature = "crc"))]
        {
            let _ = disabled;
            return Err(crate::tokens::CompileTimeGuaranteedError {
                token_name: Self::NAME,
            });
        }
        #[cfg(not(all(target_feature = "neon", target_feature = "crc")))]
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
    pub fn manually_disabled() -> Result<bool, crate::tokens::CompileTimeGuaranteedError> {
        #[cfg(all(target_feature = "neon", target_feature = "crc"))]
        {
            return Err(crate::tokens::CompileTimeGuaranteedError {
                token_name: Self::NAME,
            });
        }
        #[cfg(not(all(target_feature = "neon", target_feature = "crc")))]
        {
            Ok(NEON_CRC_DISABLED.load(Ordering::Relaxed))
        }
    }
}

/// Type alias for [`NeonToken`].
pub type Arm64 = NeonToken;

impl Has128BitSimd for NeonToken {}
impl Has128BitSimd for NeonAesToken {}
impl Has128BitSimd for NeonSha3Token {}
impl Has128BitSimd for NeonCrcToken {}
impl HasNeon for NeonToken {}
impl HasNeon for NeonAesToken {}
impl HasNeon for NeonSha3Token {}
impl HasNeon for NeonCrcToken {}
impl HasNeonAes for NeonAesToken {}
impl HasNeonSha3 for NeonSha3Token {}
