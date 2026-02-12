//! Generated from token-registry.toml — DO NOT EDIT.
//!
//! Regenerate with: cargo xtask generate

use crate::tokens::SimdToken;
use crate::tokens::{Has128BitSimd, Has256BitSimd, HasX64V2};
use core::sync::atomic::{AtomicBool, AtomicU8, Ordering};

// Cache statics: 0 = unknown, 1 = unavailable, 2 = available
pub(super) static X64_V2_CACHE: AtomicU8 = AtomicU8::new(0);
pub(super) static X64_V2_DISABLED: AtomicBool = AtomicBool::new(false);
pub(super) static X64_V3_CACHE: AtomicU8 = AtomicU8::new(0);
pub(super) static X64_V3_DISABLED: AtomicBool = AtomicBool::new(false);

/// Proof that SSE4.2 + POPCNT are available (x86-64-v2 level).
///
/// x86-64-v2 implies: SSE, SSE2, SSE3, SSSE3, SSE4.1, SSE4.2, POPCNT, CX16, SAHF.
/// This is the Nehalem (2008) / Bulldozer (2011) baseline.
#[derive(Clone, Copy, Debug)]
pub struct X64V2Token {
    _private: (),
}

impl crate::tokens::Sealed for X64V2Token {}

impl SimdToken for X64V2Token {
    const NAME: &'static str = "x86-64-v2";
    const TARGET_FEATURES: &'static str = "sse3,ssse3,sse4.1,sse4.2,popcnt";
    const ENABLE_TARGET_FEATURES: &'static str =
        "-Ctarget-feature=+sse3,+ssse3,+sse4.1,+sse4.2,+popcnt";
    const DISABLE_TARGET_FEATURES: &'static str =
        "-Ctarget-feature=-sse3,-ssse3,-sse4.1,-sse4.2,-popcnt";

    #[inline]
    fn compiled_with() -> Option<bool> {
        #[cfg(all(
            target_feature = "sse3",
            target_feature = "ssse3",
            target_feature = "sse4.1",
            target_feature = "sse4.2",
            target_feature = "popcnt",
            not(feature = "disable_compile_time_tokens")
        ))]
        {
            Some(true)
        }
        #[cfg(not(all(
            target_feature = "sse3",
            target_feature = "ssse3",
            target_feature = "sse4.1",
            target_feature = "sse4.2",
            target_feature = "popcnt",
            not(feature = "disable_compile_time_tokens")
        )))]
        {
            None
        }
    }

    #[allow(deprecated)]
    #[inline(always)]
    fn summon() -> Option<Self> {
        // Compile-time fast path (suppressed by disable_compile_time_tokens)
        #[cfg(all(
            target_feature = "sse3",
            target_feature = "ssse3",
            target_feature = "sse4.1",
            target_feature = "sse4.2",
            target_feature = "popcnt",
            not(feature = "disable_compile_time_tokens")
        ))]
        {
            return Some(unsafe { Self::forge_token_dangerously() });
        }

        // Runtime path with caching
        #[cfg(not(all(
            target_feature = "sse3",
            target_feature = "ssse3",
            target_feature = "sse4.1",
            target_feature = "sse4.2",
            target_feature = "popcnt",
            not(feature = "disable_compile_time_tokens")
        )))]
        {
            match X64_V2_CACHE.load(Ordering::Relaxed) {
                2 => Some(unsafe { Self::forge_token_dangerously() }),
                1 => None,
                _ => {
                    let available = crate::is_x86_feature_available!("sse3")
                        && crate::is_x86_feature_available!("ssse3")
                        && crate::is_x86_feature_available!("sse4.1")
                        && crate::is_x86_feature_available!("sse4.2")
                        && crate::is_x86_feature_available!("popcnt");
                    X64_V2_CACHE.store(if available { 2 } else { 1 }, Ordering::Relaxed);
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

impl X64V2Token {
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
    /// - `X64V3Token`
    /// - `X64V4Token`
    /// - `Avx512ModernToken`
    /// - `Avx512Fp16Token`
    pub fn dangerously_disable_token_process_wide(
        disabled: bool,
    ) -> Result<(), crate::tokens::CompileTimeGuaranteedError> {
        #[cfg(all(
            target_feature = "sse",
            target_feature = "sse2",
            target_feature = "sse3",
            target_feature = "ssse3",
            target_feature = "sse4.1",
            target_feature = "sse4.2",
            target_feature = "popcnt",
            not(feature = "disable_compile_time_tokens")
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
            target_feature = "sse",
            target_feature = "sse2",
            target_feature = "sse3",
            target_feature = "ssse3",
            target_feature = "sse4.1",
            target_feature = "sse4.2",
            target_feature = "popcnt",
            not(feature = "disable_compile_time_tokens")
        )))]
        {
            X64_V2_DISABLED.store(disabled, Ordering::Relaxed);
            let v = if disabled { 1 } else { 0 };
            X64_V2_CACHE.store(v, Ordering::Relaxed);
            X64_V3_DISABLED.store(disabled, Ordering::Relaxed);
            X64_V3_CACHE.store(v, Ordering::Relaxed);
            #[cfg(feature = "avx512")]
            {
                super::x86_avx512::X64_V4_DISABLED.store(disabled, Ordering::Relaxed);
                super::x86_avx512::X64_V4_CACHE.store(v, Ordering::Relaxed);
            }
            #[cfg(feature = "avx512")]
            {
                super::x86_avx512::AVX512_MODERN_DISABLED.store(disabled, Ordering::Relaxed);
                super::x86_avx512::AVX512_MODERN_CACHE.store(v, Ordering::Relaxed);
            }
            #[cfg(feature = "avx512")]
            {
                super::x86_avx512::AVX512_FP16_DISABLED.store(disabled, Ordering::Relaxed);
                super::x86_avx512::AVX512_FP16_CACHE.store(v, Ordering::Relaxed);
            }
            Ok(())
        }
    }

    /// Check if this token has been manually disabled process-wide.
    ///
    /// Returns `Err` when all required features are compile-time enabled.
    pub fn manually_disabled() -> Result<bool, crate::tokens::CompileTimeGuaranteedError> {
        #[cfg(all(
            target_feature = "sse",
            target_feature = "sse2",
            target_feature = "sse3",
            target_feature = "ssse3",
            target_feature = "sse4.1",
            target_feature = "sse4.2",
            target_feature = "popcnt",
            not(feature = "disable_compile_time_tokens")
        ))]
        {
            return Err(crate::tokens::CompileTimeGuaranteedError {
                token_name: Self::NAME,
                target_features: Self::TARGET_FEATURES,
                disable_flags: Self::DISABLE_TARGET_FEATURES,
            });
        }
        #[cfg(not(all(
            target_feature = "sse",
            target_feature = "sse2",
            target_feature = "sse3",
            target_feature = "ssse3",
            target_feature = "sse4.1",
            target_feature = "sse4.2",
            target_feature = "popcnt",
            not(feature = "disable_compile_time_tokens")
        )))]
        {
            Ok(X64_V2_DISABLED.load(Ordering::Relaxed))
        }
    }
}

/// Proof that AVX2 + FMA + BMI1/2 + F16C + LZCNT are available (x86-64-v3 level).
///
/// x86-64-v3 implies all of v2 plus: AVX, AVX2, FMA, BMI1, BMI2, F16C, LZCNT, MOVBE.
/// This is the Haswell (2013) / Zen 1 (2017) baseline.
///
/// This is the most commonly targeted level for high-performance SIMD code.
#[derive(Clone, Copy, Debug)]
pub struct X64V3Token {
    _private: (),
}

impl crate::tokens::Sealed for X64V3Token {}

impl SimdToken for X64V3Token {
    const NAME: &'static str = "x86-64-v3";
    const TARGET_FEATURES: &'static str =
        "sse3,ssse3,sse4.1,sse4.2,popcnt,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt";
    const ENABLE_TARGET_FEATURES: &'static str = "-Ctarget-feature=+sse3,+ssse3,+sse4.1,+sse4.2,+popcnt,+avx,+avx2,+fma,+bmi1,+bmi2,+f16c,+lzcnt";
    const DISABLE_TARGET_FEATURES: &'static str = "-Ctarget-feature=-sse3,-ssse3,-sse4.1,-sse4.2,-popcnt,-avx,-avx2,-fma,-bmi1,-bmi2,-f16c,-lzcnt";

    #[inline]
    fn compiled_with() -> Option<bool> {
        #[cfg(all(
            target_feature = "sse3",
            target_feature = "ssse3",
            target_feature = "sse4.1",
            target_feature = "sse4.2",
            target_feature = "popcnt",
            target_feature = "avx",
            target_feature = "avx2",
            target_feature = "fma",
            target_feature = "bmi1",
            target_feature = "bmi2",
            target_feature = "f16c",
            target_feature = "lzcnt",
            not(feature = "disable_compile_time_tokens")
        ))]
        {
            Some(true)
        }
        #[cfg(not(all(
            target_feature = "sse3",
            target_feature = "ssse3",
            target_feature = "sse4.1",
            target_feature = "sse4.2",
            target_feature = "popcnt",
            target_feature = "avx",
            target_feature = "avx2",
            target_feature = "fma",
            target_feature = "bmi1",
            target_feature = "bmi2",
            target_feature = "f16c",
            target_feature = "lzcnt",
            not(feature = "disable_compile_time_tokens")
        )))]
        {
            None
        }
    }

    #[allow(deprecated)]
    #[inline(always)]
    fn summon() -> Option<Self> {
        // Compile-time fast path (suppressed by disable_compile_time_tokens)
        #[cfg(all(
            target_feature = "sse3",
            target_feature = "ssse3",
            target_feature = "sse4.1",
            target_feature = "sse4.2",
            target_feature = "popcnt",
            target_feature = "avx",
            target_feature = "avx2",
            target_feature = "fma",
            target_feature = "bmi1",
            target_feature = "bmi2",
            target_feature = "f16c",
            target_feature = "lzcnt",
            not(feature = "disable_compile_time_tokens")
        ))]
        {
            return Some(unsafe { Self::forge_token_dangerously() });
        }

        // Runtime path with caching
        #[cfg(not(all(
            target_feature = "sse3",
            target_feature = "ssse3",
            target_feature = "sse4.1",
            target_feature = "sse4.2",
            target_feature = "popcnt",
            target_feature = "avx",
            target_feature = "avx2",
            target_feature = "fma",
            target_feature = "bmi1",
            target_feature = "bmi2",
            target_feature = "f16c",
            target_feature = "lzcnt",
            not(feature = "disable_compile_time_tokens")
        )))]
        {
            match X64_V3_CACHE.load(Ordering::Relaxed) {
                2 => Some(unsafe { Self::forge_token_dangerously() }),
                1 => None,
                _ => {
                    let available = crate::is_x86_feature_available!("sse3")
                        && crate::is_x86_feature_available!("ssse3")
                        && crate::is_x86_feature_available!("sse4.1")
                        && crate::is_x86_feature_available!("sse4.2")
                        && crate::is_x86_feature_available!("popcnt")
                        && crate::is_x86_feature_available!("avx")
                        && crate::is_x86_feature_available!("avx2")
                        && crate::is_x86_feature_available!("fma")
                        && crate::is_x86_feature_available!("bmi1")
                        && crate::is_x86_feature_available!("bmi2")
                        && crate::is_x86_feature_available!("f16c")
                        && crate::is_x86_feature_available!("lzcnt");
                    X64_V3_CACHE.store(if available { 2 } else { 1 }, Ordering::Relaxed);
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

impl X64V3Token {
    /// Get a X64V2Token (x86-64-v3 implies x86-64-v2)
    #[allow(deprecated)]
    #[inline(always)]
    pub fn v2(self) -> X64V2Token {
        unsafe { X64V2Token::forge_token_dangerously() }
    }
}

impl X64V3Token {
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
    /// - `X64V4Token`
    /// - `Avx512ModernToken`
    /// - `Avx512Fp16Token`
    pub fn dangerously_disable_token_process_wide(
        disabled: bool,
    ) -> Result<(), crate::tokens::CompileTimeGuaranteedError> {
        #[cfg(all(
            target_feature = "sse",
            target_feature = "sse2",
            target_feature = "sse3",
            target_feature = "ssse3",
            target_feature = "sse4.1",
            target_feature = "sse4.2",
            target_feature = "popcnt",
            target_feature = "avx",
            target_feature = "avx2",
            target_feature = "fma",
            target_feature = "bmi1",
            target_feature = "bmi2",
            target_feature = "f16c",
            target_feature = "lzcnt",
            not(feature = "disable_compile_time_tokens")
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
            target_feature = "sse",
            target_feature = "sse2",
            target_feature = "sse3",
            target_feature = "ssse3",
            target_feature = "sse4.1",
            target_feature = "sse4.2",
            target_feature = "popcnt",
            target_feature = "avx",
            target_feature = "avx2",
            target_feature = "fma",
            target_feature = "bmi1",
            target_feature = "bmi2",
            target_feature = "f16c",
            target_feature = "lzcnt",
            not(feature = "disable_compile_time_tokens")
        )))]
        {
            X64_V3_DISABLED.store(disabled, Ordering::Relaxed);
            let v = if disabled { 1 } else { 0 };
            X64_V3_CACHE.store(v, Ordering::Relaxed);
            #[cfg(feature = "avx512")]
            {
                super::x86_avx512::X64_V4_DISABLED.store(disabled, Ordering::Relaxed);
                super::x86_avx512::X64_V4_CACHE.store(v, Ordering::Relaxed);
            }
            #[cfg(feature = "avx512")]
            {
                super::x86_avx512::AVX512_MODERN_DISABLED.store(disabled, Ordering::Relaxed);
                super::x86_avx512::AVX512_MODERN_CACHE.store(v, Ordering::Relaxed);
            }
            #[cfg(feature = "avx512")]
            {
                super::x86_avx512::AVX512_FP16_DISABLED.store(disabled, Ordering::Relaxed);
                super::x86_avx512::AVX512_FP16_CACHE.store(v, Ordering::Relaxed);
            }
            Ok(())
        }
    }

    /// Check if this token has been manually disabled process-wide.
    ///
    /// Returns `Err` when all required features are compile-time enabled.
    pub fn manually_disabled() -> Result<bool, crate::tokens::CompileTimeGuaranteedError> {
        #[cfg(all(
            target_feature = "sse",
            target_feature = "sse2",
            target_feature = "sse3",
            target_feature = "ssse3",
            target_feature = "sse4.1",
            target_feature = "sse4.2",
            target_feature = "popcnt",
            target_feature = "avx",
            target_feature = "avx2",
            target_feature = "fma",
            target_feature = "bmi1",
            target_feature = "bmi2",
            target_feature = "f16c",
            target_feature = "lzcnt",
            not(feature = "disable_compile_time_tokens")
        ))]
        {
            return Err(crate::tokens::CompileTimeGuaranteedError {
                token_name: Self::NAME,
                target_features: Self::TARGET_FEATURES,
                disable_flags: Self::DISABLE_TARGET_FEATURES,
            });
        }
        #[cfg(not(all(
            target_feature = "sse",
            target_feature = "sse2",
            target_feature = "sse3",
            target_feature = "ssse3",
            target_feature = "sse4.1",
            target_feature = "sse4.2",
            target_feature = "popcnt",
            target_feature = "avx",
            target_feature = "avx2",
            target_feature = "fma",
            target_feature = "bmi1",
            target_feature = "bmi2",
            target_feature = "f16c",
            target_feature = "lzcnt",
            not(feature = "disable_compile_time_tokens")
        )))]
        {
            Ok(X64_V3_DISABLED.load(Ordering::Relaxed))
        }
    }
}

/// Type alias for [`X64V3Token`].
pub type Desktop64 = X64V3Token;

/// Type alias for [`X64V3Token`].
pub type Avx2FmaToken = X64V3Token;

impl Has128BitSimd for X64V2Token {}
impl Has128BitSimd for X64V3Token {}
impl Has256BitSimd for X64V3Token {}
impl HasX64V2 for X64V2Token {}
impl HasX64V2 for X64V3Token {}
