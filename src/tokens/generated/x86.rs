//! Generated from token-registry.toml — DO NOT EDIT.
//!
//! Regenerate with: cargo xtask generate

use crate::tokens::SimdToken;
use crate::tokens::{Has128BitSimd, Has256BitSimd, Has512BitSimd, HasX64V2, HasX64V4};
use core::sync::atomic::{AtomicBool, AtomicU8, Ordering};

// Cache statics: 0 = unknown, 1 = unavailable, 2 = available
#[allow(dead_code)]
pub(super) static X64_V1_CACHE: AtomicU8 = AtomicU8::new(0);
#[allow(dead_code)]
pub(super) static X64_V1_DISABLED: AtomicBool = AtomicBool::new(false);
pub(super) static X64_V2_CACHE: AtomicU8 = AtomicU8::new(0);
pub(super) static X64_V2_DISABLED: AtomicBool = AtomicBool::new(false);
pub(super) static X64_V3_CACHE: AtomicU8 = AtomicU8::new(0);
pub(super) static X64_V3_DISABLED: AtomicBool = AtomicBool::new(false);
pub(super) static X64_V4_CACHE: AtomicU8 = AtomicU8::new(0);
pub(super) static X64_V4_DISABLED: AtomicBool = AtomicBool::new(false);
pub(super) static AVX512_MODERN_CACHE: AtomicU8 = AtomicU8::new(0);
pub(super) static AVX512_MODERN_DISABLED: AtomicBool = AtomicBool::new(false);
pub(super) static AVX512_FP16_CACHE: AtomicU8 = AtomicU8::new(0);
pub(super) static AVX512_FP16_DISABLED: AtomicBool = AtomicBool::new(false);

/// Proof that SSE + SSE2 are available (x86-64-v1 baseline level).
///
/// SSE2 is the x86_64 ABI baseline — every x86_64 CPU has it. However, Rust
/// still requires #[target_feature(enable = "sse2")] for SSE2 intrinsics to
/// be safe. This token provides that via `#[arcane]`.
///
/// On x86_64, summon() always returns Some.
#[derive(Clone, Copy, Debug)]
pub struct X64V1Token {
    _private: (),
}

impl crate::tokens::Sealed for X64V1Token {}

impl SimdToken for X64V1Token {
    const NAME: &'static str = "x86-64-v1";
    const TARGET_FEATURES: &'static str = "sse,sse2";
    const ENABLE_TARGET_FEATURES: &'static str = "-Ctarget-feature=+sse,+sse2";
    const DISABLE_TARGET_FEATURES: &'static str = "-Ctarget-feature=-sse,-sse2";

    #[inline]
    fn compiled_with() -> Option<bool> {
        Some(true)
    }

    #[allow(deprecated)]
    #[inline]
    fn summon() -> Option<Self> {
        Some(unsafe { Self::forge_token_dangerously() })
    }

    #[inline(always)]
    #[allow(deprecated)]
    unsafe fn forge_token_dangerously() -> Self {
        Self { _private: () }
    }
}

impl X64V1Token {
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
    /// - `X64V2Token`
    /// - `X64V3Token`
    /// - `X64V4Token`
    /// - `Avx512ModernToken`
    /// - `Avx512Fp16Token`
    #[allow(clippy::needless_return)]
    pub fn dangerously_disable_token_process_wide(
        disabled: bool,
    ) -> Result<(), crate::tokens::CompileTimeGuaranteedError> {
        #[cfg(all(
            target_feature = "sse",
            target_feature = "sse2",
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
            target_feature = "sse",
            target_feature = "sse2",
            not(feature = "testable_dispatch")
        )))]
        {
            X64_V1_DISABLED.store(disabled, Ordering::Relaxed);
            let v = if disabled { 1 } else { 0 };
            X64_V1_CACHE.store(v, Ordering::Relaxed);
            X64_V2_DISABLED.store(disabled, Ordering::Relaxed);
            X64_V2_CACHE.store(v, Ordering::Relaxed);
            X64_V3_DISABLED.store(disabled, Ordering::Relaxed);
            X64_V3_CACHE.store(v, Ordering::Relaxed);
            X64_V4_DISABLED.store(disabled, Ordering::Relaxed);
            X64_V4_CACHE.store(v, Ordering::Relaxed);
            AVX512_MODERN_DISABLED.store(disabled, Ordering::Relaxed);
            AVX512_MODERN_CACHE.store(v, Ordering::Relaxed);
            AVX512_FP16_DISABLED.store(disabled, Ordering::Relaxed);
            AVX512_FP16_CACHE.store(v, Ordering::Relaxed);
            Ok(())
        }
    }

    /// Check if this token has been manually disabled process-wide.
    ///
    /// Returns `Err` when all required features are compile-time enabled.
    #[allow(clippy::needless_return)]
    pub fn manually_disabled() -> Result<bool, crate::tokens::CompileTimeGuaranteedError> {
        #[cfg(all(
            target_feature = "sse",
            target_feature = "sse2",
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
            target_feature = "sse",
            target_feature = "sse2",
            not(feature = "testable_dispatch")
        )))]
        {
            Ok(X64_V1_DISABLED.load(Ordering::Relaxed))
        }
    }
}

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
    const TARGET_FEATURES: &'static str = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt";
    const ENABLE_TARGET_FEATURES: &'static str =
        "-Ctarget-feature=+sse,+sse2,+sse3,+ssse3,+sse4.1,+sse4.2,+popcnt";
    const DISABLE_TARGET_FEATURES: &'static str =
        "-Ctarget-feature=-sse,-sse2,-sse3,-ssse3,-sse4.1,-sse4.2,-popcnt";

    #[inline]
    fn compiled_with() -> Option<bool> {
        #[cfg(all(
            target_feature = "sse3",
            target_feature = "ssse3",
            target_feature = "sse4.1",
            target_feature = "sse4.2",
            target_feature = "popcnt",
            not(feature = "testable_dispatch")
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
            target_feature = "sse3",
            target_feature = "ssse3",
            target_feature = "sse4.1",
            target_feature = "sse4.2",
            target_feature = "popcnt",
            not(feature = "testable_dispatch")
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
            not(feature = "testable_dispatch")
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
    /// Get a X64V1Token (x86-64-v2 implies x86-64-v1)
    #[allow(deprecated)]
    #[inline(always)]
    pub fn v1(self) -> X64V1Token {
        unsafe { X64V1Token::forge_token_dangerously() }
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
    #[allow(clippy::needless_return)]
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
            target_feature = "sse",
            target_feature = "sse2",
            target_feature = "sse3",
            target_feature = "ssse3",
            target_feature = "sse4.1",
            target_feature = "sse4.2",
            target_feature = "popcnt",
            not(feature = "testable_dispatch")
        )))]
        {
            X64_V2_DISABLED.store(disabled, Ordering::Relaxed);
            let v = if disabled { 1 } else { 0 };
            X64_V2_CACHE.store(v, Ordering::Relaxed);
            X64_V3_DISABLED.store(disabled, Ordering::Relaxed);
            X64_V3_CACHE.store(v, Ordering::Relaxed);
            X64_V4_DISABLED.store(disabled, Ordering::Relaxed);
            X64_V4_CACHE.store(v, Ordering::Relaxed);
            AVX512_MODERN_DISABLED.store(disabled, Ordering::Relaxed);
            AVX512_MODERN_CACHE.store(v, Ordering::Relaxed);
            AVX512_FP16_DISABLED.store(disabled, Ordering::Relaxed);
            AVX512_FP16_CACHE.store(v, Ordering::Relaxed);
            Ok(())
        }
    }

    /// Check if this token has been manually disabled process-wide.
    ///
    /// Returns `Err` when all required features are compile-time enabled.
    #[allow(clippy::needless_return)]
    pub fn manually_disabled() -> Result<bool, crate::tokens::CompileTimeGuaranteedError> {
        #[cfg(all(
            target_feature = "sse",
            target_feature = "sse2",
            target_feature = "sse3",
            target_feature = "ssse3",
            target_feature = "sse4.1",
            target_feature = "sse4.2",
            target_feature = "popcnt",
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
            target_feature = "sse",
            target_feature = "sse2",
            target_feature = "sse3",
            target_feature = "ssse3",
            target_feature = "sse4.1",
            target_feature = "sse4.2",
            target_feature = "popcnt",
            not(feature = "testable_dispatch")
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
        "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt";
    const ENABLE_TARGET_FEATURES: &'static str = "-Ctarget-feature=+sse,+sse2,+sse3,+ssse3,+sse4.1,+sse4.2,+popcnt,+avx,+avx2,+fma,+bmi1,+bmi2,+f16c,+lzcnt";
    const DISABLE_TARGET_FEATURES: &'static str = "-Ctarget-feature=-sse,-sse2,-sse3,-ssse3,-sse4.1,-sse4.2,-popcnt,-avx,-avx2,-fma,-bmi1,-bmi2,-f16c,-lzcnt";

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
            not(feature = "testable_dispatch")
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
            not(feature = "testable_dispatch")
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
            not(feature = "testable_dispatch")
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
    /// Get a X64V1Token (x86-64-v3 implies x86-64-v1)
    #[allow(deprecated)]
    #[inline(always)]
    pub fn v1(self) -> X64V1Token {
        unsafe { X64V1Token::forge_token_dangerously() }
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
    #[allow(clippy::needless_return)]
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
            not(feature = "testable_dispatch")
        )))]
        {
            X64_V3_DISABLED.store(disabled, Ordering::Relaxed);
            let v = if disabled { 1 } else { 0 };
            X64_V3_CACHE.store(v, Ordering::Relaxed);
            X64_V4_DISABLED.store(disabled, Ordering::Relaxed);
            X64_V4_CACHE.store(v, Ordering::Relaxed);
            AVX512_MODERN_DISABLED.store(disabled, Ordering::Relaxed);
            AVX512_MODERN_CACHE.store(v, Ordering::Relaxed);
            AVX512_FP16_DISABLED.store(disabled, Ordering::Relaxed);
            AVX512_FP16_CACHE.store(v, Ordering::Relaxed);
            Ok(())
        }
    }

    /// Check if this token has been manually disabled process-wide.
    ///
    /// Returns `Err` when all required features are compile-time enabled.
    #[allow(clippy::needless_return)]
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
            not(feature = "testable_dispatch")
        )))]
        {
            Ok(X64_V3_DISABLED.load(Ordering::Relaxed))
        }
    }
}

/// Proof that AVX-512 (F + CD + VL + DQ + BW) is available.
///
/// This is the complete x86-64-v4 AVX-512 feature set, available on:
/// - Intel Skylake-X (2017+), Ice Lake, Sapphire Rapids
/// - AMD Zen 4+ (2022+)
///
/// Note: Intel 12th-14th gen consumer CPUs do NOT have AVX-512.
#[derive(Clone, Copy, Debug)]
pub struct X64V4Token {
    _private: (),
}

impl crate::tokens::Sealed for X64V4Token {}

impl SimdToken for X64V4Token {
    const NAME: &'static str = "AVX-512";
    const TARGET_FEATURES: &'static str = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,avx512f,avx512bw,avx512cd,avx512dq,avx512vl";
    const ENABLE_TARGET_FEATURES: &'static str = "-Ctarget-feature=+sse,+sse2,+sse3,+ssse3,+sse4.1,+sse4.2,+popcnt,+avx,+avx2,+fma,+bmi1,+bmi2,+f16c,+lzcnt,+avx512f,+avx512bw,+avx512cd,+avx512dq,+avx512vl";
    const DISABLE_TARGET_FEATURES: &'static str = "-Ctarget-feature=-sse,-sse2,-sse3,-ssse3,-sse4.1,-sse4.2,-popcnt,-avx,-avx2,-fma,-bmi1,-bmi2,-f16c,-lzcnt,-avx512f,-avx512bw,-avx512cd,-avx512dq,-avx512vl";

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
            target_feature = "avx512f",
            target_feature = "avx512bw",
            target_feature = "avx512cd",
            target_feature = "avx512dq",
            target_feature = "avx512vl",
            not(feature = "testable_dispatch")
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
            target_feature = "avx512f",
            target_feature = "avx512bw",
            target_feature = "avx512cd",
            target_feature = "avx512dq",
            target_feature = "avx512vl",
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
            target_feature = "avx512f",
            target_feature = "avx512bw",
            target_feature = "avx512cd",
            target_feature = "avx512dq",
            target_feature = "avx512vl",
            not(feature = "testable_dispatch")
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
            target_feature = "avx512f",
            target_feature = "avx512bw",
            target_feature = "avx512cd",
            target_feature = "avx512dq",
            target_feature = "avx512vl",
            not(feature = "testable_dispatch")
        )))]
        {
            match X64_V4_CACHE.load(Ordering::Relaxed) {
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
                        && crate::is_x86_feature_available!("lzcnt")
                        && crate::is_x86_feature_available!("avx512f")
                        && crate::is_x86_feature_available!("avx512bw")
                        && crate::is_x86_feature_available!("avx512cd")
                        && crate::is_x86_feature_available!("avx512dq")
                        && crate::is_x86_feature_available!("avx512vl");
                    X64_V4_CACHE.store(if available { 2 } else { 1 }, Ordering::Relaxed);
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

impl X64V4Token {
    /// Get a X64V3Token (AVX-512 implies x86-64-v3)
    #[allow(deprecated)]
    #[inline(always)]
    pub fn v3(self) -> X64V3Token {
        unsafe { X64V3Token::forge_token_dangerously() }
    }
    /// Get a X64V2Token (AVX-512 implies x86-64-v2)
    #[allow(deprecated)]
    #[inline(always)]
    pub fn v2(self) -> X64V2Token {
        unsafe { X64V2Token::forge_token_dangerously() }
    }
    /// Get a X64V1Token (AVX-512 implies x86-64-v1)
    #[allow(deprecated)]
    #[inline(always)]
    pub fn v1(self) -> X64V1Token {
        unsafe { X64V1Token::forge_token_dangerously() }
    }
}

impl X64V4Token {
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
    /// - `Avx512ModernToken`
    /// - `Avx512Fp16Token`
    #[allow(clippy::needless_return)]
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
            target_feature = "avx512f",
            target_feature = "avx512bw",
            target_feature = "avx512cd",
            target_feature = "avx512dq",
            target_feature = "avx512vl",
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
            target_feature = "avx512f",
            target_feature = "avx512bw",
            target_feature = "avx512cd",
            target_feature = "avx512dq",
            target_feature = "avx512vl",
            not(feature = "testable_dispatch")
        )))]
        {
            X64_V4_DISABLED.store(disabled, Ordering::Relaxed);
            let v = if disabled { 1 } else { 0 };
            X64_V4_CACHE.store(v, Ordering::Relaxed);
            AVX512_MODERN_DISABLED.store(disabled, Ordering::Relaxed);
            AVX512_MODERN_CACHE.store(v, Ordering::Relaxed);
            AVX512_FP16_DISABLED.store(disabled, Ordering::Relaxed);
            AVX512_FP16_CACHE.store(v, Ordering::Relaxed);
            Ok(())
        }
    }

    /// Check if this token has been manually disabled process-wide.
    ///
    /// Returns `Err` when all required features are compile-time enabled.
    #[allow(clippy::needless_return)]
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
            target_feature = "avx512f",
            target_feature = "avx512bw",
            target_feature = "avx512cd",
            target_feature = "avx512dq",
            target_feature = "avx512vl",
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
            target_feature = "avx512f",
            target_feature = "avx512bw",
            target_feature = "avx512cd",
            target_feature = "avx512dq",
            target_feature = "avx512vl",
            not(feature = "testable_dispatch")
        )))]
        {
            Ok(X64_V4_DISABLED.load(Ordering::Relaxed))
        }
    }
}

/// Proof that modern AVX-512 features are available (Ice Lake / Zen 4 level).
///
/// This includes all of `X64V4Token` (F+CD+VL+DQ+BW) plus:
/// - VPOPCNTDQ, IFMA, VBMI, VBMI2, BITALG, VNNI, BF16
/// - VPCLMULQDQ, GFNI, VAES
///
/// Available on Intel Ice Lake (2019+), Sapphire Rapids, AMD Zen 4+.
/// NOT available on Skylake-X (lacks VBMI2, VNNI, BF16, etc.).
#[derive(Clone, Copy, Debug)]
pub struct Avx512ModernToken {
    _private: (),
}

impl crate::tokens::Sealed for Avx512ModernToken {}

impl SimdToken for Avx512ModernToken {
    const NAME: &'static str = "AVX-512Modern";
    const TARGET_FEATURES: &'static str = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,avx512f,avx512bw,avx512cd,avx512dq,avx512vl,avx512vpopcntdq,avx512ifma,avx512vbmi,avx512vbmi2,avx512bitalg,avx512vnni,avx512bf16,vpclmulqdq,gfni,vaes";
    const ENABLE_TARGET_FEATURES: &'static str = "-Ctarget-feature=+sse,+sse2,+sse3,+ssse3,+sse4.1,+sse4.2,+popcnt,+avx,+avx2,+fma,+bmi1,+bmi2,+f16c,+lzcnt,+avx512f,+avx512bw,+avx512cd,+avx512dq,+avx512vl,+avx512vpopcntdq,+avx512ifma,+avx512vbmi,+avx512vbmi2,+avx512bitalg,+avx512vnni,+avx512bf16,+vpclmulqdq,+gfni,+vaes";
    const DISABLE_TARGET_FEATURES: &'static str = "-Ctarget-feature=-sse,-sse2,-sse3,-ssse3,-sse4.1,-sse4.2,-popcnt,-avx,-avx2,-fma,-bmi1,-bmi2,-f16c,-lzcnt,-avx512f,-avx512bw,-avx512cd,-avx512dq,-avx512vl,-avx512vpopcntdq,-avx512ifma,-avx512vbmi,-avx512vbmi2,-avx512bitalg,-avx512vnni,-avx512bf16,-vpclmulqdq,-gfni,-vaes";

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
            target_feature = "avx512f",
            target_feature = "avx512bw",
            target_feature = "avx512cd",
            target_feature = "avx512dq",
            target_feature = "avx512vl",
            target_feature = "avx512vpopcntdq",
            target_feature = "avx512ifma",
            target_feature = "avx512vbmi",
            target_feature = "avx512vbmi2",
            target_feature = "avx512bitalg",
            target_feature = "avx512vnni",
            target_feature = "avx512bf16",
            target_feature = "vpclmulqdq",
            target_feature = "gfni",
            target_feature = "vaes",
            not(feature = "testable_dispatch")
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
            target_feature = "avx512f",
            target_feature = "avx512bw",
            target_feature = "avx512cd",
            target_feature = "avx512dq",
            target_feature = "avx512vl",
            target_feature = "avx512vpopcntdq",
            target_feature = "avx512ifma",
            target_feature = "avx512vbmi",
            target_feature = "avx512vbmi2",
            target_feature = "avx512bitalg",
            target_feature = "avx512vnni",
            target_feature = "avx512bf16",
            target_feature = "vpclmulqdq",
            target_feature = "gfni",
            target_feature = "vaes",
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
            target_feature = "avx512f",
            target_feature = "avx512bw",
            target_feature = "avx512cd",
            target_feature = "avx512dq",
            target_feature = "avx512vl",
            target_feature = "avx512vpopcntdq",
            target_feature = "avx512ifma",
            target_feature = "avx512vbmi",
            target_feature = "avx512vbmi2",
            target_feature = "avx512bitalg",
            target_feature = "avx512vnni",
            target_feature = "avx512bf16",
            target_feature = "vpclmulqdq",
            target_feature = "gfni",
            target_feature = "vaes",
            not(feature = "testable_dispatch")
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
            target_feature = "avx512f",
            target_feature = "avx512bw",
            target_feature = "avx512cd",
            target_feature = "avx512dq",
            target_feature = "avx512vl",
            target_feature = "avx512vpopcntdq",
            target_feature = "avx512ifma",
            target_feature = "avx512vbmi",
            target_feature = "avx512vbmi2",
            target_feature = "avx512bitalg",
            target_feature = "avx512vnni",
            target_feature = "avx512bf16",
            target_feature = "vpclmulqdq",
            target_feature = "gfni",
            target_feature = "vaes",
            not(feature = "testable_dispatch")
        )))]
        {
            match AVX512_MODERN_CACHE.load(Ordering::Relaxed) {
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
                        && crate::is_x86_feature_available!("lzcnt")
                        && crate::is_x86_feature_available!("avx512f")
                        && crate::is_x86_feature_available!("avx512bw")
                        && crate::is_x86_feature_available!("avx512cd")
                        && crate::is_x86_feature_available!("avx512dq")
                        && crate::is_x86_feature_available!("avx512vl")
                        && crate::is_x86_feature_available!("avx512vpopcntdq")
                        && crate::is_x86_feature_available!("avx512ifma")
                        && crate::is_x86_feature_available!("avx512vbmi")
                        && crate::is_x86_feature_available!("avx512vbmi2")
                        && crate::is_x86_feature_available!("avx512bitalg")
                        && crate::is_x86_feature_available!("avx512vnni")
                        && crate::is_x86_feature_available!("avx512bf16")
                        && crate::is_x86_feature_available!("vpclmulqdq")
                        && crate::is_x86_feature_available!("gfni")
                        && crate::is_x86_feature_available!("vaes");
                    AVX512_MODERN_CACHE.store(if available { 2 } else { 1 }, Ordering::Relaxed);
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

impl Avx512ModernToken {
    /// Get a X64V4Token (AVX-512Modern implies AVX-512)
    #[allow(deprecated)]
    #[inline(always)]
    pub fn v4(self) -> X64V4Token {
        unsafe { X64V4Token::forge_token_dangerously() }
    }

    /// Get a X64V4Token (alias for `.v4()`)
    #[allow(deprecated)]
    #[inline(always)]
    pub fn avx512(self) -> X64V4Token {
        unsafe { X64V4Token::forge_token_dangerously() }
    }
    /// Get a X64V3Token (AVX-512Modern implies x86-64-v3)
    #[allow(deprecated)]
    #[inline(always)]
    pub fn v3(self) -> X64V3Token {
        unsafe { X64V3Token::forge_token_dangerously() }
    }
    /// Get a X64V2Token (AVX-512Modern implies x86-64-v2)
    #[allow(deprecated)]
    #[inline(always)]
    pub fn v2(self) -> X64V2Token {
        unsafe { X64V2Token::forge_token_dangerously() }
    }
    /// Get a X64V1Token (AVX-512Modern implies x86-64-v1)
    #[allow(deprecated)]
    #[inline(always)]
    pub fn v1(self) -> X64V1Token {
        unsafe { X64V1Token::forge_token_dangerously() }
    }
}

impl Avx512ModernToken {
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
            target_feature = "avx512f",
            target_feature = "avx512bw",
            target_feature = "avx512cd",
            target_feature = "avx512dq",
            target_feature = "avx512vl",
            target_feature = "avx512vpopcntdq",
            target_feature = "avx512ifma",
            target_feature = "avx512vbmi",
            target_feature = "avx512vbmi2",
            target_feature = "avx512bitalg",
            target_feature = "avx512vnni",
            target_feature = "avx512bf16",
            target_feature = "vpclmulqdq",
            target_feature = "gfni",
            target_feature = "vaes",
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
            target_feature = "avx512f",
            target_feature = "avx512bw",
            target_feature = "avx512cd",
            target_feature = "avx512dq",
            target_feature = "avx512vl",
            target_feature = "avx512vpopcntdq",
            target_feature = "avx512ifma",
            target_feature = "avx512vbmi",
            target_feature = "avx512vbmi2",
            target_feature = "avx512bitalg",
            target_feature = "avx512vnni",
            target_feature = "avx512bf16",
            target_feature = "vpclmulqdq",
            target_feature = "gfni",
            target_feature = "vaes",
            not(feature = "testable_dispatch")
        )))]
        {
            AVX512_MODERN_DISABLED.store(disabled, Ordering::Relaxed);
            let v = if disabled { 1 } else { 0 };
            AVX512_MODERN_CACHE.store(v, Ordering::Relaxed);
            Ok(())
        }
    }

    /// Check if this token has been manually disabled process-wide.
    ///
    /// Returns `Err` when all required features are compile-time enabled.
    #[allow(clippy::needless_return)]
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
            target_feature = "avx512f",
            target_feature = "avx512bw",
            target_feature = "avx512cd",
            target_feature = "avx512dq",
            target_feature = "avx512vl",
            target_feature = "avx512vpopcntdq",
            target_feature = "avx512ifma",
            target_feature = "avx512vbmi",
            target_feature = "avx512vbmi2",
            target_feature = "avx512bitalg",
            target_feature = "avx512vnni",
            target_feature = "avx512bf16",
            target_feature = "vpclmulqdq",
            target_feature = "gfni",
            target_feature = "vaes",
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
            target_feature = "avx512f",
            target_feature = "avx512bw",
            target_feature = "avx512cd",
            target_feature = "avx512dq",
            target_feature = "avx512vl",
            target_feature = "avx512vpopcntdq",
            target_feature = "avx512ifma",
            target_feature = "avx512vbmi",
            target_feature = "avx512vbmi2",
            target_feature = "avx512bitalg",
            target_feature = "avx512vnni",
            target_feature = "avx512bf16",
            target_feature = "vpclmulqdq",
            target_feature = "gfni",
            target_feature = "vaes",
            not(feature = "testable_dispatch")
        )))]
        {
            Ok(AVX512_MODERN_DISABLED.load(Ordering::Relaxed))
        }
    }
}

/// Proof that AVX-512 FP16 (half-precision) is available.
///
/// AVX-512 FP16 provides native 16-bit floating-point arithmetic in 512-bit
/// vectors, enabling efficient ML inference and scientific computing.
///
/// Available on Intel Sapphire Rapids (2023+), Emerald Rapids.
/// NOT available on Skylake-X, Ice Lake, AMD Zen 4.
#[derive(Clone, Copy, Debug)]
pub struct Avx512Fp16Token {
    _private: (),
}

impl crate::tokens::Sealed for Avx512Fp16Token {}

impl SimdToken for Avx512Fp16Token {
    const NAME: &'static str = "AVX-512FP16";
    const TARGET_FEATURES: &'static str = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,popcnt,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt,avx512f,avx512bw,avx512cd,avx512dq,avx512vl,avx512fp16";
    const ENABLE_TARGET_FEATURES: &'static str = "-Ctarget-feature=+sse,+sse2,+sse3,+ssse3,+sse4.1,+sse4.2,+popcnt,+avx,+avx2,+fma,+bmi1,+bmi2,+f16c,+lzcnt,+avx512f,+avx512bw,+avx512cd,+avx512dq,+avx512vl,+avx512fp16";
    const DISABLE_TARGET_FEATURES: &'static str = "-Ctarget-feature=-sse,-sse2,-sse3,-ssse3,-sse4.1,-sse4.2,-popcnt,-avx,-avx2,-fma,-bmi1,-bmi2,-f16c,-lzcnt,-avx512f,-avx512bw,-avx512cd,-avx512dq,-avx512vl,-avx512fp16";

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
            target_feature = "avx512f",
            target_feature = "avx512bw",
            target_feature = "avx512cd",
            target_feature = "avx512dq",
            target_feature = "avx512vl",
            target_feature = "avx512fp16",
            not(feature = "testable_dispatch")
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
            target_feature = "avx512f",
            target_feature = "avx512bw",
            target_feature = "avx512cd",
            target_feature = "avx512dq",
            target_feature = "avx512vl",
            target_feature = "avx512fp16",
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
            target_feature = "avx512f",
            target_feature = "avx512bw",
            target_feature = "avx512cd",
            target_feature = "avx512dq",
            target_feature = "avx512vl",
            target_feature = "avx512fp16",
            not(feature = "testable_dispatch")
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
            target_feature = "avx512f",
            target_feature = "avx512bw",
            target_feature = "avx512cd",
            target_feature = "avx512dq",
            target_feature = "avx512vl",
            target_feature = "avx512fp16",
            not(feature = "testable_dispatch")
        )))]
        {
            match AVX512_FP16_CACHE.load(Ordering::Relaxed) {
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
                        && crate::is_x86_feature_available!("lzcnt")
                        && crate::is_x86_feature_available!("avx512f")
                        && crate::is_x86_feature_available!("avx512bw")
                        && crate::is_x86_feature_available!("avx512cd")
                        && crate::is_x86_feature_available!("avx512dq")
                        && crate::is_x86_feature_available!("avx512vl")
                        && crate::is_x86_feature_available!("avx512fp16");
                    AVX512_FP16_CACHE.store(if available { 2 } else { 1 }, Ordering::Relaxed);
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

impl Avx512Fp16Token {
    /// Get a X64V4Token (AVX-512FP16 implies AVX-512)
    #[allow(deprecated)]
    #[inline(always)]
    pub fn v4(self) -> X64V4Token {
        unsafe { X64V4Token::forge_token_dangerously() }
    }

    /// Get a X64V4Token (alias for `.v4()`)
    #[allow(deprecated)]
    #[inline(always)]
    pub fn avx512(self) -> X64V4Token {
        unsafe { X64V4Token::forge_token_dangerously() }
    }
    /// Get a X64V3Token (AVX-512FP16 implies x86-64-v3)
    #[allow(deprecated)]
    #[inline(always)]
    pub fn v3(self) -> X64V3Token {
        unsafe { X64V3Token::forge_token_dangerously() }
    }
    /// Get a X64V2Token (AVX-512FP16 implies x86-64-v2)
    #[allow(deprecated)]
    #[inline(always)]
    pub fn v2(self) -> X64V2Token {
        unsafe { X64V2Token::forge_token_dangerously() }
    }
    /// Get a X64V1Token (AVX-512FP16 implies x86-64-v1)
    #[allow(deprecated)]
    #[inline(always)]
    pub fn v1(self) -> X64V1Token {
        unsafe { X64V1Token::forge_token_dangerously() }
    }
}

impl Avx512Fp16Token {
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
            target_feature = "avx512f",
            target_feature = "avx512bw",
            target_feature = "avx512cd",
            target_feature = "avx512dq",
            target_feature = "avx512vl",
            target_feature = "avx512fp16",
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
            target_feature = "avx512f",
            target_feature = "avx512bw",
            target_feature = "avx512cd",
            target_feature = "avx512dq",
            target_feature = "avx512vl",
            target_feature = "avx512fp16",
            not(feature = "testable_dispatch")
        )))]
        {
            AVX512_FP16_DISABLED.store(disabled, Ordering::Relaxed);
            let v = if disabled { 1 } else { 0 };
            AVX512_FP16_CACHE.store(v, Ordering::Relaxed);
            Ok(())
        }
    }

    /// Check if this token has been manually disabled process-wide.
    ///
    /// Returns `Err` when all required features are compile-time enabled.
    #[allow(clippy::needless_return)]
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
            target_feature = "avx512f",
            target_feature = "avx512bw",
            target_feature = "avx512cd",
            target_feature = "avx512dq",
            target_feature = "avx512vl",
            target_feature = "avx512fp16",
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
            target_feature = "avx512f",
            target_feature = "avx512bw",
            target_feature = "avx512cd",
            target_feature = "avx512dq",
            target_feature = "avx512vl",
            target_feature = "avx512fp16",
            not(feature = "testable_dispatch")
        )))]
        {
            Ok(AVX512_FP16_DISABLED.load(Ordering::Relaxed))
        }
    }
}

/// Type alias for [`X64V1Token`].
pub type Sse2Token = X64V1Token;

/// Type alias for [`X64V3Token`].
pub type Desktop64 = X64V3Token;

/// Type alias for [`X64V3Token`].
pub type Avx2FmaToken = X64V3Token;

/// Type alias for [`X64V4Token`].
pub type Avx512Token = X64V4Token;

/// Type alias for [`X64V4Token`].
pub type Server64 = X64V4Token;

impl Has128BitSimd for X64V1Token {}
impl Has128BitSimd for X64V2Token {}
impl Has128BitSimd for X64V3Token {}
impl Has128BitSimd for X64V4Token {}
impl Has128BitSimd for Avx512ModernToken {}
impl Has128BitSimd for Avx512Fp16Token {}
impl Has256BitSimd for X64V3Token {}
impl Has256BitSimd for X64V4Token {}
impl Has256BitSimd for Avx512ModernToken {}
impl Has256BitSimd for Avx512Fp16Token {}
impl Has512BitSimd for X64V4Token {}
impl Has512BitSimd for Avx512ModernToken {}
impl Has512BitSimd for Avx512Fp16Token {}
impl HasX64V2 for X64V2Token {}
impl HasX64V2 for X64V3Token {}
impl HasX64V2 for X64V4Token {}
impl HasX64V2 for Avx512ModernToken {}
impl HasX64V2 for Avx512Fp16Token {}
impl HasX64V4 for X64V4Token {}
impl HasX64V4 for Avx512ModernToken {}
impl HasX64V4 for Avx512Fp16Token {}
