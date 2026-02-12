//! Generated from token-registry.toml — DO NOT EDIT.
//!
//! Regenerate with: cargo xtask generate

use super::x86::X64V2Token;
use super::x86::X64V3Token;
use crate::tokens::SimdToken;
use crate::tokens::{Has128BitSimd, Has256BitSimd, Has512BitSimd, HasX64V2, HasX64V4};
use core::sync::atomic::{AtomicBool, AtomicU8, Ordering};

// Cache statics: 0 = unknown, 1 = unavailable, 2 = available
pub(super) static X64_V4_CACHE: AtomicU8 = AtomicU8::new(0);
pub(super) static X64_V4_DISABLED: AtomicBool = AtomicBool::new(false);
pub(super) static AVX512_MODERN_CACHE: AtomicU8 = AtomicU8::new(0);
pub(super) static AVX512_MODERN_DISABLED: AtomicBool = AtomicBool::new(false);
pub(super) static AVX512_FP16_CACHE: AtomicU8 = AtomicU8::new(0);
pub(super) static AVX512_FP16_DISABLED: AtomicBool = AtomicBool::new(false);

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
            target_feature = "avx512vl"
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
            target_feature = "avx512vl"
        )))]
        {
            None
        }
    }

    #[allow(deprecated)]
    #[inline(always)]
    fn summon() -> Option<Self> {
        // Compile-time fast path
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
            target_feature = "avx512vl"
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
            target_feature = "avx512vl"
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
            target_feature = "avx512vl"
        ))]
        {
            let _ = disabled;
            return Err(crate::tokens::CompileTimeGuaranteedError {
                token_name: Self::NAME,
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
            target_feature = "avx512vl"
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
            target_feature = "avx512vl"
        ))]
        {
            return Err(crate::tokens::CompileTimeGuaranteedError {
                token_name: Self::NAME,
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
            target_feature = "avx512vl"
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
            target_feature = "vaes"
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
            target_feature = "vaes"
        )))]
        {
            None
        }
    }

    #[allow(deprecated)]
    #[inline(always)]
    fn summon() -> Option<Self> {
        // Compile-time fast path
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
            target_feature = "vaes"
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
            target_feature = "vaes"
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
            target_feature = "vaes"
        ))]
        {
            let _ = disabled;
            return Err(crate::tokens::CompileTimeGuaranteedError {
                token_name: Self::NAME,
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
            target_feature = "vaes"
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
            target_feature = "vaes"
        ))]
        {
            return Err(crate::tokens::CompileTimeGuaranteedError {
                token_name: Self::NAME,
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
            target_feature = "vaes"
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
            target_feature = "avx512fp16"
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
            target_feature = "avx512fp16"
        )))]
        {
            None
        }
    }

    #[allow(deprecated)]
    #[inline(always)]
    fn summon() -> Option<Self> {
        // Compile-time fast path
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
            target_feature = "avx512fp16"
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
            target_feature = "avx512fp16"
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
            target_feature = "avx512fp16"
        ))]
        {
            let _ = disabled;
            return Err(crate::tokens::CompileTimeGuaranteedError {
                token_name: Self::NAME,
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
            target_feature = "avx512fp16"
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
            target_feature = "avx512fp16"
        ))]
        {
            return Err(crate::tokens::CompileTimeGuaranteedError {
                token_name: Self::NAME,
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
            target_feature = "avx512fp16"
        )))]
        {
            Ok(AVX512_FP16_DISABLED.load(Ordering::Relaxed))
        }
    }
}

/// Type alias for [`X64V4Token`].
pub type Avx512Token = X64V4Token;

/// Type alias for [`X64V4Token`].
pub type Server64 = X64V4Token;

impl Has128BitSimd for X64V4Token {}
impl Has128BitSimd for Avx512ModernToken {}
impl Has128BitSimd for Avx512Fp16Token {}
impl Has256BitSimd for X64V4Token {}
impl Has256BitSimd for Avx512ModernToken {}
impl Has256BitSimd for Avx512Fp16Token {}
impl Has512BitSimd for X64V4Token {}
impl Has512BitSimd for Avx512ModernToken {}
impl Has512BitSimd for Avx512Fp16Token {}
impl HasX64V2 for X64V4Token {}
impl HasX64V2 for Avx512ModernToken {}
impl HasX64V2 for Avx512Fp16Token {}
impl HasX64V4 for X64V4Token {}
impl HasX64V4 for Avx512ModernToken {}
impl HasX64V4 for Avx512Fp16Token {}
