//! Generated from token-registry.toml — DO NOT EDIT.
//!
//! Regenerate with: cargo xtask generate

use crate::tokens::SimdToken;
use crate::tokens::{Has128BitSimd, Has256BitSimd, HasX64V2};
use core::sync::atomic::{AtomicU8, Ordering};

// Cache statics: 0 = unknown, 1 = unavailable, 2 = available
static X64_V2_CACHE: AtomicU8 = AtomicU8::new(0);
static X64_V3_CACHE: AtomicU8 = AtomicU8::new(0);

/// Proof that SSE4.2 + POPCNT are available (x86-64-v2 level).
///
/// x86-64-v2 implies: SSE, SSE2, SSE3, SSSE3, SSE4.1, SSE4.2, POPCNT, CX16, SAHF.
/// This is the Nehalem (2008) / Bulldozer (2011) baseline.
#[derive(Clone, Copy, Debug)]
pub struct X64V2Token {
    _private: (),
}

impl SimdToken for X64V2Token {
    const NAME: &'static str = "x86-64-v2";

    #[inline]
    fn guaranteed() -> Option<bool> {
        #[cfg(all(
            target_feature = "sse3",
            target_feature = "ssse3",
            target_feature = "sse4.1",
            target_feature = "sse4.2",
            target_feature = "popcnt"
        ))]
        {
            Some(true)
        }
        #[cfg(not(all(
            target_feature = "sse3",
            target_feature = "ssse3",
            target_feature = "sse4.1",
            target_feature = "sse4.2",
            target_feature = "popcnt"
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
            target_feature = "popcnt"
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
            target_feature = "popcnt"
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

impl SimdToken for X64V3Token {
    const NAME: &'static str = "x86-64-v3";

    #[inline]
    fn guaranteed() -> Option<bool> {
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
            target_feature = "lzcnt"
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
            target_feature = "lzcnt"
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
            target_feature = "lzcnt"
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
            target_feature = "lzcnt"
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

/// Type alias for [`X64V3Token`].
pub type Desktop64 = X64V3Token;

/// Type alias for [`X64V3Token`].
pub type Avx2FmaToken = X64V3Token;

impl Has128BitSimd for X64V2Token {}
impl Has128BitSimd for X64V3Token {}
impl Has256BitSimd for X64V3Token {}
impl HasX64V2 for X64V2Token {}
impl HasX64V2 for X64V3Token {}
