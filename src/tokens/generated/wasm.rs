//! Generated from token-registry.toml — DO NOT EDIT.
//!
//! Regenerate with: cargo xtask generate

#[allow(deprecated)]
use crate::tokens::Has128BitSimd;
use crate::tokens::SimdToken;
use core::sync::atomic::{AtomicBool, AtomicU8, Ordering};

// Cache statics: 0 = unknown, 1 = unavailable, 2 = available
#[allow(dead_code)]
pub(super) static WASM128_CACHE: AtomicU8 = AtomicU8::new(0);
#[allow(dead_code)]
pub(super) static WASM128_DISABLED: AtomicBool = AtomicBool::new(false);
#[allow(dead_code)]
pub(super) static WASM128_RELAXED_CACHE: AtomicU8 = AtomicU8::new(0);
#[allow(dead_code)]
pub(super) static WASM128_RELAXED_DISABLED: AtomicBool = AtomicBool::new(false);
#[allow(dead_code)]
pub(super) static WASM128_RELAXED_FMA_CACHE: AtomicU8 = AtomicU8::new(0);
#[allow(dead_code)]
pub(super) static WASM128_RELAXED_FMA_DISABLED: AtomicBool = AtomicBool::new(false);

/// Proof that WASM SIMD128 is available.
#[derive(Clone, Copy, Debug)]
pub struct Wasm128Token {
    _private: (),
}

impl crate::tokens::Sealed for Wasm128Token {}

impl SimdToken for Wasm128Token {
    const NAME: &'static str = "WASM SIMD128";
    const TARGET_FEATURES: &'static str = "simd128";
    const ENABLE_TARGET_FEATURES: &'static str = "-Ctarget-feature=+simd128";
    const DISABLE_TARGET_FEATURES: &'static str = "-Ctarget-feature=-simd128";

    #[inline]
    fn compiled_with() -> Option<bool> {
        #[cfg(all(
            target_arch = "wasm32",
            target_feature = "simd128",
            not(feature = "testable_dispatch")
        ))]
        {
            Some(true)
        }
        #[cfg(all(
            target_arch = "wasm32",
            target_feature = "simd128",
            feature = "testable_dispatch"
        ))]
        {
            None
        }
        #[cfg(not(all(target_arch = "wasm32", target_feature = "simd128")))]
        {
            Some(false)
        }
    }

    #[inline(always)]
    fn summon() -> Option<Self> {
        #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
        {
            #[cfg(feature = "testable_dispatch")]
            if WASM128_DISABLED.load(Ordering::Relaxed) {
                return None;
            }
            // The required wasm features are compile-time enabled; a
            // runtime that validated this module supports them (wasm has
            // no runtime feature detection). wasm32 `#[target_feature]`
            // functions are safe to call, so no `unsafe` is needed.
            Some(Self::from_context())
        }
        #[cfg(not(all(target_arch = "wasm32", target_feature = "simd128")))]
        {
            None
        }
    }
}

impl Wasm128Token {
    /// Construct a WASM SIMD128 proof from the caller's statically proven
    /// feature context.
    ///
    /// Rust permits safe calls to `#[target_feature]` functions from any
    /// WASM context: the engine validates the required instructions when the
    /// module is loaded, so a module that runs at all has the features.
    ///
    /// No runtime detection happens here, so this also bypasses
    /// process-wide token disabling (including `testable_dispatch`):
    /// the caller's feature context is already the proof. Use
    /// [`SimdToken::summon`](crate::SimdToken::summon) when the
    /// features have to be detected at runtime.
    ///
    /// Being a `#[target_feature]` function, this cannot be coerced to
    /// a safe function pointer — there would be no call site left for
    /// rustc to check.
    ///
    /// # Safety
    ///
    /// When called through an `unsafe` block, the caller must ensure
    /// every feature in this tier is available on the executing CPU.
    /// Safe calls have that obligation discharged by the compiler.
    #[inline]
    #[target_feature(enable = "simd128")]
    pub fn from_context() -> Self {
        Self { _private: () }
    }

    /// Deprecated alias for [`Wasm128Token::from_context`].
    ///
    /// Identical in every respect — same `#[target_feature]` gate, same
    /// safe-call rules. The name predates the compiler-checked design
    /// and describes only the `unsafe` half of it.
    ///
    /// # Safety
    ///
    /// Identical to [`Wasm128Token::from_context`]: when called through an
    /// `unsafe` block, the caller must ensure every feature in this
    /// tier is available on the executing CPU. Safe calls have that
    /// obligation discharged by the compiler.
    #[deprecated(
        since = "0.9.29",
        note = "Renamed to from_context() — the constructor is checked against the caller's target-feature context"
    )]
    #[inline]
    #[target_feature(enable = "simd128")]
    pub fn forge_token_dangerously() -> Self {
        // Matching features: a safe call inside the same region.
        Self::from_context()
    }
}

impl Wasm128Token {
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
    /// - `Wasm128RelaxedToken`
    /// - `Wasm128RelaxedFmaToken`
    #[allow(clippy::needless_return)]
    pub fn dangerously_disable_token_process_wide(
        disabled: bool,
    ) -> Result<(), crate::tokens::CompileTimeGuaranteedError> {
        #[cfg(all(target_feature = "simd128", not(feature = "testable_dispatch")))]
        {
            let _ = disabled;
            return Err(crate::tokens::CompileTimeGuaranteedError {
                token_name: Self::NAME,
                target_features: Self::TARGET_FEATURES,
                disable_flags: Self::DISABLE_TARGET_FEATURES,
            });
        }
        #[cfg(not(all(target_feature = "simd128", not(feature = "testable_dispatch"))))]
        {
            WASM128_DISABLED.store(disabled, Ordering::Relaxed);
            let v = if disabled { 1 } else { 0 };
            WASM128_CACHE.store(v, Ordering::Relaxed);
            WASM128_RELAXED_DISABLED.store(disabled, Ordering::Relaxed);
            WASM128_RELAXED_CACHE.store(v, Ordering::Relaxed);
            WASM128_RELAXED_FMA_DISABLED.store(disabled, Ordering::Relaxed);
            WASM128_RELAXED_FMA_CACHE.store(v, Ordering::Relaxed);
            Ok(())
        }
    }

    /// Check if this token has been manually disabled process-wide.
    ///
    /// Returns `Err` when all required features are compile-time enabled.
    #[inline(always)]
    #[allow(clippy::needless_return)]
    pub fn manually_disabled() -> Result<bool, crate::tokens::CompileTimeGuaranteedError> {
        #[cfg(all(target_feature = "simd128", not(feature = "testable_dispatch")))]
        {
            return Err(crate::tokens::CompileTimeGuaranteedError {
                token_name: Self::NAME,
                target_features: Self::TARGET_FEATURES,
                disable_flags: Self::DISABLE_TARGET_FEATURES,
            });
        }
        #[cfg(not(all(target_feature = "simd128", not(feature = "testable_dispatch"))))]
        {
            Ok(WASM128_DISABLED.load(Ordering::Relaxed))
        }
    }
}

/// Proof that WASM Relaxed SIMD is available.
///
/// Relaxed SIMD (Wasm 3.0) provides 28 instructions that trade strict
/// cross-platform determinism for performance: FMA, relaxed lane-select,
/// relaxed min/max, dot products, and relaxed truncation.
///
/// This token proves instruction availability, not single-rounding FMA.
/// Relaxed multiply-add may round once or twice, depending on the engine.
/// Modules containing relaxed instructions require relaxed-SIMD support at load
/// time; select a compatible build with WebAssembly.validate.
/// Stable in Rust since 1.82.
#[derive(Clone, Copy, Debug)]
pub struct Wasm128RelaxedToken {
    _private: (),
}

impl crate::tokens::Sealed for Wasm128RelaxedToken {}

impl SimdToken for Wasm128RelaxedToken {
    const NAME: &'static str = "WASM Relaxed SIMD";
    const TARGET_FEATURES: &'static str = "simd128,relaxed-simd";
    const ENABLE_TARGET_FEATURES: &'static str = "-Ctarget-feature=+simd128,+relaxed-simd";
    const DISABLE_TARGET_FEATURES: &'static str = "-Ctarget-feature=-simd128,-relaxed-simd";

    #[inline]
    fn compiled_with() -> Option<bool> {
        #[cfg(all(
            target_arch = "wasm32",
            target_feature = "simd128",
            target_feature = "relaxed-simd",
            not(feature = "testable_dispatch")
        ))]
        {
            Some(true)
        }
        #[cfg(all(
            target_arch = "wasm32",
            target_feature = "simd128",
            target_feature = "relaxed-simd",
            feature = "testable_dispatch"
        ))]
        {
            None
        }
        #[cfg(not(all(
            target_arch = "wasm32",
            target_feature = "simd128",
            target_feature = "relaxed-simd"
        )))]
        {
            Some(false)
        }
    }

    #[inline(always)]
    fn summon() -> Option<Self> {
        #[cfg(all(
            target_arch = "wasm32",
            target_feature = "simd128",
            target_feature = "relaxed-simd"
        ))]
        {
            #[cfg(feature = "testable_dispatch")]
            if WASM128_RELAXED_DISABLED.load(Ordering::Relaxed) {
                return None;
            }
            // The required wasm features are compile-time enabled; a
            // runtime that validated this module supports them (wasm has
            // no runtime feature detection). wasm32 `#[target_feature]`
            // functions are safe to call, so no `unsafe` is needed.
            Some(Self::from_context())
        }
        #[cfg(not(all(
            target_arch = "wasm32",
            target_feature = "simd128",
            target_feature = "relaxed-simd"
        )))]
        {
            None
        }
    }
}

impl Wasm128RelaxedToken {
    /// Construct a WASM Relaxed SIMD proof from the caller's statically proven
    /// feature context.
    ///
    /// Rust permits safe calls to `#[target_feature]` functions from any
    /// WASM context: the engine validates the required instructions when the
    /// module is loaded, so a module that runs at all has the features.
    ///
    /// No runtime detection happens here, so this also bypasses
    /// process-wide token disabling (including `testable_dispatch`):
    /// the caller's feature context is already the proof. Use
    /// [`SimdToken::summon`](crate::SimdToken::summon) when the
    /// features have to be detected at runtime.
    ///
    /// Being a `#[target_feature]` function, this cannot be coerced to
    /// a safe function pointer — there would be no call site left for
    /// rustc to check.
    ///
    /// # Safety
    ///
    /// When called through an `unsafe` block, the caller must ensure
    /// every feature in this tier is available on the executing CPU.
    /// Safe calls have that obligation discharged by the compiler.
    #[inline]
    #[target_feature(enable = "simd128,relaxed-simd")]
    pub fn from_context() -> Self {
        Self { _private: () }
    }

    /// Deprecated alias for [`Wasm128RelaxedToken::from_context`].
    ///
    /// Identical in every respect — same `#[target_feature]` gate, same
    /// safe-call rules. The name predates the compiler-checked design
    /// and describes only the `unsafe` half of it.
    ///
    /// # Safety
    ///
    /// Identical to [`Wasm128RelaxedToken::from_context`]: when called through an
    /// `unsafe` block, the caller must ensure every feature in this
    /// tier is available on the executing CPU. Safe calls have that
    /// obligation discharged by the compiler.
    #[deprecated(
        since = "0.9.29",
        note = "Renamed to from_context() — the constructor is checked against the caller's target-feature context"
    )]
    #[inline]
    #[target_feature(enable = "simd128,relaxed-simd")]
    pub fn forge_token_dangerously() -> Self {
        // Matching features: a safe call inside the same region.
        Self::from_context()
    }
}

impl Wasm128RelaxedToken {
    /// Extract a Wasm128Token — guaranteed because WASM Relaxed SIMD implies WASM SIMD128.
    ///
    /// Zero-cost: compiles away entirely.
    #[inline(always)]
    pub fn wasm128(self) -> Wasm128Token {
        // Holding `self` proves WASM Relaxed SIMD's feature set, a superset
        // of WASM SIMD128's; wasm32 target-feature calls are safe.
        Wasm128Token::from_context()
    }
}

impl Wasm128RelaxedToken {
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
    /// - `Wasm128RelaxedFmaToken`
    #[allow(clippy::needless_return)]
    pub fn dangerously_disable_token_process_wide(
        disabled: bool,
    ) -> Result<(), crate::tokens::CompileTimeGuaranteedError> {
        #[cfg(all(
            target_feature = "simd128",
            target_feature = "relaxed-simd",
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
            target_feature = "simd128",
            target_feature = "relaxed-simd",
            not(feature = "testable_dispatch")
        )))]
        {
            WASM128_RELAXED_DISABLED.store(disabled, Ordering::Relaxed);
            let v = if disabled { 1 } else { 0 };
            WASM128_RELAXED_CACHE.store(v, Ordering::Relaxed);
            WASM128_RELAXED_FMA_DISABLED.store(disabled, Ordering::Relaxed);
            WASM128_RELAXED_FMA_CACHE.store(v, Ordering::Relaxed);
            Ok(())
        }
    }

    /// Check if this token has been manually disabled process-wide.
    ///
    /// Returns `Err` when all required features are compile-time enabled.
    #[inline(always)]
    #[allow(clippy::needless_return)]
    pub fn manually_disabled() -> Result<bool, crate::tokens::CompileTimeGuaranteedError> {
        #[cfg(all(
            target_feature = "simd128",
            target_feature = "relaxed-simd",
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
            target_feature = "simd128",
            target_feature = "relaxed-simd",
            not(feature = "testable_dispatch")
        )))]
        {
            Ok(WASM128_RELAXED_DISABLED.load(Ordering::Relaxed))
        }
    }
}

/// Proof that WASM Relaxed SIMD is available AND both `f32x4.relaxed_madd` and `f64x2.relaxed_madd` fuse to single-rounding FMA on the executing engine.
///
/// Unlike [`Wasm128RelaxedToken`], which only proves the `+relaxed-simd` instruction set is enabled, `Wasm128RelaxedFmaToken::summon()` runs a cached one-time runtime probe verifying that all lanes of `f32x4.relaxed_madd` and `f64x2.relaxed_madd` round once rather than twice.
///
/// When available (e.g., Wasmtime/V8/SpiderMonkey/JavaScriptCore on x86-64 with FMA or AArch64), `magetypes` float vectors parameterized by `Wasm128RelaxedFmaToken` emit bare `f32x4.relaxed_madd` / `f64x2.relaxed_madd` for `.mul_add_portable()` and `.mul_sub_portable()` with zero per-op probe loads or fallback branches. On deterministic or non-fusing engines, `summon()` returns `None` so dispatch falls back to `Wasm128RelaxedToken` or `Wasm128Token`.
#[derive(Clone, Copy, Debug)]
pub struct Wasm128RelaxedFmaToken {
    _private: (),
}

impl crate::tokens::Sealed for Wasm128RelaxedFmaToken {}

impl SimdToken for Wasm128RelaxedFmaToken {
    const NAME: &'static str = "WASM Relaxed FMA";
    const TARGET_FEATURES: &'static str = "simd128,relaxed-simd";
    const ENABLE_TARGET_FEATURES: &'static str = "-Ctarget-feature=+simd128,+relaxed-simd";
    const DISABLE_TARGET_FEATURES: &'static str = "-Ctarget-feature=-simd128,-relaxed-simd";

    #[inline]
    fn compiled_with() -> Option<bool> {
        #[cfg(all(
            target_arch = "wasm32",
            target_feature = "simd128",
            target_feature = "relaxed-simd"
        ))]
        {
            None
        }
        #[cfg(not(all(
            target_arch = "wasm32",
            target_feature = "simd128",
            target_feature = "relaxed-simd"
        )))]
        {
            Some(false)
        }
    }

    #[inline(always)]
    fn summon() -> Option<Self> {
        #[cfg(all(
            target_arch = "wasm32",
            target_feature = "simd128",
            target_feature = "relaxed-simd"
        ))]
        {
            match WASM128_RELAXED_FMA_CACHE.load(Ordering::Relaxed) {
                // 2 is only ever stored by wasm128_relaxed_fma_detect() after confirming
                // single-rounding f32x4 and f64x2 FMA fusion across all lanes.
                2 => Some(Self::from_context()),
                1 => None,
                _ => wasm128_relaxed_fma_detect(),
            }
        }
        #[cfg(not(all(
            target_arch = "wasm32",
            target_feature = "simd128",
            target_feature = "relaxed-simd"
        )))]
        {
            None
        }
    }
}

impl Wasm128RelaxedFmaToken {
    /// Construct a WASM Relaxed FMA proof from the caller's statically proven
    /// feature context.
    ///
    /// Rust permits safe calls to `#[target_feature]` functions from any
    /// WASM context. Inside an `#[attune(wasm128_relaxed_fma)]` region the
    /// single-rounding FMA contract was already checked by the entry probe;
    /// use [`SimdToken::summon`](crate::SimdToken::summon) at dispatch boundaries
    /// to run and cache the runtime fusion probe.
    ///
    /// No runtime detection happens here, so this also bypasses
    /// process-wide token disabling (including `testable_dispatch`):
    /// the caller's feature context is already the proof. Use
    /// [`SimdToken::summon`](crate::SimdToken::summon) when the
    /// features have to be detected at runtime.
    ///
    /// Being a `#[target_feature]` function, this cannot be coerced to
    /// a safe function pointer — there would be no call site left for
    /// rustc to check.
    ///
    /// # Safety
    ///
    /// When called through an `unsafe` block, the caller must ensure
    /// every feature in this tier is available on the executing CPU.
    /// Safe calls have that obligation discharged by the compiler.
    #[inline]
    #[target_feature(enable = "simd128,relaxed-simd")]
    pub fn from_context() -> Self {
        Self { _private: () }
    }

    /// Deprecated alias for [`Wasm128RelaxedFmaToken::from_context`].
    ///
    /// Identical in every respect — same `#[target_feature]` gate, same
    /// safe-call rules. The name predates the compiler-checked design
    /// and describes only the `unsafe` half of it.
    ///
    /// # Safety
    ///
    /// Identical to [`Wasm128RelaxedFmaToken::from_context`]: when called through an
    /// `unsafe` block, the caller must ensure every feature in this
    /// tier is available on the executing CPU. Safe calls have that
    /// obligation discharged by the compiler.
    #[deprecated(
        since = "0.9.29",
        note = "Renamed to from_context() — the constructor is checked against the caller's target-feature context"
    )]
    #[inline]
    #[target_feature(enable = "simd128,relaxed-simd")]
    pub fn forge_token_dangerously() -> Self {
        // Matching features: a safe call inside the same region.
        Self::from_context()
    }
}

impl Wasm128RelaxedFmaToken {
    /// Extract a Wasm128RelaxedToken — guaranteed because WASM Relaxed FMA implies WASM Relaxed SIMD.
    ///
    /// Zero-cost: compiles away entirely.
    #[inline(always)]
    pub fn wasm128_relaxed(self) -> Wasm128RelaxedToken {
        // Holding `self` proves WASM Relaxed FMA's feature set, a superset
        // of WASM Relaxed SIMD's; wasm32 target-feature calls are safe.
        Wasm128RelaxedToken::from_context()
    }
    /// Extract a Wasm128Token — guaranteed because WASM Relaxed FMA implies WASM SIMD128.
    ///
    /// Zero-cost: compiles away entirely.
    #[inline(always)]
    pub fn wasm128(self) -> Wasm128Token {
        // Holding `self` proves WASM Relaxed FMA's feature set, a superset
        // of WASM SIMD128's; wasm32 target-feature calls are safe.
        Wasm128Token::from_context()
    }
}

impl Wasm128RelaxedFmaToken {
    /// Disable this token process-wide for testing and benchmarking.
    ///
    /// When disabled, `summon()` returns `None` even on engines where
    /// the runtime probe succeeds. Pass `false` to re-enable probing.
    pub fn dangerously_disable_token_process_wide(
        disabled: bool,
    ) -> Result<(), crate::tokens::CompileTimeGuaranteedError> {
        WASM128_RELAXED_FMA_DISABLED.store(disabled, Ordering::Relaxed);
        let v = if disabled { 1 } else { 0 };
        WASM128_RELAXED_FMA_CACHE.store(v, Ordering::Relaxed);
        Ok(())
    }

    /// Check if this token has been manually disabled process-wide.
    #[inline(always)]
    pub fn manually_disabled() -> Result<bool, crate::tokens::CompileTimeGuaranteedError> {
        Ok(WASM128_RELAXED_FMA_DISABLED.load(Ordering::Relaxed))
    }
}
#[cfg(all(
    target_arch = "wasm32",
    target_feature = "simd128",
    target_feature = "relaxed-simd"
))]
#[cold]
#[inline(never)]
#[target_feature(enable = "simd128,relaxed-simd")]
fn wasm128_relaxed_fma_detect() -> Option<Wasm128RelaxedFmaToken> {
    use core::arch::wasm32;
    use core::hint::black_box;
    // (1 + 2^-12)^2 - (1 + 2^-11) == 2^-24 with single rounding; two roundings give 0.
    let x32 = wasm32::f32x4_splat(black_box(1.0_f32 + 1.0 / 4096.0));
    let c32 = wasm32::f32x4_splat(black_box(-(1.0_f32 + 1.0 / 2048.0)));
    let r32 = wasm32::f32x4_relaxed_madd(x32, x32, c32);
    let exp32 = 1.0_f32 / 16_777_216.0;
    let f32_fuses = wasm32::f32x4_extract_lane::<0>(r32) == exp32
        && wasm32::f32x4_extract_lane::<1>(r32) == exp32
        && wasm32::f32x4_extract_lane::<2>(r32) == exp32
        && wasm32::f32x4_extract_lane::<3>(r32) == exp32;
    // (1 + 2^-27)^2 - (1 + 2^-26) == 2^-54 with single rounding; two roundings give 0.
    let e64 = 1.0_f64 / 134_217_728.0;
    let x64 = wasm32::f64x2_splat(black_box(1.0_f64 + e64));
    let c64 = wasm32::f64x2_splat(black_box(-(1.0_f64 + 2.0 * e64)));
    let r64 = wasm32::f64x2_relaxed_madd(x64, x64, c64);
    let exp64 = e64 * e64;
    let f64_fuses = wasm32::f64x2_extract_lane::<0>(r64) == exp64
        && wasm32::f64x2_extract_lane::<1>(r64) == exp64;
    let available = f32_fuses && f64_fuses;
    let _ = WASM128_RELAXED_FMA_CACHE.compare_exchange(
        0,
        if available { 2 } else { 1 },
        Ordering::Relaxed,
        Ordering::Relaxed,
    );
    if available && !WASM128_RELAXED_FMA_DISABLED.load(Ordering::Relaxed) {
        // Runtime probe just confirmed single-rounding f32x4 and f64x2
        // relaxed_madd fusion across all lanes on this engine.
        Some(Wasm128RelaxedFmaToken::from_context())
    } else {
        None
    }
}

impl Wasm128Token {
    #[doc(hidden)]
    pub const __ARCHMAGE_TIER_TAG: u32 = 0x1E0DF26B;

    #[doc(hidden)]
    pub const __ARCHMAGE_ASSERT_TIER_1E0DF26B: () =
        [()][!(Self::__ARCHMAGE_TIER_TAG == 0x1E0DF26B) as usize];
}

impl Wasm128RelaxedToken {
    #[doc(hidden)]
    pub const __ARCHMAGE_TIER_TAG: u32 = 0x821D5452;

    #[doc(hidden)]
    pub const __ARCHMAGE_ASSERT_TIER_821D5452: () =
        [()][!(Self::__ARCHMAGE_TIER_TAG == 0x821D5452) as usize];
}

impl Wasm128RelaxedFmaToken {
    #[doc(hidden)]
    pub const __ARCHMAGE_TIER_TAG: u32 = 0x735B647C;

    #[doc(hidden)]
    pub const __ARCHMAGE_ASSERT_TIER_735B647C: () =
        [()][!(Self::__ARCHMAGE_TIER_TAG == 0x735B647C) as usize];
}

#[allow(deprecated)]
impl Has128BitSimd for Wasm128Token {}
#[allow(deprecated)]
impl Has128BitSimd for Wasm128RelaxedToken {}
#[allow(deprecated)]
impl Has128BitSimd for Wasm128RelaxedFmaToken {}
