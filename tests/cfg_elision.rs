//! Tests that verify `#[cfg]` elision is working correctly.
//!
//! These tests ensure that:
//! 1. Token stubs are used on the wrong platform (summon → None)
//! 2. Feature-gated tokens are only available when features are enabled
//! 3. Platform-specific code is properly elided
//!
//! ## How cfg elision works
//!
//! `#[cfg(...)]` is a COMPILE-TIME check. Code guarded by cfg that doesn't
//! match is completely removed from the binary — it's not compiled at all.
//!
//! This is different from runtime checks like `if cfg!(...)` or `Token::summon()`.

#![allow(dead_code, unused_variables, clippy::assertions_on_constants)]

// =============================================================================
// TEST: Cross-platform token stubs return None
// =============================================================================
//
// On x86_64, ARM tokens should be stubs that always return None.
// On aarch64, x86 tokens should be stubs that always return None.
// This verifies the stub modules are being compiled in.

#[cfg(target_arch = "x86_64")]
mod x86_stub_tests {
    #[cfg(target_feature = "simd128")]
    use archmage::Wasm128Token;
    use archmage::{NeonAesToken, NeonCrcToken, NeonSha3Token, NeonToken, SimdToken};

    #[test]
    fn neon_token_is_stub_on_x86() {
        // On x86, NeonToken::summon() must return None (it's a stub)
        assert!(
            NeonToken::summon().is_none(),
            "NeonToken::summon() should return None on x86_64 (stub)"
        );
    }

    #[test]
    fn neon_aes_token_is_stub_on_x86() {
        assert!(
            NeonAesToken::summon().is_none(),
            "NeonAesToken::summon() should return None on x86_64 (stub)"
        );
    }

    #[test]
    fn neon_sha3_token_is_stub_on_x86() {
        assert!(
            NeonSha3Token::summon().is_none(),
            "NeonSha3Token::summon() should return None on x86_64 (stub)"
        );
    }

    #[test]
    fn neon_crc_token_is_stub_on_x86() {
        assert!(
            NeonCrcToken::summon().is_none(),
            "NeonCrcToken::summon() should return None on x86_64 (stub)"
        );
    }

    // WASM token doesn't exist on x86 unless simd128 target feature is set
    // (which it never is for native x86 builds)
}

#[cfg(target_arch = "aarch64")]
mod arm_stub_tests {
    #[cfg(feature = "avx512")]
    use archmage::{Avx512Fp16Token, X64V4Token, X64V4xToken};
    use archmage::{SimdToken, X64V2Token, X64V3Token};

    #[test]
    fn x64v2_token_is_stub_on_arm() {
        assert!(
            X64V2Token::summon().is_none(),
            "X64V2Token::summon() should return None on aarch64 (stub)"
        );
    }

    #[test]
    fn x64v3_token_is_stub_on_arm() {
        assert!(
            X64V3Token::summon().is_none(),
            "X64V3Token::summon() should return None on aarch64 (stub)"
        );
    }

    #[cfg(feature = "avx512")]
    #[test]
    fn x64v4_token_is_stub_on_arm() {
        assert!(
            X64V4Token::summon().is_none(),
            "X64V4Token::summon() should return None on aarch64 (stub)"
        );
    }
}

#[cfg(target_arch = "aarch64")]
mod arm_real_tests {
    use archmage::{NeonToken, SimdToken};

    #[test]
    fn neon_token_is_real_on_arm() {
        // On ARM64, NEON is always available
        assert!(
            NeonToken::summon().is_some(),
            "NeonToken::summon() should return Some on aarch64 (NEON is baseline)"
        );
    }
}

// =============================================================================
// TEST: Feature-gated tokens only exist with feature
// =============================================================================
//
// AVX-512 tokens require the `avx512` cargo feature. Without it, they shouldn't
// even be importable (this is verified by the compile_fail tests).
//
// Here we verify they DO exist when the feature is enabled.

#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
mod avx512_feature_tests {
    use archmage::{Avx512Fp16Token, SimdToken, X64V4Token, X64V4xToken};

    #[test]
    fn avx512_tokens_exist_with_feature() {
        // These should compile — tokens exist when feature is enabled
        let _v4: Option<X64V4Token> = X64V4Token::summon();
        let _v4x: Option<X64V4xToken> = X64V4xToken::summon();
        let _fp16: Option<Avx512Fp16Token> = Avx512Fp16Token::summon();

        // Whether they return Some depends on the actual CPU
        // Here we just verify the types exist and methods are callable
    }

    #[test]
    fn avx512_token_hierarchy() {
        // If we have a V4 token, we should be able to extract V3 and V2
        if let Some(v4) = X64V4Token::summon() {
            let v3 = v4.v3();
            let v2 = v3.v2();
            // Types exist and extraction works
            let _ = v2;
        }
    }
}

// =============================================================================
// TEST: Const assertions for cfg correctness
// =============================================================================
//
// Use const evaluation to verify cfg is working at compile time.

#[cfg(target_arch = "x86_64")]
const IS_X86: bool = true;
#[cfg(not(target_arch = "x86_64"))]
const IS_X86: bool = false;

#[cfg(target_arch = "aarch64")]
const IS_ARM: bool = true;
#[cfg(not(target_arch = "aarch64"))]
const IS_ARM: bool = false;

#[cfg(target_arch = "wasm32")]
const IS_WASM: bool = true;
#[cfg(not(target_arch = "wasm32"))]
const IS_WASM: bool = false;

// Exactly one platform should be true (for supported platforms)
#[cfg(any(
    target_arch = "x86_64",
    target_arch = "aarch64",
    target_arch = "wasm32"
))]
const _: () = {
    let count = IS_X86 as u8 + IS_ARM as u8 + IS_WASM as u8;
    assert!(count == 1, "Exactly one platform const should be true");
};

#[test]
fn platform_consts_are_correct() {
    #[cfg(target_arch = "x86_64")]
    {
        assert!(IS_X86);
        assert!(!IS_ARM);
        assert!(!IS_WASM);
    }

    #[cfg(target_arch = "aarch64")]
    {
        assert!(!IS_X86);
        assert!(IS_ARM);
        assert!(!IS_WASM);
    }

    #[cfg(target_arch = "wasm32")]
    {
        assert!(!IS_X86);
        assert!(!IS_ARM);
        assert!(IS_WASM);
    }
}

// =============================================================================
// TEST: Verify elided code doesn't contribute to binary
// =============================================================================
//
// This test uses a trick: define platform-specific static data, then verify
// only the right one exists at runtime.

#[cfg(target_arch = "x86_64")]
static PLATFORM_NAME: &str = "x86_64";

#[cfg(target_arch = "aarch64")]
static PLATFORM_NAME: &str = "aarch64";

#[cfg(target_arch = "wasm32")]
static PLATFORM_NAME: &str = "wasm32";

#[cfg(not(any(
    target_arch = "x86_64",
    target_arch = "aarch64",
    target_arch = "wasm32"
)))]
static PLATFORM_NAME: &str = "unknown";

#[test]
fn platform_name_matches_cfg() {
    #[cfg(target_arch = "x86_64")]
    assert_eq!(PLATFORM_NAME, "x86_64");

    #[cfg(target_arch = "aarch64")]
    assert_eq!(PLATFORM_NAME, "aarch64");

    #[cfg(target_arch = "wasm32")]
    assert_eq!(PLATFORM_NAME, "wasm32");
}

// =============================================================================
// TEST: Verify cfg_if!-style branching
// =============================================================================
//
// This tests that we can use cfg to select different implementations.

fn get_simd_width() -> usize {
    #[cfg(target_arch = "x86_64")]
    {
        // x86 can do 256-bit (AVX2) or 512-bit (AVX-512)
        #[cfg(feature = "avx512")]
        return 512;
        #[cfg(not(feature = "avx512"))]
        return 256;
    }

    #[cfg(target_arch = "aarch64")]
    {
        // ARM NEON is 128-bit
        return 128;
    }

    #[cfg(target_arch = "wasm32")]
    {
        // WASM SIMD is 128-bit
        return 128;
    }

    #[cfg(not(any(
        target_arch = "x86_64",
        target_arch = "aarch64",
        target_arch = "wasm32"
    )))]
    {
        return 0; // No SIMD
    }
}

#[test]
fn simd_width_is_correct_for_platform() {
    let width = get_simd_width();

    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    assert_eq!(
        width, 512,
        "x86_64 with avx512 feature should report 512-bit"
    );

    #[cfg(all(target_arch = "x86_64", not(feature = "avx512")))]
    assert_eq!(
        width, 256,
        "x86_64 without avx512 feature should report 256-bit"
    );

    #[cfg(target_arch = "aarch64")]
    assert_eq!(width, 128, "aarch64 should report 128-bit");

    #[cfg(target_arch = "wasm32")]
    assert_eq!(width, 128, "wasm32 should report 128-bit");
}

// =============================================================================
// TEST: Verify dead code elimination via type_name
// =============================================================================
//
// This tests that generic instantiation only happens for the right platform.

#[cfg(target_arch = "x86_64")]
mod x86_type_tests {
    use archmage::{Desktop64, X64V3Token};
    use core::any::type_name;

    #[test]
    fn desktop64_is_x64v3token() {
        // Desktop64 is a type alias for X64V3Token
        assert_eq!(
            type_name::<Desktop64>(),
            type_name::<X64V3Token>(),
            "Desktop64 should be an alias for X64V3Token"
        );
    }
}

#[cfg(target_arch = "aarch64")]
mod arm_type_tests {
    use archmage::{Arm64, NeonToken};
    use core::any::type_name;

    #[test]
    fn arm64_is_neon_token() {
        // Arm64 is a type alias for NeonToken
        assert_eq!(
            type_name::<Arm64>(),
            type_name::<NeonToken>(),
            "Arm64 should be an alias for NeonToken"
        );
    }
}

// =============================================================================
// TEST: Verify runtime vs compile-time feature detection
// =============================================================================
//
// This demonstrates the difference between cfg (compile-time) and summon (runtime).

#[cfg(target_arch = "x86_64")]
mod runtime_vs_compiletime {
    use archmage::{SimdToken, X64V3Token};

    /// This function is ALWAYS compiled on x86_64 (cfg allows it).
    /// But the token might not be available at RUNTIME (old CPU).
    #[test]
    fn cfg_vs_runtime_detection() {
        // Compile-time: this code EXISTS because we're on x86_64
        // (cfg(target_arch = "x86_64") passed)

        // Runtime: check if CPU actually supports AVX2+FMA
        let runtime_available = X64V3Token::summon().is_some();

        // On modern CPUs (Haswell 2013+), this should be true
        // On very old CPUs, it might be false
        // The point is: cfg let us COMPILE the code, summon() checks if we can RUN it

        println!(
            "X64V3Token (AVX2+FMA) runtime available: {}",
            runtime_available
        );

        // We can't assert runtime_available is true because CI might run on old hardware
        // But we CAN assert that the check itself works
        let _ = runtime_available;
    }

    /// This const is evaluated at COMPILE time, not runtime.
    /// It's true because we're compiling for x86_64.
    const COMPILED_FOR_X86: bool = cfg!(target_arch = "x86_64");

    #[test]
    fn cfg_macro_is_compile_time() {
        // cfg!() is a compile-time check that returns a bool
        // It's different from #[cfg(...)] which conditionally includes code

        assert!(
            COMPILED_FOR_X86,
            "cfg!(target_arch = \"x86_64\") should be true"
        );

        // This is evaluated at compile time, not runtime
        const HAS_AVX512_FEATURE: bool = cfg!(feature = "avx512");

        #[cfg(feature = "avx512")]
        assert!(HAS_AVX512_FEATURE);

        #[cfg(not(feature = "avx512"))]
        assert!(!HAS_AVX512_FEATURE);
    }
}

// =============================================================================
// TEST: Verify module-level cfg elision
// =============================================================================
//
// Entire modules can be elided with cfg. This tests that.

#[cfg(target_arch = "x86_64")]
mod x86_only_module {
    pub const VALUE: u32 = 0x86;
}

#[cfg(target_arch = "aarch64")]
mod arm_only_module {
    pub const VALUE: u32 = 0xA4AA; // "NEON" in spirit
}

#[test]
fn module_elision_works() {
    #[cfg(target_arch = "x86_64")]
    assert_eq!(x86_only_module::VALUE, 0x86);

    #[cfg(target_arch = "aarch64")]
    assert_eq!(arm_only_module::VALUE, 0xA4AA);

    // On x86, arm_only_module doesn't exist (elided)
    // On ARM, x86_only_module doesn't exist (elided)
    // This test passes on both because we only access the right one
}

// =============================================================================
// TEST: Function elision with different signatures
// =============================================================================
//
// Platform-specific functions can have completely different signatures.
// cfg ensures only the right one is compiled.

#[cfg(target_arch = "x86_64")]
fn platform_specific_fn() -> &'static str {
    "x86_64"
}

#[cfg(target_arch = "aarch64")]
fn platform_specific_fn() -> &'static str {
    "aarch64"
}

#[cfg(target_arch = "wasm32")]
fn platform_specific_fn() -> &'static str {
    "wasm32"
}

#[cfg(not(any(
    target_arch = "x86_64",
    target_arch = "aarch64",
    target_arch = "wasm32"
)))]
fn platform_specific_fn() -> &'static str {
    "unsupported"
}

#[test]
fn function_elision_selects_right_impl() {
    let result = platform_specific_fn();

    #[cfg(target_arch = "x86_64")]
    assert_eq!(result, "x86_64");

    #[cfg(target_arch = "aarch64")]
    assert_eq!(result, "aarch64");

    #[cfg(target_arch = "wasm32")]
    assert_eq!(result, "wasm32");
}

// =============================================================================
// TEST: Target-feature compile-time elision
// =============================================================================
//
// When compiling with `-C target-cpu=haswell` or `-C target-feature=+avx2,+fma`,
// the compiler KNOWS at compile time that AVX2+FMA are available.
// In this case, lower-tier (SSE) and scalar variants can be elided entirely.
//
// This is what `#[magetypes]` should do: generate variants but let cfg elide
// the ones that are statically known to be unnecessary.

/// Check if AVX2 is a compile-time target feature.
/// This is true when compiling with `-C target-cpu=haswell` or similar.
const AVX2_COMPILETIME: bool = cfg!(target_feature = "avx2");

/// Check if AVX-512 is a compile-time target feature.
const AVX512_COMPILETIME: bool = cfg!(target_feature = "avx512f");

/// Check if NEON is a compile-time target feature.
/// On aarch64, NEON is always available at compile time.
const NEON_COMPILETIME: bool = cfg!(target_feature = "neon");

#[test]
fn detect_compiletime_features() {
    println!("Compile-time AVX2: {}", AVX2_COMPILETIME);
    println!("Compile-time AVX-512: {}", AVX512_COMPILETIME);
    println!("Compile-time NEON: {}", NEON_COMPILETIME);

    // On default x86_64 builds without target-cpu, AVX2 is NOT compile-time
    // On aarch64, NEON IS compile-time (it's baseline)
    #[cfg(target_arch = "aarch64")]
    assert!(NEON_COMPILETIME, "NEON should be compile-time on aarch64");
}

/// Demonstrates the elision pattern for dispatch macros.
///
/// When a feature is compile-time available, the runtime check is unnecessary
/// and LLVM will optimize it away. But we can also use cfg to not even
/// generate the check.
#[cfg(target_arch = "x86_64")]
mod compiletime_dispatch_elision {
    use archmage::{SimdToken, X64V3Token};

    /// Dispatch function that uses compile-time knowledge.
    ///
    /// If AVX2 is a compile-time target feature, the runtime check is elided.
    pub fn sum_with_elision(data: &[f32]) -> f32 {
        // If AVX2 is compile-time available, this branch is always taken
        // and LLVM will eliminate the else branch entirely
        #[cfg(target_feature = "avx2")]
        {
            // AVX2 is guaranteed at compile time - no runtime check needed!
            // This is the "fast path" that completely skips summon()
            // summon().unwrap() is safe here because compiled_with() == Some(true)
            let token = X64V3Token::summon().unwrap();
            return sum_avx2(token, data);
        }

        // Only compiled when AVX2 is NOT a compile-time feature
        #[cfg(not(target_feature = "avx2"))]
        {
            // Runtime check needed
            if let Some(token) = X64V3Token::summon() {
                return sum_avx2(token, data);
            }
            sum_scalar(data)
        }
    }

    #[archmage::arcane]
    fn sum_avx2(_token: X64V3Token, data: &[f32]) -> f32 {
        // Real implementation would use SIMD
        data.iter().sum()
    }

    fn sum_scalar(data: &[f32]) -> f32 {
        data.iter().sum()
    }

    #[test]
    fn test_compiletime_dispatch() {
        let data = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let result = sum_with_elision(&data);
        assert_eq!(result, 36.0);

        // This test always passes, but the codegen differs:
        // - With -C target-cpu=haswell: Only AVX2 path is compiled
        // - Without: Both paths compiled, runtime dispatch
    }
}

/// This test verifies that `cfg!(target_feature = "...")` correctly reflects
/// compile-time knowledge, which is what `#[magetypes]`/`incant!` should use.
#[test]
fn cfg_macro_detects_target_features() {
    // These are compile-time constants
    let has_sse2 = cfg!(target_feature = "sse2");
    let has_avx = cfg!(target_feature = "avx");
    let has_avx2 = cfg!(target_feature = "avx2");
    let has_fma = cfg!(target_feature = "fma");
    let has_neon = cfg!(target_feature = "neon");

    #[cfg(target_arch = "x86_64")]
    {
        // SSE2 is baseline for x86_64
        assert!(has_sse2, "SSE2 should be compile-time on x86_64");

        println!("Compile-time AVX: {}", has_avx);
        println!("Compile-time AVX2: {}", has_avx2);
        println!("Compile-time FMA: {}", has_fma);

        // AVX/AVX2/FMA depend on build flags
        // -C target-cpu=native on Haswell+ would make these true
        // Default build would have them false
    }

    #[cfg(target_arch = "aarch64")]
    {
        assert!(has_neon, "NEON should be compile-time on aarch64");
    }
}

// =============================================================================
// TEST: Expected `#[magetypes]` elision behavior (spec compliance)
// =============================================================================
//
// The `#[magetypes]` macro SHOULD generate code like this:
//
// ```rust
// // Generated by #[magetypes]
// #[cfg(all(target_arch = "x86_64", not(target_feature = "avx2")))]
// pub fn kernel_scalar(...) { ... }
//
// #[cfg(all(target_arch = "x86_64", not(target_feature = "avx2")))]
// pub fn kernel_v2(...) { ... }
//
// #[cfg(target_arch = "x86_64")]  // Always available on x86
// pub fn kernel_v3(...) { ... }
//
// #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
// pub fn kernel_v4(...) { ... }
// ```
//
// This way:
// - Compiling with `-C target-cpu=haswell`: Only v3 (and v4 if feature) exist
// - Compiling for generic x86_64: All variants exist, runtime dispatch needed
// - Compiling for aarch64: Only neon variant exists

/// Marker to verify elision is working correctly.
///
/// When AVX2 is compile-time, this should NOT be compiled.
#[cfg(all(target_arch = "x86_64", not(target_feature = "avx2")))]
static SCALAR_VARIANT_EXISTS: bool = true;

/// When AVX2 is compile-time, this SHOULD be compiled.
#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
static AVX2_COMPILETIME_PATH: bool = true;

#[cfg(target_arch = "x86_64")]
#[test]
fn verify_elision_markers() {
    // This test verifies the cfg logic is correct

    #[cfg(target_feature = "avx2")]
    {
        // AVX2 is compile-time available
        assert!(AVX2_COMPILETIME_PATH);
        // SCALAR_VARIANT_EXISTS shouldn't even exist here
    }

    #[cfg(not(target_feature = "avx2"))]
    {
        // AVX2 is NOT compile-time available
        assert!(SCALAR_VARIANT_EXISTS);
        // AVX2_COMPILETIME_PATH shouldn't even exist here
    }
}

// =============================================================================
// TEST: compiled_with() returns correct values for token availability
// =============================================================================
//
// compiled_with() should return:
// - Some(true): Binary was compiled with these features enabled (via target_feature)
// - Some(false): Wrong architecture (this token can never be available)
// - None: Might be available, call summon() to check at runtime

#[cfg(target_arch = "x86_64")]
mod compiled_with_tests_x86 {
    #[cfg(feature = "avx512")]
    use archmage::X64V4Token;
    use archmage::{NeonToken, SimdToken, X64V2Token, X64V3Token};

    #[test]
    fn neon_compiled_with_returns_false_on_x86() {
        // NEON can never be available on x86_64, so compiled_with() should return Some(false)
        assert_eq!(
            NeonToken::compiled_with(),
            Some(false),
            "NeonToken::compiled_with() should be Some(false) on x86_64"
        );
    }

    #[test]
    fn x64v2_compiled_with_is_not_false() {
        // X64V2Token on x86_64 should be either Some(true) or None, never Some(false)
        assert_ne!(
            X64V2Token::compiled_with(),
            Some(false),
            "X64V2Token::compiled_with() should NOT be Some(false) on x86_64"
        );
    }

    #[test]
    fn x64v3_compiled_with_is_not_false() {
        // X64V3Token on x86_64 should be either Some(true) or None, never Some(false)
        assert_ne!(
            X64V3Token::compiled_with(),
            Some(false),
            "X64V3Token::compiled_with() should NOT be Some(false) on x86_64"
        );
    }

    #[test]
    #[cfg(feature = "avx512")]
    fn x64v4_compiled_with_is_not_false() {
        // X64V4Token on x86_64 should be either Some(true) or None, never Some(false)
        assert_ne!(
            X64V4Token::compiled_with(),
            Some(false),
            "X64V4Token::compiled_with() should NOT be Some(false) on x86_64"
        );
    }

    // Test that summon().unwrap() is safe when compiled_with() is Some(true)
    #[test]
    #[cfg(target_feature = "avx2")]
    fn summon_unwrap_safe_when_compiled_with_true() {
        // When compiling with -C target-cpu=haswell or similar, AVX2 is compiled_with
        assert_eq!(
            X64V3Token::compiled_with(),
            Some(true),
            "X64V3Token::compiled_with() should be Some(true) when target_feature=avx2"
        );
        // summon().unwrap() is safe and efficient - no runtime check
        let _token = X64V3Token::summon().unwrap();
    }
}

#[cfg(target_arch = "aarch64")]
mod compiled_with_tests_arm {
    use archmage::{NeonToken, SimdToken, X64V2Token, X64V3Token};

    #[test]
    fn x86_compiled_with_returns_false_on_arm() {
        // x86 tokens can never be available on ARM
        assert_eq!(
            X64V2Token::compiled_with(),
            Some(false),
            "X64V2Token::compiled_with() should be Some(false) on aarch64"
        );
        assert_eq!(
            X64V3Token::compiled_with(),
            Some(false),
            "X64V3Token::compiled_with() should be Some(false) on aarch64"
        );
    }

    #[test]
    fn neon_compiled_with_on_arm() {
        // NeonToken requires runtime detection unless compiled with +neon
        // On aarch64 without explicit -Ctarget-feature=+neon, returns None
        let result = NeonToken::compiled_with();
        assert_ne!(
            result,
            Some(false),
            "NeonToken::compiled_with() should NOT be Some(false) on aarch64"
        );
    }
}

// =============================================================================
// SUMMARY
// =============================================================================
//
// These tests verify:
//
// 1. **Stub elision**: Wrong-platform tokens return None (stubs are compiled in)
// 2. **Feature gating**: AVX-512 tokens only exist with `avx512` feature
// 3. **Const assertions**: Compile-time cfg checks work correctly
// 4. **Platform detection**: Exactly one platform is active
// 5. **Module elision**: Entire modules are removed for wrong platforms
// 6. **Function elision**: Platform-specific functions are selected correctly
// 7. **Runtime vs compile-time**: cfg is compile-time, summon() is runtime
// 8. **Target-feature elision**: Lower-tier variants elided when higher is compile-time
// 9. **`#[magetypes]` expected behavior**: Document how the macro should elide variants
// 10. **compiled_with()**: Returns Some(true) for compile-time, Some(false) for wrong arch, None for runtime
