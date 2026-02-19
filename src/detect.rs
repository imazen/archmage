//! Optimized feature detection macros.
//!
//! These macros combine compile-time and runtime feature detection,
//! avoiding redundant runtime checks when features are compile-time known.
//!
//! # Important Limitation
//!
//! `cfg!(target_feature)` is evaluated at **crate level**, not function level.
//! This means the compile-time optimization does NOT work inside functions
//! marked with `#[target_feature(enable = "...")]` unless the feature is
//! also enabled globally via `-C target-feature` or `-C target-cpu`.
//!
//! For eliminating checks inside multiversioned functions, use **tokens**
//! instead - they provide type-level proof that a feature is available.

/// Checks if an x86 CPU feature is available, with compile-time optimization.
///
/// Unlike `is_x86_feature_detected!` from std, this macro first checks
/// `cfg!(target_feature)` at compile time. If the feature is compile-time
/// known (e.g., compiled with `-C target-feature=+avx2` or `-C target-cpu=x86-64-v3`),
/// no runtime check is performed.
///
/// # When Compile-Time Optimization Works
///
/// - ✅ Compiled with `-C target-feature=+avx2`
/// - ✅ Compiled with `-C target-cpu=native` (if CPU has AVX2)
/// - ✅ Compiled with `-C target-cpu=x86-64-v3` (implies AVX2)
/// - ❌ Inside `#[target_feature(enable = "avx2")]` without global flag
/// - ❌ Inside `#[multiversion]` or `#[multiversed]` function variants
///
/// # For Multiversioned Code
///
/// Use tokens instead of this macro to avoid repeated detection:
///
/// ```ignore
/// // BAD: Check happens every time
/// fn process(data: &[f32]) {
///     if is_x86_feature_available!("avx2") {
///         inner(data);  // inner might check AGAIN
///     }
/// }
///
/// // GOOD: Check once, pass token
/// fn process(data: &[f32]) {
///     if let Some(token) = X64V3Token::summon() {
///         inner(token, data);  // inner KNOWS v3 features are available
///     }
/// }
/// ```
///
/// # Example
///
/// ```rust
/// use archmage::is_x86_feature_available;
///
/// // Inside a function compiled with +avx2, this is just `true`
/// // Inside a function without +avx2, this does runtime detection
/// if is_x86_feature_available!("avx2") {
///     println!("AVX2 is available");
/// }
/// ```
///
/// # Supported Features
///
/// All features supported by `is_x86_feature_detected!` are supported:
/// - SSE family: `"sse"`, `"sse2"`, `"sse3"`, `"ssse3"`, `"sse4.1"`, `"sse4.2"`
/// - AVX family: `"avx"`, `"avx2"`, `"avx512f"`, `"avx512bw"`, etc.
/// - Other: `"fma"`, `"bmi1"`, `"bmi2"`, `"popcnt"`, `"lzcnt"`, etc.
#[macro_export]
macro_rules! is_x86_feature_available {
    ("sse") => {
        $crate::__impl_feature_check!("sse")
    };
    ("sse2") => {
        $crate::__impl_feature_check!("sse2")
    };
    ("sse3") => {
        $crate::__impl_feature_check!("sse3")
    };
    ("ssse3") => {
        $crate::__impl_feature_check!("ssse3")
    };
    ("sse4.1") => {
        $crate::__impl_feature_check!("sse4.1")
    };
    ("sse4.2") => {
        $crate::__impl_feature_check!("sse4.2")
    };
    ("avx") => {
        $crate::__impl_feature_check!("avx")
    };
    ("avx2") => {
        $crate::__impl_feature_check!("avx2")
    };
    ("fma") => {
        $crate::__impl_feature_check!("fma")
    };
    ("avx512f") => {
        $crate::__impl_feature_check!("avx512f")
    };
    ("avx512bw") => {
        $crate::__impl_feature_check!("avx512bw")
    };
    ("avx512cd") => {
        $crate::__impl_feature_check!("avx512cd")
    };
    ("avx512dq") => {
        $crate::__impl_feature_check!("avx512dq")
    };
    ("avx512vl") => {
        $crate::__impl_feature_check!("avx512vl")
    };
    ("bmi1") => {
        $crate::__impl_feature_check!("bmi1")
    };
    ("bmi2") => {
        $crate::__impl_feature_check!("bmi2")
    };
    ("popcnt") => {
        $crate::__impl_feature_check!("popcnt")
    };
    ("lzcnt") => {
        $crate::__impl_feature_check!("lzcnt")
    };
    ("pclmulqdq") => {
        $crate::__impl_feature_check!("pclmulqdq")
    };
    ("aes") => {
        $crate::__impl_feature_check!("aes")
    };
    ("sha") => {
        $crate::__impl_feature_check!("sha")
    };
    // Fallback for unknown features - runtime only
    ($feature:tt) => {
        $crate::__impl_runtime_only_check!($feature)
    };
}

/// Implementation macro for feature check with compile-time optimization.
/// Not intended for direct use.
#[doc(hidden)]
#[macro_export]
macro_rules! __impl_feature_check {
    ("sse") => {{
        #[cfg(target_feature = "sse")]
        {
            true
        }
        #[cfg(not(target_feature = "sse"))]
        {
            $crate::__impl_runtime_only_check!("sse")
        }
    }};
    ("sse2") => {{
        #[cfg(target_feature = "sse2")]
        {
            true
        }
        #[cfg(not(target_feature = "sse2"))]
        {
            $crate::__impl_runtime_only_check!("sse2")
        }
    }};
    ("sse3") => {{
        #[cfg(target_feature = "sse3")]
        {
            true
        }
        #[cfg(not(target_feature = "sse3"))]
        {
            $crate::__impl_runtime_only_check!("sse3")
        }
    }};
    ("ssse3") => {{
        #[cfg(target_feature = "ssse3")]
        {
            true
        }
        #[cfg(not(target_feature = "ssse3"))]
        {
            $crate::__impl_runtime_only_check!("ssse3")
        }
    }};
    ("sse4.1") => {{
        #[cfg(target_feature = "sse4.1")]
        {
            true
        }
        #[cfg(not(target_feature = "sse4.1"))]
        {
            $crate::__impl_runtime_only_check!("sse4.1")
        }
    }};
    ("sse4.2") => {{
        #[cfg(target_feature = "sse4.2")]
        {
            true
        }
        #[cfg(not(target_feature = "sse4.2"))]
        {
            $crate::__impl_runtime_only_check!("sse4.2")
        }
    }};
    ("avx") => {{
        #[cfg(target_feature = "avx")]
        {
            true
        }
        #[cfg(not(target_feature = "avx"))]
        {
            $crate::__impl_runtime_only_check!("avx")
        }
    }};
    ("avx2") => {{
        #[cfg(target_feature = "avx2")]
        {
            true
        }
        #[cfg(not(target_feature = "avx2"))]
        {
            $crate::__impl_runtime_only_check!("avx2")
        }
    }};
    ("fma") => {{
        #[cfg(target_feature = "fma")]
        {
            true
        }
        #[cfg(not(target_feature = "fma"))]
        {
            $crate::__impl_runtime_only_check!("fma")
        }
    }};
    ("avx512f") => {{
        #[cfg(target_feature = "avx512f")]
        {
            true
        }
        #[cfg(not(target_feature = "avx512f"))]
        {
            $crate::__impl_runtime_only_check!("avx512f")
        }
    }};
    ("avx512bw") => {{
        #[cfg(target_feature = "avx512bw")]
        {
            true
        }
        #[cfg(not(target_feature = "avx512bw"))]
        {
            $crate::__impl_runtime_only_check!("avx512bw")
        }
    }};
    ("avx512cd") => {{
        #[cfg(target_feature = "avx512cd")]
        {
            true
        }
        #[cfg(not(target_feature = "avx512cd"))]
        {
            $crate::__impl_runtime_only_check!("avx512cd")
        }
    }};
    ("avx512dq") => {{
        #[cfg(target_feature = "avx512dq")]
        {
            true
        }
        #[cfg(not(target_feature = "avx512dq"))]
        {
            $crate::__impl_runtime_only_check!("avx512dq")
        }
    }};
    ("avx512vl") => {{
        #[cfg(target_feature = "avx512vl")]
        {
            true
        }
        #[cfg(not(target_feature = "avx512vl"))]
        {
            $crate::__impl_runtime_only_check!("avx512vl")
        }
    }};
    ("bmi1") => {{
        #[cfg(target_feature = "bmi1")]
        {
            true
        }
        #[cfg(not(target_feature = "bmi1"))]
        {
            $crate::__impl_runtime_only_check!("bmi1")
        }
    }};
    ("bmi2") => {{
        #[cfg(target_feature = "bmi2")]
        {
            true
        }
        #[cfg(not(target_feature = "bmi2"))]
        {
            $crate::__impl_runtime_only_check!("bmi2")
        }
    }};
    ("popcnt") => {{
        #[cfg(target_feature = "popcnt")]
        {
            true
        }
        #[cfg(not(target_feature = "popcnt"))]
        {
            $crate::__impl_runtime_only_check!("popcnt")
        }
    }};
    ("lzcnt") => {{
        #[cfg(target_feature = "lzcnt")]
        {
            true
        }
        #[cfg(not(target_feature = "lzcnt"))]
        {
            $crate::__impl_runtime_only_check!("lzcnt")
        }
    }};
    ("pclmulqdq") => {{
        #[cfg(target_feature = "pclmulqdq")]
        {
            true
        }
        #[cfg(not(target_feature = "pclmulqdq"))]
        {
            $crate::__impl_runtime_only_check!("pclmulqdq")
        }
    }};
    ("aes") => {{
        #[cfg(target_feature = "aes")]
        {
            true
        }
        #[cfg(not(target_feature = "aes"))]
        {
            $crate::__impl_runtime_only_check!("aes")
        }
    }};
    ("sha") => {{
        #[cfg(target_feature = "sha")]
        {
            true
        }
        #[cfg(not(target_feature = "sha"))]
        {
            $crate::__impl_runtime_only_check!("sha")
        }
    }};
}

/// Runtime-only feature check. Used when compile-time detection not possible.
#[doc(hidden)]
#[macro_export]
macro_rules! __impl_runtime_only_check {
    ($feature:tt) => {{
        #[cfg(target_arch = "x86_64")]
        {
            #[cfg(feature = "std")]
            {
                std::arch::is_x86_feature_detected!($feature)
            }
            #[cfg(not(feature = "std"))]
            {
                // In no_std, we can't do runtime detection without std
                // Fall back to compile-time only
                false
            }
        }
        #[cfg(target_arch = "x86")]
        {
            #[cfg(feature = "std")]
            {
                std::arch::is_x86_feature_detected!($feature)
            }
            #[cfg(not(feature = "std"))]
            {
                false
            }
        }
        #[cfg(not(any(target_arch = "x86_64", target_arch = "x86")))]
        {
            false
        }
    }};
}

// ============================================================================
// AArch64 Feature Detection
// ============================================================================

/// Checks if an AArch64 CPU feature is available, with compile-time optimization.
///
/// Unlike `is_aarch64_feature_detected!` from std, this macro first checks
/// `cfg!(target_feature)` at compile time. If the feature is compile-time
/// known (e.g., compiled with `-C target-feature=+sve` or `-C target-cpu=...`),
/// no runtime check is performed.
///
/// # Example
///
/// ```ignore
/// use archmage::is_aarch64_feature_available;
///
/// if is_aarch64_feature_available!("sve") {
///     println!("SVE is available");
/// }
/// ```
///
/// # Supported Features
///
/// - `"neon"` - NEON (always available on AArch64)
/// - `"dotprod"` - Dot product (ARMv8.2-A+)
/// - `"rdm"` - Rounding doubling multiply (ARMv8.1-A+)
/// - `"fp16"` - Half-precision floating point (ARMv8.2-A+)
/// - `"fhm"` - FP16 multiply-accumulate (ARMv8.4-A+)
/// - `"fcma"` - Complex number multiply-add (ARMv8.3-A+)
/// - `"i8mm"` - Int8 matrix multiply (ARMv8.6-A+)
/// - `"bf16"` - BFloat16 (ARMv8.6-A+)
/// - `"aes"` - AES encryption
/// - `"sha2"` - SHA-256
/// - `"sha3"` - SHA-3 / SHA-512
/// - `"crc"` - CRC32 (ARMv8.1-A+)
/// - `"sve"` - Scalable Vector Extension
/// - `"sve2"` - SVE2
#[macro_export]
macro_rules! is_aarch64_feature_available {
    ("neon") => {{ $crate::__impl_aarch64_feature_check!("neon") }};
    ("sve") => {{ $crate::__impl_aarch64_feature_check!("sve") }};
    ("sve2") => {{ $crate::__impl_aarch64_feature_check!("sve2") }};
    // Crypto features
    ("aes") => {{ $crate::__impl_aarch64_feature_check!("aes") }};
    ("sha2") => {{ $crate::__impl_aarch64_feature_check!("sha2") }};
    ("sha3") => {{ $crate::__impl_aarch64_feature_check!("sha3") }};
    ("crc") => {{ $crate::__impl_aarch64_feature_check!("crc") }};
    // Compute extensions (Arm64-v2 / Arm64-v3)
    ("dotprod") => {{ $crate::__impl_aarch64_feature_check!("dotprod") }};
    ("rdm") => {{ $crate::__impl_aarch64_feature_check!("rdm") }};
    ("fp16") => {{ $crate::__impl_aarch64_feature_check!("fp16") }};
    ("fhm") => {{ $crate::__impl_aarch64_feature_check!("fhm") }};
    ("fcma") => {{ $crate::__impl_aarch64_feature_check!("fcma") }};
    ("i8mm") => {{ $crate::__impl_aarch64_feature_check!("i8mm") }};
    ("bf16") => {{ $crate::__impl_aarch64_feature_check!("bf16") }};
    // Fallback for other features - runtime only
    ($feature:tt) => {{ $crate::__impl_aarch64_runtime_only_check!($feature) }};
}

/// Implementation macro for AArch64 feature check with compile-time optimization.
/// Not intended for direct use.
#[doc(hidden)]
#[macro_export]
macro_rules! __impl_aarch64_feature_check {
    ("neon") => {{
        #[cfg(target_feature = "neon")]
        {
            true
        }
        #[cfg(not(target_feature = "neon"))]
        {
            $crate::__impl_aarch64_runtime_only_check!("neon")
        }
    }};
    ("sve") => {{
        #[cfg(target_feature = "sve")]
        {
            true
        }
        #[cfg(not(target_feature = "sve"))]
        {
            $crate::__impl_aarch64_runtime_only_check!("sve")
        }
    }};
    ("sve2") => {{
        #[cfg(target_feature = "sve2")]
        {
            true
        }
        #[cfg(not(target_feature = "sve2"))]
        {
            $crate::__impl_aarch64_runtime_only_check!("sve2")
        }
    }};
    ("aes") => {{
        #[cfg(target_feature = "aes")]
        {
            true
        }
        #[cfg(not(target_feature = "aes"))]
        {
            $crate::__impl_aarch64_runtime_only_check!("aes")
        }
    }};
    ("sha2") => {{
        #[cfg(target_feature = "sha2")]
        {
            true
        }
        #[cfg(not(target_feature = "sha2"))]
        {
            $crate::__impl_aarch64_runtime_only_check!("sha2")
        }
    }};
    ("sha3") => {{
        #[cfg(target_feature = "sha3")]
        {
            true
        }
        #[cfg(not(target_feature = "sha3"))]
        {
            $crate::__impl_aarch64_runtime_only_check!("sha3")
        }
    }};
    ("crc") => {{
        #[cfg(target_feature = "crc")]
        {
            true
        }
        #[cfg(not(target_feature = "crc"))]
        {
            $crate::__impl_aarch64_runtime_only_check!("crc")
        }
    }};
    ("dotprod") => {{
        #[cfg(target_feature = "dotprod")]
        {
            true
        }
        #[cfg(not(target_feature = "dotprod"))]
        {
            $crate::__impl_aarch64_runtime_only_check!("dotprod")
        }
    }};
    ("rdm") => {{
        #[cfg(target_feature = "rdm")]
        {
            true
        }
        #[cfg(not(target_feature = "rdm"))]
        {
            $crate::__impl_aarch64_runtime_only_check!("rdm")
        }
    }};
    ("fp16") => {{
        #[cfg(target_feature = "fp16")]
        {
            true
        }
        #[cfg(not(target_feature = "fp16"))]
        {
            $crate::__impl_aarch64_runtime_only_check!("fp16")
        }
    }};
    ("fhm") => {{
        #[cfg(target_feature = "fhm")]
        {
            true
        }
        #[cfg(not(target_feature = "fhm"))]
        {
            $crate::__impl_aarch64_runtime_only_check!("fhm")
        }
    }};
    ("fcma") => {{
        #[cfg(target_feature = "fcma")]
        {
            true
        }
        #[cfg(not(target_feature = "fcma"))]
        {
            $crate::__impl_aarch64_runtime_only_check!("fcma")
        }
    }};
    ("i8mm") => {{
        #[cfg(target_feature = "i8mm")]
        {
            true
        }
        #[cfg(not(target_feature = "i8mm"))]
        {
            $crate::__impl_aarch64_runtime_only_check!("i8mm")
        }
    }};
    ("bf16") => {{
        #[cfg(target_feature = "bf16")]
        {
            true
        }
        #[cfg(not(target_feature = "bf16"))]
        {
            $crate::__impl_aarch64_runtime_only_check!("bf16")
        }
    }};
}

/// Runtime-only AArch64 feature check. Used when compile-time detection not possible.
#[doc(hidden)]
#[macro_export]
macro_rules! __impl_aarch64_runtime_only_check {
    ($feature:tt) => {{
        #[cfg(target_arch = "aarch64")]
        {
            #[cfg(feature = "std")]
            {
                std::arch::is_aarch64_feature_detected!($feature)
            }
            #[cfg(not(feature = "std"))]
            {
                // In no_std, we can't do runtime detection without std
                // Fall back to compile-time only
                false
            }
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            false
        }
    }};
}

// ============================================================================
// Assembly verification helpers
// ============================================================================

/// Helper function for verifying assembly output.
/// When compiled with +avx2, this should contain no cpuid or function calls.
#[cfg(target_arch = "x86_64")]
#[inline(never)]
pub fn check_avx2_available() -> bool {
    is_x86_feature_available!("avx2")
}

/// Helper function for verifying assembly output.
/// When compiled with +fma, this should contain no cpuid or function calls.
#[cfg(target_arch = "x86_64")]
#[inline(never)]
pub fn check_fma_available() -> bool {
    is_x86_feature_available!("fma")
}

/// Helper function for verifying assembly output.
/// When compiled with +avx512f, this should contain no cpuid or function calls.
#[cfg(target_arch = "x86_64")]
#[inline(never)]
pub fn check_avx512f_available() -> bool {
    is_x86_feature_available!("avx512f")
}

/// Helper function for verifying assembly output on AArch64.
/// When compiled with +sve, this should contain no runtime detection.
#[cfg(target_arch = "aarch64")]
#[inline(never)]
pub fn check_sve_available() -> bool {
    is_aarch64_feature_available!("sve")
}

/// Helper function for verifying assembly output on AArch64.
/// When compiled with +sve2, this should contain no runtime detection.
#[cfg(target_arch = "aarch64")]
#[inline(never)]
pub fn check_sve2_available() -> bool {
    is_aarch64_feature_available!("sve2")
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    /// Test that the macro compiles for all supported features
    #[test]
    fn test_all_features_compile() {
        let _ = is_x86_feature_available!("sse");
        let _ = is_x86_feature_available!("sse2");
        let _ = is_x86_feature_available!("sse3");
        let _ = is_x86_feature_available!("ssse3");
        let _ = is_x86_feature_available!("sse4.1");
        let _ = is_x86_feature_available!("sse4.2");
        let _ = is_x86_feature_available!("avx");
        let _ = is_x86_feature_available!("avx2");
        let _ = is_x86_feature_available!("fma");
        let _ = is_x86_feature_available!("avx512f");
        let _ = is_x86_feature_available!("avx512bw");
        let _ = is_x86_feature_available!("bmi1");
        let _ = is_x86_feature_available!("bmi2");
        let _ = is_x86_feature_available!("popcnt");
        let _ = is_x86_feature_available!("lzcnt");
        let _ = is_x86_feature_available!("aes");
    }

    /// Test that SSE2 is always available on x86_64
    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_sse2_always_available() {
        let has_sse2 = is_x86_feature_available!("sse2");
        assert!(has_sse2);
    }

    /// Test consistency with std's is_x86_feature_detected
    #[test]
    #[cfg(all(target_arch = "x86_64", feature = "std"))]
    fn test_matches_std_detection() {
        use std::arch::is_x86_feature_detected;

        assert_eq!(
            is_x86_feature_available!("avx2"),
            is_x86_feature_detected!("avx2")
        );
        assert_eq!(
            is_x86_feature_available!("fma"),
            is_x86_feature_detected!("fma")
        );
        assert_eq!(
            is_x86_feature_available!("avx512f"),
            is_x86_feature_detected!("avx512f")
        );
    }

    /// Test that compile-time known features return true
    /// This test is compiled with +avx2, so avx2 should be compile-time true
    #[test]
    #[cfg(target_feature = "avx2")]
    fn test_compile_time_avx2() {
        // This should compile to just `true`
        assert!(is_x86_feature_available!("avx2"));
    }

    /// Test that compile-time known features return true for SSE2
    /// SSE2 is baseline for x86_64, so always compile-time true
    #[test]
    #[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
    fn test_compile_time_sse2() {
        let has_sse2 = is_x86_feature_available!("sse2");
        assert!(has_sse2);
    }

    // ========================================================================
    // AArch64 Tests
    // ========================================================================

    /// Test that the aarch64 macro compiles for all supported features
    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_aarch64_features_compile() {
        let _ = is_aarch64_feature_available!("neon");
        let _ = is_aarch64_feature_available!("sve");
        let _ = is_aarch64_feature_available!("sve2");
    }

    /// Test that NEON detection works on aarch64
    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_neon_detection() {
        // NEON requires runtime detection (or compile-time target_feature)
        let has_neon = is_aarch64_feature_available!("neon");
        // On real aarch64 hardware, NEON is expected to be available
        let _ = has_neon;
    }

    /// Test consistency with std's is_aarch64_feature_detected
    #[test]
    #[cfg(all(target_arch = "aarch64", feature = "std"))]
    fn test_aarch64_matches_std_detection() {
        use std::arch::is_aarch64_feature_detected;

        assert_eq!(
            is_aarch64_feature_available!("sve"),
            is_aarch64_feature_detected!("sve")
        );
        assert_eq!(
            is_aarch64_feature_available!("sve2"),
            is_aarch64_feature_detected!("sve2")
        );
    }
}
