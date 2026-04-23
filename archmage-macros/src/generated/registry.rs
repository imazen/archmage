//! Generated from token-registry.toml — DO NOT EDIT.
//!
//! Regenerate with: cargo run -p xtask -- generate

/// Maps a token type name to its required target features.
///
/// Generated from token-registry.toml. One complete feature list per token.
pub(crate) fn token_to_features(token_name: &str) -> Option<&'static [&'static str]> {
    match token_name {
        "X64V1Token" | "Sse2Token" => Some(&["sse", "sse2"]),
        "X64V2Token" => Some(&[
            "sse",
            "sse2",
            "sse3",
            "ssse3",
            "sse4.1",
            "sse4.2",
            "popcnt",
            "cmpxchg16b",
        ]),
        "X64CryptoToken" => Some(&[
            "sse",
            "sse2",
            "sse3",
            "ssse3",
            "sse4.1",
            "sse4.2",
            "popcnt",
            "cmpxchg16b",
            "pclmulqdq",
            "aes",
        ]),
        "X64V3Token" | "Desktop64" | "Avx2FmaToken" => Some(&[
            "sse",
            "sse2",
            "sse3",
            "ssse3",
            "sse4.1",
            "sse4.2",
            "popcnt",
            "cmpxchg16b",
            "avx",
            "avx2",
            "fma",
            "bmi1",
            "bmi2",
            "f16c",
            "lzcnt",
            "movbe",
        ]),
        "X64V3CryptoToken" => Some(&[
            "sse",
            "sse2",
            "sse3",
            "ssse3",
            "sse4.1",
            "sse4.2",
            "popcnt",
            "cmpxchg16b",
            "avx",
            "avx2",
            "fma",
            "bmi1",
            "bmi2",
            "f16c",
            "lzcnt",
            "movbe",
            "pclmulqdq",
            "aes",
            "vpclmulqdq",
            "vaes",
        ]),
        "X64V4Token" | "Avx512Token" | "Server64" => Some(&[
            "sse",
            "sse2",
            "sse3",
            "ssse3",
            "sse4.1",
            "sse4.2",
            "popcnt",
            "cmpxchg16b",
            "avx",
            "avx2",
            "fma",
            "bmi1",
            "bmi2",
            "f16c",
            "lzcnt",
            "movbe",
            "pclmulqdq",
            "aes",
            "avx512f",
            "avx512bw",
            "avx512cd",
            "avx512dq",
            "avx512vl",
        ]),
        "X64V4xToken" | "Avx512ModernToken" => Some(&[
            "sse",
            "sse2",
            "sse3",
            "ssse3",
            "sse4.1",
            "sse4.2",
            "popcnt",
            "cmpxchg16b",
            "avx",
            "avx2",
            "fma",
            "bmi1",
            "bmi2",
            "f16c",
            "lzcnt",
            "movbe",
            "pclmulqdq",
            "aes",
            "avx512f",
            "avx512bw",
            "avx512cd",
            "avx512dq",
            "avx512vl",
            "avx512vpopcntdq",
            "avx512ifma",
            "avx512vbmi",
            "avx512vbmi2",
            "avx512bitalg",
            "avx512vnni",
            "vpclmulqdq",
            "gfni",
            "vaes",
        ]),
        "Avx512Fp16Token" => Some(&[
            "sse",
            "sse2",
            "sse3",
            "ssse3",
            "sse4.1",
            "sse4.2",
            "popcnt",
            "cmpxchg16b",
            "avx",
            "avx2",
            "fma",
            "bmi1",
            "bmi2",
            "f16c",
            "lzcnt",
            "movbe",
            "pclmulqdq",
            "aes",
            "avx512f",
            "avx512bw",
            "avx512cd",
            "avx512dq",
            "avx512vl",
            "avx512fp16",
        ]),
        "NeonToken" | "Arm64" => Some(&["neon"]),
        "NeonAesToken" => Some(&["neon", "aes"]),
        "NeonSha3Token" => Some(&["neon", "sha3"]),
        "NeonCrcToken" => Some(&["neon", "crc"]),
        "Arm64V2Token" => Some(&["neon", "crc", "rdm", "dotprod", "fp16", "aes", "sha2"]),
        "Arm64V3Token" => Some(&[
            "neon", "crc", "rdm", "dotprod", "fp16", "aes", "sha2", "fhm", "fcma", "sha3", "i8mm",
            "bf16",
        ]),
        "Wasm128Token" => Some(&["simd128"]),
        "Wasm128RelaxedToken" => Some(&["simd128", "relaxed-simd"]),
        "ScalarToken" => Some(&[]),
        _ => None,
    }
}

/// Maps a trait bound name to its required target features.
///
/// Generated from token-registry.toml. Includes token type names
/// so `impl TokenType` patterns work in the macro.
pub(crate) fn trait_to_features(trait_name: &str) -> Option<&'static [&'static str]> {
    match trait_name {
        "Has128BitSimd" => Some(&["sse", "sse2"]),
        "Has256BitSimd" => Some(&["sse", "sse2", "avx"]),
        "Has512BitSimd" => Some(&["sse", "sse2", "avx", "avx2", "avx512f"]),
        "HasX64V2" => Some(&[
            "sse",
            "sse2",
            "sse3",
            "ssse3",
            "sse4.1",
            "sse4.2",
            "popcnt",
            "cmpxchg16b",
        ]),
        "HasX64V4" => Some(&[
            "sse",
            "sse2",
            "sse3",
            "ssse3",
            "sse4.1",
            "sse4.2",
            "popcnt",
            "cmpxchg16b",
            "avx",
            "avx2",
            "fma",
            "bmi1",
            "bmi2",
            "f16c",
            "lzcnt",
            "movbe",
            "avx512f",
            "avx512bw",
            "avx512cd",
            "avx512dq",
            "avx512vl",
        ]),
        "HasNeon" => Some(&["neon"]),
        "HasNeonAes" => Some(&["neon", "aes"]),
        "HasNeonSha3" => Some(&["neon", "sha3"]),
        "HasArm64V2" => Some(&["neon", "crc", "rdm", "dotprod", "fp16", "aes", "sha2"]),
        "HasArm64V3" => Some(&[
            "neon", "crc", "rdm", "dotprod", "fp16", "aes", "sha2", "fhm", "fcma", "sha3", "i8mm",
            "bf16",
        ]),

        // Token types used as bounds — full feature sets
        "X64V1Token" | "Sse2Token" => Some(&["sse", "sse2"]),
        "X64V2Token" => Some(&[
            "sse",
            "sse2",
            "sse3",
            "ssse3",
            "sse4.1",
            "sse4.2",
            "popcnt",
            "cmpxchg16b",
        ]),
        "X64CryptoToken" => Some(&[
            "sse",
            "sse2",
            "sse3",
            "ssse3",
            "sse4.1",
            "sse4.2",
            "popcnt",
            "cmpxchg16b",
            "pclmulqdq",
            "aes",
        ]),
        "X64V3Token" | "Desktop64" | "Avx2FmaToken" => Some(&[
            "sse",
            "sse2",
            "sse3",
            "ssse3",
            "sse4.1",
            "sse4.2",
            "popcnt",
            "cmpxchg16b",
            "avx",
            "avx2",
            "fma",
            "bmi1",
            "bmi2",
            "f16c",
            "lzcnt",
            "movbe",
        ]),
        "X64V3CryptoToken" => Some(&[
            "sse",
            "sse2",
            "sse3",
            "ssse3",
            "sse4.1",
            "sse4.2",
            "popcnt",
            "cmpxchg16b",
            "avx",
            "avx2",
            "fma",
            "bmi1",
            "bmi2",
            "f16c",
            "lzcnt",
            "movbe",
            "pclmulqdq",
            "aes",
            "vpclmulqdq",
            "vaes",
        ]),
        "X64V4Token" | "Avx512Token" | "Server64" => Some(&[
            "sse",
            "sse2",
            "sse3",
            "ssse3",
            "sse4.1",
            "sse4.2",
            "popcnt",
            "cmpxchg16b",
            "avx",
            "avx2",
            "fma",
            "bmi1",
            "bmi2",
            "f16c",
            "lzcnt",
            "movbe",
            "pclmulqdq",
            "aes",
            "avx512f",
            "avx512bw",
            "avx512cd",
            "avx512dq",
            "avx512vl",
        ]),
        "X64V4xToken" | "Avx512ModernToken" => Some(&[
            "sse",
            "sse2",
            "sse3",
            "ssse3",
            "sse4.1",
            "sse4.2",
            "popcnt",
            "cmpxchg16b",
            "avx",
            "avx2",
            "fma",
            "bmi1",
            "bmi2",
            "f16c",
            "lzcnt",
            "movbe",
            "pclmulqdq",
            "aes",
            "avx512f",
            "avx512bw",
            "avx512cd",
            "avx512dq",
            "avx512vl",
            "avx512vpopcntdq",
            "avx512ifma",
            "avx512vbmi",
            "avx512vbmi2",
            "avx512bitalg",
            "avx512vnni",
            "vpclmulqdq",
            "gfni",
            "vaes",
        ]),
        "Avx512Fp16Token" => Some(&[
            "sse",
            "sse2",
            "sse3",
            "ssse3",
            "sse4.1",
            "sse4.2",
            "popcnt",
            "cmpxchg16b",
            "avx",
            "avx2",
            "fma",
            "bmi1",
            "bmi2",
            "f16c",
            "lzcnt",
            "movbe",
            "pclmulqdq",
            "aes",
            "avx512f",
            "avx512bw",
            "avx512cd",
            "avx512dq",
            "avx512vl",
            "avx512fp16",
        ]),
        "NeonToken" | "Arm64" => Some(&["neon"]),
        "NeonAesToken" => Some(&["neon", "aes"]),
        "NeonSha3Token" => Some(&["neon", "sha3"]),
        "NeonCrcToken" => Some(&["neon", "crc"]),
        "Arm64V2Token" => Some(&["neon", "crc", "rdm", "dotprod", "fp16", "aes", "sha2"]),
        "Arm64V3Token" => Some(&[
            "neon", "crc", "rdm", "dotprod", "fp16", "aes", "sha2", "fhm", "fcma", "sha3", "i8mm",
            "bf16",
        ]),
        "Wasm128Token" => Some(&["simd128"]),
        "Wasm128RelaxedToken" => Some(&["simd128", "relaxed-simd"]),

        _ => None,
    }
}

/// Maps a token type name to its target architecture.
///
/// Returns the `target_arch` value (e.g., "x86_64", "aarch64", "wasm32").
pub(crate) fn token_to_arch(token_name: &str) -> Option<&'static str> {
    match token_name {
        "X64V1Token" | "Sse2Token" => Some("x86_64"),
        "X64V2Token" => Some("x86_64"),
        "X64CryptoToken" => Some("x86_64"),
        "X64V3Token" | "Desktop64" | "Avx2FmaToken" => Some("x86_64"),
        "X64V3CryptoToken" => Some("x86_64"),
        "X64V4Token" | "Avx512Token" | "Server64" => Some("x86_64"),
        "X64V4xToken" | "Avx512ModernToken" => Some("x86_64"),
        "Avx512Fp16Token" => Some("x86_64"),
        "NeonToken" | "Arm64" => Some("aarch64"),
        "NeonAesToken" => Some("aarch64"),
        "NeonSha3Token" => Some("aarch64"),
        "NeonCrcToken" => Some("aarch64"),
        "Arm64V2Token" => Some("aarch64"),
        "Arm64V3Token" => Some("aarch64"),
        "Wasm128Token" => Some("wasm32"),
        "Wasm128RelaxedToken" => Some("wasm32"),
        _ => None,
    }
}

/// Maps a token type name to its magetypes width namespace.
///
/// Returns the namespace name (e.g., "v3", "v4", "neon", "wasm128", "scalar").
/// Used by `import_magetypes` to inject `use magetypes::simd::{ns}::*;`.
pub(crate) fn token_to_magetypes_namespace(token_name: &str) -> Option<&'static str> {
    match token_name {
        "X64V1Token" | "Sse2Token" => Some("v3"),
        "X64V2Token" => Some("v3"),
        "X64CryptoToken" => Some("v3"),
        "X64V3Token" | "Desktop64" | "Avx2FmaToken" => Some("v3"),
        "X64V3CryptoToken" => Some("v3"),
        "X64V4Token" | "Avx512Token" | "Server64" => Some("v4"),
        "X64V4xToken" | "Avx512ModernToken" => Some("v4x"),
        "Avx512Fp16Token" => Some("v4"),
        "NeonToken" | "Arm64" => Some("neon"),
        "NeonAesToken" => Some("neon"),
        "NeonSha3Token" => Some("neon"),
        "NeonCrcToken" => Some("neon"),
        "Arm64V2Token" => Some("neon"),
        "Arm64V3Token" => Some("neon"),
        "Wasm128Token" => Some("wasm128"),
        "Wasm128RelaxedToken" => Some("wasm128"),
        "ScalarToken" => Some("scalar"),
        _ => None,
    }
}

/// Maps a trait bound name to its magetypes width namespace.
///
/// Returns the namespace name (e.g., "v3", "v4", "neon").
/// Used by `import_magetypes` when a trait bound is used instead of a concrete token.
pub(crate) fn trait_to_magetypes_namespace(trait_name: &str) -> Option<&'static str> {
    match trait_name {
        "Has128BitSimd" => Some("v3"),
        "Has256BitSimd" => Some("v3"),
        "Has512BitSimd" => Some("v4"),
        "HasX64V2" => Some("v3"),
        "HasX64V4" => Some("v4"),
        "HasNeon" => Some("neon"),
        "HasNeonAes" => Some("neon"),
        "HasNeonSha3" => Some("neon"),
        "HasArm64V2" => Some("neon"),
        "HasArm64V3" => Some("neon"),

        // Token types used as bounds
        "X64V1Token" | "Sse2Token" => Some("v3"),
        "X64V2Token" => Some("v3"),
        "X64CryptoToken" => Some("v3"),
        "X64V3Token" | "Desktop64" | "Avx2FmaToken" => Some("v3"),
        "X64V3CryptoToken" => Some("v3"),
        "X64V4Token" | "Avx512Token" | "Server64" => Some("v4"),
        "X64V4xToken" | "Avx512ModernToken" => Some("v4x"),
        "Avx512Fp16Token" => Some("v4"),
        "NeonToken" | "Arm64" => Some("neon"),
        "NeonAesToken" => Some("neon"),
        "NeonSha3Token" => Some("neon"),
        "NeonCrcToken" => Some("neon"),
        "Arm64V2Token" => Some("neon"),
        "Arm64V3Token" => Some("neon"),
        "Wasm128Token" => Some("wasm128"),
        "Wasm128RelaxedToken" => Some("wasm128"),

        _ => None,
    }
}

/// Maps a trait bound name to its target architecture.
///
/// Returns the architecture (e.g., "x86_64", "aarch64").
/// Used by `import_intrinsics` when a trait bound is used instead of a concrete token.
pub(crate) fn trait_to_arch(trait_name: &str) -> Option<&'static str> {
    match trait_name {
        "Has128BitSimd" => Some("x86_64"),
        "Has256BitSimd" => Some("x86_64"),
        "Has512BitSimd" => Some("x86_64"),
        "HasX64V2" => Some("x86_64"),
        "HasX64V4" => Some("x86_64"),
        "HasNeon" => Some("aarch64"),
        "HasNeonAes" => Some("aarch64"),
        "HasNeonSha3" => Some("aarch64"),
        "HasArm64V2" => Some("aarch64"),
        "HasArm64V3" => Some("aarch64"),

        // Token types used as bounds
        "X64V1Token" | "Sse2Token" => Some("x86_64"),
        "X64V2Token" => Some("x86_64"),
        "X64CryptoToken" => Some("x86_64"),
        "X64V3Token" | "Desktop64" | "Avx2FmaToken" => Some("x86_64"),
        "X64V3CryptoToken" => Some("x86_64"),
        "X64V4Token" | "Avx512Token" | "Server64" => Some("x86_64"),
        "X64V4xToken" | "Avx512ModernToken" => Some("x86_64"),
        "Avx512Fp16Token" => Some("x86_64"),
        "NeonToken" | "Arm64" => Some("aarch64"),
        "NeonAesToken" => Some("aarch64"),
        "NeonSha3Token" => Some("aarch64"),
        "NeonCrcToken" => Some("aarch64"),
        "Arm64V2Token" => Some("aarch64"),
        "Arm64V3Token" => Some("aarch64"),
        "Wasm128Token" => Some("wasm32"),
        "Wasm128RelaxedToken" => Some("wasm32"),

        _ => None,
    }
}

/// Maps a tier short name to its canonical token type name.
///
/// Used by `#[rite(v3)]` to resolve the tier to a token without
/// requiring a token parameter in the function signature.
///
/// Accepts `_v3` as well as `v3` — the leading `_` matches name-mangling suffixes.
pub(crate) fn tier_to_canonical_token(tier_name: &str) -> Option<&'static str> {
    let tier_name = tier_name.strip_prefix('_').unwrap_or(tier_name);
    match tier_name {
        "v1" => Some("X64V1Token"),
        "v2" => Some("X64V2Token"),
        "x64_crypto" => Some("X64CryptoToken"),
        "v3" => Some("X64V3Token"),
        "v3_crypto" => Some("X64V3CryptoToken"),
        "v4" => Some("X64V4Token"),
        "avx512" => Some("X64V4Token"),
        "v4x" => Some("X64V4xToken"),
        "fp16" => Some("Avx512Fp16Token"),
        "neon" => Some("NeonToken"),
        "neon_aes" => Some("NeonAesToken"),
        "neon_sha3" => Some("NeonSha3Token"),
        "neon_crc" => Some("NeonCrcToken"),
        "arm_v2" => Some("Arm64V2Token"),
        "arm_v3" => Some("Arm64V3Token"),
        "wasm128" => Some("Wasm128Token"),
        "wasm128_relaxed" => Some("Wasm128RelaxedToken"),
        "scalar" => Some("ScalarToken"),
        _ => None,
    }
}

/// Maps a canonical token type name to its tier suffix.
///
/// Used by `#[rite(v3, v4, neon)]` to generate suffixed function names
/// (e.g., `fn_v3`, `fn_v4`, `fn_neon`).
pub(crate) fn canonical_token_to_tier_suffix(token_name: &str) -> Option<&'static str> {
    match token_name {
        "X64V1Token" | "Sse2Token" => Some("v1"),
        "X64V2Token" => Some("v2"),
        "X64CryptoToken" => Some("x64_crypto"),
        "X64V3Token" | "Desktop64" | "Avx2FmaToken" => Some("v3"),
        "X64V3CryptoToken" => Some("v3_crypto"),
        "X64V4Token" | "Avx512Token" | "Server64" => Some("v4"),
        "X64V4xToken" | "Avx512ModernToken" => Some("v4x"),
        "Avx512Fp16Token" => Some("fp16"),
        "NeonToken" | "Arm64" => Some("neon"),
        "NeonAesToken" => Some("neon_aes"),
        "NeonSha3Token" => Some("neon_sha3"),
        "NeonCrcToken" => Some("neon_crc"),
        "Arm64V2Token" => Some("arm_v2"),
        "Arm64V3Token" => Some("arm_v3"),
        "Wasm128Token" => Some("wasm128"),
        "Wasm128RelaxedToken" => Some("wasm128_relaxed"),
        "ScalarToken" => Some("scalar"),
        _ => None,
    }
}

/// Check if tier `from_suffix` can downgrade to tier `to_suffix`.
///
/// Derived from feature set math: true when `from.features ⊃ to.features`.
/// Identity (from == to) returns false (use direct pass, no method needed).
pub(crate) fn can_downgrade_tier(from_suffix: &str, to_suffix: &str) -> bool {
    if from_suffix == to_suffix {
        return false;
    }
    matches!(
        (from_suffix, to_suffix),
        ("v2", "v1")
            | ("x64_crypto", "v1" | "v2")
            | ("v3", "v1" | "v2")
            | ("v3_crypto", "v1" | "v2" | "v3" | "x64_crypto")
            | ("v4", "v1" | "v2" | "v3" | "x64_crypto")
            | (
                "v4x",
                "v1" | "v2" | "v3" | "v3_crypto" | "v4" | "x64_crypto"
            )
            | ("fp16", "v1" | "v2" | "v3" | "v4" | "x64_crypto")
            | ("neon_aes", "neon")
            | ("neon_sha3", "neon")
            | ("neon_crc", "neon")
            | ("arm_v2", "neon" | "neon_aes" | "neon_crc")
            | (
                "arm_v3",
                "arm_v2" | "neon" | "neon_aes" | "neon_crc" | "neon_sha3"
            )
            | ("wasm128_relaxed", "wasm128")
    )
}

/// Returns the expected tier tag for a concrete token type name.
///
/// Used by `#[arcane]` to emit compile-time assertions.
/// Generated from token-registry.toml.
pub(crate) fn expected_tier_tag(token_name: &str) -> Option<u32> {
    match token_name {
        "ScalarToken" => Some(0xD141EACA),
        "X64V1Token" | "Sse2Token" => Some(0x510E0DCD),
        "X64V2Token" => Some(0x8CCD97C6),
        "X64CryptoToken" => Some(0xF5F0FBF5),
        "X64V3Token" | "Desktop64" | "Avx2FmaToken" => Some(0xF38B284B),
        "X64V3CryptoToken" => Some(0x01EAE708),
        "X64V4Token" | "Avx512Token" | "Server64" => Some(0xFE1B900C),
        "X64V4xToken" | "Avx512ModernToken" => Some(0x8F63232A),
        "Avx512Fp16Token" => Some(0x2F39FFC0),
        "NeonToken" | "Arm64" => Some(0x72CB52B2),
        "NeonAesToken" => Some(0x8C16863D),
        "NeonSha3Token" => Some(0x8215198F),
        "NeonCrcToken" => Some(0x5C2B1B4E),
        "Arm64V2Token" => Some(0xB0231590),
        "Arm64V3Token" => Some(0xB2F6E2D5),
        "Wasm128Token" => Some(0x1E0DF26B),
        "Wasm128RelaxedToken" => Some(0x821D5452),
        _ => None,
    }
}

/// All concrete token names that exist in the runtime crate.
#[cfg(test)]
pub(crate) const ALL_CONCRETE_TOKENS: &[&str] = &[
    "X64V1Token",
    "Sse2Token",
    "X64V2Token",
    "X64CryptoToken",
    "X64V3Token",
    "Desktop64",
    "Avx2FmaToken",
    "X64V3CryptoToken",
    "X64V4Token",
    "Avx512Token",
    "Server64",
    "X64V4xToken",
    "Avx512ModernToken",
    "Avx512Fp16Token",
    "NeonToken",
    "Arm64",
    "NeonAesToken",
    "NeonSha3Token",
    "NeonCrcToken",
    "Arm64V2Token",
    "Arm64V3Token",
    "Wasm128Token",
    "Wasm128RelaxedToken",
];

/// All trait names that exist in the runtime crate.
#[cfg(test)]
pub(crate) const ALL_TRAIT_NAMES: &[&str] = &[
    "Has128BitSimd",
    "Has256BitSimd",
    "Has512BitSimd",
    "HasX64V2",
    "HasX64V4",
    "HasNeon",
    "HasNeonAes",
    "HasNeonSha3",
    "HasArm64V2",
    "HasArm64V3",
];
