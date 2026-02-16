//! Generated from token-registry.toml — DO NOT EDIT.
//!
//! Regenerate with: cargo run -p xtask -- generate

/// Maps a token type name to its required target features.
///
/// Generated from token-registry.toml. One complete feature list per token.
pub(crate) fn token_to_features(token_name: &str) -> Option<&'static [&'static str]> {
    match token_name {
        "X64V1Token" | "Sse2Token" => Some(&["sse", "sse2"]),
        "X64V2Token" => Some(&["sse", "sse2", "sse3", "ssse3", "sse4.1", "sse4.2", "popcnt"]),
        "X64V3Token" | "Desktop64" | "Avx2FmaToken" => Some(&[
            "sse", "sse2", "sse3", "ssse3", "sse4.1", "sse4.2", "popcnt", "avx", "avx2", "fma",
            "bmi1", "bmi2", "f16c", "lzcnt",
        ]),
        "X64V4Token" | "Avx512Token" | "Server64" => Some(&[
            "sse", "sse2", "sse3", "ssse3", "sse4.1", "sse4.2", "popcnt", "avx", "avx2", "fma",
            "bmi1", "bmi2", "f16c", "lzcnt", "avx512f", "avx512bw", "avx512cd", "avx512dq",
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
            "avx",
            "avx2",
            "fma",
            "bmi1",
            "bmi2",
            "f16c",
            "lzcnt",
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
            "avx",
            "avx2",
            "fma",
            "bmi1",
            "bmi2",
            "f16c",
            "lzcnt",
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
        "Wasm128Token" => Some(&["simd128"]),
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
        "HasX64V2" => Some(&["sse", "sse2", "sse3", "ssse3", "sse4.1", "sse4.2", "popcnt"]),
        "HasX64V4" => Some(&[
            "sse", "sse2", "sse3", "ssse3", "sse4.1", "sse4.2", "popcnt", "avx", "avx2", "fma",
            "bmi1", "bmi2", "f16c", "lzcnt", "avx512f", "avx512bw", "avx512cd", "avx512dq",
            "avx512vl",
        ]),
        "HasNeon" => Some(&["neon"]),
        "HasNeonAes" => Some(&["neon", "aes"]),
        "HasNeonSha3" => Some(&["neon", "sha3"]),

        // Token types used as bounds — full feature sets
        "X64V1Token" | "Sse2Token" => Some(&["sse", "sse2"]),
        "X64V2Token" => Some(&["sse", "sse2", "sse3", "ssse3", "sse4.1", "sse4.2", "popcnt"]),
        "X64V3Token" | "Desktop64" | "Avx2FmaToken" => Some(&[
            "sse", "sse2", "sse3", "ssse3", "sse4.1", "sse4.2", "popcnt", "avx", "avx2", "fma",
            "bmi1", "bmi2", "f16c", "lzcnt",
        ]),
        "X64V4Token" | "Avx512Token" | "Server64" => Some(&[
            "sse", "sse2", "sse3", "ssse3", "sse4.1", "sse4.2", "popcnt", "avx", "avx2", "fma",
            "bmi1", "bmi2", "f16c", "lzcnt", "avx512f", "avx512bw", "avx512cd", "avx512dq",
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
            "avx",
            "avx2",
            "fma",
            "bmi1",
            "bmi2",
            "f16c",
            "lzcnt",
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
            "avx",
            "avx2",
            "fma",
            "bmi1",
            "bmi2",
            "f16c",
            "lzcnt",
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
        "Wasm128Token" => Some(&["simd128"]),

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
        "X64V3Token" | "Desktop64" | "Avx2FmaToken" => Some("x86_64"),
        "X64V4Token" | "Avx512Token" | "Server64" => Some("x86_64"),
        "X64V4xToken" | "Avx512ModernToken" => Some("x86_64"),
        "Avx512Fp16Token" => Some("x86_64"),
        "NeonToken" | "Arm64" => Some("aarch64"),
        "NeonAesToken" => Some("aarch64"),
        "NeonSha3Token" => Some("aarch64"),
        "NeonCrcToken" => Some("aarch64"),
        "Wasm128Token" => Some("wasm32"),
        _ => None,
    }
}

/// All concrete token names that exist in the runtime crate.
#[cfg(test)]
pub(crate) const ALL_CONCRETE_TOKENS: &[&str] = &[
    "X64V1Token",
    "Sse2Token",
    "X64V2Token",
    "X64V3Token",
    "Desktop64",
    "Avx2FmaToken",
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
    "Wasm128Token",
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
];
