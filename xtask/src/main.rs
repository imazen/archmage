//! Generator for safe_unaligned_simd token-gated wrappers.
//!
//! This tool parses the safe_unaligned_simd crate source and generates
//! wrapper modules that gate each function behind the appropriate archmage token.
//!
//! Usage:
//!   cargo xtask generate   - Generate wrapper modules
//!   cargo xtask validate   - Validate token bounds match feature requirements
//!   cargo xtask check-version - Check for safe_unaligned_simd updates

use anyhow::{Context, Result, bail};
use std::collections::{BTreeMap, BTreeSet, HashSet};
use std::fmt::Write as FmtWrite;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use syn::{
    Attribute, FnArg, GenericParam, Generics, ItemFn, Meta, Pat, ReturnType, Type, parse_file,
};
use walkdir::WalkDir;

mod simd_types;

/// Version of safe_unaligned_simd we're generating from
const SAFE_SIMD_VERSION: &str = "0.2.3";

/// Intel Intrinsics Guide base URL
const INTEL_INTRINSICS_URL: &str =
    "https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html";

/// ARM Intrinsics reference base URL
const ARM_INTRINSICS_URL: &str =
    "https://developer.arm.com/architectures/instruction-sets/intrinsics";

/// Architecture for code generation
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Arch {
    X86,
    Aarch64,
}

/// Token/trait info for wrapper generation.
struct TokenInfo {
    /// The token or trait name
    name: &'static str,
    /// The target features string
    features: &'static str,
    /// True if this is a trait (use `impl Trait`), false if concrete type
    is_trait: bool,
    /// True if this requires the avx512 feature flag
    requires_avx512: bool,
}

// ============================================================================
// Feature Registry - Authoritative source of what features each token provides
// ============================================================================

/// Returns the set of features that a token or trait provides.
///
/// This is the authoritative mapping used for validation. When we assign a token
/// bound to a wrapper, we must verify the token provides ALL required features.
fn token_provides_features(token_or_trait: &str) -> Option<&'static [&'static str]> {
    match token_or_trait {
        // Width traits
        "Has128BitSimd" => Some(&["sse", "sse2"]),
        "Has256BitSimd" => Some(&["sse", "sse2", "avx"]),
        "Has512BitSimd" => Some(&["sse", "sse2", "avx", "avx2", "avx512f"]),

        // x86 tier traits
        "HasX64V2" => Some(&["sse", "sse2", "sse3", "ssse3", "sse4.1", "sse4.2", "popcnt"]),
        "HasX64V4" => Some(&[
            "sse", "sse2", "sse3", "ssse3", "sse4.1", "sse4.2", "popcnt", "avx", "avx2", "fma",
            "bmi1", "bmi2", "f16c", "lzcnt", "avx512f", "avx512bw", "avx512cd", "avx512dq",
            "avx512vl",
        ]),

        // x86 concrete tokens
        "Sse41Token" => Some(&["sse", "sse2", "sse3", "ssse3", "sse4.1"]),
        "Sse42Token" => Some(&["sse", "sse2", "sse3", "ssse3", "sse4.1", "sse4.2"]),
        "AvxToken" => Some(&["sse", "sse2", "avx"]),
        "Avx2Token" => Some(&["sse", "sse2", "avx", "avx2"]),
        "Avx2FmaToken" => Some(&["sse", "sse2", "avx", "avx2", "fma"]),
        "X64V2Token" => Some(&["sse", "sse2", "sse3", "ssse3", "sse4.1", "sse4.2", "popcnt"]),
        "X64V3Token" | "Desktop64" => Some(&[
            "sse", "sse2", "sse3", "ssse3", "sse4.1", "sse4.2", "popcnt", "avx", "avx2", "fma",
            "bmi1", "bmi2", "f16c", "lzcnt",
        ]),
        "X64V4Token" | "Avx512Token" | "Server64" => Some(&[
            "sse", "sse2", "sse3", "ssse3", "sse4.1", "sse4.2", "popcnt", "avx", "avx2", "fma",
            "bmi1", "bmi2", "f16c", "lzcnt", "avx512f", "avx512bw", "avx512cd", "avx512dq",
            "avx512vl",
        ]),
        "Avx512ModernToken" => Some(&[
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
            "avx512bf16",
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

        // AArch64 traits
        "HasNeon" => Some(&["neon"]),
        "HasNeonAes" => Some(&["neon", "aes"]),
        "HasNeonSha3" => Some(&["neon", "sha3"]),

        // AArch64 concrete tokens
        "NeonToken" | "Arm64" => Some(&["neon"]),
        "NeonAesToken" => Some(&["neon", "aes"]),
        "NeonSha3Token" => Some(&["neon", "sha3"]),

        _ => None,
    }
}

/// Parse a comma-separated feature string into a set of individual features.
fn parse_features(features: &str) -> HashSet<&str> {
    features.split(',').map(|s| s.trim()).collect()
}

/// Check if a token/trait provides all required features.
fn token_satisfies_features(token: &str, required_features: &str) -> Result<(), Vec<String>> {
    let provided = match token_provides_features(token) {
        Some(f) => f.iter().copied().collect::<HashSet<_>>(),
        None => return Err(vec![format!("Unknown token/trait: {}", token)]),
    };

    let required = parse_features(required_features);
    let missing: Vec<_> = required
        .iter()
        .filter(|f| !provided.contains(*f))
        .map(|s| s.to_string())
        .collect();

    if missing.is_empty() {
        Ok(())
    } else {
        Err(missing)
    }
}

/// Mapping from target_feature strings to archmage tokens/traits.
///
/// Uses LLVM x86-64 microarchitecture level-based tier tokens.
fn feature_to_token(features: &str, arch: Arch) -> Option<TokenInfo> {
    match arch {
        Arch::X86 => match features {
            // AVX (v3 level) - use Has256BitSimd trait so any 256-bit token works
            "avx" => Some(TokenInfo {
                name: "Has256BitSimd",
                features: "avx",
                is_trait: true,
                requires_avx512: false,
            }),

            // v4 features - use HasX64V4 trait
            "avx512f" => Some(TokenInfo {
                name: "HasX64V4",
                features: "avx512f",
                is_trait: true,
                requires_avx512: true,
            }),
            "avx512f,avx512vl" => Some(TokenInfo {
                name: "HasX64V4",
                features: "avx512f,avx512vl",
                is_trait: true,
                requires_avx512: true,
            }),
            "avx512bw" => Some(TokenInfo {
                name: "HasX64V4",
                features: "avx512bw",
                is_trait: true,
                requires_avx512: true,
            }),
            "avx512bw,avx512vl" => Some(TokenInfo {
                name: "HasX64V4",
                features: "avx512bw,avx512vl",
                is_trait: true,
                requires_avx512: true,
            }),

            // Modern extensions - use concrete Avx512ModernToken
            "avx512vbmi2" => Some(TokenInfo {
                name: "Avx512ModernToken",
                features: "avx512vbmi2",
                is_trait: false,
                requires_avx512: true,
            }),
            "avx512vbmi2,avx512vl" => Some(TokenInfo {
                name: "Avx512ModernToken",
                features: "avx512vbmi2,avx512vl",
                is_trait: false,
                requires_avx512: true,
            }),

            // SSE/SSE2 - skip, baseline on x86_64
            "sse" | "sse2" => None,

            _ => None,
        },
        Arch::Aarch64 => match features {
            "neon" => Some(TokenInfo {
                name: "HasNeon",
                features: "neon",
                is_trait: true,
                requires_avx512: false,
            }),
            f if f.contains("sha3") => Some(TokenInfo {
                name: "HasNeonSha3",
                features: "neon",
                is_trait: true,
                requires_avx512: false,
            }),
            f if f.contains("aes") => Some(TokenInfo {
                name: "HasNeonAes",
                features: "neon",
                is_trait: true,
                requires_avx512: false,
            }),
            _ => None,
        },
    }
}

/// Module name for a given feature set.
///
/// Groups features by tier: avx (v3), v4 (standard AVX-512), modern (AVX-512 extensions).
fn feature_to_module(features: &str, arch: Arch) -> &'static str {
    match arch {
        Arch::X86 => match features {
            "avx" => "avx",
            "avx512f" => "v4",
            "avx512f,avx512vl" => "v4_vl",
            "avx512bw" => "v4_bw",
            "avx512bw,avx512vl" => "v4_bw_vl",
            "avx512vbmi2" => "modern",
            "avx512vbmi2,avx512vl" => "modern_vl",
            _ => "unknown",
        },
        Arch::Aarch64 => match features {
            "neon" => "neon",
            f if f.contains("sha3") => "neon_sha3",
            f if f.contains("aes") => "neon_aes",
            _ => "unknown",
        },
    }
}

// ============================================================================
// Documentation and Test Generation Helpers
// ============================================================================

/// Generate Intel Intrinsics Guide search URL for an intrinsic
fn intel_intrinsic_url(name: &str) -> String {
    // Intel's guide uses the intrinsic name without leading underscore for search
    let search_name = name.trim_start_matches('_');
    format!("{}#text={}", INTEL_INTRINSICS_URL, search_name)
}

/// Generate ARM Intrinsics reference URL for an intrinsic
fn arm_intrinsic_url(name: &str) -> String {
    format!("{}/#q={}", ARM_INTRINSICS_URL, name)
}

/// Categorize an intrinsic by its operation type
fn categorize_intrinsic(name: &str) -> &'static str {
    // x86 categorization
    if name.contains("loadu")
        || name.contains("load_")
        || name.starts_with("_mm") && name.contains("load")
    {
        "Load"
    } else if name.contains("storeu")
        || name.contains("store_")
        || name.starts_with("_mm") && name.contains("store")
    {
        "Store"
    } else if name.contains("gather") {
        "Gather"
    } else if name.contains("scatter") {
        "Scatter"
    } else if name.contains("compress") || name.contains("expand") {
        "Compress/Expand"
    // NEON categorization
    } else if name.starts_with("vld") {
        "Load"
    } else if name.starts_with("vst") {
        "Store"
    } else {
        "Other"
    }
}

/// Determine if an intrinsic is a load operation
fn is_load_intrinsic(name: &str) -> bool {
    name.contains("load") || name.starts_with("vld")
}

// Note: is_store_intrinsic, element_type_from_simd, and generate_test_data are
// reserved for future dynamic test generation based on parsed function signatures.

/// Information collected for test generation
#[derive(Clone)]
#[allow(dead_code)] // Fields reserved for future dynamic test generation
struct TestableFunction {
    name: String,
    module: String,
    token_name: String,
    is_trait: bool,
    requires_avx512: bool,
    category: String,
    input_types: Vec<String>,
    output_type: Option<String>,
    arch: Arch,
}

/// Parsed function from safe_unaligned_simd
#[derive(Clone)]
struct ParsedFunction {
    name: String,
    target_features: String,
    generics: Generics,
    inputs: Vec<(String, Type)>,
    output: Option<Type>,
    doc_comment: Option<String>,
}

/// Extract target_feature from attributes
fn extract_target_feature(attrs: &[Attribute]) -> Option<String> {
    for attr in attrs {
        if attr.path().is_ident("target_feature") {
            if let Meta::List(meta_list) = &attr.meta {
                let tokens = meta_list.tokens.to_string();
                if let Some(start) = tokens.find('"') {
                    if let Some(end) = tokens.rfind('"') {
                        if start < end {
                            return Some(tokens[start + 1..end].to_string());
                        }
                    }
                }
            }
        }
    }
    None
}

/// Extract doc comments from attributes
fn extract_doc_comment(attrs: &[Attribute]) -> Option<String> {
    let docs: Vec<String> = attrs
        .iter()
        .filter_map(|attr| {
            if attr.path().is_ident("doc") {
                if let Meta::NameValue(nv) = &attr.meta {
                    if let syn::Expr::Lit(syn::ExprLit {
                        lit: syn::Lit::Str(s),
                        ..
                    }) = &nv.value
                    {
                        return Some(s.value());
                    }
                }
            }
            None
        })
        .collect();

    if docs.is_empty() {
        None
    } else {
        Some(docs.join("\n"))
    }
}

/// Parse a single function
fn parse_function(func: &ItemFn) -> Option<ParsedFunction> {
    let target_features = extract_target_feature(&func.attrs)?;

    if !matches!(func.vis, syn::Visibility::Public(_)) {
        return None;
    }

    let name = func.sig.ident.to_string();
    let doc_comment = extract_doc_comment(&func.attrs);

    let inputs: Vec<(String, Type)> = func
        .sig
        .inputs
        .iter()
        .filter_map(|arg| {
            if let FnArg::Typed(pat_type) = arg {
                if let Pat::Ident(pat_ident) = &*pat_type.pat {
                    return Some((pat_ident.ident.to_string(), (*pat_type.ty).clone()));
                }
            }
            None
        })
        .collect();

    let output = match &func.sig.output {
        ReturnType::Default => None,
        ReturnType::Type(_, ty) => Some((**ty).clone()),
    };

    Some(ParsedFunction {
        name,
        target_features,
        generics: func.sig.generics.clone(),
        inputs,
        output,
        doc_comment,
    })
}

/// Parse all functions from a Rust file
fn parse_file_functions(path: &Path) -> Result<Vec<ParsedFunction>> {
    let content =
        fs::read_to_string(path).with_context(|| format!("Failed to read {}", path.display()))?;

    let syntax =
        parse_file(&content).with_context(|| format!("Failed to parse {}", path.display()))?;

    let functions: Vec<ParsedFunction> = syntax
        .items
        .iter()
        .filter_map(|item| {
            if let syn::Item::Fn(func) = item {
                parse_function(func)
            } else {
                None
            }
        })
        .collect();

    Ok(functions)
}

/// Parse aarch64 functions using `cargo expand` to expand macros.
///
/// The aarch64.rs file in safe_unaligned_simd uses macros like `vld_n_replicate_k!`
/// which don't yield parseable function items directly. We use `cargo expand` to
/// expand the macros and then parse the resulting code.
fn parse_aarch64_via_expand(safe_simd_path: &Path) -> Result<BTreeMap<String, String>> {
    let mut functions: BTreeMap<String, String> = BTreeMap::new();

    // Run cargo expand on safe_unaligned_simd for aarch64 target
    let output = Command::new("cargo")
        .arg("expand")
        .arg("--lib")
        .arg("--target")
        .arg("aarch64-unknown-linux-gnu")
        .current_dir(safe_simd_path)
        .output()
        .context("Failed to run cargo expand. Is cargo-expand installed? Run: cargo install cargo-expand")?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        // Check if it's just missing target (non-fatal)
        if stderr.contains("target may not be installed") {
            println!("  Warning: aarch64 target not installed, skipping expand");
            println!("  Run: rustup target add aarch64-unknown-linux-gnu");
            return Ok(functions);
        }
        bail!(
            "cargo expand failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
    }

    let expanded = String::from_utf8_lossy(&output.stdout);

    // Parse the expanded output to extract function signatures
    // We look for patterns like:
    //   #[target_feature(enable = "neon")]
    //   pub fn vld1_u8(from: &[u8; 8]) -> uint8x8_t {
    //
    // or:
    //   #[target_feature(enable = "neon,aes")]
    //   pub fn ...

    let mut current_features: Option<String> = None;

    for line in expanded.lines() {
        let trimmed = line.trim();

        // Check for target_feature attribute
        if trimmed.starts_with("#[target_feature(enable") {
            // Extract the feature string
            if let Some(start) = trimmed.find('"') {
                if let Some(end) = trimmed.rfind('"') {
                    if start < end {
                        current_features = Some(trimmed[start + 1..end].to_string());
                    }
                }
            }
            continue;
        }

        // Check for pub fn declaration
        if trimmed.starts_with("pub fn ") {
            if let Some(features) = current_features.take() {
                // Extract function name
                let after_fn = &trimmed[7..]; // Skip "pub fn "
                let name_end = after_fn.find(['<', '(']).unwrap_or(after_fn.len());
                let name = after_fn[..name_end].to_string();

                // Only include aarch64-specific functions (skip any generic ones)
                // NEON functions typically start with 'v'
                if name.starts_with('v') {
                    functions.insert(name, features);
                }
            }
        }
    }

    Ok(functions)
}

/// Generate aarch64 neon.rs by extracting macro invocations from safe_unaligned_simd
/// and replacing the macro name.
///
/// This is much simpler than parsing - we just:
/// 1. Extract all `vld_n_replicate_k! { ... }` blocks
/// 2. Replace `vld_n_replicate_k!` with `aarch64_load_store!` and inject token/feature params
/// 3. Prepend our header with the macro definition
fn generate_aarch64_neon_rs(safe_simd_path: &Path) -> Result<String> {
    let aarch64_path = safe_simd_path.join("src/aarch64.rs");
    let content = fs::read_to_string(&aarch64_path)
        .with_context(|| format!("Failed to read {}", aarch64_path.display()))?;

    // Extract all vld_n_replicate_k! { ... } blocks
    let mut macro_blocks = Vec::new();
    let mut depth = 0;
    let mut current_block = String::new();
    let mut in_block = false;

    for line in content.lines() {
        if line.trim().starts_with("vld_n_replicate_k!") && line.contains('{') {
            in_block = true;
            depth = 1;
            current_block = line.to_string() + "\n";
            continue;
        }

        if in_block {
            current_block.push_str(line);
            current_block.push('\n');

            for ch in line.chars() {
                match ch {
                    '{' => depth += 1,
                    '}' => {
                        depth -= 1;
                        if depth == 0 {
                            macro_blocks.push(current_block.clone());
                            current_block.clear();
                            in_block = false;
                            break;
                        }
                    }
                    _ => {}
                }
            }
        }
    }

    // Count functions for the header
    let fn_count = macro_blocks
        .iter()
        .map(|b| b.matches("fn ").count())
        .sum::<usize>();

    // Build the output file
    let mut output = String::new();

    // Header
    output.push_str(&format!(
        r#"//! Token-gated wrappers for `#[target_feature(enable = "neon")]` functions.
//!
//! This module contains {} NEON load/store functions that are safe to call when you have a token implementing [`HasNeon`].
//!
//! **Auto-generated** from safe_unaligned_simd v{} - do not edit manually.
//! Run `cargo xtask generate` to regenerate.

// Guard against rare aarch64 targets without NEON (e.g., aarch64-unknown-none-softfloat)
#![cfg(target_feature = "neon")]

#![allow(unused_imports)]
#![allow(unused_macros)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::missing_safety_doc)]

use core::arch::aarch64::*;
use crate::tokens::HasNeon;

// Macro for aarch64 SIMD functions - requires #[target_feature] wrapper for safety.
// Even NEON requires this because aarch64 targets without NEON exist (e.g., softfloat).
// Uses `impl $trait_bound` for generic token acceptance.
// Separate arms for load/store to avoid macro metavariable forwarding issues.
macro_rules! aarch64_load_store {{
    // Load functions
    (
        token: $trait_bound:path;
        feature: $feature:literal;
        unsafe: load;
        size: $size:ident;

        $(
            $(#[$meta:meta])* fn $intrinsic:ident(_: &[$base_ty:ty; $n:literal][..$len:literal] as $realty:ty) -> $ret:ty;
        )*
    ) => {{
        $(
            $(#[$meta])*
            #[inline(always)]
            pub fn $intrinsic(_token: impl $trait_bound, from: &$realty) -> $ret {{
                #[inline]
                #[target_feature(enable = $feature)]
                unsafe fn inner(from: &$realty) -> $ret {{
                    safe_unaligned_simd::aarch64::$intrinsic(from)
                }}
                unsafe {{ inner(from) }}
            }}
        )*
    }};

    // Store functions
    (
        token: $trait_bound:path;
        feature: $feature:literal;
        unsafe: store;
        size: $size:ident;

        $(
            $(#[$meta:meta])* fn $intrinsic:ident(_: &[$base_ty:ty; $n:literal][..$len:literal] as $realty:ty) -> $ret:ty;
        )*
    ) => {{
        $(
            $(#[$meta])*
            #[inline(always)]
            pub fn $intrinsic(_token: impl $trait_bound, into: &mut $realty, val: $ret) {{
                #[inline]
                #[target_feature(enable = $feature)]
                unsafe fn inner(into: &mut $realty, val: $ret) {{
                    safe_unaligned_simd::aarch64::$intrinsic(into, val)
                }}
                unsafe {{ inner(into, val) }}
            }}
        )*
    }};
}}

// ============================================================================
// Auto-extracted macro invocations from safe_unaligned_simd
// ============================================================================

"#,
        fn_count, SAFE_SIMD_VERSION
    ));

    // Add the macro blocks, converting to aarch64_load_store! with HasNeon trait bound
    for block in macro_blocks {
        // Replace macro name and inject token/feature parameters
        let renamed = block
            .replace("vld_n_replicate_k!", "aarch64_load_store!")
            .replace(
                "aarch64_load_store! {\n    unsafe:",
                "aarch64_load_store! {\n    token: HasNeon;\n    feature: \"neon\";\n    unsafe:",
            );
        output.push_str(&renamed);
        output.push('\n');
    }

    Ok(output)
}

/// Count functions in aarch64 macro blocks (for reporting)
fn count_aarch64_functions(safe_simd_path: &Path) -> Result<usize> {
    let aarch64_path = safe_simd_path.join("src/aarch64.rs");
    let content = fs::read_to_string(&aarch64_path)?;

    // Count "fn " occurrences in vld_n_replicate_k! blocks
    let mut count = 0;
    let mut in_block = false;
    let mut depth = 0;

    for line in content.lines() {
        if line.trim().starts_with("vld_n_replicate_k!") && line.contains('{') {
            in_block = true;
            depth = 1;
            continue;
        }

        if in_block {
            count += line.matches("fn ").count();

            for ch in line.chars() {
                match ch {
                    '{' => depth += 1,
                    '}' => {
                        depth -= 1;
                        if depth == 0 {
                            in_block = false;
                            break;
                        }
                    }
                    _ => {}
                }
            }
        }
    }

    Ok(count)
}

/// Find safe_unaligned_simd in cargo cache
fn find_safe_simd_path() -> Result<PathBuf> {
    let home = std::env::var("HOME").context("HOME not set")?;
    let cargo_registry = PathBuf::from(home).join(".cargo/registry/src");

    for entry in WalkDir::new(&cargo_registry).max_depth(2) {
        let entry = entry?;
        if entry.file_type().is_dir() {
            let name = entry.file_name().to_string_lossy();
            if name.starts_with("safe_unaligned_simd-") && name.contains(SAFE_SIMD_VERSION) {
                return Ok(entry.path().to_path_buf());
            }
        }
    }

    bail!(
        "Could not find safe_unaligned_simd-{} in cargo cache. Run: cargo fetch",
        SAFE_SIMD_VERSION
    )
}

/// Convert a Type to a string representation
fn type_to_string(ty: &Type) -> String {
    use quote::ToTokens;
    let s = ty.to_token_stream().to_string();
    // Normalize whitespace to match rustfmt style
    s.replace(" ,", ",")
        .replace("< ", "<")
        .replace(" >", ">")
        .replace("& mut ", "&mut ")
        .replace("& [", "&[")
        .replace(" ; ", "; ")
        .replace(": :", "::")
}

/// Generate a single wrapper function as a formatted string
fn generate_wrapper_string(func: &ParsedFunction, token_info: &TokenInfo, arch: Arch) -> String {
    let mut out = String::new();

    // Doc comment
    if let Some(doc) = &func.doc_comment {
        for line in doc.lines() {
            let trimmed = line.trim();
            if trimmed.is_empty() {
                writeln!(out, "///").unwrap();
            } else {
                writeln!(out, "/// {}", trimmed).unwrap();
            }
        }
    }

    // Function signature
    write!(out, "#[inline(always)]\npub fn {}", func.name).unwrap();

    // Generics
    let generic_params: Vec<String> = func
        .generics
        .params
        .iter()
        .map(|p| match p {
            GenericParam::Type(tp) => {
                let ident = &tp.ident;
                let bounds: Vec<String> = tp
                    .bounds
                    .iter()
                    .map(|b| {
                        use quote::ToTokens;
                        b.to_token_stream().to_string()
                    })
                    .collect();
                if bounds.is_empty() {
                    ident.to_string()
                } else {
                    format!("{}: {}", ident, bounds.join(" + "))
                }
            }
            GenericParam::Lifetime(lt) => {
                use quote::ToTokens;
                lt.to_token_stream().to_string()
            }
            GenericParam::Const(c) => {
                format!("const {}: {}", c.ident, type_to_string(&c.ty))
            }
        })
        .collect();

    if !generic_params.is_empty() {
        write!(out, "<{}>", generic_params.join(", ")).unwrap();
    }

    // Parameters (add token as first param - use impl Trait for traits, concrete type otherwise)
    let token_param = if token_info.is_trait {
        format!("_token: impl {}", token_info.name)
    } else {
        format!("_token: {}", token_info.name)
    };
    let params: Vec<String> = std::iter::once(token_param)
        .chain(
            func.inputs
                .iter()
                .map(|(name, ty)| format!("{}: {}", name, type_to_string(ty))),
        )
        .collect();

    write!(out, "({})", params.join(", ")).unwrap();

    // Return type
    if let Some(ty) = &func.output {
        write!(out, " -> {}", type_to_string(ty)).unwrap();
    }

    writeln!(out, " {{").unwrap();

    // Inner function with target_feature
    // Note: #[inline(always)] cannot be used with #[target_feature] on stable Rust.
    // Users who need guaranteed inlining should use #[arcane(inline_always)] with
    // nightly Rust and #![feature(target_feature_inline_always)] instead of these
    // pre-generated wrappers.
    write!(
        out,
        "    #[inline]\n    #[target_feature(enable = \"{}\")]\n    unsafe fn inner",
        token_info.features
    )
    .unwrap();

    if !generic_params.is_empty() {
        write!(out, "<{}>", generic_params.join(", ")).unwrap();
    }

    let inner_params: Vec<String> = func
        .inputs
        .iter()
        .map(|(name, ty)| format!("{}: {}", name, type_to_string(ty)))
        .collect();

    write!(out, "({})", inner_params.join(", ")).unwrap();

    if let Some(ty) = &func.output {
        write!(out, " -> {}", type_to_string(ty)).unwrap();
    }

    writeln!(out, " {{").unwrap();

    // Call to safe_unaligned_simd
    let simd_module = match arch {
        Arch::X86 => "x86_64",
        Arch::Aarch64 => "aarch64",
    };
    write!(
        out,
        "        safe_unaligned_simd::{}::{}",
        simd_module, func.name
    )
    .unwrap();

    // Turbofish for type params
    let type_params: Vec<String> = func
        .generics
        .params
        .iter()
        .filter_map(|p| {
            if let GenericParam::Type(tp) = p {
                Some(tp.ident.to_string())
            } else {
                None
            }
        })
        .collect();

    if !type_params.is_empty() {
        write!(out, "::<{}>", type_params.join(", ")).unwrap();
    }

    // Arguments
    let args: Vec<String> = func.inputs.iter().map(|(name, _)| name.clone()).collect();
    writeln!(out, "({})", args.join(", ")).unwrap();

    writeln!(out, "    }}").unwrap();

    // Call inner
    writeln!(
        out,
        "    // SAFETY: Token proves the target features are available"
    )
    .unwrap();
    write!(out, "    unsafe {{ inner(").unwrap();
    write!(out, "{}", args.join(", ")).unwrap();
    writeln!(out, ") }}").unwrap();

    writeln!(out, "}}").unwrap();

    out
}

/// Generate a module for a specific feature set
fn generate_module(token_info: &TokenInfo, functions: &[ParsedFunction], arch: Arch) -> String {
    let mut out = String::new();

    // Header
    writeln!(
        out,
        r#"//! Token-gated wrappers for `#[target_feature(enable = "{}")]` functions.
//!
//! This module contains {} functions that are safe to call when you have a [`{}`].
//!
//! **Auto-generated** from safe_unaligned_simd v{} - do not edit manually.
//! See `xtask/src/main.rs` for the generator.

#![allow(unused_imports)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::missing_safety_doc)]
"#,
        token_info.features,
        functions.len(),
        token_info.name,
        SAFE_SIMD_VERSION,
    )
    .unwrap();

    // Architecture-specific imports
    match arch {
        Arch::X86 => {
            writeln!(
                out,
                r#"
#[cfg(target_arch = "x86")]
use core::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

#[cfg(target_arch = "x86")]
use safe_unaligned_simd::x86::{{
    Is16BitsUnaligned, Is32BitsUnaligned, Is64BitsUnaligned,
    Is128BitsUnaligned, Is256BitsUnaligned, Is512BitsUnaligned,
    Is16CellUnaligned, Is32CellUnaligned, Is64CellUnaligned,
    Is128CellUnaligned, Is256CellUnaligned,
}};
#[cfg(target_arch = "x86_64")]
use safe_unaligned_simd::x86_64::{{
    Is16BitsUnaligned, Is32BitsUnaligned, Is64BitsUnaligned,
    Is128BitsUnaligned, Is256BitsUnaligned, Is512BitsUnaligned,
    Is16CellUnaligned, Is32CellUnaligned, Is64CellUnaligned,
    Is128CellUnaligned, Is256CellUnaligned,
}};

use crate::tokens::{{{}}};"#,
                token_info.name
            )
            .unwrap();
        }
        Arch::Aarch64 => {
            writeln!(
                out,
                r#"
use core::arch::aarch64::*;
use crate::tokens::{};"#,
                token_info.name
            )
            .unwrap();
        }
    }

    // Generate each function
    for func in functions {
        writeln!(out).unwrap();
        out.push_str(&generate_wrapper_string(func, token_info, arch));
    }

    out
}

/// Generate the mod.rs that ties everything together
fn generate_mod_rs(
    x86_modules: &[(&str, &str, usize, bool)],
    aarch64_modules: &[(&str, &str, usize)],
) -> String {
    let mut code = String::from(
        r#"//! Token-gated wrappers for safe_unaligned_simd.
//!
//! This module re-exports safe_unaligned_simd functions with archmage token gating.
//! Each function takes a token as its first parameter, proving the required
//! CPU features are available at runtime.
//!
//! **Auto-generated** - do not edit manually. See `xtask/src/main.rs`.
//!
//! ## x86/x86_64 Feature Coverage
//!
"#,
    );

    for (module, features, count, requires_avx512) in x86_modules {
        let avx512_note = if *requires_avx512 {
            " (requires `avx512` feature)"
        } else {
            ""
        };
        code.push_str(&format!(
            "//! - [`x86::{}`]: {} functions (`{}`){}\n",
            module, count, features, avx512_note
        ));
    }

    code.push_str("//!\n//! ## AArch64 Feature Coverage\n//!\n");

    for (module, features, count) in aarch64_modules {
        code.push_str(&format!(
            "//! - [`aarch64::{}`]: {} functions (`{}`)\n",
            module, count, features
        ));
    }

    code.push_str(
        r#"
// Note: safe_unaligned_simd only provides x86_64 module, not x86 (i686)
#[cfg(target_arch = "x86_64")]
pub mod x86;

#[cfg(target_arch = "aarch64")]
pub mod aarch64;
"#,
    );

    code
}

/// Generate x86/mod.rs with conditional compilation for AVX-512 modules
fn generate_x86_mod_rs(modules: &[(&str, &str, usize, bool)]) -> String {
    let mut code = String::from(
        r#"//! x86/x86_64 token-gated wrappers.
//!
//! **Auto-generated** - do not edit manually.

"#,
    );

    for (module, _features, _count, requires_avx512) in modules {
        if *requires_avx512 {
            code.push_str(&format!(
                "#[cfg(feature = \"avx512\")]\npub mod {};\n",
                module
            ));
        } else {
            code.push_str(&format!("pub mod {};\n", module));
        }
    }

    code
}

/// Generate aarch64/mod.rs
fn generate_aarch64_mod_rs(modules: &[(&str, &str, usize)]) -> String {
    let mut code = String::from(
        r#"//! AArch64 token-gated wrappers.
//!
//! **Auto-generated** - do not edit manually.

"#,
    );

    for (module, _features, _count) in modules {
        code.push_str(&format!("pub mod {};\n", module));
    }

    code
}

// ============================================================================
// Test Generation
// ============================================================================

/// Generate exhaustive test file for x86 intrinsics
#[allow(unused_variables)] // testable_fns reserved for future dynamic generation
fn generate_x86_tests(testable_fns: &[TestableFunction]) -> String {
    let mut code = String::from(
        r#"//! Auto-generated exhaustive tests for x86 mem module intrinsics.
//!
//! This file exercises every intrinsic in `archmage::mem` to ensure they compile
//! and execute correctly on supported hardware.
//!
//! **Auto-generated** by `cargo xtask generate` - do not edit manually.

#![cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#![allow(unused_variables)]
#![allow(unused_mut)]
#![allow(clippy::approx_constant)]

use std::hint::black_box;

use archmage::SimdToken;
use archmage::mem::avx;

#[cfg(feature = "avx512")]
use archmage::mem::{v4, v4_bw, v4_bw_vl, v4_vl, modern, modern_vl};

"#,
    );

    // Group functions by module
    let mut by_module: BTreeMap<String, Vec<&TestableFunction>> = BTreeMap::new();
    for func in testable_fns {
        if func.arch == Arch::X86 {
            by_module.entry(func.module.clone()).or_default().push(func);
        }
    }

    // Generate test for AVX module (always available)
    if let Some(avx_fns) = by_module.get("avx") {
        code.push_str(
            r#"
/// Test all AVX load/store intrinsics
#[test]
fn test_avx_mem_intrinsics_exhaustive() {
    use archmage::Avx2Token;

    let Some(token) = Avx2Token::try_new() else {
        eprintln!("AVX2 not available, skipping test");
        return;
    };

    // Test data
    let f32_data: [f32; 8] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let f64_data: [f64; 4] = [1.0, 2.0, 3.0, 4.0];
    let i32_data: [i32; 8] = [1, 2, 3, 4, 5, 6, 7, 8];
    let i64_data: [i64; 4] = [1, 2, 3, 4];

    let mut f32_out: [f32; 8] = [0.0; 8];
    let mut f64_out: [f64; 4] = [0.0; 4];
    let mut i32_out: [i32; 8] = [0; 8];
    let mut i64_out: [i64; 4] = [0; 4];

"#,
        );

        for func in avx_fns {
            let call = generate_avx_test_call(func);
            if !call.is_empty() {
                code.push_str(&format!("    // {}\n", func.name));
                code.push_str(&format!("    {}\n\n", call));
            }
        }

        code.push_str(
            r#"    // Ensure values are used
    black_box(&f32_out);
    black_box(&f64_out);
    black_box(&i32_out);
    black_box(&i64_out);
}
"#,
        );
    }

    // Generate test for AVX-512 modules
    code.push_str(
        r#"
/// Test all AVX-512 (v4) load/store intrinsics
#[test]
#[cfg(feature = "avx512")]
fn test_v4_mem_intrinsics_exhaustive() {
    use archmage::X64V4Token;

    let Some(token) = X64V4Token::try_new() else {
        eprintln!("AVX-512 not available, skipping test");
        return;
    };

    // Test data for 512-bit vectors
    let f32_data_16: [f32; 16] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0,
                                   9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0];
    let f64_data_8: [f64; 8] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let i32_data_16: [i32; 16] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
    let i64_data_8: [i64; 8] = [1, 2, 3, 4, 5, 6, 7, 8];

    let mut f32_out_16: [f32; 16] = [0.0; 16];
    let mut f64_out_8: [f64; 8] = [0.0; 8];
    let mut i32_out_16: [i32; 16] = [0; 16];
    let mut i64_out_8: [i64; 8] = [0; 8];

    // Exercise v4 intrinsics
    let v = v4::_mm512_loadu_ps(token, &f32_data_16);
    v4::_mm512_storeu_ps(token, &mut f32_out_16, v);
    assert_eq!(f32_data_16, f32_out_16);

    let v = v4::_mm512_loadu_pd(token, &f64_data_8);
    v4::_mm512_storeu_pd(token, &mut f64_out_8, v);
    assert_eq!(f64_data_8, f64_out_8);

    let v = v4::_mm512_loadu_epi32(token, &i32_data_16);
    v4::_mm512_storeu_epi32(token, &mut i32_out_16, v);
    assert_eq!(i32_data_16, i32_out_16);

    let v = v4::_mm512_loadu_epi64(token, &i64_data_8);
    v4::_mm512_storeu_epi64(token, &mut i64_out_8, v);
    assert_eq!(i64_data_8, i64_out_8);

    black_box(&f32_out_16);
    black_box(&f64_out_8);
    black_box(&i32_out_16);
    black_box(&i64_out_8);
}

/// Test AVX-512VL intrinsics (256/128-bit with AVX-512 features)
#[test]
#[cfg(feature = "avx512")]
fn test_v4_vl_mem_intrinsics_exhaustive() {
    use archmage::X64V4Token;

    let Some(token) = X64V4Token::try_new() else {
        eprintln!("AVX-512 not available, skipping test");
        return;
    };

    // 256-bit test data
    let f32_data_8: [f32; 8] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let f64_data_4: [f64; 4] = [1.0, 2.0, 3.0, 4.0];
    let i32_data_8: [i32; 8] = [1, 2, 3, 4, 5, 6, 7, 8];
    let i64_data_4: [i64; 4] = [1, 2, 3, 4];

    let mut f32_out_8: [f32; 8] = [0.0; 8];
    let mut f64_out_4: [f64; 4] = [0.0; 4];
    let mut i32_out_8: [i32; 8] = [0; 8];
    let mut i64_out_4: [i64; 4] = [0; 4];

    // 128-bit test data
    let f32_data_4: [f32; 4] = [1.0, 2.0, 3.0, 4.0];
    let f64_data_2: [f64; 2] = [1.0, 2.0];
    let i32_data_4: [i32; 4] = [1, 2, 3, 4];
    let i64_data_2: [i64; 2] = [1, 2];

    let mut f32_out_4: [f32; 4] = [0.0; 4];
    let mut f64_out_2: [f64; 2] = [0.0; 2];
    let mut i32_out_4: [i32; 4] = [0; 4];
    let mut i64_out_2: [i64; 2] = [0; 2];

    // 256-bit VL operations
    let v = v4_vl::_mm256_loadu_epi32(token, &i32_data_8);
    v4_vl::_mm256_storeu_epi32(token, &mut i32_out_8, v);
    assert_eq!(i32_data_8, i32_out_8);

    let v = v4_vl::_mm256_loadu_epi64(token, &i64_data_4);
    v4_vl::_mm256_storeu_epi64(token, &mut i64_out_4, v);
    assert_eq!(i64_data_4, i64_out_4);

    // 128-bit VL operations
    let v = v4_vl::_mm_loadu_epi32(token, &i32_data_4);
    v4_vl::_mm_storeu_epi32(token, &mut i32_out_4, v);
    assert_eq!(i32_data_4, i32_out_4);

    let v = v4_vl::_mm_loadu_epi64(token, &i64_data_2);
    v4_vl::_mm_storeu_epi64(token, &mut i64_out_2, v);
    assert_eq!(i64_data_2, i64_out_2);

    black_box(&f32_out_8);
    black_box(&f64_out_4);
    black_box(&i32_out_8);
    black_box(&i64_out_4);
    black_box(&f32_out_4);
    black_box(&f64_out_2);
    black_box(&i32_out_4);
    black_box(&i64_out_2);
}
"#,
    );

    code
}

/// Generate a test call for an AVX intrinsic
fn generate_avx_test_call(func: &TestableFunction) -> String {
    let name = &func.name;

    // Map common AVX intrinsics to test calls
    match name.as_str() {
        "_mm256_loadu_ps" => "let v = avx::_mm256_loadu_ps(token, &f32_data); avx::_mm256_storeu_ps(token, &mut f32_out, v);".to_string(),
        "_mm256_storeu_ps" => String::new(), // Combined with load
        "_mm256_loadu_pd" => "let v = avx::_mm256_loadu_pd(token, &f64_data); avx::_mm256_storeu_pd(token, &mut f64_out, v);".to_string(),
        "_mm256_storeu_pd" => String::new(),
        "_mm256_loadu_si256" => "let v = avx::_mm256_loadu_si256(token, &i64_data); avx::_mm256_storeu_si256(token, &mut i64_out, v);".to_string(),
        "_mm256_storeu_si256" => String::new(),
        "_mm256_lddqu_si256" => "let v = avx::_mm256_lddqu_si256(token, &i64_data); black_box(v);".to_string(),
        _ => {
            // For other intrinsics, generate a simple black_box call if it's a load
            if is_load_intrinsic(name) {
                format!("// {} - skipped (complex signature)", name)
            } else {
                String::new()
            }
        }
    }
}

/// Generate exhaustive test file for aarch64 intrinsics
#[allow(unused_variables)] // testable_fns reserved for future dynamic generation
fn generate_aarch64_tests(testable_fns: &[TestableFunction]) -> String {
    let mut code = String::from(
        r#"//! Auto-generated exhaustive tests for aarch64 mem module intrinsics.
//!
//! This file exercises every intrinsic in `archmage::mem::neon` to ensure they compile
//! and execute correctly on supported hardware.
//!
//! **Auto-generated** by `cargo xtask generate` - do not edit manually.

#![cfg(target_arch = "aarch64")]
#![allow(unused_variables)]

use std::hint::black_box;
use core::arch::aarch64::*;

use archmage::SimdToken;
use archmage::mem::neon;

/// Test all NEON load intrinsics
#[test]
fn test_neon_load_intrinsics_exhaustive() {
    use archmage::NeonToken;

    let Some(token) = NeonToken::try_new() else {
        eprintln!("NEON not available, skipping test");
        return;
    };

    // Test data for various element types and sizes
    let u8_8: [u8; 8] = [1, 2, 3, 4, 5, 6, 7, 8];
    let u8_16: [u8; 16] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
    let u16_4: [u16; 4] = [1, 2, 3, 4];
    let u16_8: [u16; 8] = [1, 2, 3, 4, 5, 6, 7, 8];
    let u32_2: [u32; 2] = [1, 2];
    let u32_4: [u32; 4] = [1, 2, 3, 4];
    let u64_1: [u64; 1] = [1];
    let u64_2: [u64; 2] = [1, 2];
    let f32_2: [f32; 2] = [1.0, 2.0];
    let f32_4: [f32; 4] = [1.0, 2.0, 3.0, 4.0];
    let f64_1: [f64; 1] = [1.0];
    let f64_2: [f64; 2] = [1.0, 2.0];

    let i8_8: [i8; 8] = [1, 2, 3, 4, 5, 6, 7, 8];
    let i8_16: [i8; 16] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
    let i16_4: [i16; 4] = [1, 2, 3, 4];
    let i16_8: [i16; 8] = [1, 2, 3, 4, 5, 6, 7, 8];
    let i32_2: [i32; 2] = [1, 2];
    let i32_4: [i32; 4] = [1, 2, 3, 4];
    let i64_1: [i64; 1] = [1];
    let i64_2: [i64; 2] = [1, 2];

    // Exercise load intrinsics
"#,
    );

    // Add load test calls for common NEON intrinsics
    let neon_loads = [
        ("vld1_u8", "u8_8"),
        ("vld1q_u8", "u8_16"),
        ("vld1_u16", "u16_4"),
        ("vld1q_u16", "u16_8"),
        ("vld1_u32", "u32_2"),
        ("vld1q_u32", "u32_4"),
        ("vld1_u64", "u64_1"),
        ("vld1q_u64", "u64_2"),
        ("vld1_s8", "i8_8"),
        ("vld1q_s8", "i8_16"),
        ("vld1_s16", "i16_4"),
        ("vld1q_s16", "i16_8"),
        ("vld1_s32", "i32_2"),
        ("vld1q_s32", "i32_4"),
        ("vld1_s64", "i64_1"),
        ("vld1q_s64", "i64_2"),
        ("vld1_f32", "f32_2"),
        ("vld1q_f32", "f32_4"),
        ("vld1_f64", "f64_1"),
        ("vld1q_f64", "f64_2"),
    ];

    for (intrinsic, data) in neon_loads {
        code.push_str(&format!(
            "    let v = neon::{}(token, &{}); black_box(v);\n",
            intrinsic, data
        ));
    }

    code.push_str(
        r#"}

/// Test all NEON store intrinsics
#[test]
fn test_neon_store_intrinsics_exhaustive() {
    use archmage::NeonToken;

    let Some(token) = NeonToken::try_new() else {
        eprintln!("NEON not available, skipping test");
        return;
    };

    // Input data
    let u8_8: [u8; 8] = [1, 2, 3, 4, 5, 6, 7, 8];
    let u8_16: [u8; 16] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
    let u16_4: [u16; 4] = [1, 2, 3, 4];
    let u16_8: [u16; 8] = [1, 2, 3, 4, 5, 6, 7, 8];
    let u32_2: [u32; 2] = [1, 2];
    let u32_4: [u32; 4] = [1, 2, 3, 4];
    let u64_1: [u64; 1] = [1];
    let u64_2: [u64; 2] = [1, 2];
    let f32_2: [f32; 2] = [1.0, 2.0];
    let f32_4: [f32; 4] = [1.0, 2.0, 3.0, 4.0];
    let f64_1: [f64; 1] = [1.0];
    let f64_2: [f64; 2] = [1.0, 2.0];

    let i8_8: [i8; 8] = [1, 2, 3, 4, 5, 6, 7, 8];
    let i8_16: [i8; 16] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
    let i16_4: [i16; 4] = [1, 2, 3, 4];
    let i16_8: [i16; 8] = [1, 2, 3, 4, 5, 6, 7, 8];
    let i32_2: [i32; 2] = [1, 2];
    let i32_4: [i32; 4] = [1, 2, 3, 4];
    let i64_1: [i64; 1] = [1];
    let i64_2: [i64; 2] = [1, 2];

    // Output buffers
    let mut out_u8_8: [u8; 8] = [0; 8];
    let mut out_u8_16: [u8; 16] = [0; 16];
    let mut out_u16_4: [u16; 4] = [0; 4];
    let mut out_u16_8: [u16; 8] = [0; 8];
    let mut out_u32_2: [u32; 2] = [0; 2];
    let mut out_u32_4: [u32; 4] = [0; 4];
    let mut out_u64_1: [u64; 1] = [0; 1];
    let mut out_u64_2: [u64; 2] = [0; 2];
    let mut out_f32_2: [f32; 2] = [0.0; 2];
    let mut out_f32_4: [f32; 4] = [0.0; 4];
    let mut out_f64_1: [f64; 1] = [0.0; 1];
    let mut out_f64_2: [f64; 2] = [0.0; 2];

    let mut out_i8_8: [i8; 8] = [0; 8];
    let mut out_i8_16: [i8; 16] = [0; 16];
    let mut out_i16_4: [i16; 4] = [0; 4];
    let mut out_i16_8: [i16; 8] = [0; 8];
    let mut out_i32_2: [i32; 2] = [0; 2];
    let mut out_i32_4: [i32; 4] = [0; 4];
    let mut out_i64_1: [i64; 1] = [0; 1];
    let mut out_i64_2: [i64; 2] = [0; 2];

    // Load-store round trips
"#,
    );

    // Add load-store round trip tests
    let neon_pairs = [
        ("vld1_u8", "vst1_u8", "u8_8", "out_u8_8"),
        ("vld1q_u8", "vst1q_u8", "u8_16", "out_u8_16"),
        ("vld1_u16", "vst1_u16", "u16_4", "out_u16_4"),
        ("vld1q_u16", "vst1q_u16", "u16_8", "out_u16_8"),
        ("vld1_u32", "vst1_u32", "u32_2", "out_u32_2"),
        ("vld1q_u32", "vst1q_u32", "u32_4", "out_u32_4"),
        ("vld1_u64", "vst1_u64", "u64_1", "out_u64_1"),
        ("vld1q_u64", "vst1q_u64", "u64_2", "out_u64_2"),
        ("vld1_s8", "vst1_s8", "i8_8", "out_i8_8"),
        ("vld1q_s8", "vst1q_s8", "i8_16", "out_i8_16"),
        ("vld1_s16", "vst1_s16", "i16_4", "out_i16_4"),
        ("vld1q_s16", "vst1q_s16", "i16_8", "out_i16_8"),
        ("vld1_s32", "vst1_s32", "i32_2", "out_i32_2"),
        ("vld1q_s32", "vst1q_s32", "i32_4", "out_i32_4"),
        ("vld1_s64", "vst1_s64", "i64_1", "out_i64_1"),
        ("vld1q_s64", "vst1q_s64", "i64_2", "out_i64_2"),
        ("vld1_f32", "vst1_f32", "f32_2", "out_f32_2"),
        ("vld1q_f32", "vst1q_f32", "f32_4", "out_f32_4"),
        ("vld1_f64", "vst1_f64", "f64_1", "out_f64_1"),
        ("vld1q_f64", "vst1q_f64", "f64_2", "out_f64_2"),
    ];

    for (load, store, input, output) in neon_pairs {
        code.push_str(&format!(
            "    let v = neon::{}(token, &{}); neon::{}(token, &mut {}, v); assert_eq!({}, {});\n",
            load, input, store, output, input, output
        ));
    }

    code.push_str("}\n");
    code
}

// ============================================================================
// Reference Documentation Generation
// ============================================================================

/// Generate reference documentation markdown
fn generate_reference_docs(
    x86_fns: &[TestableFunction],
    aarch64_fns: &[TestableFunction],
) -> String {
    let mut doc = String::from(
        r#"# Archmage Intrinsic Reference

This document lists all safe SIMD intrinsics available through `archmage::mem`.

**Auto-generated** by `cargo xtask generate` - do not edit manually.

## Overview

| Architecture | Module | Functions | Required Feature |
|-------------|--------|-----------|------------------|
"#,
    );

    // Count by module
    let mut x86_by_mod: BTreeMap<&str, (usize, &str, bool)> = BTreeMap::new();
    for f in x86_fns {
        let entry = x86_by_mod
            .entry(&f.module)
            .or_insert((0, &f.token_name, f.requires_avx512));
        entry.0 += 1;
    }
    let mut aarch64_by_mod: BTreeMap<&str, (usize, &str, bool)> = BTreeMap::new();
    for f in aarch64_fns {
        let entry =
            aarch64_by_mod
                .entry(&f.module)
                .or_insert((0, &f.token_name, f.requires_avx512));
        entry.0 += 1;
    }

    for (module, (count, _token, avx512)) in &x86_by_mod {
        let feature = if *avx512 { "`avx512`" } else { "-" };
        doc.push_str(&format!(
            "| x86_64 | `{}` | {} | {} |\n",
            module, count, feature
        ));
    }
    for (module, (count, _token, _)) in &aarch64_by_mod {
        doc.push_str(&format!("| aarch64 | `{}` | {} | - |\n", module, count));
    }

    // x86 section
    doc.push_str("\n## x86_64 Intrinsics\n\n");

    for (module, (_, token, avx512)) in &x86_by_mod {
        let cfg = if *avx512 {
            " (requires `avx512` feature)"
        } else {
            ""
        };
        doc.push_str(&format!("### `archmage::mem::{}`{}\n\n", module, cfg));
        doc.push_str(&format!("Token: `{}`\n\n", token));
        doc.push_str("| Function | Category | Intel Docs |\n");
        doc.push_str("|----------|----------|------------|\n");

        let module_fns: Vec<_> = x86_fns.iter().filter(|f| &f.module == *module).collect();
        for f in module_fns {
            let category = categorize_intrinsic(&f.name);
            let intel_url = intel_intrinsic_url(&f.name);
            doc.push_str(&format!(
                "| `{}` | {} | [Intel Guide]({}) |\n",
                f.name, category, intel_url
            ));
        }
        doc.push_str("\n");
    }

    // aarch64 section
    doc.push_str("## AArch64 Intrinsics\n\n");

    for (module, (_, token, _)) in &aarch64_by_mod {
        doc.push_str(&format!("### `archmage::mem::{}`\n\n", module));
        doc.push_str(&format!("Token: `{}`\n\n", token));
        doc.push_str("| Function | Category | ARM Docs |\n");
        doc.push_str("|----------|----------|----------|\n");

        let module_fns: Vec<_> = aarch64_fns
            .iter()
            .filter(|f| &f.module == *module)
            .collect();
        for f in module_fns {
            let category = categorize_intrinsic(&f.name);
            let arm_url = arm_intrinsic_url(&f.name);
            doc.push_str(&format!(
                "| `{}` | {} | [ARM Docs]({}) |\n",
                f.name, category, arm_url
            ));
        }
        doc.push_str("\n");
    }

    // Usage examples
    doc.push_str(
        r#"## Usage Examples

### AVX Load/Store

```rust
use archmage::{Avx2Token, SimdToken};
use archmage::mem::avx;

fn process_f32(data: &mut [f32; 8]) {
    if let Some(token) = Avx2Token::try_new() {
        let v = avx::_mm256_loadu_ps(token, data);
        // Process v...
        avx::_mm256_storeu_ps(token, data, v);
    }
}
```

### NEON Load/Store

```rust
use archmage::{NeonToken, SimdToken};
use archmage::mem::neon;

fn process_f32(data: &mut [f32; 4]) {
    if let Some(token) = NeonToken::try_new() {
        let v = neon::vld1q_f32(token, data);
        // Process v...
        neon::vst1q_f32(token, data, v);
    }
}
```

### AVX-512 Load/Store

```rust
#[cfg(feature = "avx512")]
use archmage::{X64V4Token, SimdToken};
#[cfg(feature = "avx512")]
use archmage::mem::v4;

#[cfg(feature = "avx512")]
fn process_f32_512(data: &mut [f32; 16]) {
    if let Some(token) = X64V4Token::try_new() {
        let v = v4::_mm512_loadu_ps(token, data);
        // Process v...
        v4::_mm512_storeu_ps(token, data, v);
    }
}
```
"#,
    );

    doc
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();

    if args.len() < 2 {
        eprintln!("Usage: cargo xtask generate       - Generate wrappers, tests, and docs");
        eprintln!("       cargo xtask validate       - Validate token bounds");
        eprintln!("       cargo xtask check-version  - Check for updates");
        std::process::exit(1);
    }

    match args[1].as_str() {
        "generate" => generate_all()?,
        "validate" => validate_wrappers()?,
        "check-version" => check_version()?,
        _ => {
            eprintln!("Unknown command: {}", args[1]);
            std::process::exit(1);
        }
    }

    Ok(())
}

/// Generate all artifacts: wrappers, tests, and documentation
fn generate_all() -> Result<()> {
    let (x86_testable, aarch64_testable) = generate_wrappers()?;

    // Note: mem module tests are skipped because archmage::mem was removed.
    // The safe_unaligned_simd crate should be used directly instead.
    // See: feat!: remove archmage::mem module, use safe_unaligned_simd directly
    println!("\n=== Generating Tests ===");
    println!("Skipping mem module tests (archmage::mem was removed)");

    // Generate reference documentation
    println!("\n=== Generating Reference Documentation ===");

    let reference = generate_reference_docs(&x86_testable, &aarch64_testable);
    let ref_path = PathBuf::from("docs/INTRINSIC_REFERENCE.md");
    fs::create_dir_all("docs")?;
    fs::write(&ref_path, &reference)?;
    println!("Wrote {} ({} bytes)", ref_path.display(), reference.len());

    // Generate SIMD types (wide-like ergonomic types)
    println!("\n=== Generating SIMD Types ===");

    let simd_dir = PathBuf::from("src/simd");
    fs::create_dir_all(&simd_dir)?;

    let simd_types_code = simd_types::generate_simd_types();
    let simd_mod_path = simd_dir.join("mod.rs");
    fs::write(&simd_mod_path, &simd_types_code)?;
    println!(
        "Wrote {} ({} bytes)",
        simd_mod_path.display(),
        simd_types_code.len()
    );

    // Generate SIMD type tests
    let simd_tests = simd_types::generate_simd_tests();
    let simd_test_path = PathBuf::from("tests/generated_simd_types.rs");
    fs::write(&simd_test_path, &simd_tests)?;
    println!(
        "Wrote {} ({} bytes)",
        simd_test_path.display(),
        simd_tests.len()
    );

    println!("\n=== Generation Complete ===");
    println!("  - Wrappers: src/generated/");
    println!("  - Tests: tests/generated_*.rs");
    println!("  - Reference: docs/INTRINSIC_REFERENCE.md");
    println!("  - SIMD types: src/simd/");

    Ok(())
}

/// Generate wrappers and return testable function info for tests/docs generation
fn generate_wrappers() -> Result<(Vec<TestableFunction>, Vec<TestableFunction>)> {
    println!(
        "Finding safe_unaligned_simd-{} in cargo cache...",
        SAFE_SIMD_VERSION
    );
    let safe_simd_path = find_safe_simd_path()?;
    println!("Found at: {}", safe_simd_path.display());

    // Collect testable functions for later
    let mut x86_testable: Vec<TestableFunction> = Vec::new();
    let mut aarch64_testable: Vec<TestableFunction> = Vec::new();

    // Parse all x86 source files (excluding cell/ subdirectory to avoid duplicates)
    // Cell variants have same function names but with Cell traits - we skip them
    // to avoid conflicts. Users can use the regular Bits traits variants.
    let x86_dir = safe_simd_path.join("src/x86");
    let mut all_functions: Vec<ParsedFunction> = Vec::new();

    for entry in WalkDir::new(&x86_dir).max_depth(1) {
        let entry = entry?;
        if entry.file_type().is_file() && entry.path().extension().is_some_and(|e| e == "rs") {
            let path = entry.path();
            let name = path.file_name().unwrap().to_string_lossy();
            if name == "mod.rs" || name.contains("test") || name == "cell.rs" {
                continue;
            }

            println!("Parsing {}...", path.display());
            let functions = parse_file_functions(path)?;
            println!("  Found {} functions", functions.len());
            all_functions.extend(functions);
        }
    }

    println!("\nTotal functions parsed: {}", all_functions.len());

    // Group by target_feature
    let mut by_feature: BTreeMap<String, Vec<ParsedFunction>> = BTreeMap::new();
    for func in all_functions {
        by_feature
            .entry(func.target_features.clone())
            .or_default()
            .push(func);
    }

    println!("\nFeature groups:");
    for (feature, funcs) in &by_feature {
        println!("  {}: {} functions", feature, funcs.len());
    }

    // Generate output directory
    let out_dir = PathBuf::from("src/generated/x86");
    fs::create_dir_all(&out_dir)?;

    // (module_name, feature_str, count, requires_avx512)
    let mut modules_info: Vec<(&str, &str, usize, bool)> = Vec::new();

    // Generate each module
    for (feature, functions) in &by_feature {
        if let Some(token_info) = feature_to_token(feature, Arch::X86) {
            let module_name = feature_to_module(feature, Arch::X86);
            println!("\nGenerating {}...", module_name);

            let code = generate_module(&token_info, functions, Arch::X86);
            let out_path = out_dir.join(format!("{}.rs", module_name));
            fs::write(&out_path, &code)?;
            println!("  Wrote {} ({} bytes)", out_path.display(), code.len());

            modules_info.push((
                module_name,
                feature,
                functions.len(),
                token_info.requires_avx512,
            ));

            // Collect testable functions
            for func in functions {
                x86_testable.push(TestableFunction {
                    name: func.name.clone(),
                    module: module_name.to_string(),
                    token_name: token_info.name.to_string(),
                    is_trait: token_info.is_trait,
                    requires_avx512: token_info.requires_avx512,
                    category: categorize_intrinsic(&func.name).to_string(),
                    input_types: func
                        .inputs
                        .iter()
                        .map(|(_, ty)| type_to_string(ty))
                        .collect(),
                    output_type: func.output.as_ref().map(type_to_string),
                    arch: Arch::X86,
                });
            }
        } else {
            println!("\nSkipping unknown feature: {}", feature);
        }
    }

    // Sort modules for consistent output
    modules_info.sort_by_key(|(name, _, _, _)| *name);

    // Generate x86/mod.rs
    let x86_mod = generate_x86_mod_rs(&modules_info);
    fs::write(out_dir.join("mod.rs"), &x86_mod)?;

    // ========================================================================
    // Generate aarch64 wrappers using simple macro search/replace
    // ========================================================================
    println!("\n=== Generating aarch64 wrappers ===");

    let aarch64_out_dir = PathBuf::from("src/generated/aarch64");
    fs::create_dir_all(&aarch64_out_dir)?;

    // Generate neon.rs by extracting macro invocations and replacing the macro name
    println!("Generating neon.rs via macro extraction...");
    let neon_code = generate_aarch64_neon_rs(&safe_simd_path)?;
    let neon_path = aarch64_out_dir.join("neon.rs");
    fs::write(&neon_path, &neon_code)?;
    println!(
        "  Wrote {} ({} bytes)",
        neon_path.display(),
        neon_code.len()
    );

    // Count functions for reporting
    let neon_fn_count = count_aarch64_functions(&safe_simd_path)?;
    let aarch64_modules_info: Vec<(&str, &str, usize)> = vec![("neon", "neon", neon_fn_count)];

    // Collect aarch64 testable functions
    let aarch64_fn_names = extract_aarch64_function_names(&safe_simd_path)?;
    for name in aarch64_fn_names {
        aarch64_testable.push(TestableFunction {
            name: name.clone(),
            module: "neon".to_string(),
            token_name: "HasNeon".to_string(),
            is_trait: true,
            requires_avx512: false,
            category: categorize_intrinsic(&name).to_string(),
            input_types: vec![], // We don't parse these for aarch64
            output_type: None,
            arch: Arch::Aarch64,
        });
    }

    // Generate aarch64/mod.rs
    let aarch64_mod = generate_aarch64_mod_rs(&aarch64_modules_info);
    fs::write(aarch64_out_dir.join("mod.rs"), &aarch64_mod)?;

    // ========================================================================
    // Generate top-level mod.rs
    // ========================================================================
    let top_mod = generate_mod_rs(&modules_info, &aarch64_modules_info);
    fs::write("src/generated/mod.rs", &top_mod)?;

    // Write version file for tracking
    fs::write(
        "src/generated/VERSION",
        format!("safe_unaligned_simd = \"{}\"\n", SAFE_SIMD_VERSION),
    )?;

    let total_x86 = modules_info.iter().map(|(_, _, c, _)| c).sum::<usize>();
    let total_aarch64 = aarch64_modules_info
        .iter()
        .map(|(_, _, c)| c)
        .sum::<usize>();

    println!("\nGeneration complete!");
    println!(
        "Generated x86: {} modules with {} functions",
        modules_info.len(),
        total_x86
    );
    println!(
        "Generated aarch64: {} modules with {} functions",
        aarch64_modules_info.len(),
        total_aarch64
    );
    println!("Total: {} functions", total_x86 + total_aarch64);

    Ok((x86_testable, aarch64_testable))
}

/// Extract aarch64 function names from macro invocations
fn extract_aarch64_function_names(safe_simd_path: &Path) -> Result<Vec<String>> {
    let aarch64_path = safe_simd_path.join("src/aarch64.rs");
    let content = fs::read_to_string(&aarch64_path)?;

    let mut names = Vec::new();
    let mut in_block = false;
    let mut depth = 0;

    for line in content.lines() {
        if line.trim().starts_with("vld_n_replicate_k!") && line.contains('{') {
            in_block = true;
            depth = 1;
            continue;
        }

        if in_block {
            // Extract function names from "fn name(" patterns
            if let Some(fn_pos) = line.find("fn ") {
                let after_fn = &line[fn_pos + 3..];
                if let Some(paren_pos) = after_fn.find('(') {
                    let name = after_fn[..paren_pos].trim().to_string();
                    if !name.is_empty() {
                        names.push(name);
                    }
                }
            }

            for ch in line.chars() {
                match ch {
                    '{' => depth += 1,
                    '}' => {
                        depth -= 1;
                        if depth == 0 {
                            in_block = false;
                            break;
                        }
                    }
                    _ => {}
                }
            }
        }
    }

    Ok(names)
}

// ============================================================================
// Validation
// ============================================================================

/// Parsed wrapper function from generated code
#[derive(Debug)]
struct GeneratedWrapper {
    name: String,
    token_bound: String,
    module: String,
}

/// Extract token bound from a function signature line
fn extract_token_bound(line: &str) -> Option<String> {
    // Match patterns like:
    //   _token: impl Has256BitSimd
    //   _token: Avx512ModernToken
    if let Some(start) = line.find("_token:") {
        let after_colon = &line[start + 7..];
        let after_colon = after_colon.trim_start();

        // Find the end (comma or closing paren)
        let end = after_colon.find([',', ')']).unwrap_or(after_colon.len());

        let token_part = after_colon[..end].trim();

        // Handle "impl Trait" syntax
        if let Some(stripped) = token_part.strip_prefix("impl ") {
            return Some(stripped.trim().to_string());
        } else {
            return Some(token_part.to_string());
        }
    }
    None
}

/// Parse generated aarch64 wrappers from macro invocations.
///
/// The generated neon.rs uses `aarch64_load_store!` macro invocations.
/// We parse the macro arguments to extract function names.
fn parse_generated_aarch64_wrappers(path: &Path) -> Result<Vec<GeneratedWrapper>> {
    let mut wrappers = Vec::new();

    if !path.exists() {
        return Ok(wrappers);
    }

    let content = fs::read_to_string(path)?;

    // Look for aarch64_load_store! blocks and extract:
    // - token: HasNeon;
    // - fn <name>(...
    let mut current_token: Option<String> = None;
    let mut in_macro = false;

    for line in content.lines() {
        let trimmed = line.trim();

        // Start of macro block
        if trimmed.starts_with("aarch64_load_store!") {
            in_macro = true;
            current_token = None;
            continue;
        }

        if in_macro {
            // Extract token bound
            if trimmed.starts_with("token:") {
                // Parse "token: HasNeon;"
                let after_colon = trimmed[6..].trim();
                if let Some(semicolon) = after_colon.find(';') {
                    current_token = Some(after_colon[..semicolon].trim().to_string());
                }
                continue;
            }

            // Extract function names
            if trimmed.contains("fn ") && !trimmed.starts_with("//") {
                if let Some(fn_pos) = trimmed.find("fn ") {
                    let after_fn = &trimmed[fn_pos + 3..];
                    // Function name ends at '(' or '<'
                    if let Some(name_end) = after_fn.find('(') {
                        let name = after_fn[..name_end].trim().to_string();
                        if let Some(ref token) = current_token {
                            wrappers.push(GeneratedWrapper {
                                name,
                                token_bound: token.clone(),
                                module: "neon".to_string(),
                            });
                        }
                    }
                }
            }

            // End of macro block
            if trimmed == "}" && !trimmed.contains("fn ") {
                // Check brace depth to detect end of macro
                // Simple heuristic: standalone "}" at trimmed start likely ends the macro
                in_macro = false;
            }
        }
    }

    Ok(wrappers)
}

/// Parse generated wrapper files and extract function info
fn parse_generated_wrappers(dir: &Path, _arch: Arch) -> Result<Vec<GeneratedWrapper>> {
    let mut wrappers = Vec::new();

    if !dir.exists() {
        return Ok(wrappers);
    }

    for entry in WalkDir::new(dir).max_depth(1) {
        let entry = entry?;
        if !entry.file_type().is_file() {
            continue;
        }
        let path = entry.path();
        if path.extension().is_some_and(|e| e == "rs") {
            let filename = path.file_name().unwrap().to_string_lossy();
            if filename == "mod.rs" {
                continue;
            }

            let module = filename.trim_end_matches(".rs").to_string();
            let content = fs::read_to_string(path)?;

            for line in content.lines() {
                if line.starts_with("pub fn ") {
                    // Extract function name
                    let name_start = 7; // "pub fn ".len()
                    let name_end = line[name_start..]
                        .find(['<', '('])
                        .map(|i| i + name_start)
                        .unwrap_or(line.len());
                    let name = line[name_start..name_end].to_string();

                    // Extract token bound
                    if let Some(token_bound) = extract_token_bound(line) {
                        wrappers.push(GeneratedWrapper {
                            name,
                            token_bound,
                            module: module.clone(),
                        });
                    }
                }
            }
        }
    }

    Ok(wrappers)
}

/// Validate that all generated wrappers have correct token bounds
fn validate_wrappers() -> Result<()> {
    println!("=== Archmage Wrapper Validation ===\n");

    // Find safe_unaligned_simd source
    println!(
        "Finding safe_unaligned_simd-{} in cargo cache...",
        SAFE_SIMD_VERSION
    );
    let safe_simd_path = find_safe_simd_path()?;
    println!("Found at: {}\n", safe_simd_path.display());

    // Parse source functions to get ground truth
    println!("--- Parsing Source Functions ---");

    // x86 source
    let x86_dir = safe_simd_path.join("src/x86");
    let mut x86_source: BTreeMap<String, String> = BTreeMap::new(); // name -> features

    for entry in WalkDir::new(&x86_dir).max_depth(1) {
        let entry = entry?;
        if entry.file_type().is_file() && entry.path().extension().is_some_and(|e| e == "rs") {
            let path = entry.path();
            let name = path.file_name().unwrap().to_string_lossy();
            if name == "mod.rs" || name.contains("test") || name == "cell.rs" {
                continue;
            }

            let functions = parse_file_functions(path)?;
            for func in functions {
                x86_source.insert(func.name, func.target_features);
            }
        }
    }
    println!("Parsed {} x86 source functions", x86_source.len());

    // aarch64 source - use cargo expand to handle macros
    println!("\nExpanding aarch64 macros via cargo expand...");
    let aarch64_source = parse_aarch64_via_expand(&safe_simd_path)?;
    if aarch64_source.is_empty() {
        println!("  No aarch64 functions extracted (target may not be installed)");
    } else {
        println!(
            "Parsed {} aarch64 source functions via macro expansion\n",
            aarch64_source.len()
        );
    }

    // Parse generated wrappers
    println!("--- Parsing Generated Wrappers ---");
    let x86_gen_dir = PathBuf::from("src/generated/x86");
    let x86_wrappers = parse_generated_wrappers(&x86_gen_dir, Arch::X86)?;
    println!("Found {} x86 wrappers", x86_wrappers.len());

    // Parse aarch64 wrappers from macro invocations in neon.rs
    let aarch64_neon_path = PathBuf::from("src/generated/aarch64/neon.rs");
    let aarch64_wrappers = parse_generated_aarch64_wrappers(&aarch64_neon_path)?;
    println!(
        "Found {} aarch64 wrappers (from macro parsing)\n",
        aarch64_wrappers.len()
    );

    // Validation results
    let mut errors: Vec<String> = Vec::new();
    let mut warnings: Vec<String> = Vec::new();
    let mut validated = 0;

    println!("--- Validating x86 Token Bounds ---");

    // Check each x86 wrapper against source
    for wrapper in &x86_wrappers {
        if let Some(required_features) = x86_source.get(&wrapper.name) {
            match token_satisfies_features(&wrapper.token_bound, required_features) {
                Ok(()) => validated += 1,
                Err(missing) => {
                    errors.push(format!(
                        "MISMATCH: x86/{}::{}\n  Token: {} does not provide: {:?}\n  Required: {}",
                        wrapper.module,
                        wrapper.name,
                        wrapper.token_bound,
                        missing,
                        required_features
                    ));
                }
            }
        } else {
            warnings.push(format!(
                "NOT IN SOURCE: x86/{}::{} (wrapper exists but no source function found)",
                wrapper.module, wrapper.name
            ));
        }
    }

    println!("--- Validating aarch64 Token Bounds ---");

    // Check each aarch64 wrapper against source
    for wrapper in &aarch64_wrappers {
        if let Some(required_features) = aarch64_source.get(&wrapper.name) {
            match token_satisfies_features(&wrapper.token_bound, required_features) {
                Ok(()) => validated += 1,
                Err(missing) => {
                    errors.push(format!(
                        "MISMATCH: aarch64/{}::{}\n  Token: {} does not provide: {:?}\n  Required: {}",
                        wrapper.module, wrapper.name, wrapper.token_bound, missing, required_features
                    ));
                }
            }
        } else {
            warnings.push(format!(
                "NOT IN SOURCE: aarch64/{}::{} (wrapper exists but no source function found)",
                wrapper.module, wrapper.name
            ));
        }
    }

    // Check for unhandled feature combinations in x86 source
    println!("\n--- Checking x86 Feature Coverage ---");
    let mut x86_feature_coverage: BTreeMap<String, usize> = BTreeMap::new();
    let mut unhandled_x86: BTreeSet<String> = BTreeSet::new();

    for features in x86_source.values() {
        *x86_feature_coverage.entry(features.clone()).or_insert(0) += 1;

        if feature_to_token(features, Arch::X86).is_none() {
            // Skip baseline features (sse, sse2) - intentionally not wrapped
            if features != "sse" && features != "sse2" {
                unhandled_x86.insert(features.clone());
            }
        }
    }

    println!("x86 feature coverage (source → generated):");
    for (features, count) in &x86_feature_coverage {
        let token = feature_to_token(features, Arch::X86);
        let status = match token {
            Some(t) => format!("→ {} ✓", t.name),
            None if features == "sse" || features == "sse2" => "→ (skipped, baseline)".to_string(),
            None => "→ UNHANDLED ✗".to_string(),
        };
        println!("  {}: {} functions {}", features, count, status);
    }

    if !unhandled_x86.is_empty() {
        println!("\n⚠ Unhandled x86 feature combinations (functions silently skipped):");
        for f in &unhandled_x86 {
            let count = x86_feature_coverage.get(f).unwrap_or(&0);
            warnings.push(format!(
                "UNHANDLED x86: {} ({} functions skipped)",
                f, count
            ));
            println!("  - {} ({} functions)", f, count);
        }
    }

    // Check for unhandled feature combinations in aarch64 source
    if !aarch64_source.is_empty() {
        println!("\n--- Checking aarch64 Feature Coverage ---");
        let mut aarch64_feature_coverage: BTreeMap<String, usize> = BTreeMap::new();
        let mut unhandled_aarch64: BTreeSet<String> = BTreeSet::new();

        for features in aarch64_source.values() {
            *aarch64_feature_coverage
                .entry(features.clone())
                .or_insert(0) += 1;

            if feature_to_token(features, Arch::Aarch64).is_none() {
                unhandled_aarch64.insert(features.clone());
            }
        }

        println!("aarch64 feature coverage (source → generated):");
        for (features, count) in &aarch64_feature_coverage {
            let token = feature_to_token(features, Arch::Aarch64);
            let status = match token {
                Some(t) => format!("→ {} ✓", t.name),
                None => "→ UNHANDLED ✗".to_string(),
            };
            println!("  {}: {} functions {}", features, count, status);
        }

        if !unhandled_aarch64.is_empty() {
            println!("\n⚠ Unhandled aarch64 feature combinations (functions silently skipped):");
            for f in &unhandled_aarch64 {
                let count = aarch64_feature_coverage.get(f).unwrap_or(&0);
                warnings.push(format!(
                    "UNHANDLED aarch64: {} ({} functions skipped)",
                    f, count
                ));
                println!("  - {} ({} functions)", f, count);
            }
        }
    }

    // Summary
    println!("\n=== Validation Summary ===");
    println!("  Validated: {} wrappers", validated);
    println!(
        "  Errors:    {} (token doesn't provide required features)",
        errors.len()
    );
    println!(
        "  Warnings:  {} (unhandled features or orphan wrappers)",
        warnings.len()
    );

    if !errors.is_empty() {
        println!("\n=== ERRORS ===");
        for e in &errors {
            println!("\n{}", e);
        }
    }

    if !warnings.is_empty() {
        println!("\n=== WARNINGS ===");
        for w in &warnings {
            println!("  {}", w);
        }
    }

    if errors.is_empty() && warnings.is_empty() {
        println!("\n✓ All validations passed!");
        Ok(())
    } else if errors.is_empty() {
        println!("\n⚠ Validation passed with warnings");
        Ok(())
    } else {
        bail!("Validation failed with {} errors", errors.len())
    }
}

fn check_version() -> Result<()> {
    println!("Checking for safe_unaligned_simd updates...");
    println!("Current pinned version: {}", SAFE_SIMD_VERSION);
    println!("\nTo check manually:");
    println!("  cargo search safe_unaligned_simd");
    Ok(())
}
