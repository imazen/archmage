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

/// Version of safe_unaligned_simd we're generating from
const SAFE_SIMD_VERSION: &str = "0.2.3";

/// Architecture for code generation
#[derive(Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)] // Aarch64 reserved for future SVE support
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
            "sse", "sse2", "sse3", "ssse3", "sse4.1", "sse4.2", "popcnt",
            "avx", "avx2", "fma", "bmi1", "bmi2", "f16c", "lzcnt",
            "avx512f", "avx512bw", "avx512cd", "avx512dq", "avx512vl",
        ]),

        // x86 concrete tokens
        "Sse41Token" => Some(&["sse", "sse2", "sse3", "ssse3", "sse4.1"]),
        "Sse42Token" => Some(&["sse", "sse2", "sse3", "ssse3", "sse4.1", "sse4.2"]),
        "AvxToken" => Some(&["sse", "sse2", "avx"]),
        "Avx2Token" => Some(&["sse", "sse2", "avx", "avx2"]),
        "Avx2FmaToken" => Some(&["sse", "sse2", "avx", "avx2", "fma"]),
        "X64V2Token" => Some(&["sse", "sse2", "sse3", "ssse3", "sse4.1", "sse4.2", "popcnt"]),
        "X64V3Token" | "Desktop64" => Some(&[
            "sse", "sse2", "sse3", "ssse3", "sse4.1", "sse4.2", "popcnt",
            "avx", "avx2", "fma", "bmi1", "bmi2", "f16c", "lzcnt",
        ]),
        "X64V4Token" | "Avx512Token" | "Server64" => Some(&[
            "sse", "sse2", "sse3", "ssse3", "sse4.1", "sse4.2", "popcnt",
            "avx", "avx2", "fma", "bmi1", "bmi2", "f16c", "lzcnt",
            "avx512f", "avx512bw", "avx512cd", "avx512dq", "avx512vl",
        ]),
        "Avx512ModernToken" => Some(&[
            "sse", "sse2", "sse3", "ssse3", "sse4.1", "sse4.2", "popcnt",
            "avx", "avx2", "fma", "bmi1", "bmi2", "f16c", "lzcnt",
            "avx512f", "avx512bw", "avx512cd", "avx512dq", "avx512vl",
            "avx512vpopcntdq", "avx512ifma", "avx512vbmi", "avx512vbmi2",
            "avx512bitalg", "avx512vnni", "avx512bf16", "vpclmulqdq", "gfni", "vaes",
        ]),
        "Avx512Fp16Token" => Some(&[
            "sse", "sse2", "sse3", "ssse3", "sse4.1", "sse4.2", "popcnt",
            "avx", "avx2", "fma", "bmi1", "bmi2", "f16c", "lzcnt",
            "avx512f", "avx512bw", "avx512cd", "avx512dq", "avx512vl",
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
                let name_end = after_fn
                    .find(['<', '('])
                    .unwrap_or(after_fn.len());
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
fn generate_wrapper_string(
    func: &ParsedFunction,
    token_info: &TokenInfo,
    arch: Arch,
) -> String {
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
fn generate_module(
    token_info: &TokenInfo,
    functions: &[ParsedFunction],
    arch: Arch,
) -> String {
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
        let avx512_note = if *requires_avx512 { " (requires `avx512` feature)" } else { "" };
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
            code.push_str(&format!("#[cfg(feature = \"avx512\")]\npub mod {};\n", module));
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

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();

    if args.len() < 2 {
        eprintln!("Usage: cargo xtask generate");
        eprintln!("       cargo xtask validate");
        eprintln!("       cargo xtask check-version");
        std::process::exit(1);
    }

    match args[1].as_str() {
        "generate" => generate_wrappers()?,
        "validate" => validate_wrappers()?,
        "check-version" => check_version()?,
        _ => {
            eprintln!("Unknown command: {}", args[1]);
            std::process::exit(1);
        }
    }

    Ok(())
}

fn generate_wrappers() -> Result<()> {
    println!(
        "Finding safe_unaligned_simd-{} in cargo cache...",
        SAFE_SIMD_VERSION
    );
    let safe_simd_path = find_safe_simd_path()?;
    println!("Found at: {}", safe_simd_path.display());

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

            modules_info.push((module_name, feature, functions.len(), token_info.requires_avx512));
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

    Ok(())
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
        let end = after_colon
            .find([',', ')'])
            .unwrap_or(after_colon.len());

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
    println!("Finding safe_unaligned_simd-{} in cargo cache...", SAFE_SIMD_VERSION);
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
        println!("Parsed {} aarch64 source functions via macro expansion\n", aarch64_source.len());
    }

    // Parse generated wrappers
    println!("--- Parsing Generated Wrappers ---");
    let x86_gen_dir = PathBuf::from("src/generated/x86");
    let x86_wrappers = parse_generated_wrappers(&x86_gen_dir, Arch::X86)?;
    println!("Found {} x86 wrappers", x86_wrappers.len());

    // Parse aarch64 wrappers from macro invocations in neon.rs
    let aarch64_neon_path = PathBuf::from("src/generated/aarch64/neon.rs");
    let aarch64_wrappers = parse_generated_aarch64_wrappers(&aarch64_neon_path)?;
    println!("Found {} aarch64 wrappers (from macro parsing)\n", aarch64_wrappers.len());

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
                        wrapper.module, wrapper.name, wrapper.token_bound, missing, required_features
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
            warnings.push(format!("UNHANDLED x86: {} ({} functions skipped)", f, count));
            println!("  - {} ({} functions)", f, count);
        }
    }

    // Check for unhandled feature combinations in aarch64 source
    if !aarch64_source.is_empty() {
        println!("\n--- Checking aarch64 Feature Coverage ---");
        let mut aarch64_feature_coverage: BTreeMap<String, usize> = BTreeMap::new();
        let mut unhandled_aarch64: BTreeSet<String> = BTreeSet::new();

        for features in aarch64_source.values() {
            *aarch64_feature_coverage.entry(features.clone()).or_insert(0) += 1;

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
                warnings.push(format!("UNHANDLED aarch64: {} ({} functions skipped)", f, count));
                println!("  - {} ({} functions)", f, count);
            }
        }
    }

    // Summary
    println!("\n=== Validation Summary ===");
    println!("  Validated: {} wrappers", validated);
    println!("  Errors:    {} (token doesn't provide required features)", errors.len());
    println!("  Warnings:  {} (unhandled features or orphan wrappers)", warnings.len());

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
