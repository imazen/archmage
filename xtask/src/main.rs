//! Generator for safe_unaligned_simd token-gated wrappers.
//!
//! This tool parses the safe_unaligned_simd crate source and generates
//! wrapper modules that gate each function behind the appropriate archmage token.
//!
//! Usage: cargo xtask generate

use anyhow::{Context, Result, bail};
use std::collections::BTreeMap;
use std::fmt::Write as FmtWrite;
use std::fs;
use std::path::{Path, PathBuf};
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

/// Mapping from target_feature strings to archmage trait bounds
/// Returns (trait_name, features_string)
///
/// Note: We avoid redundant bounds by using trait hierarchy knowledge.
/// - HasAvx512vl: HasAvx512f - so HasAvx512vl implies HasAvx512f
/// - HasAvx512bw: HasAvx512f - sibling of HasAvx512vl, need both
/// - HasAvx512vbmi2: HasAvx512bw - need to add HasAvx512vl separately
fn feature_to_token(features: &str, arch: Arch) -> Option<(&'static str, &'static str)> {
    match arch {
        Arch::X86 => match features {
            "sse" => Some(("HasSse", "sse")),
            "sse2" => Some(("HasSse2", "sse2")),
            "avx" => Some(("HasAvx", "avx")),
            "avx512f" => Some(("HasAvx512f", "avx512f")),
            // HasAvx512vl: HasAvx512f, so just HasAvx512vl is sufficient
            "avx512f,avx512vl" => Some(("HasAvx512vl", "avx512f,avx512vl")),
            "avx512bw" => Some(("HasAvx512bw", "avx512bw")),
            // HasAvx512bw and HasAvx512vl are siblings (both extend HasAvx512f)
            "avx512bw,avx512vl" => Some(("HasAvx512bw + HasAvx512vl", "avx512bw,avx512vl")),
            "avx512vbmi2" => Some(("HasAvx512vbmi2", "avx512vbmi2")),
            // HasAvx512vbmi2: HasAvx512bw, but doesn't imply VL
            "avx512vbmi2,avx512vl" => {
                Some(("HasAvx512vbmi2 + HasAvx512vl", "avx512vbmi2,avx512vl"))
            }
            _ => None,
        },
        Arch::Aarch64 => match features {
            "neon" => Some(("HasNeon", "neon")),
            _ => None,
        },
    }
}

/// Module name for a given feature set
fn feature_to_module(features: &str, arch: Arch) -> &'static str {
    match arch {
        Arch::X86 => match features {
            "sse" => "sse",
            "sse2" => "sse2",
            "avx" => "avx",
            "avx512f" => "avx512f",
            "avx512f,avx512vl" => "avx512f_vl",
            "avx512bw" => "avx512bw",
            "avx512bw,avx512vl" => "avx512bw_vl",
            "avx512vbmi2" => "avx512vbmi2",
            "avx512vbmi2,avx512vl" => "avx512vbmi2_vl",
            _ => "unknown",
        },
        Arch::Aarch64 => match features {
            "neon" => "neon",
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
    token_name: &str,
    features: &str,
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

    // Parameters (add token as first param with impl Trait bound)
    let params: Vec<String> = std::iter::once(format!("_token: impl {}", token_name))
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
        features
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
    token_name: &str,
    features: &str,
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
        features,
        functions.len(),
        token_name,
        SAFE_SIMD_VERSION,
    )
    .unwrap();

    // Architecture-specific imports
    match arch {
        Arch::X86 => {
            // Extract trait names from the token_name (e.g., "HasAvx512f + HasAvx512vl" -> ["HasAvx512f", "HasAvx512vl"])
            let traits: Vec<&str> = token_name.split(" + ").map(|s| s.trim()).collect();
            let trait_imports = traits.join(", ");

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
                trait_imports
            )
            .unwrap();
        }
        Arch::Aarch64 => {
            writeln!(
                out,
                r#"
use core::arch::aarch64::*;
use crate::tokens::{};"#,
                token_name
            )
            .unwrap();
        }
    }

    // Generate each function
    for func in functions {
        writeln!(out).unwrap();
        out.push_str(&generate_wrapper_string(func, token_name, features, arch));
    }

    out
}

/// Generate the mod.rs that ties everything together
fn generate_mod_rs(
    x86_modules: &[(&str, &str, usize)],
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

    for (module, features, count) in x86_modules {
        code.push_str(&format!(
            "//! - [`x86::{}`]: {} functions (`{}`)\n",
            module, count, features
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

/// Generate x86/mod.rs
fn generate_x86_mod_rs(modules: &[(&str, &str, usize)]) -> String {
    let mut code = String::from(
        r#"//! x86/x86_64 token-gated wrappers.
//!
//! **Auto-generated** - do not edit manually.

"#,
    );

    for (module, _features, _count) in modules {
        code.push_str(&format!("pub mod {};\n", module));
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
        eprintln!("       cargo xtask check-version");
        std::process::exit(1);
    }

    match args[1].as_str() {
        "generate" => generate_wrappers()?,
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

    let mut modules_info: Vec<(&str, &str, usize)> = Vec::new();

    // Generate each module
    for (feature, functions) in &by_feature {
        if let Some((token_name, features)) = feature_to_token(feature, Arch::X86) {
            let module_name = feature_to_module(feature, Arch::X86);
            println!("\nGenerating {}...", module_name);

            let code = generate_module(token_name, features, functions, Arch::X86);
            let out_path = out_dir.join(format!("{}.rs", module_name));
            fs::write(&out_path, &code)?;
            println!("  Wrote {} ({} bytes)", out_path.display(), code.len());

            modules_info.push((module_name, feature, functions.len()));
        } else {
            println!("\nSkipping unknown feature: {}", feature);
        }
    }

    // Sort modules for consistent output
    modules_info.sort_by_key(|(name, _, _)| *name);

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

    let total_x86 = modules_info.iter().map(|(_, _, c)| c).sum::<usize>();
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

fn check_version() -> Result<()> {
    println!("Checking for safe_unaligned_simd updates...");
    println!("Current pinned version: {}", SAFE_SIMD_VERSION);
    println!("\nTo check manually:");
    println!("  cargo search safe_unaligned_simd");
    Ok(())
}
