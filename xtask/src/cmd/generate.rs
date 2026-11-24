//! `cargo xtask generate` — regenerate tokens, SIMD types, re-exports, and reference docs.

use anyhow::{Context, Result, bail};
use regex::Regex;
use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use crate::cmd::docs::{find_safe_simd_path_simple, generate_reference_docs_new};
use crate::cmd::validate::validate_summon;
use crate::{registry, simd_types, soundness, token_gen};

// ============================================================================
// Intrinsics Re-export Module Generation
// ============================================================================

/// A function extracted from safe_unaligned_simd, with its feature gate.
struct SafeSimdFn {
    name: String,
    /// If Some, this function requires `#[cfg(feature = "...")]` on the re-export.
    cfg_feature: Option<String>,
}

/// Extract function names from safe_unaligned_simd source for a given architecture.
///
/// Returns a sorted, deduplicated list of functions with their feature gates.
fn extract_safe_simd_function_names(safe_simd_path: &Path, arch: &str) -> Result<Vec<SafeSimdFn>> {
    let fn_re = Regex::new(r"pub\s+fn\s+(\w+)").expect("invalid fn regex");
    let macro_fn_re = Regex::new(r"(?m)^\s{4}fn\s+(\w+)\(").expect("invalid macro fn regex");
    let mut names: BTreeMap<String, Option<String>> = BTreeMap::new();

    // x86 source files and their cargo feature gates (matching safe_unaligned_simd's x86.rs)
    let x86_file_features: &[(&str, Option<&str>)] = &[
        ("sse.rs", None),
        ("sse2.rs", None),
        ("avx.rs", None),
        ("avx512f.rs", Some("avx512")),
        ("avx512bw.rs", Some("avx512")),
        ("avx512vbmi2.rs", Some("avx512")),
    ];

    match arch {
        "x86_64" | "x86" => {
            let x86_dir = safe_simd_path.join("src/x86");
            if x86_dir.exists() {
                for (filename, cfg_feature) in x86_file_features {
                    let path = x86_dir.join(filename);
                    if path.exists() {
                        let content = fs::read_to_string(&path)?;
                        for cap in fn_re.captures_iter(&content) {
                            names.insert(cap[1].to_string(), cfg_feature.map(|s| s.to_string()));
                        }
                    }
                }
            }
        }
        "aarch64" => {
            let aarch64_path = safe_simd_path.join("src/aarch64.rs");
            if aarch64_path.exists() {
                let content = fs::read_to_string(&aarch64_path)?;
                for cap in macro_fn_re.captures_iter(&content) {
                    let name = &cap[1];
                    // Filter out helper functions that aren't intrinsic wrappers
                    if name.starts_with("vld") || name.starts_with("vst") {
                        names.insert(name.to_string(), None);
                    }
                }
            }
        }
        "wasm32" => {
            let wasm_path = safe_simd_path.join("src/wasm32.rs");
            if wasm_path.exists() {
                let content = fs::read_to_string(&wasm_path)?;
                for cap in fn_re.captures_iter(&content) {
                    names.insert(cap[1].to_string(), None);
                }
            }
        }
        _ => bail!("Unknown architecture: {}", arch),
    }

    Ok(names
        .into_iter()
        .map(|(name, cfg_feature)| SafeSimdFn { name, cfg_feature })
        .collect())
}

/// Generate the combined intrinsics re-export module for one architecture.
///
/// The module glob-imports `core::arch::{arch}::*` (all types + value intrinsics),
/// then explicitly imports each `safe_unaligned_simd` function. Rust's name
/// resolution makes explicit imports shadow glob imports, so the safe memory ops
/// win without ambiguity.
///
/// Functions gated behind cargo features (like AVX-512) get their own
/// `#[cfg(feature = "...")]` block.
fn generate_intrinsics_module(arch: &str, functions: &[SafeSimdFn]) -> String {
    use indoc::formatdoc;

    // Group functions by their cfg_feature
    let mut ungated: Vec<&str> = Vec::new();
    let mut gated: BTreeMap<&str, Vec<&str>> = BTreeMap::new();

    for f in functions {
        match &f.cfg_feature {
            None => ungated.push(&f.name),
            Some(feature) => gated.entry(feature.as_str()).or_default().push(&f.name),
        }
    }

    let mut code = formatdoc! {r#"
        //! Combined `core::arch` + `safe_unaligned_simd` intrinsics for `{arch}`.
        //!
        //! **Auto-generated** by `cargo xtask generate` — do not edit manually.
        //!
        //! This module glob-imports all of `core::arch::{arch}` (types, value intrinsics,
        //! and unsafe memory ops), then explicitly re-exports the safe reference-based
        //! memory operations from `safe_unaligned_simd`. Rust's name resolution rules
        //! make explicit imports shadow glob imports, so `_mm256_loadu_ps` etc. resolve
        //! to the safe versions automatically.

        #[allow(unused_imports)]
        pub use core::arch::{arch}::*;

    "#};

    let x86 = matches!(arch, "x86_64" | "x86");

    // Ungated imports
    if !ungated.is_empty() {
        if x86 {
            code.push_str("#[cfg(target_feature = \"sse2\")]\n");
        }
        let imports = ungated
            .iter()
            .map(|n| format!("    {},", n))
            .collect::<Vec<_>>()
            .join("\n");
        code.push_str(&formatdoc! {r#"
            #[allow(unused_imports)]
            pub use safe_unaligned_simd::{arch}::{{
            {imports}
            }};

        "#});
    }

    // Feature-gated imports
    for (feature, names) in &gated {
        let imports = names
            .iter()
            .map(|n| format!("    {},", n))
            .collect::<Vec<_>>()
            .join("\n");
        let cfg = if x86 {
            format!(r#"all(target_feature = "sse2", feature = "{feature}")"#)
        } else {
            format!(r#"feature = "{feature}""#)
        };
        code.push_str(&formatdoc! {r#"
            #[cfg({cfg})]
            #[allow(unused_imports)]
            pub use safe_unaligned_simd::{arch}::{{
            {imports}
            }};

        "#});
    }

    code
}

/// Generate intrinsics re-export modules for all architectures.
///
/// Creates `src/intrinsics/generated/{x86_64,x86,aarch64,wasm32}.rs` files that
/// combine `core::arch` and `safe_unaligned_simd` into a single namespace where
/// safe memory ops shadow unsafe pointer-based versions.
fn generate_intrinsics_reexport_modules() -> Result<()> {
    println!("=== Generating Intrinsics Re-export Modules ===");

    let safe_simd_path = find_safe_simd_path_simple()?;
    let gen_dir = PathBuf::from("src/intrinsics/generated");
    fs::create_dir_all(&gen_dir)?;

    let architectures = [
        ("x86_64", "x86_64"),
        ("x86", "x86"), // Same functions, different arch module
        ("aarch64", "aarch64"),
        ("wasm32", "wasm32"),
    ];

    for (arch, source_arch) in &architectures {
        let functions = extract_safe_simd_function_names(&safe_simd_path, source_arch)?;
        if functions.is_empty() {
            println!("  Warning: no functions found for {}", arch);
            continue;
        }

        let content = generate_intrinsics_module(arch, &functions);
        let file_path = gen_dir.join(format!("{}.rs", arch));
        fs::write(&file_path, &content)?;
        println!(
            "  Wrote {} ({} functions, {} bytes)",
            file_path.display(),
            functions.len(),
            content.len()
        );

        // Format generated file
        let _ = std::process::Command::new("rustfmt")
            .arg("--edition")
            .arg("2024")
            .arg(&file_path)
            .status();
    }

    Ok(())
}

/// Generate all artifacts: SIMD types, macro registry, and documentation
pub(crate) fn generate_all() -> Result<()> {
    // crates.io uses the same README body with the existing badge-free title.
    for (name, directory) in [("archmage", "."), ("magetypes", "magetypes")] {
        let source = fs::read_to_string(format!("{directory}/README.md"))?;
        let (_, body) = source
            .split_once('\n')
            .context("README needs a title line")?;
        fs::write(
            format!("{directory}/README.crates.md"),
            format!("# {name}\n{body}"),
        )?;
    }

    // Keep the website's complete reference in sync with the owning guide.
    // Release-validation history stays in the repository, outside the user guide.
    let guide = fs::read_to_string("docs/ATTUNE.md")?;
    let (_, body) = guide
        .split_once('\n')
        .context("attune guide needs a title")?;
    let (reference, _) = body
        .split_once("\n## Beta scope and validation")
        .context("attune guide needs its release-validation boundary")?;
    fs::write(
        "docs/site/content/attunement/reference.md",
        format!(
            r#"+++
title = "Attunement Reference"
description = "Unified macro forms, call rules, and legacy mappings"
weight = 4
+++

<!-- Generated by just generate from docs/ATTUNE.md. -->
{reference}"#
        ),
    )?;

    // Resolve the ONE external precondition before destroying anything.
    //
    // The purge below deletes six directories, and the safe_unaligned_simd
    // lookup that can fail used to happen AFTER it (inside
    // generate_intrinsics_reexport_modules). A cache miss therefore left the
    // repo with every generated file deleted and nothing written back —
    // measured at 41,561 lines across 64 files on a run where the version
    // constant did not match the cache. Checking first turns that into a clean
    // refusal that changes nothing on disk.
    let _ = find_safe_simd_path_simple()
        .context("cannot generate: the safe_unaligned_simd source is unavailable")?;

    // Purge all generated directories first to handle renamed/removed files
    println!("=== Purging Generated Directories ===");
    let generated_dirs = [
        "archmage-macros/src/generated",
        "src/tokens/generated",
        "src/intrinsics/generated",
        "magetypes/src/simd/generated",
        "magetypes/src/simd/generic/generated",
        "docs/generated",
    ];
    for dir in &generated_dirs {
        let path = PathBuf::from(dir);
        if path.exists() {
            fs::remove_dir_all(&path)?;
            println!("  Purged {}", dir);
        }
    }
    println!();

    // Generate intrinsics re-export modules (core::arch + safe_unaligned_simd combined)
    generate_intrinsics_reexport_modules()?;

    // Generate macro registry from token-registry.toml
    println!("=== Generating Macro Registry ===");
    let registry_path = PathBuf::from("token-registry.toml");
    let reg = registry::Registry::load(&registry_path)?;

    // Parse major version from workspace Cargo.toml for tier tag hashing
    let cargo_toml = fs::read_to_string("Cargo.toml")?;
    let major_version: u32 = cargo_toml
        .lines()
        .find(|l| l.starts_with("version") || l.contains("version ="))
        .and_then(|l| l.split('"').nth(1))
        .and_then(|v| v.split('.').next())
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);
    println!("  Major version for tier tags: {major_version}");

    let generated = reg.generate_macro_registry(major_version);
    let gen_dir = PathBuf::from("archmage-macros/src/generated");
    fs::create_dir_all(&gen_dir)?;
    let gen_path = gen_dir.join("registry.rs");
    fs::write(&gen_path, &generated)?;
    println!("  Wrote {} ({} bytes)", gen_path.display(), generated.len());

    let dispatch_path = gen_dir.join("dispatch.rs");
    fs::write(&dispatch_path, reg.generate_dispatch_registry())?;

    // Write generated/mod.rs for the macro crate
    let gen_mod = r#"//! Generated code from token-registry.toml.
//!
//! **Auto-generated** by `cargo xtask generate` - do not edit manually.

mod registry;
mod dispatch;
pub(crate) use registry::*;
pub(crate) use dispatch::*;
"#;
    let gen_mod_path = gen_dir.join("mod.rs");
    fs::write(&gen_mod_path, gen_mod)?;
    println!(
        "  Wrote {} ({} bytes)",
        gen_mod_path.display(),
        gen_mod.len()
    );

    // Run rustfmt on the generated files so they stay fmt-clean
    // archmage-macros uses edition 2021
    for path in [&gen_path, &gen_mod_path, &dispatch_path] {
        let fmt_status = std::process::Command::new("rustfmt")
            .arg("--edition")
            .arg("2021")
            .arg(path)
            .status();
        match fmt_status {
            Ok(s) if s.success() => println!("  Formatted {}", path.display()),
            Ok(s) => println!("  Warning: rustfmt exited with {}", s),
            Err(e) => println!("  Warning: could not run rustfmt: {}", e),
        }
    }

    // Generate token implementations from registry
    println!("\n=== Generating Token Implementations ===");
    let token_dir = PathBuf::from("src/tokens/generated");
    fs::create_dir_all(&token_dir)?;
    let token_files = token_gen::generate_token_files(&reg, major_version);
    let mut token_total_bytes = 0;
    for (rel_path, content) in &token_files {
        let full_path = token_dir.join(rel_path);
        fs::write(&full_path, content)?;
        token_total_bytes += content.len();
        println!("  Wrote {} ({} bytes)", full_path.display(), content.len());
    }
    println!(
        "Total token files: {} files, {} bytes",
        token_files.len(),
        token_total_bytes
    );

    // Run rustfmt on generated token files
    for (rel_path, _) in &token_files {
        let full_path = token_dir.join(rel_path);
        let _ = std::process::Command::new("rustfmt")
            .arg("--edition")
            .arg("2024")
            .arg(&full_path)
            .status();
    }
    println!("  Formatted {} token files", token_files.len());

    // Generate SIMD types (wide-like ergonomic types) into magetypes crate
    println!("\n=== Generating SIMD Types (magetypes) ===");

    let simd_dir = PathBuf::from("magetypes/src/simd");
    fs::create_dir_all(&simd_dir)?;
    fs::create_dir_all(simd_dir.join("generated"))?;

    // Generate split files
    let simd_files = simd_types::generate_simd_types_split();
    let mut total_bytes = 0;
    for (rel_path, content) in &simd_files {
        let full_path = simd_dir.join(rel_path);
        fs::write(&full_path, content)?;
        total_bytes += content.len();
        println!("  Wrote {} ({} bytes)", full_path.display(), content.len());
    }
    println!(
        "Total SIMD types: {} files, {} bytes",
        simd_files.len(),
        total_bytes
    );

    // Generate backend traits + implementations (generic SIMD strategy pattern)
    println!("\n=== Generating Backend Traits ===");
    fs::create_dir_all(simd_dir.join("backends"))?;
    fs::create_dir_all(simd_dir.join("impls"))?;
    let backend_files = simd_types::backend_gen::generate_backend_files();
    let mut backend_total_bytes = 0;
    for (rel_path, content) in &backend_files {
        let full_path = simd_dir.join(rel_path);
        fs::write(&full_path, content)?;
        backend_total_bytes += content.len();
        println!("  Wrote {} ({} bytes)", full_path.display(), content.len());
    }
    println!(
        "Total backend files: {} files, {} bytes",
        backend_files.len(),
        backend_total_bytes
    );

    // Generate generic wrapper types (strategy-pattern types parameterized by backend)
    println!("\n=== Generating Generic Wrapper Types ===");
    fs::create_dir_all(simd_dir.join("generic/generated"))?;
    let generic_files = simd_types::generic_gen::generate_generic_files(&reg);
    fs::create_dir_all("docs/constructors")?;
    for (name, signatures) in simd_types::generic_gen::constructor_inventory(&generic_files) {
        fs::write(format!("docs/constructors/{name}.md"), signatures)?;
    }
    let mut generic_total_bytes = 0;
    for (rel_path, content) in &generic_files {
        let full_path = simd_dir.join(rel_path);
        fs::write(&full_path, content)?;
        generic_total_bytes += content.len();
        println!("  Wrote {} ({} bytes)", full_path.display(), content.len());
    }
    println!(
        "Total generic files: {} files, {} bytes",
        generic_files.len(),
        generic_total_bytes
    );

    // Generate SIMD type tests
    let simd_tests = simd_types::generate_simd_tests();
    let simd_test_path = PathBuf::from("magetypes/tests/generated_simd_types.rs");
    fs::create_dir_all("magetypes/tests")?;
    fs::write(&simd_test_path, &simd_tests)?;
    println!(
        "Wrote {} ({} bytes)",
        simd_test_path.display(),
        simd_tests.len()
    );

    // Generate parity tests (cross-architecture + polyfill vs native)
    println!("\n=== Generating Parity Tests ===");
    let parity_tests = simd_types::parity_tests::generate_parity_tests();
    for (filename, content) in &parity_tests {
        let path = PathBuf::from("magetypes/tests").join(filename);
        fs::write(&path, content)?;
        println!("  Wrote {} ({} bytes)", path.display(), content.len());
    }

    // Generate width dispatch (WidthDispatch trait + impls)
    println!("\n=== Generating Width Dispatch ===");
    let width_code = simd_types::width_dispatch::generate_width_dispatch();
    let width_path = PathBuf::from("magetypes/src/width.rs");
    fs::write(&width_path, &width_code)?;
    println!(
        "  Wrote {} ({} bytes)",
        width_path.display(),
        width_code.len()
    );

    // Run rustfmt on all generated .rs files for idempotent output
    let mut fmt_paths: Vec<PathBuf> = simd_files.keys().map(|rel| simd_dir.join(rel)).collect();
    fmt_paths.push(simd_test_path.clone());
    for (filename, _) in &parity_tests {
        fmt_paths.push(PathBuf::from("magetypes/tests").join(filename));
    }
    fmt_paths.push(width_path);
    for rel in backend_files.keys() {
        fmt_paths.push(simd_dir.join(rel));
    }
    for rel in generic_files.keys() {
        fmt_paths.push(simd_dir.join(rel));
    }
    for path in &fmt_paths {
        let _ = std::process::Command::new("rustfmt")
            .arg("--edition")
            .arg("2024")
            .arg(path)
            .status();
    }
    println!("  Formatted {} generated files", fmt_paths.len());

    // Shorten fully qualified paths the file already imports (after rustfmt,
    // so token-stream-built text is normalized), then reformat what changed.
    let mut shortened = 0;
    for rel in backend_files.keys().chain(generic_files.keys()) {
        let path = simd_dir.join(rel);
        let text = fs::read_to_string(&path)?;
        let short = simd_types::import_paths::shorten_imported_paths(&text);
        if short != text {
            fs::write(&path, &short)?;
            let _ = std::process::Command::new("rustfmt")
                .arg("--edition")
                .arg("2024")
                .arg(&path)
                .status();
            shortened += 1;
        }
    }
    println!("  Shortened imported paths in {shortened} generated files");

    // Generate reference documentation
    println!("\n=== Generating Reference Documentation ===");
    generate_reference_docs_new()?;

    // Run safety validation on generated code (hard failure)
    println!("\n=== Validating Magetypes Safety ===");
    soundness::verify(&reg)?;

    // Verify summon() implementations match registry
    println!();
    validate_summon(&reg)?;

    println!("\n=== Generation Complete ===");
    println!("  - SIMD types: magetypes/src/simd/");
    println!("  - Reference docs: docs/*.md");

    Ok(())
}
