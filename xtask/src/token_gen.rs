//! Token code generator — generates `src/tokens/generated/` from `token-registry.toml`.
//!
//! Produces:
//! - Real implementations for native arch (with runtime detection)
//! - Stub implementations for cross-platform (summon → None)
//! - Trait definitions
//! - Module file with cfg-gated imports

use crate::registry::{Registry, TokenDef, TraitDef};
use std::fmt::Write;

/// Convert token name to a screaming snake case variable name with given suffix.
fn screaming_snake(token_name: &str, suffix: &str) -> String {
    let mut result = String::new();
    let mut prev_was_upper = false;
    for (i, c) in token_name.chars().enumerate() {
        if c.is_uppercase() {
            if i > 0 && !prev_was_upper {
                result.push('_');
            }
            result.push(c);
            prev_was_upper = true;
        } else if c.is_lowercase() {
            result.push(c.to_ascii_uppercase());
            prev_was_upper = false;
        } else {
            result.push(c);
            prev_was_upper = false;
        }
    }
    // Remove "TOKEN" suffix
    let result = result.trim_end_matches("_TOKEN").to_string();
    format!("{result}_{suffix}")
}

/// Convert token name to a screaming snake case cache variable name.
fn cache_var_name(token_name: &str) -> String {
    screaming_snake(token_name, "CACHE")
}

/// Convert token name to a screaming snake case disabled variable name.
fn disabled_var_name(token_name: &str) -> String {
    screaming_snake(token_name, "DISABLED")
}

/// All generated token files as (relative_path, content) pairs.
///
/// Relative to `src/tokens/generated/`.
pub fn generate_token_files(reg: &Registry) -> Vec<(String, String)> {
    let mut files = Vec::new();

    // Group tokens by arch
    let x86_tokens: Vec<&TokenDef> = reg.token.iter().filter(|t| t.arch == "x86").collect();
    let arm_tokens: Vec<&TokenDef> = reg.token.iter().filter(|t| t.arch == "aarch64").collect();
    let wasm_tokens: Vec<&TokenDef> = reg.token.iter().filter(|t| t.arch == "wasm").collect();

    // Split x86 into base and avx512 (gated on cargo feature)
    let x86_base: Vec<&TokenDef> = x86_tokens
        .iter()
        .filter(|t| t.cargo_feature.is_none())
        .copied()
        .collect();
    let x86_avx512: Vec<&TokenDef> = x86_tokens
        .iter()
        .filter(|t| t.cargo_feature.is_some())
        .copied()
        .collect();

    // Real implementations
    files.push(("x86.rs".into(), gen_real_tokens(reg, &x86_base, "x86")));
    files.push((
        "x86_avx512.rs".into(),
        gen_real_tokens(reg, &x86_avx512, "x86"),
    ));
    files.push((
        "arm.rs".into(),
        gen_real_tokens(reg, &arm_tokens, "aarch64"),
    ));
    files.push(("wasm.rs".into(), gen_real_tokens(reg, &wasm_tokens, "wasm")));

    // Stubs
    files.push(("x86_stubs.rs".into(), gen_stub_tokens(reg, &x86_base)));
    files.push((
        "x86_avx512_stubs.rs".into(),
        gen_stub_tokens(reg, &x86_avx512),
    ));
    files.push(("arm_stubs.rs".into(), gen_stub_tokens(reg, &arm_tokens)));
    files.push(("wasm_stubs.rs".into(), gen_stub_tokens(reg, &wasm_tokens)));

    // Traits
    files.push(("traits.rs".into(), gen_traits(reg)));

    // Module file
    files.push(("mod.rs".into(), gen_mod_rs()));

    files
}

// ============================================================================
// Real token implementations (native arch)
// ============================================================================

fn gen_real_tokens(reg: &Registry, tokens: &[&TokenDef], arch: &str) -> String {
    let mut out = String::with_capacity(4096);
    writeln!(out, "//! Generated from token-registry.toml — DO NOT EDIT.").unwrap();
    writeln!(out, "//!").unwrap();
    writeln!(out, "//! Regenerate with: cargo xtask generate").unwrap();
    writeln!(out).unwrap();

    // Imports
    writeln!(out, "use crate::tokens::SimdToken;").unwrap();

    // Check if we need atomic imports (for tokens with runtime detection)
    if arch != "wasm" && !tokens.is_empty() {
        writeln!(
            out,
            "use core::sync::atomic::{{AtomicBool, AtomicU8, Ordering}};"
        )
        .unwrap();
    }

    // Collect all trait names used by tokens in this file
    let trait_names: Vec<&str> = tokens
        .iter()
        .flat_map(|t| t.traits.iter().map(|s| s.as_str()))
        .collect::<std::collections::BTreeSet<_>>()
        .into_iter()
        .collect();

    if !trait_names.is_empty() {
        write!(out, "use crate::tokens::{{").unwrap();
        for (i, name) in trait_names.iter().enumerate() {
            if i > 0 {
                write!(out, ", ").unwrap();
            }
            write!(out, "{name}").unwrap();
        }
        writeln!(out, "}};").unwrap();
    }

    // Import ancestor tokens from other files (for extraction methods)
    let mut external_imports: std::collections::BTreeSet<(&str, &str)> =
        std::collections::BTreeSet::new();
    for token in tokens {
        let ancestors = collect_ancestors(reg, token);
        for ancestor in &ancestors {
            let in_file = tokens.iter().any(|other| other.name == ancestor.name);
            if !in_file {
                let mod_name = file_module_for_token(ancestor);
                external_imports.insert((mod_name, &ancestor.name));
            }
        }
    }
    for (mod_name, type_name) in &external_imports {
        writeln!(out, "use super::{mod_name}::{type_name};").unwrap();
    }

    writeln!(out).unwrap();

    // Generate cache and disabled statics (skip WASM - it's compile-time only)
    if arch != "wasm" {
        writeln!(
            out,
            "// Cache statics: 0 = unknown, 1 = unavailable, 2 = available"
        )
        .unwrap();
        for token in tokens {
            let cache_name = cache_var_name(&token.name);
            let disabled_name = disabled_var_name(&token.name);
            writeln!(
                out,
                "pub(super) static {cache_name}: AtomicU8 = AtomicU8::new(0);"
            )
            .unwrap();
            writeln!(
                out,
                "pub(super) static {disabled_name}: AtomicBool = AtomicBool::new(false);"
            )
            .unwrap();
        }
        writeln!(out).unwrap();
    }

    // Generate each token
    for token in tokens {
        gen_real_token_struct(&mut out, reg, token, arch, tokens);
        writeln!(out).unwrap();
    }

    // Aliases
    for token in tokens {
        gen_aliases(&mut out, token);
    }

    // Trait impls
    gen_trait_impls(&mut out, tokens);

    out
}

fn gen_real_token_struct(
    out: &mut String,
    reg: &Registry,
    token: &TokenDef,
    arch: &str,
    all_tokens_in_file: &[&TokenDef],
) {
    // Doc comment
    if let Some(doc) = &token.doc {
        for line in doc.lines() {
            if line.is_empty() {
                writeln!(out, "///").unwrap();
            } else {
                writeln!(out, "/// {line}").unwrap();
            }
        }
    }

    let name = &token.name;
    let display = token.display_name.as_deref().unwrap_or(name);

    writeln!(out, "#[derive(Clone, Copy, Debug)]").unwrap();
    writeln!(out, "pub struct {name} {{").unwrap();
    writeln!(out, "    _private: (),").unwrap();
    writeln!(out, "}}").unwrap();
    writeln!(out).unwrap();

    writeln!(out, "impl crate::tokens::Sealed for {name} {{}}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "impl SimdToken for {name} {{").unwrap();
    writeln!(out, "    const NAME: &'static str = \"{display}\";").unwrap();
    writeln!(out).unwrap();

    // compiled_with()
    gen_compiled_with(out, token, arch);

    // Note: guaranteed() has a default impl in the trait that calls compiled_with()

    // summon()
    writeln!(out).unwrap();
    gen_summon(out, token, arch);

    // forge_token_dangerously()
    writeln!(out).unwrap();
    writeln!(out, "    #[inline(always)]").unwrap();
    writeln!(out, "    #[allow(deprecated)]").unwrap();
    writeln!(out, "    unsafe fn forge_token_dangerously() -> Self {{").unwrap();
    writeln!(out, "        Self {{ _private: () }}").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out, "}}").unwrap();

    // Extraction methods
    gen_extraction_methods(out, reg, token);

    // Disable/getter methods (not for WASM — compile-time only)
    if arch != "wasm" {
        gen_disable_methods(out, reg, token, all_tokens_in_file);
    }
}

/// Generate the `compiled_with()` method for a real token.
fn gen_compiled_with(out: &mut String, token: &TokenDef, arch: &str) {
    writeln!(out, "    #[inline]").unwrap();
    writeln!(out, "    fn compiled_with() -> Option<bool> {{").unwrap();

    match arch {
        "x86" => {
            // Check if all required features are compile-time target_feature
            // Filter out sse/sse2 — they are x86_64 baseline
            let check_features: Vec<&str> = token
                .features
                .iter()
                .filter(|f| *f != "sse" && *f != "sse2")
                .map(|s| s.as_str())
                .collect();

            if check_features.is_empty() {
                // All features are baseline
                writeln!(out, "        Some(true)").unwrap();
            } else {
                let mut conditions = Vec::new();
                for feat in &check_features {
                    conditions.push(format!("target_feature = \"{feat}\""));
                }
                let all_conditions = conditions.join(", ");

                writeln!(out, "        #[cfg(all({all_conditions}))]").unwrap();
                writeln!(out, "        {{ Some(true) }}").unwrap();
                writeln!(out, "        #[cfg(not(all({all_conditions})))]").unwrap();
                writeln!(out, "        {{ None }}").unwrap();
            }
        }
        "aarch64" => {
            // Check ALL features including neon — neon is NOT assumed baseline
            let check_features: Vec<&str> = token.features.iter().map(|s| s.as_str()).collect();

            let mut conditions = Vec::new();
            for feat in &check_features {
                conditions.push(format!("target_feature = \"{feat}\""));
            }
            let all_conditions = conditions.join(", ");

            writeln!(out, "        #[cfg(all({all_conditions}))]").unwrap();
            writeln!(out, "        {{ Some(true) }}").unwrap();
            writeln!(out, "        #[cfg(not(all({all_conditions})))]").unwrap();
            writeln!(out, "        {{ None }}").unwrap();
        }
        "wasm" => {
            // WASM SIMD is compile-time only (no runtime detection)
            writeln!(
                out,
                "        #[cfg(all(target_arch = \"wasm32\", target_feature = \"simd128\"))]"
            )
            .unwrap();
            writeln!(out, "        {{ Some(true) }}").unwrap();
            writeln!(
                out,
                "        #[cfg(not(all(target_arch = \"wasm32\", target_feature = \"simd128\")))]"
            )
            .unwrap();
            writeln!(out, "        {{ None }}").unwrap();
        }
        _ => unreachable!("unknown arch: {arch}"),
    }

    writeln!(out, "    }}").unwrap();
}

fn gen_summon(out: &mut String, token: &TokenDef, arch: &str) {
    // Add #[allow(deprecated)] since summon() internally uses forge_token_dangerously()
    writeln!(out, "    #[allow(deprecated)]").unwrap();
    match arch {
        "x86" => gen_summon_x86(out, token),
        "aarch64" => gen_summon_aarch64(out, token),
        "wasm" => gen_summon_wasm(out),
        _ => unreachable!("unknown arch: {arch}"),
    }
}

fn gen_summon_x86(out: &mut String, token: &TokenDef) {
    // Filter out sse/sse2 (x86_64 baseline, always available)
    let check_features: Vec<&str> = token
        .features
        .iter()
        .filter(|f| *f != "sse" && *f != "sse2")
        .map(|s| s.as_str())
        .collect();

    if check_features.is_empty() {
        writeln!(out, "    #[inline]").unwrap();
        writeln!(out, "    fn summon() -> Option<Self> {{").unwrap();
        writeln!(
            out,
            "        Some(unsafe {{ Self::forge_token_dangerously() }})"
        )
        .unwrap();
        writeln!(out, "    }}").unwrap();
        return;
    }

    let cache_name = cache_var_name(&token.name);
    let all_features = check_features
        .iter()
        .map(|f| format!("target_feature = \"{f}\""))
        .collect::<Vec<_>>()
        .join(", ");

    writeln!(out, "    #[inline(always)]").unwrap();
    writeln!(out, "    fn summon() -> Option<Self> {{").unwrap();

    // Compile-time fast path: if all features are guaranteed, return immediately
    writeln!(out, "        // Compile-time fast path").unwrap();
    writeln!(out, "        #[cfg(all({all_features}))]").unwrap();
    writeln!(out, "        {{").unwrap();
    writeln!(
        out,
        "            return Some(unsafe {{ Self::forge_token_dangerously() }});"
    )
    .unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(out).unwrap();

    // Runtime path with caching
    writeln!(out, "        // Runtime path with caching").unwrap();
    writeln!(out, "        #[cfg(not(all({all_features})))]").unwrap();
    writeln!(out, "        {{").unwrap();
    writeln!(
        out,
        "            match {cache_name}.load(Ordering::Relaxed) {{"
    )
    .unwrap();
    writeln!(
        out,
        "                2 => Some(unsafe {{ Self::forge_token_dangerously() }}),"
    )
    .unwrap();
    writeln!(out, "                1 => None,").unwrap();
    writeln!(out, "                _ => {{").unwrap();

    // Feature detection
    write!(out, "                    let available = ").unwrap();
    for (i, feat) in check_features.iter().enumerate() {
        if i > 0 {
            write!(out, "                        && ").unwrap();
        }
        writeln!(out, "crate::is_x86_feature_available!(\"{feat}\")").unwrap();
    }
    writeln!(out, "                    ;").unwrap();
    writeln!(out, "                    {cache_name}.store(if available {{ 2 }} else {{ 1 }}, Ordering::Relaxed);").unwrap();
    writeln!(out, "                    if available {{").unwrap();
    writeln!(
        out,
        "                        Some(unsafe {{ Self::forge_token_dangerously() }})"
    )
    .unwrap();
    writeln!(out, "                    }} else {{").unwrap();
    writeln!(out, "                        None").unwrap();
    writeln!(out, "                    }}").unwrap();
    writeln!(out, "                }}").unwrap();
    writeln!(out, "            }}").unwrap();
    writeln!(out, "        }}").unwrap();

    writeln!(out, "    }}").unwrap();
}

fn gen_summon_aarch64(out: &mut String, token: &TokenDef) {
    // Check ALL features including neon — neon is NOT assumed baseline
    let check_features: Vec<&str> = token.features.iter().map(|s| s.as_str()).collect();

    let cache_name = cache_var_name(&token.name);
    let all_features = check_features
        .iter()
        .map(|f| format!("target_feature = \"{f}\""))
        .collect::<Vec<_>>()
        .join(", ");

    writeln!(out, "    #[inline(always)]").unwrap();
    writeln!(out, "    fn summon() -> Option<Self> {{").unwrap();

    // Compile-time fast path
    writeln!(out, "        // Compile-time fast path").unwrap();
    writeln!(out, "        #[cfg(all({all_features}))]").unwrap();
    writeln!(out, "        {{").unwrap();
    writeln!(
        out,
        "            return Some(unsafe {{ Self::forge_token_dangerously() }});"
    )
    .unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(out).unwrap();

    // Runtime path with caching
    writeln!(out, "        // Runtime path with caching").unwrap();
    writeln!(out, "        #[cfg(not(all({all_features})))]").unwrap();
    writeln!(out, "        {{").unwrap();
    writeln!(
        out,
        "            match {cache_name}.load(Ordering::Relaxed) {{"
    )
    .unwrap();
    writeln!(
        out,
        "                2 => Some(unsafe {{ Self::forge_token_dangerously() }}),"
    )
    .unwrap();
    writeln!(out, "                1 => None,").unwrap();
    writeln!(out, "                _ => {{").unwrap();

    // Feature detection
    write!(out, "                    let available = ").unwrap();
    for (i, feat) in check_features.iter().enumerate() {
        if i > 0 {
            write!(out, "                        && ").unwrap();
        }
        writeln!(out, "crate::is_aarch64_feature_available!(\"{feat}\")").unwrap();
    }
    writeln!(out, "                    ;").unwrap();
    writeln!(out, "                    {cache_name}.store(if available {{ 2 }} else {{ 1 }}, Ordering::Relaxed);").unwrap();
    writeln!(out, "                    if available {{").unwrap();
    writeln!(
        out,
        "                        Some(unsafe {{ Self::forge_token_dangerously() }})"
    )
    .unwrap();
    writeln!(out, "                    }} else {{").unwrap();
    writeln!(out, "                        None").unwrap();
    writeln!(out, "                    }}").unwrap();
    writeln!(out, "                }}").unwrap();
    writeln!(out, "            }}").unwrap();
    writeln!(out, "        }}").unwrap();

    writeln!(out, "    }}").unwrap();
}

fn gen_summon_wasm(out: &mut String) {
    // WASM has no runtime detection - it's all compile-time
    writeln!(out, "    #[inline]").unwrap();
    writeln!(out, "    fn summon() -> Option<Self> {{").unwrap();
    writeln!(
        out,
        "        #[cfg(all(target_arch = \"wasm32\", target_feature = \"simd128\"))]"
    )
    .unwrap();
    writeln!(out, "        {{").unwrap();
    writeln!(
        out,
        "            Some(unsafe {{ Self::forge_token_dangerously() }})"
    )
    .unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(
        out,
        "        #[cfg(not(all(target_arch = \"wasm32\", target_feature = \"simd128\")))]"
    )
    .unwrap();
    writeln!(out, "        {{").unwrap();
    writeln!(out, "            None").unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(out, "    }}").unwrap();
}

fn gen_extraction_methods(out: &mut String, reg: &Registry, token: &TokenDef) {
    // Collect all ancestors by walking the parent chain
    let ancestors = collect_ancestors(reg, token);

    if ancestors.is_empty() {
        return;
    }

    writeln!(out).unwrap();
    writeln!(out, "impl {} {{", token.name).unwrap();

    for ancestor in &ancestors {
        let anc_name = &ancestor.name;
        let short = ancestor
            .short_name
            .as_deref()
            .unwrap_or("MISSING_SHORT_NAME");
        let token_display = token.display_name.as_deref().unwrap_or(&token.name);
        let anc_display = ancestor.display_name.as_deref().unwrap_or(anc_name);

        writeln!(
            out,
            "    /// Get a {anc_name} ({token_display} implies {anc_display})"
        )
        .unwrap();
        writeln!(out, "    #[allow(deprecated)]").unwrap();
        writeln!(out, "    #[inline(always)]").unwrap();
        writeln!(out, "    pub fn {short}(self) -> {anc_name} {{").unwrap();
        writeln!(
            out,
            "        unsafe {{ {anc_name}::forge_token_dangerously() }}"
        )
        .unwrap();
        writeln!(out, "    }}").unwrap();

        // Extraction aliases for this ancestor (e.g., .avx512() for X64V4Token)
        for alias_name in &ancestor.extraction_aliases {
            writeln!(out).unwrap();
            writeln!(out, "    /// Get a {anc_name} (alias for `.{short}()`)").unwrap();
            writeln!(out, "    #[allow(deprecated)]").unwrap();
            writeln!(out, "    #[inline(always)]").unwrap();
            writeln!(out, "    pub fn {alias_name}(self) -> {anc_name} {{").unwrap();
            writeln!(
                out,
                "        unsafe {{ {anc_name}::forge_token_dangerously() }}"
            )
            .unwrap();
            writeln!(out, "    }}").unwrap();
        }
    }

    writeln!(out, "}}").unwrap();
}

/// Generate disable/getter methods for a real token.
fn gen_disable_methods(
    out: &mut String,
    reg: &Registry,
    token: &TokenDef,
    all_tokens_in_file: &[&TokenDef],
) {
    let name = &token.name;
    let cache_name = cache_var_name(&token.name);
    let disabled_name = disabled_var_name(&token.name);

    // For the compile-time guard, use the FULL unfiltered feature list
    let all_feature_conditions: Vec<String> = token
        .features
        .iter()
        .map(|f| format!("target_feature = \"{f}\""))
        .collect();
    let all_conditions = all_feature_conditions.join(", ");

    // Collect descendants (tokens that have this token in their ancestor chain)
    let descendants = collect_descendants(reg, token);

    writeln!(out).unwrap();
    writeln!(out, "impl {name} {{").unwrap();

    // dangerously_disable_token_process_wide
    writeln!(
        out,
        "    /// Disable this token process-wide for testing and benchmarking."
    )
    .unwrap();
    writeln!(out, "    ///").unwrap();
    writeln!(
        out,
        "    /// When disabled, `summon()` will return `None` even if the CPU supports"
    )
    .unwrap();
    writeln!(out, "    /// the required features.").unwrap();
    writeln!(out, "    ///").unwrap();
    writeln!(
        out,
        "    /// Returns `Err` when all required features are compile-time enabled"
    )
    .unwrap();
    writeln!(
        out,
        "    /// (e.g., via `-Ctarget-cpu=native`), since the compiler has already"
    )
    .unwrap();
    writeln!(out, "    /// elided the runtime checks.").unwrap();
    if !descendants.is_empty() {
        writeln!(out, "    ///").unwrap();
        writeln!(out, "    /// **Cascading:** Also affects descendants:").unwrap();
        for desc in &descendants {
            writeln!(out, "    /// - `{}`", desc.name).unwrap();
        }
    }
    writeln!(
        out,
        "    pub fn dangerously_disable_token_process_wide(disabled: bool) -> Result<(), crate::tokens::CompileTimeGuaranteedError> {{"
    )
    .unwrap();
    writeln!(out, "        #[cfg(all({all_conditions}))]").unwrap();
    writeln!(out, "        {{").unwrap();
    writeln!(out, "            let _ = disabled;").unwrap();
    writeln!(
        out,
        "            return Err(crate::tokens::CompileTimeGuaranteedError {{ token_name: Self::NAME }});"
    )
    .unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(out, "        #[cfg(not(all({all_conditions})))]").unwrap();
    writeln!(out, "        {{").unwrap();
    writeln!(
        out,
        "            {disabled_name}.store(disabled, Ordering::Relaxed);"
    )
    .unwrap();
    writeln!(out, "            let v = if disabled {{ 1 }} else {{ 0 }};").unwrap();
    writeln!(out, "            {cache_name}.store(v, Ordering::Relaxed);").unwrap();

    // Cascade to descendants
    for desc in &descendants {
        let desc_cache = cache_var_name(&desc.name);
        let desc_disabled = disabled_var_name(&desc.name);
        let desc_mod = file_module_for_token(desc);
        let in_same_file = all_tokens_in_file.iter().any(|t| t.name == desc.name);

        if in_same_file {
            writeln!(
                out,
                "            {desc_disabled}.store(disabled, Ordering::Relaxed);"
            )
            .unwrap();
            writeln!(out, "            {desc_cache}.store(v, Ordering::Relaxed);").unwrap();
        } else {
            // Descendants in another file need a cfg guard for cargo feature
            if let Some(cargo_feat) = &desc.cargo_feature {
                writeln!(out, "            #[cfg(feature = \"{cargo_feat}\")]").unwrap();
                writeln!(out, "            {{").unwrap();
                writeln!(
                    out,
                    "                super::{desc_mod}::{desc_disabled}.store(disabled, Ordering::Relaxed);"
                )
                .unwrap();
                writeln!(
                    out,
                    "                super::{desc_mod}::{desc_cache}.store(v, Ordering::Relaxed);"
                )
                .unwrap();
                writeln!(out, "            }}").unwrap();
            } else {
                writeln!(
                    out,
                    "            super::{desc_mod}::{desc_disabled}.store(disabled, Ordering::Relaxed);"
                )
                .unwrap();
                writeln!(
                    out,
                    "            super::{desc_mod}::{desc_cache}.store(v, Ordering::Relaxed);"
                )
                .unwrap();
            }
        }
    }

    writeln!(out, "            Ok(())").unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out).unwrap();

    // manually_disabled
    writeln!(
        out,
        "    /// Check if this token has been manually disabled process-wide."
    )
    .unwrap();
    writeln!(out, "    ///").unwrap();
    writeln!(
        out,
        "    /// Returns `Err` when all required features are compile-time enabled."
    )
    .unwrap();
    writeln!(
        out,
        "    pub fn manually_disabled() -> Result<bool, crate::tokens::CompileTimeGuaranteedError> {{"
    )
    .unwrap();
    writeln!(out, "        #[cfg(all({all_conditions}))]").unwrap();
    writeln!(out, "        {{").unwrap();
    writeln!(
        out,
        "            return Err(crate::tokens::CompileTimeGuaranteedError {{ token_name: Self::NAME }});"
    )
    .unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(out, "        #[cfg(not(all({all_conditions})))]").unwrap();
    writeln!(out, "        {{").unwrap();
    writeln!(
        out,
        "            Ok({disabled_name}.load(Ordering::Relaxed))"
    )
    .unwrap();
    writeln!(out, "        }}").unwrap();
    writeln!(out, "    }}").unwrap();

    writeln!(out, "}}").unwrap();
}

/// Walk parent chain and collect all ancestors (parent, grandparent, ...).
fn collect_ancestors<'a>(reg: &'a Registry, token: &'a TokenDef) -> Vec<&'a TokenDef> {
    let mut ancestors = Vec::new();
    let mut current = token;
    while let Some(parent_name) = &current.parent {
        if let Some(parent) = reg.token.iter().find(|t| t.name == *parent_name) {
            ancestors.push(parent);
            current = parent;
        } else {
            break;
        }
    }
    ancestors
}

/// Collect all descendants (tokens that have this token in their ancestor chain).
fn collect_descendants<'a>(reg: &'a Registry, token: &'a TokenDef) -> Vec<&'a TokenDef> {
    let mut descendants = Vec::new();
    for other in &reg.token {
        if other.name == token.name {
            continue;
        }
        // Check if token is an ancestor of other
        let ancestors = collect_ancestors(reg, other);
        if ancestors.iter().any(|a| a.name == token.name) {
            descendants.push(other);
        }
    }
    descendants
}

// ============================================================================
// Stub token implementations (cross-platform)
// ============================================================================

fn gen_stub_tokens(reg: &Registry, tokens: &[&TokenDef]) -> String {
    let _ = reg; // available if needed later
    let mut out = String::with_capacity(2048);
    writeln!(out, "//! Generated from token-registry.toml — DO NOT EDIT.").unwrap();
    writeln!(out, "//!").unwrap();
    writeln!(out, "//! Stub tokens: `summon()` always returns `None`.").unwrap();
    writeln!(out).unwrap();

    writeln!(out, "use crate::tokens::SimdToken;").unwrap();

    // Collect all trait names
    let trait_names: Vec<&str> = tokens
        .iter()
        .flat_map(|t| t.traits.iter().map(|s| s.as_str()))
        .collect::<std::collections::BTreeSet<_>>()
        .into_iter()
        .collect();

    if !trait_names.is_empty() {
        write!(out, "use crate::tokens::{{").unwrap();
        for (i, name) in trait_names.iter().enumerate() {
            if i > 0 {
                write!(out, ", ").unwrap();
            }
            write!(out, "{name}").unwrap();
        }
        writeln!(out, "}};").unwrap();
    }

    writeln!(out).unwrap();

    // Generate struct + SimdToken impl for each token
    for token in tokens {
        gen_stub_token_struct(&mut out, token);
        writeln!(out).unwrap();
    }

    // Aliases
    for token in tokens {
        gen_aliases(&mut out, token);
    }

    // Trait impls (same as real — traits apply to stubs too for generic code)
    gen_trait_impls(&mut out, tokens);

    out
}

fn gen_stub_token_struct(out: &mut String, token: &TokenDef) {
    let name = &token.name;
    let display = token.display_name.as_deref().unwrap_or(name);

    writeln!(
        out,
        "/// Stub for {display} token (not available on this architecture)."
    )
    .unwrap();
    writeln!(out, "#[derive(Clone, Copy, Debug)]").unwrap();
    writeln!(out, "pub struct {name} {{").unwrap();
    writeln!(out, "    _private: (),").unwrap();
    writeln!(out, "}}").unwrap();
    writeln!(out).unwrap();

    writeln!(out, "impl crate::tokens::Sealed for {name} {{}}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "impl SimdToken for {name} {{").unwrap();
    writeln!(out, "    const NAME: &'static str = \"{display}\";").unwrap();
    writeln!(out).unwrap();
    // compiled_with() returns Some(false) for stubs — wrong architecture
    writeln!(out, "    #[inline]").unwrap();
    writeln!(out, "    fn compiled_with() -> Option<bool> {{").unwrap();
    writeln!(out, "        Some(false) // Wrong architecture").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out).unwrap();
    // Note: guaranteed() has a default impl in the trait that calls compiled_with()
    writeln!(out, "    #[inline]").unwrap();
    writeln!(out, "    fn summon() -> Option<Self> {{").unwrap();
    writeln!(out, "        None // Not available on this architecture").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "    #[allow(deprecated)]").unwrap();
    writeln!(out, "    #[inline(always)]").unwrap();
    writeln!(out, "    unsafe fn forge_token_dangerously() -> Self {{").unwrap();
    writeln!(out, "        Self {{ _private: () }}").unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out, "}}").unwrap();

    // Disable methods for stubs — always Err(wrong architecture ≈ compile-time guaranteed to not exist)
    writeln!(out).unwrap();
    writeln!(out, "impl {name} {{").unwrap();
    writeln!(
        out,
        "    /// This token is not available on this architecture."
    )
    .unwrap();
    writeln!(
        out,
        "    pub fn dangerously_disable_token_process_wide(_disabled: bool) -> Result<(), crate::tokens::CompileTimeGuaranteedError> {{"
    )
    .unwrap();
    writeln!(
        out,
        "        Err(crate::tokens::CompileTimeGuaranteedError {{ token_name: Self::NAME }})"
    )
    .unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out).unwrap();
    writeln!(
        out,
        "    /// This token is not available on this architecture."
    )
    .unwrap();
    writeln!(
        out,
        "    pub fn manually_disabled() -> Result<bool, crate::tokens::CompileTimeGuaranteedError> {{"
    )
    .unwrap();
    writeln!(
        out,
        "        Err(crate::tokens::CompileTimeGuaranteedError {{ token_name: Self::NAME }})"
    )
    .unwrap();
    writeln!(out, "    }}").unwrap();
    writeln!(out, "}}").unwrap();
}

// ============================================================================
// Aliases (shared between real and stub)
// ============================================================================

fn gen_aliases(out: &mut String, token: &TokenDef) {
    for alias in &token.aliases {
        writeln!(out, "/// Type alias for [`{}`].", token.name).unwrap();
        writeln!(out, "pub type {alias} = {};", token.name).unwrap();
        writeln!(out).unwrap();
    }
}

// ============================================================================
// Trait impls (shared between real and stub)
// ============================================================================

fn gen_trait_impls(out: &mut String, tokens: &[&TokenDef]) {
    // Group by trait for readability
    let mut trait_to_tokens: std::collections::BTreeMap<&str, Vec<&str>> =
        std::collections::BTreeMap::new();

    for token in tokens {
        for trait_name in &token.traits {
            trait_to_tokens
                .entry(trait_name.as_str())
                .or_default()
                .push(&token.name);
        }
    }

    if trait_to_tokens.is_empty() {
        return;
    }

    writeln!(out).unwrap();

    for (trait_name, token_names) in &trait_to_tokens {
        for token_name in token_names {
            writeln!(out, "impl {trait_name} for {token_name} {{}}").unwrap();
        }
    }
}

// ============================================================================
// Trait definitions
// ============================================================================

fn gen_traits(reg: &Registry) -> String {
    let mut out = String::with_capacity(2048);
    writeln!(out, "//! Generated from token-registry.toml — DO NOT EDIT.").unwrap();
    writeln!(out, "//!").unwrap();
    writeln!(out, "//! Marker traits for SIMD capability levels.").unwrap();
    writeln!(out).unwrap();
    writeln!(out, "use crate::tokens::SimdToken;").unwrap();
    writeln!(out).unwrap();

    for trait_def in &reg.traits {
        gen_trait_def(&mut out, trait_def);
        writeln!(out).unwrap();
    }

    out
}

fn gen_trait_def(out: &mut String, trait_def: &TraitDef) {
    // Doc comment
    if let Some(doc) = &trait_def.doc {
        for line in doc.lines() {
            if line.is_empty() {
                writeln!(out, "///").unwrap();
            } else {
                writeln!(out, "/// {line}").unwrap();
            }
        }
    }

    // Trait definition with parent bounds
    let name = &trait_def.name;
    if trait_def.parents.is_empty() {
        writeln!(out, "pub trait {name}: SimdToken {{}}").unwrap();
    } else {
        let bounds = trait_def.parents.join(" + ");
        writeln!(out, "pub trait {name}: {bounds} {{}}").unwrap();
    }
}

// ============================================================================
// Module file
// ============================================================================

fn gen_mod_rs() -> String {
    let mut out = String::with_capacity(2048);
    writeln!(out, "//! Generated from token-registry.toml — DO NOT EDIT.").unwrap();
    writeln!(out, "//!").unwrap();
    writeln!(
        out,
        "//! cfg-gated module imports and re-exports for all token types."
    )
    .unwrap();
    writeln!(out).unwrap();

    // Traits (always available)
    writeln!(out, "mod traits;").unwrap();
    writeln!(out, "pub use traits::*;").unwrap();
    writeln!(out).unwrap();

    // x86 real + avx512
    writeln!(
        out,
        "// x86: real implementations on x86_64, stubs elsewhere"
    )
    .unwrap();
    writeln!(
        out,
        "#[cfg(any(target_arch = \"x86_64\", target_arch = \"x86\"))]"
    )
    .unwrap();
    writeln!(out, "mod x86;").unwrap();
    writeln!(
        out,
        "#[cfg(all(any(target_arch = \"x86_64\", target_arch = \"x86\"), feature = \"avx512\"))]"
    )
    .unwrap();
    writeln!(out, "mod x86_avx512;").unwrap();
    writeln!(
        out,
        "#[cfg(any(target_arch = \"x86_64\", target_arch = \"x86\"))]"
    )
    .unwrap();
    writeln!(out, "pub use x86::*;").unwrap();
    writeln!(
        out,
        "#[cfg(all(any(target_arch = \"x86_64\", target_arch = \"x86\"), feature = \"avx512\"))]"
    )
    .unwrap();
    writeln!(out, "pub use x86_avx512::*;").unwrap();
    writeln!(out).unwrap();

    // x86 stubs
    writeln!(
        out,
        "#[cfg(not(any(target_arch = \"x86_64\", target_arch = \"x86\")))]"
    )
    .unwrap();
    writeln!(out, "mod x86_stubs;").unwrap();
    writeln!(
        out,
        "#[cfg(all(not(any(target_arch = \"x86_64\", target_arch = \"x86\")), feature = \"avx512\"))]"
    )
    .unwrap();
    writeln!(out, "mod x86_avx512_stubs;").unwrap();
    writeln!(
        out,
        "#[cfg(not(any(target_arch = \"x86_64\", target_arch = \"x86\")))]"
    )
    .unwrap();
    writeln!(out, "pub use x86_stubs::*;").unwrap();
    writeln!(
        out,
        "#[cfg(all(not(any(target_arch = \"x86_64\", target_arch = \"x86\")), feature = \"avx512\"))]"
    )
    .unwrap();
    writeln!(out, "pub use x86_avx512_stubs::*;").unwrap();
    writeln!(out).unwrap();

    // ARM real
    writeln!(
        out,
        "// aarch64: real implementations on aarch64, stubs elsewhere"
    )
    .unwrap();
    writeln!(out, "#[cfg(target_arch = \"aarch64\")]").unwrap();
    writeln!(out, "mod arm;").unwrap();
    writeln!(out, "#[cfg(target_arch = \"aarch64\")]").unwrap();
    writeln!(out, "pub use arm::*;").unwrap();
    writeln!(out).unwrap();

    // ARM stubs
    writeln!(out, "#[cfg(not(target_arch = \"aarch64\"))]").unwrap();
    writeln!(out, "mod arm_stubs;").unwrap();
    writeln!(out, "#[cfg(not(target_arch = \"aarch64\"))]").unwrap();
    writeln!(out, "pub use arm_stubs::*;").unwrap();
    writeln!(out).unwrap();

    // WASM real
    writeln!(
        out,
        "// wasm32: real implementations on wasm32, stubs elsewhere"
    )
    .unwrap();
    writeln!(out, "#[cfg(target_arch = \"wasm32\")]").unwrap();
    writeln!(out, "mod wasm;").unwrap();
    writeln!(out, "#[cfg(target_arch = \"wasm32\")]").unwrap();
    writeln!(out, "pub use wasm::*;").unwrap();
    writeln!(out).unwrap();

    // WASM stubs
    writeln!(out, "#[cfg(not(target_arch = \"wasm32\"))]").unwrap();
    writeln!(out, "mod wasm_stubs;").unwrap();
    writeln!(out, "#[cfg(not(target_arch = \"wasm32\"))]").unwrap();
    writeln!(out, "pub use wasm_stubs::*;").unwrap();

    out
}

// ============================================================================
// Helpers
// ============================================================================

/// Determine which generated module file a token lives in.
fn file_module_for_token(token: &TokenDef) -> &str {
    match token.arch.as_str() {
        "x86" => {
            if token.cargo_feature.is_some() {
                "x86_avx512"
            } else {
                "x86"
            }
        }
        "aarch64" => "arm",
        "wasm" => "wasm",
        _ => "unknown",
    }
}
