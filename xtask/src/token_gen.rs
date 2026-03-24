//! Token code generator — generates `src/tokens/generated/` from `token-registry.toml`.
//!
//! Produces:
//! - Real implementations for native arch (with runtime detection)
//! - Stub implementations for cross-platform (summon → None)
//! - Trait definitions
//! - Module file with cfg-gated imports

use crate::registry::{Registry, TokenDef, TraitDef};
use indoc::formatdoc;

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

    // Collect deprecated trait names for #[allow(deprecated)] on impls
    let deprecated_traits: Vec<&str> = reg
        .traits
        .iter()
        .filter(|t| t.deprecated.is_some())
        .map(|t| t.name.as_str())
        .collect();

    // Group tokens by arch
    let x86_tokens: Vec<&TokenDef> = reg.token.iter().filter(|t| t.arch == "x86").collect();
    let arm_tokens: Vec<&TokenDef> = reg.token.iter().filter(|t| t.arch == "aarch64").collect();
    let wasm_tokens: Vec<&TokenDef> = reg.token.iter().filter(|t| t.arch == "wasm").collect();

    // Real implementations (all x86 tokens in one file — no base/avx512 split)
    files.push((
        "x86.rs".into(),
        gen_real_tokens(reg, &x86_tokens, "x86", &deprecated_traits),
    ));
    files.push((
        "arm.rs".into(),
        gen_real_tokens(reg, &arm_tokens, "aarch64", &deprecated_traits),
    ));
    files.push((
        "wasm.rs".into(),
        gen_real_tokens(reg, &wasm_tokens, "wasm", &deprecated_traits),
    ));

    // Stubs (all x86 tokens in one file)
    files.push((
        "x86_stubs.rs".into(),
        gen_stub_tokens(reg, &x86_tokens, &deprecated_traits),
    ));
    files.push((
        "arm_stubs.rs".into(),
        gen_stub_tokens(reg, &arm_tokens, &deprecated_traits),
    ));
    files.push((
        "wasm_stubs.rs".into(),
        gen_stub_tokens(reg, &wasm_tokens, &deprecated_traits),
    ));

    // Traits
    files.push(("traits.rs".into(), gen_traits(reg)));

    // Module file
    files.push(("mod.rs".into(), gen_mod_rs()));

    files
}

// ============================================================================
// Real token implementations (native arch)
// ============================================================================

fn gen_real_tokens(
    reg: &Registry,
    tokens: &[&TokenDef],
    arch: &str,
    deprecated_traits: &[&str],
) -> String {
    let mut out = String::with_capacity(4096);

    out.push_str(&formatdoc! {"
        //! Generated from token-registry.toml — DO NOT EDIT.
        //!
        //! Regenerate with: cargo xtask generate

        use crate::tokens::SimdToken;
    "});

    // Atomic imports (for tokens with runtime detection)
    if arch != "wasm" && !tokens.is_empty() {
        out.push_str("use core::sync::atomic::{AtomicBool, AtomicU8, Ordering};\n");
    }

    // Collect all trait names used by tokens in this file
    let trait_names: Vec<&str> = tokens
        .iter()
        .flat_map(|t| t.traits.iter().map(|s| s.as_str()))
        .collect::<std::collections::BTreeSet<_>>()
        .into_iter()
        .collect();

    if !trait_names.is_empty() {
        if trait_names.iter().any(|t| deprecated_traits.contains(t)) {
            out.push_str("#[allow(deprecated)]\n");
        }
        let traits_list = trait_names.join(", ");
        out.push_str(&format!("use crate::tokens::{{{traits_list}}};\n"));
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
        out.push_str(&format!("use super::{mod_name}::{type_name};\n"));
    }

    out.push('\n');

    // Generate cache and disabled statics (skip WASM - it's compile-time only)
    if arch != "wasm" {
        out.push_str("// Cache statics: 0 = unknown, 1 = unavailable, 2 = available\n");
        for token in tokens {
            let cache_name = cache_var_name(&token.name);
            let disabled_name = disabled_var_name(&token.name);

            // Tokens whose features are all baseline for their architecture have
            // their cache/disabled statics only used when `testable_dispatch`
            // is enabled. Suppress dead_code warnings for the default case.
            // - x86: SSE/SSE2 are always available on x86_64
            // - aarch64: NEON is always available (part of the AArch64 spec)
            let needs_allow = (arch == "x86" && is_x86_baseline_only(token))
                || (arch == "aarch64" && is_arm_baseline_only(token));
            let allow_attr = if needs_allow {
                "#[allow(dead_code)]\n"
            } else {
                ""
            };

            out.push_str(&formatdoc! {"
                {allow_attr}pub(super) static {cache_name}: AtomicU8 = AtomicU8::new(0);
                {allow_attr}pub(super) static {disabled_name}: AtomicBool = AtomicBool::new(false);
            "});
        }
        out.push('\n');
    }

    // Generate each token
    for token in tokens {
        gen_real_token_struct(&mut out, reg, token, arch, tokens);
        out.push('\n');
    }

    // Aliases
    for token in tokens {
        gen_aliases(&mut out, token);
    }

    // Trait impls
    gen_trait_impls(&mut out, tokens, deprecated_traits);

    out
}

fn gen_real_token_struct(
    out: &mut String,
    reg: &Registry,
    token: &TokenDef,
    arch: &str,
    all_tokens_in_file: &[&TokenDef],
) {
    let name = &token.name;
    let display = token.display_name.as_deref().unwrap_or(name);

    // Doc comment
    let doc_block = gen_doc_comment(&token.doc);

    // Feature flag strings
    let (_, target_features, enable_flags, disable_flags) = feature_flag_strings(token);

    out.push_str(&formatdoc! {"
        {doc_block}#[derive(Clone, Copy, Debug)]
        pub struct {name} {{
            _private: (),
        }}

        impl crate::tokens::Sealed for {name} {{}}

        impl SimdToken for {name} {{
            const NAME: &'static str = \"{display}\";
            const TARGET_FEATURES: &'static str = \"{target_features}\";
            const ENABLE_TARGET_FEATURES: &'static str = \"{enable_flags}\";
            const DISABLE_TARGET_FEATURES: &'static str = \"{disable_flags}\";

    "});

    // compiled_with()
    out.push_str(&gen_compiled_with(token, arch));

    // Note: guaranteed() has a default impl in the trait that calls compiled_with()

    // summon()
    out.push('\n');
    out.push_str(&gen_summon(token, arch));

    // forge_token_dangerously()
    out.push_str(&formatdoc! {"

            #[inline(always)]
            #[allow(deprecated)]
            unsafe fn forge_token_dangerously() -> Self {{
                Self {{ _private: () }}
            }}
        }}
    "});

    // Extraction methods
    gen_extraction_methods(out, reg, token);

    // Disable/getter methods (not for WASM — compile-time only)
    if arch != "wasm" {
        gen_disable_methods(out, reg, token, all_tokens_in_file);
    }
}

/// Generate the `compiled_with()` method for a real token.
///
/// When the `testable_dispatch` feature is active, compile-time
/// detection always returns `None` so that runtime disable can work.
fn gen_compiled_with(token: &TokenDef, arch: &str) -> String {
    // The "testable_dispatch" feature suppresses the compile-time fast path:
    // compiled_with() returns None instead of Some(true), forcing runtime detection.
    let dct_guard = ", not(feature = \"testable_dispatch\")";

    match arch {
        "x86" => {
            // Filter out sse/sse2 — they are x86_64 baseline
            let check_features: Vec<&str> = token
                .features
                .iter()
                .filter(|f| *f != "sse" && *f != "sse2")
                .map(|s| s.as_str())
                .collect();

            if check_features.is_empty() {
                // Baseline-only token (V1): always available on x86_64,
                // but testable_dispatch must still allow runtime disabling.
                formatdoc! {"
                    {INDENT}#[inline]
                    {INDENT}fn compiled_with() -> Option<bool> {{
                    {INDENT}    #[cfg(not(feature = \"testable_dispatch\"))]
                    {INDENT}    {{ Some(true) }}
                    {INDENT}    #[cfg(feature = \"testable_dispatch\")]
                    {INDENT}    {{ None }}
                    {INDENT}}}
                "}
            } else {
                let all_conditions = cfg_all_features(&check_features);
                formatdoc! {"
                    {INDENT}#[inline]
                    {INDENT}fn compiled_with() -> Option<bool> {{
                    {INDENT}    #[cfg(all({all_conditions}{dct_guard}))]
                    {INDENT}    {{ Some(true) }}
                    {INDENT}    #[cfg(not(all({all_conditions}{dct_guard})))]
                    {INDENT}    {{ None }}
                    {INDENT}}}
                "}
            }
        }
        "aarch64" => {
            // Check ALL features including neon — neon is NOT assumed baseline
            let check_features: Vec<&str> = token.features.iter().map(|s| s.as_str()).collect();
            let all_conditions = cfg_all_features(&check_features);

            formatdoc! {"
                {INDENT}#[inline]
                {INDENT}fn compiled_with() -> Option<bool> {{
                {INDENT}    #[cfg(all({all_conditions}{dct_guard}))]
                {INDENT}    {{ Some(true) }}
                {INDENT}    #[cfg(not(all({all_conditions}{dct_guard})))]
                {INDENT}    {{ None }}
                {INDENT}}}
            "}
        }
        "wasm" => {
            // No testable_dispatch guard for WASM — compile-time only,
            // no runtime detection to toggle.
            let check_features: Vec<&str> = token.features.iter().map(|s| s.as_str()).collect();
            let all_conditions = cfg_all_features(&check_features);
            formatdoc! {"
                {INDENT}#[inline]
                {INDENT}fn compiled_with() -> Option<bool> {{
                {INDENT}    #[cfg(all(target_arch = \"wasm32\", {all_conditions}))]
                {INDENT}    {{ Some(true) }}
                {INDENT}    #[cfg(not(all(target_arch = \"wasm32\", {all_conditions})))]
                {INDENT}    {{ None }}
                {INDENT}}}
            "}
        }
        _ => unreachable!("unknown arch: {arch}"),
    }
}

fn gen_summon(token: &TokenDef, arch: &str) -> String {
    match arch {
        "x86" => gen_summon_x86(token),
        "aarch64" => gen_summon_aarch64(token),
        "wasm" => gen_summon_wasm(token),
        _ => unreachable!("unknown arch: {arch}"),
    }
}

fn gen_summon_x86(token: &TokenDef) -> String {
    let dct_guard = ", not(feature = \"testable_dispatch\")";

    // Filter out sse/sse2 (x86_64 baseline, always available)
    let check_features: Vec<&str> = token
        .features
        .iter()
        .filter(|f| *f != "sse" && *f != "sse2")
        .map(|s| s.as_str())
        .collect();

    if check_features.is_empty() {
        // Baseline-only token (V1): always available on x86_64.
        // With testable_dispatch, use cache so runtime disabling works.
        let cache_name = cache_var_name(&token.name);
        return formatdoc! {"
            {INDENT}#[allow(deprecated)]
            {INDENT}#[inline]
            {INDENT}fn summon() -> Option<Self> {{
            {INDENT}    #[cfg(not(feature = \"testable_dispatch\"))]
            {INDENT}    {{
            {INDENT}        Some(unsafe {{ Self::forge_token_dangerously() }})
            {INDENT}    }}
            {INDENT}    #[cfg(feature = \"testable_dispatch\")]
            {INDENT}    {{
            {INDENT}        match {cache_name}.load(Ordering::Relaxed) {{
            {INDENT}            1 => None,
            {INDENT}            _ => Some(unsafe {{ Self::forge_token_dangerously() }}),
            {INDENT}        }}
            {INDENT}    }}
            {INDENT}}}
        "};
    }

    let cache_name = cache_var_name(&token.name);
    let all_features = cfg_all_features(&check_features);
    let detect_expr = gen_x86_detect_expr(&check_features);

    formatdoc! {"
        {INDENT}#[allow(deprecated)]
        {INDENT}#[inline(always)]
        {INDENT}fn summon() -> Option<Self> {{
        {INDENT}    // Compile-time fast path (suppressed by testable_dispatch)
        {INDENT}    #[cfg(all({all_features}{dct_guard}))]
        {INDENT}    {{
        {INDENT}        Some(unsafe {{ Self::forge_token_dangerously() }})
        {INDENT}    }}

        {INDENT}    // Runtime path with caching
        {INDENT}    #[cfg(not(all({all_features}{dct_guard})))]
        {INDENT}    {{
        {INDENT}        match {cache_name}.load(Ordering::Relaxed) {{
        {INDENT}            2 => Some(unsafe {{ Self::forge_token_dangerously() }}),
        {INDENT}            1 => None,
        {INDENT}            _ => {{
        {INDENT}                let available = {detect_expr};
        {INDENT}                {cache_name}.store(if available {{ 2 }} else {{ 1 }}, Ordering::Relaxed);
        {INDENT}                if available {{
        {INDENT}                    Some(unsafe {{ Self::forge_token_dangerously() }})
        {INDENT}                }} else {{
        {INDENT}                    None
        {INDENT}                }}
        {INDENT}            }}
        {INDENT}        }}
        {INDENT}    }}
        {INDENT}}}
    "}
}

fn gen_summon_aarch64(token: &TokenDef) -> String {
    let dct_guard = ", not(feature = \"testable_dispatch\")";

    // Check ALL features including neon — neon is NOT assumed baseline
    let check_features: Vec<&str> = token.features.iter().map(|s| s.as_str()).collect();

    let cache_name = cache_var_name(&token.name);
    let all_features = cfg_all_features(&check_features);
    let detect_expr = gen_aarch64_detect_expr(&check_features);

    formatdoc! {"
        {INDENT}#[allow(deprecated)]
        {INDENT}#[inline(always)]
        {INDENT}fn summon() -> Option<Self> {{
        {INDENT}    // Compile-time fast path (suppressed by testable_dispatch)
        {INDENT}    #[cfg(all({all_features}{dct_guard}))]
        {INDENT}    {{
        {INDENT}        Some(unsafe {{ Self::forge_token_dangerously() }})
        {INDENT}    }}

        {INDENT}    // Runtime path with caching
        {INDENT}    #[cfg(not(all({all_features}{dct_guard})))]
        {INDENT}    {{
        {INDENT}        match {cache_name}.load(Ordering::Relaxed) {{
        {INDENT}            2 => Some(unsafe {{ Self::forge_token_dangerously() }}),
        {INDENT}            1 => None,
        {INDENT}            _ => {{
        {INDENT}                let available = {detect_expr};
        {INDENT}                {cache_name}.store(if available {{ 2 }} else {{ 1 }}, Ordering::Relaxed);
        {INDENT}                if available {{
        {INDENT}                    Some(unsafe {{ Self::forge_token_dangerously() }})
        {INDENT}                }} else {{
        {INDENT}                    None
        {INDENT}                }}
        {INDENT}            }}
        {INDENT}        }}
        {INDENT}    }}
        {INDENT}}}
    "}
}

fn gen_summon_wasm(token: &TokenDef) -> String {
    // No testable_dispatch guard for WASM — compile-time only,
    // no runtime detection to toggle.
    let check_features: Vec<&str> = token.features.iter().map(|s| s.as_str()).collect();
    let all_conditions = cfg_all_features(&check_features);
    formatdoc! {"
        {INDENT}#[allow(deprecated)]
        {INDENT}#[inline]
        {INDENT}fn summon() -> Option<Self> {{
        {INDENT}    #[cfg(all(target_arch = \"wasm32\", {all_conditions}))]
        {INDENT}    {{
        {INDENT}        Some(unsafe {{ Self::forge_token_dangerously() }})
        {INDENT}    }}
        {INDENT}    #[cfg(not(all(target_arch = \"wasm32\", {all_conditions})))]
        {INDENT}    {{
        {INDENT}        None
        {INDENT}    }}
        {INDENT}}}
    "}
}

fn gen_extraction_methods(out: &mut String, reg: &Registry, token: &TokenDef) {
    let ancestors = collect_ancestors(reg, token);
    if ancestors.is_empty() {
        return;
    }

    let name = &token.name;
    let token_display = token.display_name.as_deref().unwrap_or(name);

    out.push_str(&format!("\nimpl {name} {{\n"));

    for ancestor in &ancestors {
        let anc_name = &ancestor.name;
        let short = ancestor
            .short_name
            .as_deref()
            .unwrap_or("MISSING_SHORT_NAME");
        let anc_display = ancestor.display_name.as_deref().unwrap_or(anc_name);

        out.push_str(&formatdoc! {"
            {INDENT}/// Get a {anc_name} ({token_display} implies {anc_display})
            {INDENT}#[allow(deprecated)]
            {INDENT}#[inline(always)]
            {INDENT}pub fn {short}(self) -> {anc_name} {{
            {INDENT}    unsafe {{ {anc_name}::forge_token_dangerously() }}
            {INDENT}}}
        "});

        // Extraction aliases for this ancestor (e.g., .avx512() for X64V4Token)
        for alias_name in &ancestor.extraction_aliases {
            out.push_str(&formatdoc! {"

                {INDENT}/// Get a {anc_name} (alias for `.{short}()`)
                {INDENT}#[allow(deprecated)]
                {INDENT}#[inline(always)]
                {INDENT}pub fn {alias_name}(self) -> {anc_name} {{
                {INDENT}    unsafe {{ {anc_name}::forge_token_dangerously() }}
                {INDENT}}}
            "});
        }
    }

    out.push_str("}\n");
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

    // For the compile-time guard, use the FULL unfiltered feature list.
    // The testable_dispatch feature suppresses the guard.
    let base_conditions = cfg_all_features_full(&token.features);
    let all_conditions = format!("{base_conditions}, not(feature = \"testable_dispatch\")");

    // Collect descendants (tokens that have this token in their ancestor chain)
    let descendants = collect_descendants(reg, token);

    // Build cascade code
    let cascade_code = gen_cascade_code(&descendants, all_tokens_in_file);

    // Build descendant doc lines
    let descendant_docs = if !descendants.is_empty() {
        let mut lines = String::from("    ///\n    /// **Cascading:** Also affects descendants:\n");
        for desc in &descendants {
            lines.push_str(&format!("    /// - `{}`\n", desc.name));
        }
        lines
    } else {
        String::new()
    };

    out.push_str(&formatdoc! {"

        impl {name} {{
            /// Disable this token process-wide for testing and benchmarking.
            ///
            /// When disabled, `summon()` will return `None` even if the CPU supports
            /// the required features.
            ///
            /// Returns `Err` when all required features are compile-time enabled
            /// (e.g., via `-Ctarget-cpu=native`), since the compiler has already
            /// elided the runtime checks.
        {descendant_docs}    #[allow(clippy::needless_return)]
            pub fn dangerously_disable_token_process_wide(disabled: bool) -> Result<(), crate::tokens::CompileTimeGuaranteedError> {{
                #[cfg(all({all_conditions}))]
                {{
                    let _ = disabled;
                    return Err(crate::tokens::CompileTimeGuaranteedError {{ token_name: Self::NAME, target_features: Self::TARGET_FEATURES, disable_flags: Self::DISABLE_TARGET_FEATURES }});
                }}
                #[cfg(not(all({all_conditions})))]
                {{
                    {disabled_name}.store(disabled, Ordering::Relaxed);
                    let v = if disabled {{ 1 }} else {{ 0 }};
                    {cache_name}.store(v, Ordering::Relaxed);
        {cascade_code}            Ok(())
                }}
            }}

            /// Check if this token has been manually disabled process-wide.
            ///
            /// Returns `Err` when all required features are compile-time enabled.
            #[allow(clippy::needless_return)]
            pub fn manually_disabled() -> Result<bool, crate::tokens::CompileTimeGuaranteedError> {{
                #[cfg(all({all_conditions}))]
                {{
                    return Err(crate::tokens::CompileTimeGuaranteedError {{ token_name: Self::NAME, target_features: Self::TARGET_FEATURES, disable_flags: Self::DISABLE_TARGET_FEATURES }});
                }}
                #[cfg(not(all({all_conditions})))]
                {{
                    Ok({disabled_name}.load(Ordering::Relaxed))
                }}
            }}
        }}
    "});
}

/// Generate cascade code for disabling descendant tokens.
fn gen_cascade_code(descendants: &[&TokenDef], all_tokens_in_file: &[&TokenDef]) -> String {
    let mut cascade = String::new();
    for desc in descendants {
        let desc_cache = cache_var_name(&desc.name);
        let desc_disabled = disabled_var_name(&desc.name);
        let desc_mod = file_module_for_token(desc);
        let in_same_file = all_tokens_in_file.iter().any(|t| t.name == desc.name);

        if in_same_file {
            cascade.push_str(&formatdoc! {"
                {INDENT}{INDENT}{INDENT}{desc_disabled}.store(disabled, Ordering::Relaxed);
                {INDENT}{INDENT}{INDENT}{desc_cache}.store(v, Ordering::Relaxed);
            "});
        } else if let Some(cargo_feat) = &desc.cargo_feature {
            cascade.push_str(&formatdoc! {"
                {INDENT}{INDENT}{INDENT}#[cfg(feature = \"{cargo_feat}\")]
                {INDENT}{INDENT}{INDENT}{{
                {INDENT}{INDENT}{INDENT}    super::{desc_mod}::{desc_disabled}.store(disabled, Ordering::Relaxed);
                {INDENT}{INDENT}{INDENT}    super::{desc_mod}::{desc_cache}.store(v, Ordering::Relaxed);
                {INDENT}{INDENT}{INDENT}}}
            "});
        } else {
            cascade.push_str(&formatdoc! {"
                {INDENT}{INDENT}{INDENT}super::{desc_mod}::{desc_disabled}.store(disabled, Ordering::Relaxed);
                {INDENT}{INDENT}{INDENT}super::{desc_mod}::{desc_cache}.store(v, Ordering::Relaxed);
            "});
        }
    }
    cascade
}

/// BFS through parent DAG and collect all ancestors (deduplicated, stable order).
fn collect_ancestors<'a>(reg: &'a Registry, token: &'a TokenDef) -> Vec<&'a TokenDef> {
    let mut ancestors = Vec::new();
    let mut visited = std::collections::HashSet::new();
    let mut queue: std::collections::VecDeque<&str> =
        token.parents.iter().map(|s| s.as_str()).collect();
    while let Some(name) = queue.pop_front() {
        if !visited.insert(name) {
            continue;
        }
        if let Some(parent) = reg.token.iter().find(|t| t.name == name) {
            ancestors.push(parent);
            for gp in &parent.parents {
                queue.push_back(gp.as_str());
            }
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

fn gen_stub_tokens(reg: &Registry, tokens: &[&TokenDef], deprecated_traits: &[&str]) -> String {
    let _ = reg; // available if needed later
    let mut out = String::with_capacity(2048);

    out.push_str(&formatdoc! {"
        //! Generated from token-registry.toml — DO NOT EDIT.
        //!
        //! Stub tokens: `summon()` always returns `None`.

        use crate::tokens::SimdToken;
    "});

    // Collect all trait names
    let trait_names: Vec<&str> = tokens
        .iter()
        .flat_map(|t| t.traits.iter().map(|s| s.as_str()))
        .collect::<std::collections::BTreeSet<_>>()
        .into_iter()
        .collect();

    if !trait_names.is_empty() {
        if trait_names.iter().any(|t| deprecated_traits.contains(t)) {
            out.push_str("#[allow(deprecated)]\n");
        }
        let traits_list = trait_names.join(", ");
        out.push_str(&format!("use crate::tokens::{{{traits_list}}};\n"));
    }

    out.push('\n');

    // Generate struct + SimdToken impl for each token
    for token in tokens {
        gen_stub_token_struct(&mut out, token);
        out.push('\n');
    }

    // Aliases
    for token in tokens {
        gen_aliases(&mut out, token);
    }

    // Trait impls (same as real — traits apply to stubs too for generic code)
    gen_trait_impls(&mut out, tokens, deprecated_traits);

    out
}

fn gen_stub_token_struct(out: &mut String, token: &TokenDef) {
    let name = &token.name;
    let display = token.display_name.as_deref().unwrap_or(name);

    // Feature flag strings — stubs carry the SAME strings as real tokens
    let (_, target_features, enable_flags, disable_flags) = feature_flag_strings(token);

    out.push_str(&formatdoc! {"
        /// Stub for {display} token (not available on this architecture).
        #[derive(Clone, Copy, Debug)]
        pub struct {name} {{
            _private: (),
        }}

        impl crate::tokens::Sealed for {name} {{}}

        impl SimdToken for {name} {{
            const NAME: &'static str = \"{display}\";
            const TARGET_FEATURES: &'static str = \"{target_features}\";
            const ENABLE_TARGET_FEATURES: &'static str = \"{enable_flags}\";
            const DISABLE_TARGET_FEATURES: &'static str = \"{disable_flags}\";

            #[inline]
            fn compiled_with() -> Option<bool> {{
                Some(false) // Wrong architecture
            }}

            // Note: guaranteed() has a default impl in the trait that calls compiled_with()

            #[inline]
            fn summon() -> Option<Self> {{
                None // Not available on this architecture
            }}

            #[allow(deprecated)]
            #[inline(always)]
            unsafe fn forge_token_dangerously() -> Self {{
                Self {{ _private: () }}
            }}
        }}

        impl {name} {{
            /// This token is not available on this architecture.
            pub fn dangerously_disable_token_process_wide(_disabled: bool) -> Result<(), crate::tokens::CompileTimeGuaranteedError> {{
                Err(crate::tokens::CompileTimeGuaranteedError {{ token_name: Self::NAME, target_features: Self::TARGET_FEATURES, disable_flags: Self::DISABLE_TARGET_FEATURES }})
            }}

            /// This token is not available on this architecture.
            pub fn manually_disabled() -> Result<bool, crate::tokens::CompileTimeGuaranteedError> {{
                Err(crate::tokens::CompileTimeGuaranteedError {{ token_name: Self::NAME, target_features: Self::TARGET_FEATURES, disable_flags: Self::DISABLE_TARGET_FEATURES }})
            }}
        }}
    "});
}

// ============================================================================
// Aliases (shared between real and stub)
// ============================================================================

fn gen_aliases(out: &mut String, token: &TokenDef) {
    for alias in &token.aliases {
        if let Some(msg) = token.deprecated_aliases.get(alias) {
            let escaped = msg.replace('"', "\\\"");
            out.push_str(&formatdoc! {"
                /// Type alias for [`{name}`].
                #[deprecated(since = \"0.9.9\", note = \"{escaped}\")]
                pub type {alias} = {name};

            ", name = token.name});
        } else {
            out.push_str(&formatdoc! {"
                /// Type alias for [`{name}`].
                pub type {alias} = {name};

            ", name = token.name});
        }
    }
}

// ============================================================================
// Trait impls (shared between real and stub)
// ============================================================================

fn gen_trait_impls(out: &mut String, tokens: &[&TokenDef], deprecated_trait_names: &[&str]) {
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

    out.push('\n');

    for (trait_name, token_names) in &trait_to_tokens {
        let needs_allow = deprecated_trait_names.contains(trait_name);
        for token_name in token_names {
            if needs_allow {
                out.push_str("#[allow(deprecated)]\n");
            }
            out.push_str(&format!("impl {trait_name} for {token_name} {{}}\n"));
        }
    }
}

// ============================================================================
// Trait definitions
// ============================================================================

fn gen_traits(reg: &Registry) -> String {
    let mut out = String::with_capacity(2048);

    out.push_str(&formatdoc! {"
        //! Generated from token-registry.toml — DO NOT EDIT.
        //!
        //! Marker traits for SIMD capability levels.

        use crate::tokens::SimdToken;

    "});

    let deprecated_names: Vec<&str> = reg
        .traits
        .iter()
        .filter(|t| t.deprecated.is_some())
        .map(|t| t.name.as_str())
        .collect();

    for trait_def in &reg.traits {
        gen_trait_def(&mut out, trait_def, &deprecated_names);
        out.push('\n');
    }

    out
}

fn gen_trait_def(out: &mut String, trait_def: &TraitDef, deprecated_trait_names: &[&str]) {
    let doc_block = gen_doc_comment(&trait_def.doc);
    let name = &trait_def.name;

    // If this trait or any parent is deprecated, add #[allow(deprecated)]
    let has_deprecated_parent = trait_def
        .parents
        .iter()
        .any(|p| deprecated_trait_names.contains(&p.as_str()));
    let allow_deprecated = if has_deprecated_parent || trait_def.deprecated.is_some() {
        "#[allow(deprecated)]\n"
    } else {
        ""
    };

    let dep_attr = if let Some(msg) = &trait_def.deprecated {
        format!(
            "#[deprecated(since = \"0.9.9\", note = \"{}\")]\n",
            msg.replace('"', "\\\"")
        )
    } else {
        String::new()
    };

    if trait_def.parents.is_empty() {
        out.push_str(&formatdoc! {"
            {doc_block}{allow_deprecated}{dep_attr}pub trait {name}: SimdToken {{}}
        "});
    } else {
        let bounds = trait_def.parents.join(" + ");
        out.push_str(&formatdoc! {"
            {doc_block}{allow_deprecated}{dep_attr}pub trait {name}: {bounds} {{}}
        "});
    }
}

// ============================================================================
// Module file
// ============================================================================

fn gen_mod_rs() -> String {
    formatdoc! {r#"
        //! Generated from token-registry.toml — DO NOT EDIT.
        //!
        //! cfg-gated module imports and re-exports for all token types.

        mod traits;
        pub use traits::*;

        // x86: real implementations on x86_64, stubs elsewhere
        #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
        mod x86;
        #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
        pub use x86::*;

        #[cfg(not(any(target_arch = "x86_64", target_arch = "x86")))]
        mod x86_stubs;
        #[cfg(not(any(target_arch = "x86_64", target_arch = "x86")))]
        pub use x86_stubs::*;

        // aarch64: real implementations on aarch64, stubs elsewhere
        #[cfg(target_arch = "aarch64")]
        mod arm;
        #[cfg(target_arch = "aarch64")]
        pub use arm::*;

        #[cfg(not(target_arch = "aarch64"))]
        mod arm_stubs;
        #[cfg(not(target_arch = "aarch64"))]
        pub use arm_stubs::*;

        // wasm32: real implementations on wasm32, stubs elsewhere
        #[cfg(target_arch = "wasm32")]
        mod wasm;
        #[cfg(target_arch = "wasm32")]
        pub use wasm::*;

        #[cfg(not(target_arch = "wasm32"))]
        mod wasm_stubs;
        #[cfg(not(target_arch = "wasm32"))]
        pub use wasm_stubs::*;
    "#}
}

// ============================================================================
// Helpers
// ============================================================================

/// Indent constant for generated code inside impl blocks.
const INDENT: &str = "    ";

/// Compute the feature flag strings for a token:
/// (target_features, enable_flags, disable_flags).
///
/// All features are included (sse/sse2 are NOT stripped for x86).
/// For aarch64: includes all features (neon is NOT baseline).
/// For wasm: uses "simd128".
fn feature_flag_strings(token: &TokenDef) -> (&'static str, String, String, String) {
    let filtered: Vec<&str> = match token.arch.as_str() {
        "x86" => token.features.iter().map(|s| s.as_str()).collect(),
        "aarch64" => token.features.iter().map(|s| s.as_str()).collect(),
        "wasm" => token.features.iter().map(|s| s.as_str()).collect(),
        _ => vec![],
    };

    if filtered.is_empty() {
        return ("", String::new(), String::new(), String::new());
    }

    let target_features = filtered.join(",");
    let enable = format!(
        "-Ctarget-feature={}",
        filtered
            .iter()
            .map(|f| format!("+{f}"))
            .collect::<Vec<_>>()
            .join(",")
    );
    let disable = format!(
        "-Ctarget-feature={}",
        filtered
            .iter()
            .map(|f| format!("-{f}"))
            .collect::<Vec<_>>()
            .join(",")
    );

    // We return a &'static str "" placeholder — caller must use the Strings
    ("", target_features, enable, disable)
}

/// Returns true if an x86 token's features are all baseline (sse/sse2 only).
/// Such tokens are always available on x86_64 and don't need runtime detection.
fn is_x86_baseline_only(token: &TokenDef) -> bool {
    token.features.iter().all(|f| f == "sse" || f == "sse2")
}

fn is_arm_baseline_only(token: &TokenDef) -> bool {
    token.features.iter().all(|f| f == "neon")
}

/// Determine which generated module file a token lives in.
fn file_module_for_token(token: &TokenDef) -> &str {
    match token.arch.as_str() {
        "x86" => "x86",
        "aarch64" => "arm",
        "wasm" => "wasm",
        _ => "unknown",
    }
}

/// Generate `target_feature = "feat1", target_feature = "feat2", ...` for cfg(all(...)).
fn cfg_all_features(features: &[&str]) -> String {
    features
        .iter()
        .map(|f| format!("target_feature = \"{f}\""))
        .collect::<Vec<_>>()
        .join(", ")
}

/// Generate cfg conditions from full (owned) feature list.
fn cfg_all_features_full(features: &[String]) -> String {
    features
        .iter()
        .map(|f| format!("target_feature = \"{f}\""))
        .collect::<Vec<_>>()
        .join(", ")
}

/// Generate x86 feature detection expression.
fn gen_x86_detect_expr(features: &[&str]) -> String {
    if features.len() == 1 {
        format!("crate::is_x86_feature_available!(\"{}\")", features[0])
    } else {
        let parts: Vec<String> = features
            .iter()
            .map(|f| format!("crate::is_x86_feature_available!(\"{f}\")"))
            .collect();
        parts.join("\n                        && ")
    }
}

/// Generate aarch64 feature detection expression.
fn gen_aarch64_detect_expr(features: &[&str]) -> String {
    if features.len() == 1 {
        format!("crate::is_aarch64_feature_available!(\"{}\")", features[0])
    } else {
        let parts: Vec<String> = features
            .iter()
            .map(|f| format!("crate::is_aarch64_feature_available!(\"{f}\")"))
            .collect();
        parts.join("\n                        && ")
    }
}

/// Generate a doc comment block from an optional doc string.
fn gen_doc_comment(doc: &Option<String>) -> String {
    match doc {
        Some(doc) => {
            let mut block = String::new();
            for line in doc.lines() {
                if line.is_empty() {
                    block.push_str("///\n");
                } else {
                    block.push_str(&format!("/// {line}\n"));
                }
            }
            block
        }
        None => String::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::registry::Registry;
    use std::path::Path;

    fn load_test_registry() -> Registry {
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .join("token-registry.toml");
        Registry::load(&path).expect("Failed to load token-registry.toml")
    }

    #[test]
    fn generates_expected_file_set() {
        let reg = load_test_registry();
        let files = generate_token_files(&reg);
        let names: Vec<&str> = files.iter().map(|(n, _)| n.as_str()).collect();

        // Must produce exactly these files
        let expected = [
            "x86.rs",
            "arm.rs",
            "wasm.rs",
            "x86_stubs.rs",
            "arm_stubs.rs",
            "wasm_stubs.rs",
            "traits.rs",
            "mod.rs",
        ];
        for exp in &expected {
            assert!(
                names.contains(exp),
                "Missing expected file: {exp}. Got: {names:?}"
            );
        }
        assert_eq!(
            files.len(),
            expected.len(),
            "Unexpected file count. Got: {names:?}"
        );
    }

    #[test]
    fn x86_real_tokens_contain_all_x86_tokens() {
        let reg = load_test_registry();
        let files = generate_token_files(&reg);
        let x86_code = &files.iter().find(|(n, _)| n == "x86.rs").unwrap().1;

        let x86_tokens: Vec<&str> = reg
            .token
            .iter()
            .filter(|t| t.arch == "x86")
            .map(|t| t.name.as_str())
            .collect();

        for token_name in &x86_tokens {
            assert!(
                x86_code.contains(&format!("pub struct {token_name}")),
                "x86.rs missing struct definition for {token_name}"
            );
            assert!(
                x86_code.contains(&format!("impl SimdToken for {token_name}")),
                "x86.rs missing SimdToken impl for {token_name}"
            );
        }
    }

    #[test]
    fn arm_real_tokens_contain_all_arm_tokens() {
        let reg = load_test_registry();
        let files = generate_token_files(&reg);
        let arm_code = &files.iter().find(|(n, _)| n == "arm.rs").unwrap().1;

        let arm_tokens: Vec<&str> = reg
            .token
            .iter()
            .filter(|t| t.arch == "aarch64")
            .map(|t| t.name.as_str())
            .collect();

        for token_name in &arm_tokens {
            assert!(
                arm_code.contains(&format!("pub struct {token_name}")),
                "arm.rs missing struct definition for {token_name}"
            );
            assert!(
                arm_code.contains(&format!("impl SimdToken for {token_name}")),
                "arm.rs missing SimdToken impl for {token_name}"
            );
        }
    }

    #[test]
    fn stubs_return_none_for_summon() {
        let reg = load_test_registry();
        let files = generate_token_files(&reg);

        for (name, code) in &files {
            if name.ends_with("_stubs.rs") {
                // Every stub's summon must return None
                assert!(
                    !code.contains("Some(Self"),
                    "{name} contains Some(Self — stubs must only return None"
                );
            }
        }
    }

    #[test]
    fn traits_file_contains_all_traits() {
        let reg = load_test_registry();
        let files = generate_token_files(&reg);
        let traits_code = &files.iter().find(|(n, _)| n == "traits.rs").unwrap().1;

        for trait_def in &reg.traits {
            assert!(
                traits_code.contains(&format!("pub trait {}", trait_def.name)),
                "traits.rs missing trait: {}",
                trait_def.name
            );
        }
    }

    #[test]
    fn real_tokens_have_summon_and_compiled_with() {
        let reg = load_test_registry();
        let files = generate_token_files(&reg);

        for (name, code) in &files {
            if name == "x86.rs" || name == "arm.rs" || name == "wasm.rs" {
                // Every real token file must implement summon() and compiled_with()
                let arch_tokens: Vec<&str> = reg
                    .token
                    .iter()
                    .filter(|t| match name.as_str() {
                        "x86.rs" => t.arch == "x86",
                        "arm.rs" => t.arch == "aarch64",
                        "wasm.rs" => t.arch == "wasm",
                        _ => false,
                    })
                    .map(|t| t.name.as_str())
                    .collect();

                for token_name in &arch_tokens {
                    assert!(
                        code.contains("fn summon()"),
                        "{name} missing summon() for {token_name}"
                    );
                    assert!(
                        code.contains("fn compiled_with()"),
                        "{name} missing compiled_with() for {token_name}"
                    );
                }
            }
        }
    }

    #[test]
    fn parent_extraction_methods_generated() {
        let reg = load_test_registry();
        let files = generate_token_files(&reg);
        let x86_code = &files.iter().find(|(n, _)| n == "x86.rs").unwrap().1;

        // X64V3Token has parent X64V2Token — should generate extraction method
        let v3 = reg.find_token("X64V3Token").unwrap();
        assert!(
            !v3.parents.is_empty(),
            "X64V3Token should have parents for this test"
        );
        // The short_name of the parent becomes the extraction method
        let v2 = reg.find_token("X64V2Token").unwrap();
        let short = v2.short_name.as_deref().expect("X64V2Token has short_name");
        assert!(
            x86_code.contains(&format!("pub fn {short}(self)")),
            "x86.rs missing parent extraction method {short}() on X64V3Token",
        );
    }

    #[test]
    fn cache_variable_names_are_unique() {
        let reg = load_test_registry();

        let mut cache_names = std::collections::HashSet::new();
        for token in &reg.token {
            if token.arch == "any" {
                continue;
            }
            let name = cache_var_name(&token.name);
            assert!(
                cache_names.insert(name.clone()),
                "Duplicate cache variable: {name} (from {})",
                token.name
            );
        }
    }

    #[test]
    fn mod_rs_routes_all_architectures() {
        let reg = load_test_registry();
        let files = generate_token_files(&reg);
        let mod_code = &files.iter().find(|(n, _)| n == "mod.rs").unwrap().1;

        // Must have cfg routing for all architectures
        assert!(
            mod_code.contains("target_arch = \"x86_64\""),
            "mod.rs missing x86_64 routing"
        );
        assert!(
            mod_code.contains("target_arch = \"aarch64\""),
            "mod.rs missing aarch64 routing"
        );
        assert!(
            mod_code.contains("target_arch = \"wasm32\""),
            "mod.rs missing wasm32 routing"
        );
    }
}
