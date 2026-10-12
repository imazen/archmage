//! Generate the macro lookup tables from the token registry.
use super::{Registry, TokenDef, tier_tag};

impl Registry {
    /// One descriptor table serves definition and call expansion. Wildcard
    /// policies remain explicit frontend choices, not registry defaults.
    pub fn generate_dispatch_registry(&self) -> String {
        let mut tokens: Vec<_> = self
            .token
            .iter()
            .filter(|t| t.dispatch_priority.is_some())
            .collect();
        let arch_order = |arch: &str| match arch {
            "x86" => 0,
            "aarch64" => 1,
            "wasm" => 2,
            _ => 3,
        };
        tokens.sort_by(|a, b| {
            arch_order(&a.arch)
                .cmp(&arch_order(&b.arch))
                .then_with(|| b.dispatch_priority.cmp(&a.dispatch_priority))
        });
        let mut out = String::from(
            "//! Generated from token-registry.toml — DO NOT EDIT.\n\nuse crate::tiers::TierDescriptor;\n\npub(crate) const DISPATCH_TIERS: &[TierDescriptor] = &[\n",
        );
        for token in tokens {
            let name = token
                .short_name
                .as_ref()
                .expect("validated dispatch short name");
            let method = token
                .dispatch_as
                .as_ref()
                .expect("validated dispatch extraction");
            let arch = match token.arch.as_str() {
                "x86" => "x86_64",
                "wasm" => "wasm32",
                other => other,
            };
            let gate = token
                .legacy_dispatch_gate
                .as_ref()
                .map_or("None".to_string(), |g| format!("Some({g:?})"));
            out.push_str(&format!("    TierDescriptor {{ name: {name:?}, suffix: {name:?}, token_path: \"archmage::{}\", as_method: {method:?}, target_arch: Some({arch:?}), cfg_feature: {gate}, priority: {} }},\n", token.name, token.dispatch_priority.unwrap()));
        }
        out.push_str("    TierDescriptor { name: \"scalar\", suffix: \"scalar\", token_path: \"archmage::ScalarToken\", as_method: \"as_scalar\", target_arch: None, cfg_feature: None, priority: 0 },\n");
        out.push_str("    TierDescriptor { name: \"default\", suffix: \"default\", token_path: \"\", as_method: \"\", target_arch: None, cfg_feature: None, priority: 0 },\n];\n");
        out
    }

    /// Generate `generated_registry.rs` content for `archmage-macros`.
    ///
    /// This produces:
    /// - `token_to_features()` — maps token names (including aliases) to feature lists
    /// - `trait_to_features()` — maps trait and token names to feature lists (for bounds)
    /// - `ALL_CONCRETE_TOKENS` — all token names including aliases
    /// - `ALL_TRAIT_NAMES` — all trait names
    pub fn generate_macro_registry(&self, major_version: u32) -> String {
        use indoc::formatdoc;
        let mut out = String::with_capacity(8192);

        out.push_str(&formatdoc! {"
            //! Generated from token-registry.toml — DO NOT EDIT.
            //!
            //! Regenerate with: cargo run -p xtask -- generate

        "});

        self.gen_token_to_features(&mut out);
        out.push('\n');
        self.gen_feature_csv(&mut out);
        out.push('\n');
        self.gen_trait_to_features(&mut out);
        out.push('\n');
        self.gen_token_to_arch(&mut out);
        out.push('\n');
        self.gen_token_to_magetypes_namespace(&mut out);
        out.push('\n');
        self.gen_trait_to_magetypes_namespace(&mut out);
        out.push('\n');
        self.gen_trait_to_arch(&mut out);
        out.push('\n');
        self.gen_tier_to_canonical_token(&mut out);
        out.push('\n');
        self.gen_canonical_token_to_tier_suffix(&mut out);
        out.push('\n');
        self.gen_can_downgrade_tier(&mut out);
        out.push('\n');
        self.gen_expected_tier_tag(&mut out, major_version);
        out.push('\n');
        self.gen_all_concrete_tokens(&mut out);
        out.push('\n');
        self.gen_all_trait_names(&mut out);
        out.push('\n');

        out
    }

    fn gen_token_to_features(&self, out: &mut String) {
        use indoc::formatdoc;
        out.push_str(&formatdoc! {"
            /// Maps a token type name to its required target features.
            ///
            /// Generated from token-registry.toml. One complete feature list per token.
            pub(crate) fn token_to_features(token_name: &str) -> Option<&'static [&'static str]> {{
                match token_name {{
        "});

        for token in &self.token {
            let pattern = Self::match_pattern(token);

            // All features including sse/sse2 — needed for #[target_feature] on X64V1Token
            let macro_features: Vec<&str> = token.features.iter().map(|s| s.as_str()).collect();

            out.push_str(&Self::format_feature_arm(&pattern, &macro_features));
        }

        // ScalarToken — always available, no target features. Enables
        // `#[rite(scalar)]` (tokenful, `token: ScalarToken`) and
        // `#[rite(default)]` (tokenless) as fallback tiers that slot into
        // incant!'s suffix convention alongside target-feature-bearing tiers.
        out.push_str("        \"ScalarToken\" => Some(&[]),\n");

        out.push_str("        _ => None,\n");
        out.push_str("    }\n}\n");
    }

    /// Keep the lookup keyed by token name, like the other registry lookups.
    /// Matching whole feature slices adds unnecessary comparisons and compiler work.
    fn gen_feature_csv(&self, out: &mut String) {
        out.push_str(
            "/// Precomputed target-feature CSV for concrete tokens, including aliases.\n",
        );
        // Keep the generated CSV table stable across rustfmt versions. The
        // feature strings are indivisible; wrapping match arms adds no clarity.
        out.push_str("#[rustfmt::skip]\n");
        out.push_str("pub(crate) fn token_to_features_csv(name: &str) -> Option<&'static str> {\n    match name {\n");
        for token in &self.token {
            out.push_str(&format!(
                "        {} => Some({:?}),\n",
                Self::match_pattern(token),
                token.features.join(",")
            ));
        }
        out.push_str("        \"ScalarToken\" => Some(\"\"),\n        _ => None,\n    }\n}\n");
    }

    fn gen_trait_to_features(&self, out: &mut String) {
        use indoc::formatdoc;
        out.push_str(&formatdoc! {"
            /// Maps a trait bound name to its required target features.
            ///
            /// Generated from token-registry.toml. Includes token type names
            /// so `impl TokenType` patterns work in the macro.
            pub(crate) fn trait_to_features(trait_name: &str) -> Option<&'static [&'static str]> {{
                match trait_name {{
        "});

        // Traits first — do NOT strip sse/sse2, these are used for #[target_feature]
        // in codegen where the baseline is needed for generic bounds
        for trait_def in &self.traits {
            let features: Vec<&str> = if !trait_def.x86_features.is_empty() {
                trait_def.x86_features.iter().map(|s| s.as_str()).collect()
            } else {
                trait_def.features.iter().map(|s| s.as_str()).collect()
            };

            let pattern = format!("\"{}\"", trait_def.name);
            out.push_str(&Self::format_feature_arm(&pattern, &features));
        }

        out.push('\n');
        out.push_str("        // Token types used as bounds — full feature sets\n");

        // Token types as bounds — full feature lists
        for token in &self.token {
            let pattern = Self::match_pattern(token);
            let features: Vec<&str> = token.features.iter().map(|s| s.as_str()).collect();
            out.push_str(&Self::format_feature_arm(&pattern, &features));
        }

        out.push('\n');
        out.push_str("        _ => None,\n");
        out.push_str("    }\n}\n");
    }

    fn gen_token_to_arch(&self, out: &mut String) {
        use indoc::formatdoc;
        out.push_str(&formatdoc! {"
            /// Maps a token type name to its target architecture.
            ///
            /// Returns the `target_arch` value (e.g., \"x86_64\", \"aarch64\", \"wasm32\").
            pub(crate) fn token_to_arch(token_name: &str) -> Option<&'static str> {{
                match token_name {{
        "});

        for token in &self.token {
            let pattern = Self::match_pattern(token);
            let target_arch = match token.arch.as_str() {
                "x86" => "x86_64",
                "aarch64" => "aarch64",
                "wasm" => "wasm32",
                other => other,
            };
            out.push_str(&format!("        {pattern} => Some(\"{target_arch}\"),\n"));
        }

        out.push_str("        _ => None,\n");
        out.push_str("    }\n}\n");
    }

    fn gen_token_to_magetypes_namespace(&self, out: &mut String) {
        use indoc::formatdoc;
        out.push_str(&formatdoc! {"
            /// Maps a token type name to its magetypes width namespace.
            ///
            /// Returns the namespace name (e.g., \"v3\", \"v4\", \"neon\", \"wasm128\", \"scalar\").
            /// Used by `import_magetypes` to inject `use magetypes::simd::{{ns}}::*;`.
            pub(crate) fn token_to_magetypes_namespace(token_name: &str) -> Option<&'static str> {{
                match token_name {{
        "});

        for token in &self.token {
            if let Some(ns) = &token.magetypes_namespace {
                let pattern = Self::match_pattern(token);
                out.push_str(&format!("        {pattern} => Some(\"{ns}\"),\n"));
            }
        }

        // ScalarToken — always-available fallback namespace.
        out.push_str("        \"ScalarToken\" => Some(\"scalar\"),\n");

        out.push_str("        _ => None,\n");
        out.push_str("    }\n}\n");
    }

    fn gen_trait_to_magetypes_namespace(&self, out: &mut String) {
        use indoc::formatdoc;
        out.push_str(&formatdoc! {"
            /// Maps a trait bound name to its magetypes width namespace.
            ///
            /// Returns the namespace name (e.g., \"v3\", \"v4\", \"neon\").
            /// Used by `import_magetypes` when a trait bound is used instead of a concrete token.
            pub(crate) fn trait_to_magetypes_namespace(trait_name: &str) -> Option<&'static str> {{
                match trait_name {{
        "});

        // Traits
        for trait_def in &self.traits {
            if let Some(ns) = &trait_def.magetypes_namespace {
                out.push_str(&format!(
                    "        \"{}\" => Some(\"{ns}\"),\n",
                    trait_def.name
                ));
            }
        }

        out.push('\n');
        out.push_str("        // Token types used as bounds\n");

        // Token types used as bounds (same as trait_to_features pattern)
        for token in &self.token {
            if let Some(ns) = &token.magetypes_namespace {
                let pattern = Self::match_pattern(token);
                out.push_str(&format!("        {pattern} => Some(\"{ns}\"),\n"));
            }
        }

        out.push('\n');
        out.push_str("        _ => None,\n");
        out.push_str("    }\n}\n");
    }

    fn gen_trait_to_arch(&self, out: &mut String) {
        use indoc::formatdoc;
        out.push_str(&formatdoc! {"
            /// Maps a trait bound name to its target architecture.
            ///
            /// Returns the architecture (e.g., \"x86_64\", \"aarch64\").
            /// Used by `import_intrinsics` when a trait bound is used instead of a concrete token.
            pub(crate) fn trait_to_arch(trait_name: &str) -> Option<&'static str> {{
                match trait_name {{
        "});

        // Traits
        for trait_def in &self.traits {
            if let Some(arch) = &trait_def.arch {
                out.push_str(&format!(
                    "        \"{}\" => Some(\"{arch}\"),\n",
                    trait_def.name
                ));
            }
        }

        out.push('\n');
        out.push_str("        // Token types used as bounds\n");

        // Token types used as bounds (same as other trait_to_* functions)
        for token in &self.token {
            let pattern = Self::match_pattern(token);
            let target_arch = match token.arch.as_str() {
                "x86" => "x86_64",
                "aarch64" => "aarch64",
                "wasm" => "wasm32",
                other => other,
            };
            out.push_str(&format!("        {pattern} => Some(\"{target_arch}\"),\n"));
        }

        out.push('\n');
        out.push_str("        _ => None,\n");
        out.push_str("    }\n}\n");
    }

    fn gen_tier_to_canonical_token(&self, out: &mut String) {
        use indoc::formatdoc;
        out.push_str(&formatdoc! {"
            /// Maps a tier short name to its canonical token type name.
            ///
            /// Used by `#[rite(v3)]` to resolve the tier to a token without
            /// requiring a token parameter in the function signature.
            ///
            /// Accepts `_v3` as well as `v3` — the leading `_` matches name-mangling suffixes.
            pub(crate) fn tier_to_canonical_token(tier_name: &str) -> Option<&'static str> {{
                let tier_name = tier_name.strip_prefix('_').unwrap_or(tier_name);
                match tier_name {{
        "});

        for token in &self.token {
            if let Some(short) = &token.short_name {
                out.push_str(&format!(
                    "        \"{short}\" => Some(\"{}\"),\n",
                    token.name
                ));
                // Also add extraction_aliases (e.g., "avx512" for v4)
                for alias in &token.extraction_aliases {
                    out.push_str(&format!(
                        "        \"{alias}\" => Some(\"{}\"),\n",
                        token.name
                    ));
                }
            }
        }

        // ScalarToken — tierless fallback. Enables `#[rite(scalar)]` and
        // incant!'s scalar routing to share the suffix convention.
        out.push_str("        \"scalar\" => Some(\"ScalarToken\"),\n");

        out.push_str("        _ => None,\n");
        out.push_str("    }\n}\n");
    }

    fn gen_canonical_token_to_tier_suffix(&self, out: &mut String) {
        use indoc::formatdoc;
        out.push_str(&formatdoc! {"
            /// Maps a canonical token type name to its tier suffix.
            ///
            /// Used by `#[rite(v3, v4, neon)]` to generate suffixed function names
            /// (e.g., `fn_v3`, `fn_v4`, `fn_neon`).
            pub(crate) fn canonical_token_to_tier_suffix(token_name: &str) -> Option<&'static str> {{
                match token_name {{
        "});

        for token in &self.token {
            if let Some(short) = &token.short_name {
                let pattern = Self::match_pattern(token);
                out.push_str(&format!("        {pattern} => Some(\"{short}\"),\n"));
            }
        }

        // ScalarToken — tierless fallback.
        out.push_str("        \"ScalarToken\" => Some(\"scalar\"),\n");

        out.push_str("        _ => None,\n");
        out.push_str("    }\n}\n");
    }

    /// Generate `can_downgrade_tier(from_suffix, to_suffix) -> bool`.
    ///
    /// Derived from feature set math: `from` can downgrade to `to` when
    /// `from.features ⊇ to.features` (strict superset). Computed at codegen
    /// time from the actual feature lists — not from the parent DAG.
    fn gen_can_downgrade_tier(&self, out: &mut String) {
        use indoc::formatdoc;
        out.push_str(&formatdoc! {"
            /// Check if tier `from_suffix` can downgrade to tier `to_suffix`.
            ///
            /// Derived from feature set math: true when `from.features ⊃ to.features`.
            /// Identity (from == to) returns false (use direct pass, no method needed).
            pub(crate) fn can_downgrade_tier(from_suffix: &str, to_suffix: &str) -> bool {{
                if from_suffix == to_suffix {{ return false; }}
                matches!((from_suffix, to_suffix),
        "});

        // Feature subset computation: from can downgrade to to when
        // from.features ⊇ to.features AND same architecture
        for from_token in &self.token {
            let from_suffix = match from_token.short_name.as_deref() {
                Some(s) => s,
                None => continue,
            };
            let from_features: std::collections::BTreeSet<&str> =
                from_token.features.iter().map(|s| s.as_str()).collect();

            let mut downgradable = Vec::new();
            for to_token in &self.token {
                if from_token.name == to_token.name {
                    continue;
                }
                let to_suffix = match to_token.short_name.as_deref() {
                    Some(s) => s,
                    None => continue,
                };
                if from_token.arch != to_token.arch {
                    continue;
                }
                let to_features: std::collections::BTreeSet<&str> =
                    to_token.features.iter().map(|s| s.as_str()).collect();

                // Strict superset: from ⊇ to AND from ≠ to
                if to_features.is_subset(&from_features) && to_features != from_features {
                    downgradable.push(to_suffix);
                }
            }

            if !downgradable.is_empty() {
                downgradable.sort();
                let pattern: Vec<String> =
                    downgradable.iter().map(|s| format!("\"{s}\"")).collect();
                let pattern = pattern.join(" | ");
                out.push_str(&format!("        (\"{from_suffix}\", {pattern}) |\n"));
            }
        }

        // Remove trailing " |\n" and close the matches! macro
        let trimmed = out.trim_end_matches(" |\n").len();
        out.truncate(trimmed);
        out.push_str("\n    )\n}\n");
    }

    /// Generate `expected_tier_tag()` — maps token names (including aliases) to
    /// their FNV-1a tier tag constants.
    ///
    /// Used by `#[arcane]` to emit compile-time `const` assertions that verify
    /// a concrete token type is genuinely the expected archmage type.
    fn gen_expected_tier_tag(&self, out: &mut String, major_version: u32) {
        use indoc::formatdoc;
        out.push_str(&formatdoc! {"
            /// Returns the expected tier tag for a concrete token type name.
            ///
            /// Used by `#[arcane]` to emit compile-time assertions.
            /// Generated from token-registry.toml.
            pub(crate) fn expected_tier_tag(token_name: &str) -> Option<u32> {{
                match token_name {{
        "});

        // ScalarToken first
        let scalar_tag = tier_tag("ScalarToken", major_version);
        out.push_str(&format!(
            "        \"ScalarToken\" => Some(0x{scalar_tag:08X}),\n"
        ));

        for token in &self.token {
            let tag = tier_tag(&token.name, major_version);
            let pattern = Self::match_pattern(token);
            out.push_str(&format!("        {pattern} => Some(0x{tag:08X}),\n"));
        }

        out.push_str("        _ => None,\n");
        out.push_str("    }\n}\n");
    }

    fn gen_all_concrete_tokens(&self, out: &mut String) {
        out.push_str("/// All concrete token names that exist in the runtime crate.\n");
        out.push_str("#[cfg(test)]\n");
        out.push_str("pub(crate) const ALL_CONCRETE_TOKENS: &[&str] = &[\n");
        for token in &self.token {
            out.push_str(&format!("    \"{}\",\n", token.name));
            for a in &token.aliases {
                out.push_str(&format!("    \"{a}\",\n"));
            }
        }
        out.push_str("];\n");
    }

    fn gen_all_trait_names(&self, out: &mut String) {
        out.push_str("/// All trait names that exist in the runtime crate.\n");
        out.push_str("#[cfg(test)]\n");
        out.push_str("pub(crate) const ALL_TRAIT_NAMES: &[&str] = &[\n");
        for trait_def in &self.traits {
            out.push_str(&format!("    \"{}\",\n", trait_def.name));
        }
        out.push_str("];\n");
    }

    fn gen_token_requires_avx512(&self, out: &mut String) {
        out.push_str(concat!(
            "/// Returns true if this token's features include any AVX-512 features.\n",
            "///\n",
            "/// Used by `#[arcane]`/`#[rite]` to error when `import_intrinsics` is used\n",
            "/// with a token that needs 512-bit safe memory ops but the `avx512` feature\n",
            "/// is not enabled on archmage.\n",
            "///\n",
            "/// Generated from token-registry.toml.\n",
            "#[cfg_attr(feature = \"avx512\", allow(dead_code))]\n",
            "pub(crate) fn token_requires_avx512(token_name: &str) -> bool {\n",
        ));

        // Collect all avx512 token patterns into a single matches!() call
        let mut patterns = Vec::new();
        for token in &self.token {
            let has_avx512 = token.features.iter().any(|f| f.starts_with("avx512"));
            if has_avx512 {
                patterns.push(Self::match_pattern(token));
            }
        }
        let all_patterns = patterns.join(" | ");
        out.push_str(&format!("    matches!(token_name, {all_patterns})\n"));
        out.push_str("}\n");
    }

    /// Build a match pattern like `"Name" | "Alias1" | "Alias2"` for a token.
    fn match_pattern(token: &TokenDef) -> String {
        let mut names: Vec<&str> = vec![&token.name];
        for a in &token.aliases {
            names.push(a);
        }
        names
            .iter()
            .map(|n| format!("\"{n}\""))
            .collect::<Vec<_>>()
            .join(" | ")
    }

    /// Format a match arm for a feature list (short inline or multi-line).
    fn format_feature_arm(pattern: &str, features: &[&str]) -> String {
        if features.len() <= 5 {
            let features_str: String = features
                .iter()
                .map(|f| format!("\"{f}\""))
                .collect::<Vec<_>>()
                .join(", ");
            format!("        {pattern} => Some(&[{features_str}]),\n")
        } else {
            let mut s = format!("        {pattern} => Some(&[\n");
            for f in features {
                s.push_str(&format!("            \"{f}\",\n"));
            }
            s.push_str("        ]),\n");
            s
        }
    }
}
