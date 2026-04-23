//! Token registry — loads and validates `token-registry.toml`.
//!
//! This is the single source of truth for all token definitions, feature
//! sets, traits, width namespaces, magetypes file mappings, and polyfill
//! configurations.

use anyhow::{Context, Result, bail};
use serde::Deserialize;
use std::collections::{BTreeSet, HashSet};
use std::path::Path;

// ============================================================================
// Serde Structs
// ============================================================================

/// Top-level registry file.
#[derive(Debug, Deserialize)]
pub struct Registry {
    pub token: Vec<TokenDef>,
    #[serde(rename = "trait")]
    pub traits: Vec<TraitDef>,
    pub width_namespace: Vec<WidthNamespace>,
    pub magetypes_file: Vec<MagetypesFile>,
    #[serde(default)]
    pub polyfill_w256: Vec<PolyfillW256>,
    #[serde(default)]
    pub polyfill_w512: Vec<PolyfillW512>,
}

/// A token definition.
#[derive(Debug, Deserialize)]
pub struct TokenDef {
    pub name: String,
    pub arch: String,
    #[serde(default)]
    pub aliases: Vec<String>,
    /// Aliases that are deprecated. Map of alias name → deprecation message.
    /// These still generate type aliases but with `#[deprecated]`.
    #[serde(default)]
    pub deprecated_aliases: std::collections::HashMap<String, String>,
    pub features: Vec<String>,
    pub traits: Vec<String>,
    #[serde(default)]
    pub cargo_feature: Option<String>,
    #[serde(default)]
    #[allow(dead_code)]
    pub always_available: bool,
    /// SimdToken::NAME const value (human-readable).
    #[serde(default)]
    pub display_name: Option<String>,
    /// Extraction method name (e.g., "v2", "v3", "neon").
    #[serde(default)]
    pub short_name: Option<String>,
    /// Parent tokens in the hierarchy (for extraction method chain, DAG).
    #[serde(default)]
    pub parents: Vec<String>,
    /// Extra extraction method names (e.g., ["avx512"] for X64V4Token).
    #[serde(default)]
    pub extraction_aliases: Vec<String>,
    /// Doc comment for the struct.
    #[serde(default)]
    pub doc: Option<String>,
    /// Magetypes width namespace for this token (e.g., "v3", "v4", "neon").
    /// Used by `import_magetypes` parameter in `#[arcane]`/`#[rite]`.
    #[serde(default)]
    pub magetypes_namespace: Option<String>,
}

/// A trait definition.
#[derive(Debug, Deserialize)]
pub struct TraitDef {
    pub name: String,
    /// Target architecture (e.g., "x86_64", "aarch64").
    /// Used by `import_intrinsics` to determine which `core::arch::` module to import.
    #[serde(default)]
    pub arch: Option<String>,
    /// Features for x86 arch (used when trait is a generic bound in macro).
    #[serde(default)]
    pub x86_features: Vec<String>,
    /// Features for the trait's primary arch (non-x86).
    #[serde(default)]
    pub features: Vec<String>,
    #[serde(default)]
    pub parents: Vec<String>,
    /// Doc comment for the trait.
    #[serde(default)]
    pub doc: Option<String>,
    /// Magetypes width namespace for this trait (e.g., "v3", "neon").
    /// Used by `import_magetypes` parameter in `#[arcane]`/`#[rite]`.
    #[serde(default)]
    pub magetypes_namespace: Option<String>,
    /// Deprecation message. When set, generates `#[deprecated(since = "...", note = "...")]`.
    #[serde(default)]
    pub deprecated: Option<String>,
}

/// A width namespace for simd type re-exports.
#[derive(Debug, Deserialize)]
pub struct WidthNamespace {
    pub name: String,
    #[allow(dead_code)]
    pub arch: String,
    #[allow(dead_code)]
    pub width: u32,
    pub token: String,
    #[serde(default)]
    #[allow(dead_code)]
    pub cargo_feature: Option<String>,
}

/// A magetypes file-to-token validation mapping.
#[derive(Debug, Deserialize)]
pub struct MagetypesFile {
    pub rel_path: String,
    pub token: String,
    pub arch: String,
}

/// A polyfill_w256 platform configuration.
#[derive(Debug, Deserialize)]
pub struct PolyfillW256 {
    pub mod_name: String,
    #[allow(dead_code)]
    pub cfg: String,
    pub token: String,
    #[allow(dead_code)]
    pub w128_import: String,
}

/// A polyfill_w512 platform configuration.
#[derive(Debug, Deserialize)]
pub struct PolyfillW512 {
    pub mod_name: String,
    #[allow(dead_code)]
    pub cfg: String,
    pub token: String,
    #[allow(dead_code)]
    pub w256_import: String,
}

// ============================================================================
// Loading
// ============================================================================

impl Registry {
    /// Load and validate the registry from a TOML file.
    pub fn load(path: &Path) -> Result<Self> {
        let content =
            std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
        let registry: Registry =
            toml::from_str(&content).with_context(|| format!("parsing {}", path.display()))?;
        registry.validate()?;
        Ok(registry)
    }

    /// Look up a token by name (including aliases).
    pub fn find_token(&self, name: &str) -> Option<&TokenDef> {
        self.token
            .iter()
            .find(|t| t.name == name || t.aliases.iter().any(|a| a == name))
    }

    /// Get the feature set for a token or trait by name.
    ///
    /// For tokens, returns the token's features. For traits, returns the
    /// trait's features (preferring x86_features for x86 width traits).
    pub fn features_for(&self, name: &str) -> Option<Vec<&str>> {
        // Try tokens first (including aliases)
        if let Some(token) = self.find_token(name) {
            return Some(token.features.iter().map(|s| s.as_str()).collect());
        }
        // Try traits
        if let Some(trait_def) = self.traits.iter().find(|t| t.name == name) {
            if !trait_def.x86_features.is_empty() {
                return Some(trait_def.x86_features.iter().map(|s| s.as_str()).collect());
            }
            if !trait_def.features.is_empty() {
                return Some(trait_def.features.iter().map(|s| s.as_str()).collect());
            }
        }
        None
    }

    /// All token names including aliases.
    #[allow(dead_code)]
    pub fn all_token_names(&self) -> Vec<&str> {
        let mut names = Vec::new();
        for t in &self.token {
            names.push(t.name.as_str());
            for a in &t.aliases {
                names.push(a.as_str());
            }
        }
        names
    }

    /// All trait names.
    #[allow(dead_code)]
    pub fn all_trait_names(&self) -> Vec<&str> {
        self.traits.iter().map(|t| t.name.as_str()).collect()
    }
}

// ============================================================================
// Validation
// ============================================================================

impl Registry {
    fn validate(&self) -> Result<()> {
        self.validate_no_duplicate_names()?;
        self.validate_trait_references()?;
        self.validate_token_trait_features()?;
        self.validate_width_namespace_tokens()?;
        self.validate_magetypes_file_tokens()?;
        self.validate_polyfill_tokens()?;
        self.validate_token_parents()?;
        Ok(())
    }

    fn validate_no_duplicate_names(&self) -> Result<()> {
        let mut seen = HashSet::new();
        for t in &self.token {
            if !seen.insert(&t.name) {
                bail!("Duplicate token name: {}", t.name);
            }
            for a in &t.aliases {
                if !seen.insert(a) {
                    bail!("Duplicate token alias: {} (on {})", a, t.name);
                }
            }
        }
        let mut trait_seen = HashSet::new();
        for t in &self.traits {
            if !trait_seen.insert(&t.name) {
                bail!("Duplicate trait name: {}", t.name);
            }
        }
        Ok(())
    }

    fn validate_trait_references(&self) -> Result<()> {
        let trait_names: HashSet<&str> = self.traits.iter().map(|t| t.name.as_str()).collect();

        // Tokens reference valid traits
        for token in &self.token {
            for trait_name in &token.traits {
                if !trait_names.contains(trait_name.as_str()) {
                    bail!(
                        "Token {} references unknown trait: {}",
                        token.name,
                        trait_name
                    );
                }
            }
        }

        // Trait parents reference valid traits
        for trait_def in &self.traits {
            for parent in &trait_def.parents {
                if !trait_names.contains(parent.as_str()) {
                    bail!(
                        "Trait {} references unknown parent: {}",
                        trait_def.name,
                        parent
                    );
                }
            }
        }
        Ok(())
    }

    fn validate_token_trait_features(&self) -> Result<()> {
        // For each token, verify its features are a superset of each claimed trait's features
        for token in &self.token {
            let token_features: BTreeSet<&str> =
                token.features.iter().map(|s| s.as_str()).collect();

            for trait_name in &token.traits {
                if let Some(trait_def) = self.traits.iter().find(|t| t.name == *trait_name) {
                    // Determine which features the trait requires
                    let trait_features: Vec<&str> =
                        if token.arch == "x86" && !trait_def.x86_features.is_empty() {
                            trait_def.x86_features.iter().map(|s| s.as_str()).collect()
                        } else if !trait_def.features.is_empty() {
                            trait_def.features.iter().map(|s| s.as_str()).collect()
                        } else {
                            continue; // No features to check
                        };

                    for f in &trait_features {
                        if !token_features.contains(f) {
                            bail!(
                                "Token {} claims trait {} but is missing feature '{}'",
                                token.name,
                                trait_name,
                                f
                            );
                        }
                    }
                }
            }
        }
        Ok(())
    }

    fn validate_width_namespace_tokens(&self) -> Result<()> {
        for ns in &self.width_namespace {
            if self.find_token(&ns.token).is_none() {
                bail!(
                    "Width namespace '{}' references unknown token: {}",
                    ns.name,
                    ns.token
                );
            }
        }
        Ok(())
    }

    fn validate_magetypes_file_tokens(&self) -> Result<()> {
        for mf in &self.magetypes_file {
            if self.find_token(&mf.token).is_none() {
                bail!(
                    "Magetypes file '{}' references unknown token: {}",
                    mf.rel_path,
                    mf.token
                );
            }
        }
        Ok(())
    }

    fn validate_polyfill_tokens(&self) -> Result<()> {
        for p in &self.polyfill_w256 {
            if self.find_token(&p.token).is_none() {
                bail!(
                    "Polyfill w256 '{}' references unknown token: {}",
                    p.mod_name,
                    p.token
                );
            }
        }
        for p in &self.polyfill_w512 {
            if self.find_token(&p.token).is_none() {
                bail!(
                    "Polyfill w512 '{}' references unknown token: {}",
                    p.mod_name,
                    p.token
                );
            }
        }
        Ok(())
    }

    fn validate_token_parents(&self) -> Result<()> {
        let token_names: HashSet<&str> = self.token.iter().map(|t| t.name.as_str()).collect();

        for token in &self.token {
            for parent in &token.parents {
                if !token_names.contains(parent.as_str()) {
                    bail!("Token {} references unknown parent: {}", token.name, parent);
                }
                // Parent must be same arch
                if let Some(parent_def) = self.token.iter().find(|t| t.name == *parent) {
                    if parent_def.arch != token.arch {
                        bail!(
                            "Token {} (arch={}) has parent {} (arch={}) — must be same arch",
                            token.name,
                            token.arch,
                            parent,
                            parent_def.arch
                        );
                    }
                }
            }

            // Cycle detection: BFS from this token through parents must not revisit itself
            let mut visited = HashSet::new();
            let mut queue: std::collections::VecDeque<&str> =
                token.parents.iter().map(|s| s.as_str()).collect();
            while let Some(name) = queue.pop_front() {
                if name == token.name {
                    bail!("Token {} has a cycle in its parent hierarchy", token.name);
                }
                if !visited.insert(name) {
                    continue;
                }
                if let Some(ancestor) = self.token.iter().find(|t| t.name == name) {
                    for gp in &ancestor.parents {
                        queue.push_back(gp.as_str());
                    }
                }
            }
        }
        Ok(())
    }
}

// ============================================================================
// Display
// ============================================================================

impl std::fmt::Display for Registry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Token Registry:")?;
        writeln!(f, "  Tokens: {}", self.token.len())?;
        for t in &self.token {
            let aliases = if t.aliases.is_empty() {
                String::new()
            } else {
                format!(" (aliases: {})", t.aliases.join(", "))
            };
            writeln!(
                f,
                "    {} [{}] — {} features, {} traits{}",
                t.name,
                t.arch,
                t.features.len(),
                t.traits.len(),
                aliases,
            )?;
        }
        writeln!(f, "  Traits: {}", self.traits.len())?;
        for t in &self.traits {
            writeln!(f, "    {}", t.name)?;
        }
        writeln!(f, "  Width namespaces: {}", self.width_namespace.len())?;
        writeln!(f, "  Magetypes files: {}", self.magetypes_file.len())?;
        writeln!(
            f,
            "  Polyfill platforms: {} w256 + {} w512",
            self.polyfill_w256.len(),
            self.polyfill_w512.len()
        )?;
        Ok(())
    }
}

// ============================================================================
// Tier Tags
// ============================================================================

/// FNV-1a hash of token name seeded with major version.
///
/// Produces a unique tag for each token struct name, used for compile-time
/// assertion that a concrete token type is genuinely the expected archmage
/// type (not shadowed or aliased).
pub fn tier_tag(token_name: &str, major_version: u32) -> u32 {
    let mut hash: u32 = 0x811c_9dc5 ^ major_version.wrapping_mul(0x0100_0193);
    for byte in token_name.bytes() {
        hash ^= byte as u32;
        hash = hash.wrapping_mul(0x0100_0193);
    }
    hash
}

// ============================================================================
// Code Generation
// ============================================================================

impl Registry {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_load_registry() {
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .join("token-registry.toml");
        let registry = Registry::load(&path).expect("Failed to load token-registry.toml");

        // Basic counts
        assert_eq!(registry.token.len(), 16, "Expected 16 tokens");
        assert_eq!(registry.traits.len(), 10, "Expected 10 traits");
        assert_eq!(
            registry.width_namespace.len(),
            4,
            "Expected 4 width namespaces"
        );
        assert_eq!(
            registry.magetypes_file.len(),
            5,
            "Expected 5 magetypes file mappings"
        );

        // Spot-check X64V3Token
        let v3 = registry
            .find_token("X64V3Token")
            .expect("X64V3Token not found");
        assert!(v3.features.contains(&"avx2".to_string()));
        assert!(v3.features.contains(&"fma".to_string()));
        assert!(v3.features.contains(&"f16c".to_string()));
        assert!(v3.features.contains(&"lzcnt".to_string()));
        assert_eq!(v3.features.len(), 16); // sse, sse2, + v2 (5+cmpxchg16b) + v3 (avx2, fma, bmi1, bmi2, f16c, lzcnt, movbe, avx)

        // Spot-check aliases
        assert!(registry.find_token("Desktop64").is_some());
        assert!(registry.find_token("Avx2FmaToken").is_some());
        assert!(registry.find_token("Arm64").is_some());
        assert!(registry.find_token("Server64").is_some());

        // Spot-check NeonCrcToken
        let crc = registry
            .find_token("NeonCrcToken")
            .expect("NeonCrcToken not found");
        assert_eq!(crc.features, vec!["neon", "crc"]);
    }

    fn load_test_registry() -> Registry {
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .join("token-registry.toml");
        Registry::load(&path).expect("Failed to load token-registry.toml")
    }

    #[test]
    fn macro_registry_contains_all_tokens() {
        let registry = load_test_registry();
        let output = registry.generate_macro_registry(0);

        // Every token should appear in the generated registry
        for token in &registry.token {
            assert!(
                output.contains(&token.name),
                "Token {} missing from macro registry output",
                token.name
            );
            // Every alias should also appear
            for alias in &token.aliases {
                assert!(
                    output.contains(alias),
                    "Alias {} for {} missing from macro registry",
                    alias,
                    token.name
                );
            }
        }
    }

    #[test]
    fn macro_registry_contains_features() {
        let registry = load_test_registry();
        let output = registry.generate_macro_registry(0);

        // Key features should appear in the output
        for feature in &["avx2", "fma", "neon", "simd128"] {
            assert!(
                output.contains(feature),
                "Feature {feature} missing from macro registry output",
            );
        }
    }

    #[test]
    fn token_hierarchy_is_consistent() {
        let registry = load_test_registry();

        // Every parent must exist
        for token in &registry.token {
            for parent_name in &token.parents {
                assert!(
                    registry.find_token(parent_name).is_some(),
                    "Token {} declares parent {} which doesn't exist",
                    token.name,
                    parent_name
                );
            }
        }

        // Every child's features must be a superset of parent's features
        for token in &registry.token {
            for parent_name in &token.parents {
                if let Some(parent) = registry.find_token(parent_name) {
                    for parent_feature in &parent.features {
                        assert!(
                            token.features.contains(parent_feature),
                            "Token {} missing parent feature {} from {}",
                            token.name,
                            parent_feature,
                            parent_name
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn all_traits_referenced_by_tokens_exist() {
        let registry = load_test_registry();
        let trait_names: Vec<&str> = registry.traits.iter().map(|t| t.name.as_str()).collect();

        for token in &registry.token {
            for trait_name in &token.traits {
                assert!(
                    trait_names.contains(&trait_name.as_str()),
                    "Token {} references trait {} which isn't defined",
                    token.name,
                    trait_name
                );
            }
        }
    }

    #[test]
    fn features_for_returns_correct_data() {
        let registry = load_test_registry();

        // X64V3Token should include inherited features
        let v3_features = registry.features_for("X64V3Token").unwrap();
        assert!(v3_features.contains(&"avx2"));
        assert!(v3_features.contains(&"fma"));
        assert!(v3_features.contains(&"sse2")); // inherited from V1

        // Alias should work too
        let desktop = registry.features_for("Desktop64").unwrap();
        assert_eq!(v3_features, desktop);

        // Nonexistent token
        assert!(registry.features_for("FakeToken").is_none());
    }

    #[test]
    fn arch_grouping_covers_all_tokens() {
        let registry = load_test_registry();
        let valid_arches = ["x86", "aarch64", "wasm"];

        for token in &registry.token {
            if token.name == "ScalarToken" {
                continue; // ScalarToken has arch "any"
            }
            assert!(
                valid_arches.contains(&token.arch.as_str()) || token.arch == "any",
                "Token {} has unexpected arch: {}",
                token.name,
                token.arch
            );
        }
    }
}
