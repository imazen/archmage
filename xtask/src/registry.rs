//! Token registry — loads and validates `token-registry.toml`.
//!
//! This is the single source of truth for all token definitions, feature
//! sets, traits, width namespaces, magetypes file mappings, and polyfill
//! configurations.

use anyhow::{Context, Result, bail};
use serde::Deserialize;
use std::collections::{BTreeSet, HashMap, HashSet};
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
    pub features: Vec<String>,
    pub traits: Vec<String>,
    #[serde(default)]
    pub cargo_feature: Option<String>,
    #[serde(default)]
    pub always_available: bool,
    /// SimdToken::NAME const value (human-readable).
    #[serde(default)]
    pub display_name: Option<String>,
    /// Extraction method name (e.g., "v2", "v3", "neon").
    #[serde(default)]
    pub short_name: Option<String>,
    /// Parent token in the hierarchy (for extraction method chain).
    #[serde(default)]
    pub parent: Option<String>,
    /// Extra extraction method names (e.g., ["avx512"] for X64V4Token).
    #[serde(default)]
    pub extraction_aliases: Vec<String>,
    /// Doc comment for the struct.
    #[serde(default)]
    pub doc: Option<String>,
}

/// A trait definition.
#[derive(Debug, Deserialize)]
pub struct TraitDef {
    pub name: String,
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
}

/// A width namespace for simd type re-exports.
#[derive(Debug, Deserialize)]
pub struct WidthNamespace {
    pub name: String,
    pub arch: String,
    pub width: u32,
    pub token: String,
    #[serde(default)]
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
    pub cfg: String,
    pub token: String,
    pub w128_import: String,
}

/// A polyfill_w512 platform configuration.
#[derive(Debug, Deserialize)]
pub struct PolyfillW512 {
    pub mod_name: String,
    pub cfg: String,
    pub token: String,
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
            if let Some(parent) = &token.parent {
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
    pub fn generate_macro_registry(&self) -> String {
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
            let macro_features: Vec<&str> =
                token.features.iter().map(|s| s.as_str()).collect();

            out.push_str(&Self::format_feature_arm(&pattern, &macro_features));
        }

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
        assert_eq!(registry.token.len(), 11, "Expected 11 tokens");
        assert_eq!(registry.traits.len(), 8, "Expected 8 traits");
        assert_eq!(
            registry.width_namespace.len(),
            5,
            "Expected 5 width namespaces"
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
        assert_eq!(v3.features.len(), 14); // sse, sse2 + 12 v3 features

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
}
