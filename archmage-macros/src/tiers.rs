//! Tier descriptors and resolution for dispatch macros.
//!
//! Shared by `incant!`, `#[autoversion]`, `#[magetypes]`, and `#[rite]` multi-tier.

use syn::{Ident, Token, parse::ParseStream};

/// Describes a dispatch tier for incant! and #[magetypes].
pub(crate) struct TierDescriptor {
    /// Tier name as written in user code (e.g., "v3", "neon")
    pub name: &'static str,
    /// Function suffix (e.g., "v3", "neon", "scalar")
    pub suffix: &'static str,
    /// Token type path (e.g., "archmage::X64V3Token")
    pub token_path: &'static str,
    /// IntoConcreteToken method name (e.g., "as_x64v3")
    pub as_method: &'static str,
    /// Target architecture for cfg guard (None = no guard)
    pub target_arch: Option<&'static str>,
    /// Cargo feature required for this tier's functions to exist.
    pub cfg_feature: Option<&'static str>,
    /// Dispatch priority (higher = tried first within same arch)
    pub priority: u32,
}

/// All known tiers in dispatch-priority order (highest first within arch).
pub(crate) const ALL_TIERS: &[TierDescriptor] = &[
    // x86: highest to lowest
    TierDescriptor {
        name: "v4x",
        suffix: "v4x",
        token_path: "archmage::X64V4xToken",
        as_method: "as_x64v4x",
        target_arch: Some("x86_64"),
        cfg_feature: Some("avx512"),
        priority: 50,
    },
    TierDescriptor {
        name: "v4",
        suffix: "v4",
        token_path: "archmage::X64V4Token",
        as_method: "as_x64v4",
        target_arch: Some("x86_64"),
        cfg_feature: Some("avx512"),
        priority: 40,
    },
    TierDescriptor {
        name: "v3_crypto",
        suffix: "v3_crypto",
        token_path: "archmage::X64V3CryptoToken",
        as_method: "as_x64v3_crypto",
        target_arch: Some("x86_64"),
        cfg_feature: None,
        priority: 35,
    },
    TierDescriptor {
        name: "v3",
        suffix: "v3",
        token_path: "archmage::X64V3Token",
        as_method: "as_x64v3",
        target_arch: Some("x86_64"),
        cfg_feature: None,
        priority: 30,
    },
    TierDescriptor {
        name: "x64_crypto",
        suffix: "x64_crypto",
        token_path: "archmage::X64CryptoToken",
        as_method: "as_x64_crypto",
        target_arch: Some("x86_64"),
        cfg_feature: None,
        priority: 25,
    },
    TierDescriptor {
        name: "v2",
        suffix: "v2",
        token_path: "archmage::X64V2Token",
        as_method: "as_x64v2",
        target_arch: Some("x86_64"),
        cfg_feature: None,
        priority: 20,
    },
    TierDescriptor {
        name: "v1",
        suffix: "v1",
        token_path: "archmage::X64V1Token",
        as_method: "as_x64v1",
        target_arch: Some("x86_64"),
        cfg_feature: None,
        priority: 10,
    },
    // ARM: highest to lowest
    TierDescriptor {
        name: "arm_v3",
        suffix: "arm_v3",
        token_path: "archmage::Arm64V3Token",
        as_method: "as_arm_v3",
        target_arch: Some("aarch64"),
        cfg_feature: None,
        priority: 50,
    },
    TierDescriptor {
        name: "arm_v2",
        suffix: "arm_v2",
        token_path: "archmage::Arm64V2Token",
        as_method: "as_arm_v2",
        target_arch: Some("aarch64"),
        cfg_feature: None,
        priority: 40,
    },
    TierDescriptor {
        name: "neon_aes",
        suffix: "neon_aes",
        token_path: "archmage::NeonAesToken",
        as_method: "as_neon_aes",
        target_arch: Some("aarch64"),
        cfg_feature: None,
        priority: 30,
    },
    TierDescriptor {
        name: "neon_sha3",
        suffix: "neon_sha3",
        token_path: "archmage::NeonSha3Token",
        as_method: "as_neon_sha3",
        target_arch: Some("aarch64"),
        cfg_feature: None,
        priority: 30,
    },
    TierDescriptor {
        name: "neon_crc",
        suffix: "neon_crc",
        token_path: "archmage::NeonCrcToken",
        as_method: "as_neon_crc",
        target_arch: Some("aarch64"),
        cfg_feature: None,
        priority: 30,
    },
    TierDescriptor {
        name: "neon",
        suffix: "neon",
        token_path: "archmage::NeonToken",
        as_method: "as_neon",
        target_arch: Some("aarch64"),
        cfg_feature: None,
        priority: 20,
    },
    // WASM
    TierDescriptor {
        name: "wasm128_relaxed",
        suffix: "wasm128_relaxed",
        token_path: "archmage::Wasm128RelaxedToken",
        as_method: "as_wasm128_relaxed",
        target_arch: Some("wasm32"),
        cfg_feature: None,
        priority: 21,
    },
    TierDescriptor {
        name: "wasm128",
        suffix: "wasm128",
        token_path: "archmage::Wasm128Token",
        as_method: "as_wasm128",
        target_arch: Some("wasm32"),
        cfg_feature: None,
        priority: 20,
    },
    // Scalar (always last — takes ScalarToken)
    TierDescriptor {
        name: "scalar",
        suffix: "scalar",
        token_path: "archmage::ScalarToken",
        as_method: "as_scalar",
        target_arch: None,
        cfg_feature: None,
        priority: 0,
    },
    // Default (always last — tokenless, for incant! nesting with autoversion)
    TierDescriptor {
        name: "default",
        suffix: "default",
        token_path: "", // not used — default is called without a token
        as_method: "",  // not used — default is not dispatched via IntoConcreteToken
        target_arch: None,
        cfg_feature: None,
        priority: 0,
    },
];

/// Default tiers for all dispatch macros.
pub(crate) const DEFAULT_TIER_NAMES: &[&str] = &["v4", "v3", "neon", "wasm128", "scalar"];

/// Parse a comma-separated list of tier names, each optionally followed by
/// `(feature)` for cfg-gating: `v4(avx512), v3, neon(simd), scalar`.
pub(crate) fn parse_tier_names(input: ParseStream) -> syn::Result<Vec<String>> {
    let mut names = Vec::new();
    while !input.is_empty() {
        let ident: Ident = input.parse()?;
        let name = if input.peek(syn::token::Paren) {
            let paren_content;
            syn::parenthesized!(paren_content in input);
            let feat: Ident = paren_content.parse()?;
            format!("{}({})", ident, feat)
        } else {
            ident.to_string()
        };
        names.push(name);
        if input.peek(Token![,]) {
            let _: Token![,] = input.parse()?;
        }
    }
    Ok(names)
}

/// Look up a tier by name.
pub(crate) fn find_tier(name: &str) -> Option<&'static TierDescriptor> {
    ALL_TIERS.iter().find(|t| t.name == name)
}

/// A resolved tier with optional feature gate.
#[derive(Clone)]
pub(crate) struct ResolvedTier {
    pub tier: &'static TierDescriptor,
    /// When Some, dispatch/generation is wrapped in `#[cfg(feature = "...")]`
    pub feature_gate: Option<String>,
    /// When true, `#[allow(unexpected_cfgs)]` is added before the `#[cfg]`.
    pub allow_unexpected_cfg: bool,
}

impl core::ops::Deref for ResolvedTier {
    type Target = TierDescriptor;
    fn deref(&self) -> &TierDescriptor {
        self.tier
    }
}

/// Resolve tier names to descriptors, sorted by dispatch priority (highest first).
///
/// When `default_feature_gates` is true, tiers with `cfg_feature` in their
/// descriptor automatically get that as their feature gate.
pub(crate) fn resolve_tiers(
    tier_names: &[String],
    error_span: proc_macro2::Span,
    default_feature_gates: bool,
) -> syn::Result<Vec<ResolvedTier>> {
    let mut tiers = Vec::new();
    for raw_name in tier_names {
        let (name, explicit_gate) = if let Some(paren_pos) = raw_name.find('(') {
            let tier_name = &raw_name[..paren_pos];
            let feat = raw_name[paren_pos + 1..].trim_end_matches(')');
            (tier_name, Some(feat.to_string()))
        } else {
            (raw_name.as_str(), None)
        };
        match find_tier(name) {
            Some(tier) => {
                let is_explicit = explicit_gate.is_some();
                let feature_gate = explicit_gate.or_else(|| {
                    if default_feature_gates {
                        tier.cfg_feature.map(String::from)
                    } else {
                        None
                    }
                });
                tiers.push(ResolvedTier {
                    tier,
                    allow_unexpected_cfg: feature_gate.is_some() && !is_explicit,
                    feature_gate,
                });
            }
            None => {
                let known: Vec<&str> = ALL_TIERS.iter().map(|t| t.name).collect();
                return Err(syn::Error::new(
                    error_span,
                    format!("unknown tier `{}`. Known tiers: {}", name, known.join(", ")),
                ));
            }
        }
    }

    let has_scalar = tiers.iter().any(|rt| rt.tier.name == "scalar");
    let has_default = tiers.iter().any(|rt| rt.tier.name == "default");
    if has_scalar && has_default {
        return Err(syn::Error::new(
            error_span,
            "`scalar` and `default` are mutually exclusive fallback tiers. \
             Use `scalar` (takes ScalarToken) or `default` (tokenless).",
        ));
    }

    if !has_scalar && !has_default {
        tiers.push(ResolvedTier {
            tier: find_tier("scalar").unwrap(),
            feature_gate: None,
            allow_unexpected_cfg: false,
        });
    }

    tiers.sort_by(|a, b| b.tier.priority.cmp(&a.tier.priority));

    Ok(tiers)
}
