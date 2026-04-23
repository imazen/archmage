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

/// Parse an optional cfg-gate after a tier name.
///
/// Accepts both `tier(cfg(feature))` (canonical) and `tier(feature)` (shorthand).
/// Returns the combined `"tier(feature)"` string, or just `"tier"` if no gate.
pub(crate) fn parse_tier_name_with_gate(ident: &Ident, input: ParseStream) -> syn::Result<String> {
    if input.peek(syn::token::Paren) {
        let paren_content;
        syn::parenthesized!(paren_content in input);
        // Check for cfg(feature) syntax: v4(cfg(avx512))
        let feat_name = if paren_content.peek(Ident) && paren_content.peek2(syn::token::Paren) {
            let kw: Ident = paren_content.parse()?;
            if kw != "cfg" {
                return Err(syn::Error::new(
                    kw.span(),
                    format!("expected `cfg` in tier gate, got `{kw}`"),
                ));
            }
            let inner;
            syn::parenthesized!(inner in paren_content);
            let feat: Ident = inner.parse()?;
            feat.to_string()
        } else {
            // Bare feature name shorthand: v4(avx512)
            let feat: Ident = paren_content.parse()?;
            feat.to_string()
        };
        Ok(format!("{}({})", ident, feat_name))
    } else {
        Ok(ident.to_string())
    }
}

/// Parse a single tier entry: optional `+`/`-` prefix, ident, optional `(cfg(feature))` gate.
///
/// Returns the tier string with prefix preserved:
/// - `"+arm_v2"` — add/override this tier
/// - `"-neon"` — remove this tier from defaults
/// - `"v3"` — plain tier (override mode)
pub(crate) fn parse_one_tier(input: ParseStream) -> syn::Result<String> {
    let prefix = if input.peek(Token![+]) {
        let _: Token![+] = input.parse()?;
        "+"
    } else if input.peek(Token![-]) {
        let _: Token![-] = input.parse()?;
        "-"
    } else {
        ""
    };
    let ident: Ident = input.parse()?;
    let name = parse_tier_name_with_gate(&ident, input)?;
    Ok(format!("{prefix}{name}"))
}

/// Look up a tier by name.
///
/// Accepts `_v3` as well as `v3` — the leading `_` matches the name-mangling
/// suffix (`fn_v3`), so users can write whichever form they find natural.
pub(crate) fn find_tier(name: &str) -> Option<&'static TierDescriptor> {
    let name = name.strip_prefix('_').unwrap_or(name);
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
///
/// Tier names prefixed with `+` trigger **additive mode**: defaults are included
/// first, then the `+` entries are appended. All entries must be `+` or none.
pub(crate) fn resolve_tiers(
    tier_names: &[String],
    error_span: proc_macro2::Span,
    default_feature_gates: bool,
) -> syn::Result<Vec<ResolvedTier>> {
    let any_modifier = tier_names
        .iter()
        .any(|n| n.starts_with('+') || n.starts_with('-'));
    let any_plain = tier_names
        .iter()
        .any(|n| !n.starts_with('+') && !n.starts_with('-'));
    if any_modifier && any_plain {
        return Err(syn::Error::new(
            error_span,
            "Cannot mix `+tier`/`-tier` (modify defaults) with plain `tier` (override).\n\
             Use all `+`/`-` to modify defaults, or none to replace them entirely.",
        ));
    }

    // In additive mode, start with defaults then merge user entries:
    // - Same base name → replace (e.g., +v4 overrides the default v4(avx512) gate)
    // - New base name → append (e.g., +arm_v2 adds a tier)
    //
    // This lets users write [+default] to swap scalar→default, [+v4] to make
    // v4 unconditional, or [+neon(cfg(neon))] to gate a default tier.
    //
    // User-overridden tiers are tracked so that default_feature_gates doesn't
    // re-apply the descriptor's cfg_feature on them. Writing +v4 means "I want
    // v4 exactly as written, without the automatic avx512 gate."
    let mut user_overrides: Vec<String> = Vec::new();
    let effective_names: Vec<String> = if any_modifier {
        let mut names: Vec<String> = DEFAULT_TIER_NAMES.iter().map(|s| s.to_string()).collect();
        for raw in tier_names {
            let is_removal = raw.starts_with('-');
            let stripped = raw
                .strip_prefix('+')
                .or_else(|| raw.strip_prefix('-'))
                .unwrap_or(raw);
            let base = stripped.split('(').next().unwrap_or(stripped);
            // Strip leading _ for matching (find_tier does this too)
            let base = base.strip_prefix('_').unwrap_or(base);

            // `default` and `scalar` are interchangeable fallback slots.
            let is_fallback = base == "default" || base == "scalar";

            let pos = names.iter().position(|n| {
                let n_base = n.split('(').next().unwrap_or(n);
                if is_fallback {
                    n_base == "default" || n_base == "scalar"
                } else {
                    n_base == base
                }
            });

            if is_removal {
                if let Some(pos) = pos {
                    names.remove(pos);
                }
                // Removing a tier that's not in defaults is a silent no-op
            } else if let Some(pos) = pos {
                // Replace existing default with user's version
                names[pos] = stripped.to_string();
                user_overrides.push(base.to_string());
            } else {
                names.push(stripped.to_string());
                user_overrides.push(base.to_string());
            }
        }
        names
    } else {
        tier_names.to_vec()
    };

    let mut tiers = Vec::new();
    for raw_name in &effective_names {
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
                // User-overridden tiers (from + entries) get exactly what the user
                // wrote — no auto-gating from the descriptor. +v4 means unconditional.
                let is_user_override = user_overrides.iter().any(|o| o == tier.name);
                let feature_gate = explicit_gate.or_else(|| {
                    if default_feature_gates && !is_user_override {
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

    tiers.sort_by_key(|rt| core::cmp::Reverse(rt.tier.priority));

    Ok(tiers)
}
