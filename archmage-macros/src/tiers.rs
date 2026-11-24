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
pub(crate) use crate::generated::DISPATCH_TIERS as ALL_TIERS;

/// Default tiers for all dispatch macros.
pub(crate) const DEFAULT_TIER_NAMES: &[&str] = &["v4", "v3", "neon", "wasm128", "scalar"];

/// Modifier prefix on a legacy tier entry (`+tier`, `-tier`, or plain `tier`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum TierModifier {
    Plain,
    Add,
    Remove,
}

/// A parsed legacy tier entry with modifier, identifier name, and optional
/// explicit `cfg(feature)` gate kept structured (no stringify/re-split).
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct ParsedTier {
    pub modifier: TierModifier,
    pub name: String,
    pub feature_gate: Option<String>,
}

impl ParsedTier {
    pub(crate) fn base_name(&self) -> &str {
        self.name.strip_prefix('_').unwrap_or(&self.name)
    }

    pub(crate) fn is_fallback(&self) -> bool {
        matches!(self.base_name(), "default" | "scalar")
    }

    /// Parse a single tier spec string (e.g. `"+v4(avx512)"`, `"-neon"`) for unit tests.
    #[cfg(test)]
    pub(crate) fn parse_test_spec(raw: &str) -> Self {
        let (modifier, rest) = if let Some(s) = raw.strip_prefix('+') {
            (TierModifier::Add, s)
        } else if let Some(s) = raw.strip_prefix('-') {
            (TierModifier::Remove, s)
        } else {
            (TierModifier::Plain, raw)
        };
        let (name, feature_gate) = if let Some(paren_pos) = rest.find('(') {
            let feat = rest[paren_pos + 1..].trim_end_matches(')');
            (&rest[..paren_pos], Some(feat.to_string()))
        } else {
            (rest, None)
        };
        Self {
            modifier,
            name: name.to_string(),
            feature_gate,
        }
    }
}

/// Parse an optional cfg-gate after a tier name.
///
/// Accepts both `tier(cfg(feature))` (canonical) and `tier(feature)` (shorthand).
pub(crate) fn parse_tier_name_with_gate(
    ident: &Ident,
    input: ParseStream<'_>,
) -> syn::Result<ParsedTier> {
    let feature_gate = if input.peek(syn::token::Paren) {
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
        Some(feat_name)
    } else {
        None
    };
    Ok(ParsedTier {
        modifier: TierModifier::Plain,
        name: ident.to_string(),
        feature_gate,
    })
}

/// Parse a single tier entry: optional `+`/`-` prefix, ident, optional `(cfg(feature))` gate.
pub(crate) fn parse_one_tier(input: ParseStream<'_>) -> syn::Result<ParsedTier> {
    let modifier = if input.peek(Token![+]) {
        let _: Token![+] = input.parse()?;
        TierModifier::Add
    } else if input.peek(Token![-]) {
        let _: Token![-] = input.parse()?;
        TierModifier::Remove
    } else {
        TierModifier::Plain
    };
    let ident: Ident = input.parse()?;
    let mut entry = parse_tier_name_with_gate(&ident, input)?;
    entry.modifier = modifier;
    Ok(entry)
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

impl ResolvedTier {
    /// The fallback tiers take no `summon()` check: `scalar` is always
    /// available and `default` is tokenless.
    pub(crate) fn is_fallback(&self) -> bool {
        self.name == "scalar" || self.name == "default"
    }

    /// The `#[cfg]` attributes for a generated variant of this tier: its
    /// architecture, then its feature gate (preceded by
    /// `#[allow(unexpected_cfgs)]` when the gate is an implied default).
    pub(crate) fn variant_cfg_guard(&self) -> proc_macro2::TokenStream {
        let arch = crate::common::gen_cfg_guard(self.target_arch, None);
        let gate = self.feature_gate.as_ref().map(|feat| {
            let allow_attr = if self.allow_unexpected_cfg {
                quote::quote! { #[allow(unexpected_cfgs)] }
            } else {
                quote::quote! {}
            };
            quote::quote! {
                #allow_attr
                #[cfg(feature = #feat)]
            }
        });
        quote::quote! { #arch #gate }
    }
}

/// Build the dispatch arms for the non-fallback tiers. `check(tier)` produces
/// one tier's test-and-call; each is wrapped in its `#[cfg(feature)]` gate,
/// and the tiers of one architecture share a `#[cfg(target_arch)]` block, in
/// list order. `incant!`'s two forms and `#[autoversion]` differ only in the
/// check, so they share this.
pub(crate) fn gen_dispatch_arms(
    tiers: &[ResolvedTier],
    mut check: impl FnMut(&ResolvedTier) -> proc_macro2::TokenStream,
) -> Vec<proc_macro2::TokenStream> {
    let mut arch_groups: Vec<(Option<&str>, Vec<&ResolvedTier>)> = Vec::new();
    for rt in tiers.iter().filter(|rt| !rt.is_fallback()) {
        if let Some(group) = arch_groups.iter_mut().find(|(a, _)| *a == rt.target_arch) {
            group.1.push(rt);
        } else {
            arch_groups.push((rt.target_arch, vec![rt]));
        }
    }

    let mut dispatch_arms = Vec::new();
    for (target_arch, group_tiers) in &arch_groups {
        let mut tier_checks = Vec::new();
        for rt in group_tiers {
            let check = check(rt);
            if let Some(feat) = &rt.feature_gate {
                let allow_attr = if rt.allow_unexpected_cfg {
                    quote::quote! { #[allow(unexpected_cfgs)] }
                } else {
                    quote::quote! {}
                };
                tier_checks.push(quote::quote! {
                    #allow_attr
                    #[cfg(feature = #feat)]
                    { #check }
                });
            } else {
                tier_checks.push(check);
            }
        }

        let inner = quote::quote! { #(#tier_checks)* };
        if target_arch.is_some() {
            let guard = crate::common::gen_cfg_guard(*target_arch, None);
            dispatch_arms.push(quote::quote! {
                #guard
                { #inner }
            });
        } else {
            dispatch_arms.push(inner);
        }
    }
    dispatch_arms
}

/// The fixed defaults are already in stable descending priority order (tested
/// below). No modifier processing, name copies, or sorting is needed here.
pub(crate) fn default_tiers(default_feature_gates: bool) -> Vec<ResolvedTier> {
    DEFAULT_TIER_NAMES
        .iter()
        .map(|name| {
            let tier = find_tier(name).expect("registered default tier");
            let feature_gate = if default_feature_gates {
                tier.cfg_feature.map(String::from)
            } else {
                None
            };
            ResolvedTier {
                tier,
                allow_unexpected_cfg: feature_gate.is_some(),
                feature_gate,
            }
        })
        .collect()
}

/// Resolve tier names to descriptors, sorted by dispatch priority (highest first).
///
/// When `default_feature_gates` is true, tiers with `cfg_feature` in their
/// descriptor automatically get that as their feature gate.
///
/// Tier names prefixed with `+` trigger **additive mode**: defaults are included
/// first, then the `+` entries are appended. Plain names may be mixed in: with
/// any `+` present they are treated as `+name`; `-` alone with plain names
/// keeps override mode and removes from that set (issue #48).
pub(crate) fn resolve_tiers(
    entries: &[ParsedTier],
    error_span: proc_macro2::Span,
    default_feature_gates: bool,
) -> syn::Result<Vec<ResolvedTier>> {
    let any_plus = entries.iter().any(|e| e.modifier == TierModifier::Add);
    let any_minus = entries.iter().any(|e| e.modifier == TierModifier::Remove);
    let any_plain = entries.iter().any(|e| e.modifier == TierModifier::Plain);

    // Mode selection (issue #48 — plain and `+`/`-` may now be mixed):
    //
    // * **Additive** — start from `DEFAULT_TIER_NAMES` and adjust. Triggered by
    //   any `+` entry, OR by `-` removals when there are no plain tiers to form
    //   an override set (e.g. `[-neon, -wasm128]` = "defaults minus those").
    //   A plain `tier` here is treated as `+tier` (ensure present / override
    //   the default's gate), so `[+v4, v3]` is additive.
    // * **Override** — the plain tiers replace the defaults verbatim. `-tier`
    //   removes a tier from that set if present; `-scalar` / `-default`
    //   additionally suppress the auto-appended fallback. This is the
    //   `(v3, -scalar)` → `[v3]` case: list what you want, drop the fallback.
    //
    // No-modifier all-plain lists stay in override mode exactly as before.
    let additive_mode = any_plus || (any_minus && !any_plain);

    // Tracks whether the user explicitly removed the scalar/default fallback
    // via `-scalar` or `-default`. Used to suppress the auto-append below so
    // that `[-scalar]` (or `[v3, -scalar]`) actually drops the scalar variant.
    let mut fallback_explicitly_removed = false;

    // Effective tier entry after applying `+`/`-` adjustments.
    // `user_override` is true for additive user additions/replacements: writing
    // `+v4` means "I want v4 exactly as written, without the descriptor's
    // automatic `avx512` gate."
    struct EffectiveTier<'a> {
        name: &'a str,
        explicit_gate: Option<&'a str>,
        user_override: bool,
    }
    let matches_slot = |slot: &EffectiveTier<'_>, base: &str, is_fallback: bool| {
        let slot_base = slot.name.strip_prefix('_').unwrap_or(slot.name);
        if is_fallback {
            slot_base == "default" || slot_base == "scalar"
        } else {
            slot_base == base
        }
    };

    let effective: Vec<EffectiveTier<'_>> = if additive_mode {
        // In additive mode, start with defaults then merge user entries:
        // - Same base name → replace (e.g., +v4 overrides the default v4(avx512) gate)
        // - New base name → append (e.g., +arm_v2 adds a tier)
        //
        // This lets users write [+default] to swap scalar→default, [+v4] to make
        // v4 unconditional, or [+neon(cfg(neon))] to gate a default tier.
        let mut list: Vec<EffectiveTier<'_>> = DEFAULT_TIER_NAMES
            .iter()
            .map(|&name| EffectiveTier {
                name,
                explicit_gate: None,
                user_override: false,
            })
            .collect();
        for entry in entries {
            let base = entry.base_name();
            let is_fallback = entry.is_fallback();
            let pos = list
                .iter()
                .position(|slot| matches_slot(slot, base, is_fallback));

            if entry.modifier == TierModifier::Remove {
                if let Some(pos) = pos {
                    list.remove(pos);
                }
                if is_fallback {
                    fallback_explicitly_removed = true;
                }
                // Removing a tier that's not in defaults is a silent no-op
            } else {
                let replacement = EffectiveTier {
                    name: &entry.name,
                    explicit_gate: entry.feature_gate.as_deref(),
                    user_override: true,
                };
                if let Some(pos) = pos {
                    list[pos] = replacement;
                } else {
                    list.push(replacement);
                }
            }
        }
        list
    } else {
        // Override mode: plain tiers form the set (never sees `+` — that would
        // force additive mode). `-tier` removes from the set / drops the fallback.
        let mut list: Vec<EffectiveTier<'_>> = Vec::new();
        for entry in entries {
            if entry.modifier == TierModifier::Remove {
                let base = entry.base_name();
                let is_fallback = entry.is_fallback();
                if let Some(pos) = list
                    .iter()
                    .position(|slot| matches_slot(slot, base, is_fallback))
                {
                    list.remove(pos);
                }
                if is_fallback {
                    fallback_explicitly_removed = true;
                }
            } else {
                list.push(EffectiveTier {
                    name: &entry.name,
                    explicit_gate: entry.feature_gate.as_deref(),
                    user_override: false,
                });
            }
        }
        list
    };

    let mut tiers = Vec::with_capacity(effective.len() + 1);
    for entry in &effective {
        match find_tier(entry.name) {
            Some(tier) => {
                let is_explicit = entry.explicit_gate.is_some();
                // User-overridden tiers (from + entries) get exactly what the user
                // wrote — no auto-gating from the descriptor. +v4 means unconditional.
                let feature_gate = entry.explicit_gate.map(String::from).or_else(|| {
                    if default_feature_gates && !entry.user_override {
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
                    format!(
                        "unknown tier `{}`. Known tiers: {}",
                        entry.name,
                        known.join(", ")
                    ),
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

    if !has_scalar && !has_default && !fallback_explicitly_removed {
        tiers.push(ResolvedTier {
            tier: find_tier("scalar").unwrap(),
            feature_gate: None,
            allow_unexpected_cfg: false,
        });
    }

    tiers.sort_by_key(|rt| core::cmp::Reverse(rt.tier.priority));

    Ok(tiers)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn specs(names: &[&str]) -> Vec<ParsedTier> {
        names
            .iter()
            .map(|s| ParsedTier::parse_test_spec(s))
            .collect()
    }

    /// Resolve and return the tier suffixes in dispatch (priority) order.
    fn resolved(names: &[&str], gates: bool) -> Vec<String> {
        resolve_tiers(&specs(names), proc_macro2::Span::call_site(), gates)
            .unwrap()
            .iter()
            .map(|t| t.tier.suffix.to_string())
            .collect()
    }

    /// Same, but sorted alphabetically — for set-equality assertions where
    /// priority order is incidental.
    fn resolved_set(names: &[&str], gates: bool) -> Vec<String> {
        let mut v = resolved(names, gates);
        v.sort();
        v
    }

    // ---- issue #48 test plan ----

    #[test]
    fn plain_minus_scalar_drops_fallback() {
        // (v3, -scalar) → [v3]: list what you want, drop the auto-scalar.
        assert_eq!(resolved(&["v3", "-scalar"], false), vec!["v3"]);
    }

    #[test]
    fn two_plain_minus_scalar() {
        // (v3, v4, -scalar) → {v3, v4}, no scalar. Priority order: v4 then v3.
        assert_eq!(resolved(&["v3", "v4", "-scalar"], false), vec!["v4", "v3"]);
    }

    #[test]
    fn plus_and_plain_is_additive() {
        // (+v4, v3): any `+` ⇒ additive; plain v3 treated as +v3. Defaults stay.
        assert_eq!(
            resolved(&["+v4", "v3"], false),
            vec!["v4", "v3", "neon", "wasm128", "scalar"]
        );
    }

    #[test]
    fn mixing_plain_and_modifier_no_longer_errors() {
        // The old hard error is gone for every combination.
        assert!(
            resolve_tiers(
                &specs(&["v3", "-scalar"]),
                proc_macro2::Span::call_site(),
                false
            )
            .is_ok()
        );
        assert!(
            resolve_tiers(
                &specs(&["+v4", "v3"]),
                proc_macro2::Span::call_site(),
                false
            )
            .is_ok()
        );
    }

    // ---- preserved legacy behavior ----

    #[test]
    fn pure_plain_auto_appends_scalar() {
        assert_eq!(resolved(&["v3"], false), vec!["v3", "scalar"]);
        assert_eq!(resolved(&["v3", "v4"], false), vec!["v4", "v3", "scalar"]);
    }

    #[test]
    fn minus_only_starts_from_defaults() {
        // [-neon, -wasm128] = defaults minus those (additive mode, no plain).
        assert_eq!(
            resolved(&["-neon", "-wasm128"], false),
            vec!["v4", "v3", "scalar"]
        );
        // [-scalar] alone = defaults minus the fallback.
        assert_eq!(
            resolved(&["-scalar"], false),
            vec!["v4", "v3", "neon", "wasm128"]
        );
    }

    #[test]
    fn plus_appends_new_tier() {
        assert!(resolved_set(&["+arm_v2"], false).contains(&"arm_v2".to_string()));
        // and keeps the default fallback
        assert!(resolved_set(&["+arm_v2"], false).contains(&"scalar".to_string()));
    }

    // ---- feature gating interaction ----

    #[test]
    fn override_plain_v4_gets_auto_gate_when_enabled() {
        let resolved =
            resolve_tiers(&specs(&["v4", "v3"]), proc_macro2::Span::call_site(), true).unwrap();
        let v4 = resolved.iter().find(|t| t.tier.suffix == "v4").unwrap();
        assert_eq!(v4.feature_gate.as_deref(), Some("avx512"));
    }

    #[test]
    fn override_minus_scalar_keeps_gate_semantics() {
        // (v4, -scalar) with gates on: v4 still auto-gated, no scalar.
        let resolved = resolve_tiers(
            &specs(&["v4", "-scalar"]),
            proc_macro2::Span::call_site(),
            true,
        )
        .unwrap();
        assert!(resolved.iter().all(|t| t.tier.suffix != "scalar"));
        let v4 = resolved.iter().find(|t| t.tier.suffix == "v4").unwrap();
        assert_eq!(v4.feature_gate.as_deref(), Some("avx512"));
    }

    #[test]
    fn additive_plus_v4_is_unconditional() {
        // +v4 means "v4 exactly as written" — no auto avx512 gate.
        let resolved =
            resolve_tiers(&specs(&["+v4"]), proc_macro2::Span::call_site(), true).unwrap();
        let v4 = resolved.iter().find(|t| t.tier.suffix == "v4").unwrap();
        assert_eq!(v4.feature_gate, None);
    }

    // ---- errors still fire ----

    #[test]
    fn unknown_tier_errors() {
        assert!(resolve_tiers(&specs(&["nope"]), proc_macro2::Span::call_site(), false).is_err());
    }

    #[test]
    fn scalar_and_default_mutually_exclusive() {
        assert!(
            resolve_tiers(
                &specs(&["v3", "scalar", "default"]),
                proc_macro2::Span::call_site(),
                false,
            )
            .is_err()
        );
    }
}
