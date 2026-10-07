//! Unit tests for token and trait recognition, argument parsing and the
//! generated registry maps. Integration-level expansion tests live in
//! `expansion_tests.rs`; snapshot tests in the crate's `tests/expand/`.

use super::*;

use super::generated::{ALL_CONCRETE_TOKENS, ALL_TRAIT_NAMES};
use syn::{ItemFn, ReturnType};

#[test]
fn every_concrete_token_is_in_token_to_features() {
    for &name in ALL_CONCRETE_TOKENS {
        assert!(
            token_to_features(name).is_some(),
            "Token `{}` exists in runtime crate but is NOT recognized by \
             token_to_features() in the proc macro. Add it!",
            name
        );
    }
}

#[test]
fn every_trait_is_in_trait_to_features() {
    for &name in ALL_TRAIT_NAMES {
        assert!(
            trait_to_features(name).is_some(),
            "Trait `{}` exists in runtime crate but is NOT recognized by \
             trait_to_features() in the proc macro. Add it!",
            name
        );
    }
}

#[test]
fn token_aliases_map_to_same_features() {
    // Desktop64 = X64V3Token
    assert_eq!(
        token_to_features("Desktop64"),
        token_to_features("X64V3Token"),
        "Desktop64 and X64V3Token should map to identical features"
    );

    // Server64 = X64V4Token = Avx512Token
    assert_eq!(
        token_to_features("Server64"),
        token_to_features("X64V4Token"),
        "Server64 and X64V4Token should map to identical features"
    );
    assert_eq!(
        token_to_features("X64V4Token"),
        token_to_features("Avx512Token"),
        "X64V4Token and Avx512Token should map to identical features"
    );

    // Arm64 = NeonToken
    assert_eq!(
        token_to_features("Arm64"),
        token_to_features("NeonToken"),
        "Arm64 and NeonToken should map to identical features"
    );
}

#[test]
fn trait_to_features_includes_tokens_as_bounds() {
    // Tier tokens should also work as trait bounds
    // (for `impl X64V3Token` patterns, even though Rust won't allow it,
    // the macro processes AST before type checking)
    let tier_tokens = [
        "X64V2Token",
        "X64CryptoToken",
        "X64V3Token",
        "Desktop64",
        "Avx2FmaToken",
        "X64V4Token",
        "Avx512Token",
        "Server64",
        "X64V4xToken",
        "Avx512Fp16Token",
        "NeonToken",
        "Arm64",
        "NeonAesToken",
        "NeonSha3Token",
        "NeonCrcToken",
        "Arm64V2Token",
        "Arm64V3Token",
    ];

    for &name in &tier_tokens {
        assert!(
            trait_to_features(name).is_some(),
            "Tier token `{}` should also be recognized in trait_to_features() \
             for use as a generic bound. Add it!",
            name
        );
    }
}

#[test]
fn trait_features_are_cumulative() {
    // HasX64V4 should include all HasX64V2 features plus more
    let v2_features = trait_to_features("HasX64V2").unwrap();
    let v4_features = trait_to_features("HasX64V4").unwrap();

    for &f in v2_features {
        assert!(
            v4_features.contains(&f),
            "HasX64V4 should include v2 feature `{}` but doesn't",
            f
        );
    }

    // v4 should have more features than v2
    assert!(
        v4_features.len() > v2_features.len(),
        "HasX64V4 should have more features than HasX64V2"
    );
}

#[test]
fn x64v3_trait_features_include_v2() {
    // X64V3Token as trait bound should include v2 features
    let v2 = trait_to_features("HasX64V2").unwrap();
    let v3 = trait_to_features("X64V3Token").unwrap();

    for &f in v2 {
        assert!(
            v3.contains(&f),
            "X64V3Token trait features should include v2 feature `{}` but don't",
            f
        );
    }
}

#[test]
fn has_neon_aes_includes_neon() {
    let neon = trait_to_features("HasNeon").unwrap();
    let neon_aes = trait_to_features("HasNeonAes").unwrap();

    for &f in neon {
        assert!(
            neon_aes.contains(&f),
            "HasNeonAes should include NEON feature `{}`",
            f
        );
    }
}

#[test]
fn no_removed_traits_are_recognized() {
    // These traits were removed in 0.3.0 and should NOT be recognized
    let removed = [
        "HasSse",
        "HasSse2",
        "HasSse41",
        "HasSse42",
        "HasAvx",
        "HasAvx2",
        "HasFma",
        "HasAvx512f",
        "HasAvx512bw",
        "HasAvx512vl",
        "HasAvx512vbmi2",
        "HasSve",
        "HasSve2",
    ];

    for &name in &removed {
        assert!(
            trait_to_features(name).is_none(),
            "Removed trait `{}` should NOT be in trait_to_features(). \
             It was removed in 0.3.0 — users should migrate to tier traits.",
            name
        );
    }
}

#[test]
fn no_nonexistent_tokens_are_recognized() {
    // These tokens don't exist and should NOT be recognized
    let fake = [
        "SveToken",
        "Sve2Token",
        "Avx512VnniToken",
        "X64V4ModernToken",
        "NeonFp16Token",
    ];

    for &name in &fake {
        assert!(
            token_to_features(name).is_none(),
            "Non-existent token `{}` should NOT be in token_to_features()",
            name
        );
    }
}

#[test]
fn featureless_traits_are_not_in_registries() {
    // SimdToken and IntoConcreteToken should NOT be in any feature registry
    // because they don't map to CPU features
    for &name in FEATURELESS_TRAIT_NAMES {
        assert!(
            token_to_features(name).is_none(),
            "`{}` should NOT be in token_to_features() — it has no CPU features",
            name
        );
        assert!(
            trait_to_features(name).is_none(),
            "`{}` should NOT be in trait_to_features() — it has no CPU features",
            name
        );
    }
}

#[test]
fn find_featureless_trait_detects_simdtoken() {
    let names = vec!["SimdToken".to_string()];
    assert_eq!(find_featureless_trait(&names), Some("SimdToken"));

    let names = vec!["IntoConcreteToken".to_string()];
    assert_eq!(find_featureless_trait(&names), Some("IntoConcreteToken"));

    // Feature-bearing traits should NOT be detected
    let names = vec!["HasX64V2".to_string()];
    assert_eq!(find_featureless_trait(&names), None);

    let names = vec!["HasNeon".to_string()];
    assert_eq!(find_featureless_trait(&names), None);

    // Mixed: if SimdToken is among real traits, still detected
    let names = vec!["SimdToken".to_string(), "HasX64V2".to_string()];
    assert_eq!(find_featureless_trait(&names), Some("SimdToken"));
}

#[test]
fn arm64_v2_v3_traits_are_cumulative() {
    let v2_features = trait_to_features("HasArm64V2").unwrap();
    let v3_features = trait_to_features("HasArm64V3").unwrap();

    for &f in v2_features {
        assert!(
            v3_features.contains(&f),
            "HasArm64V3 should include v2 feature `{}` but doesn't",
            f
        );
    }

    assert!(
        v3_features.len() > v2_features.len(),
        "HasArm64V3 should have more features than HasArm64V2"
    );
}

// =========================================================================
// resolve_tiers — additive / subtractive / override
// =========================================================================

fn resolve_tier_names(names: &[&str], default_gates: bool) -> Vec<String> {
    let names: Vec<String> = names.iter().map(|s| s.to_string()).collect();
    resolve_tiers(&names, proc_macro2::Span::call_site(), default_gates)
        .unwrap()
        .iter()
        .map(|rt| {
            if let Some(ref gate) = rt.feature_gate {
                format!("{}({})", rt.name, gate)
            } else {
                rt.name.to_string()
            }
        })
        .collect()
}

#[test]
fn resolve_defaults() {
    let tiers = resolve_tier_names(&["v4", "v3", "neon", "wasm128", "scalar"], true);
    assert!(tiers.contains(&"v3".to_string()));
    assert!(tiers.contains(&"scalar".to_string()));
    // v4 gets auto-gated when default_feature_gates=true
    assert!(tiers.contains(&"v4(avx512)".to_string()));
}

#[test]
fn resolve_additive_appends() {
    let tiers = resolve_tier_names(&["+v1"], true);
    assert!(tiers.contains(&"v1".to_string()));
    assert!(tiers.contains(&"v3".to_string())); // from defaults
    assert!(tiers.contains(&"scalar".to_string())); // from defaults
}

#[test]
fn resolve_additive_v4_overrides_gate() {
    // +v4 should replace v4(avx512) with plain v4 (no gate)
    let tiers = resolve_tier_names(&["+v4"], true);
    assert!(tiers.contains(&"v4".to_string())); // no gate
    assert!(!tiers.iter().any(|t| t == "v4(avx512)")); // gated version gone
}

#[test]
fn resolve_additive_default_replaces_scalar() {
    let tiers = resolve_tier_names(&["+default"], true);
    assert!(tiers.contains(&"default".to_string()));
    assert!(!tiers.iter().any(|t| t == "scalar")); // scalar replaced
}

#[test]
fn resolve_subtractive_removes() {
    let tiers = resolve_tier_names(&["-neon", "-wasm128"], true);
    assert!(!tiers.iter().any(|t| t == "neon"));
    assert!(!tiers.iter().any(|t| t == "wasm128"));
    assert!(tiers.contains(&"v3".to_string())); // others remain
}

#[test]
fn resolve_subtractive_removes_scalar() {
    // -scalar must actually remove scalar — auto-append must not undo it
    let tiers = resolve_tier_names(&["-scalar"], true);
    assert!(
        !tiers.iter().any(|t| t == "scalar"),
        "expected scalar to be removed, got {tiers:?}"
    );
}

#[test]
fn resolve_subtractive_removes_scalar_with_other_modifiers() {
    // [-scalar, +arm_v2] — scalar removed, arm_v2 added, no scalar re-injected
    let tiers = resolve_tier_names(&["-scalar", "+arm_v2"], true);
    assert!(
        !tiers.iter().any(|t| t == "scalar"),
        "expected scalar to be removed, got {tiers:?}"
    );
    assert!(tiers.contains(&"arm_v2".to_string()));
}

#[test]
fn resolve_mixed_add_remove() {
    let tiers = resolve_tier_names(&["-neon", "-wasm128", "+v1"], true);
    assert!(tiers.contains(&"v1".to_string()));
    assert!(!tiers.iter().any(|t| t == "neon"));
    assert!(!tiers.iter().any(|t| t == "wasm128"));
    assert!(tiers.contains(&"v3".to_string()));
    assert!(tiers.contains(&"scalar".to_string()));
}

#[test]
fn resolve_additive_duplicate_is_noop() {
    // +v3 when v3 is already in defaults — no duplicate
    let tiers = resolve_tier_names(&["+v3"], true);
    let v3_count = tiers.iter().filter(|t| t.as_str() == "v3").count();
    assert_eq!(v3_count, 1);
}

#[test]
fn resolve_mixing_plus_and_plain_is_additive() {
    // #48: mixing `+tier` with plain `tier` is now allowed. Any `+` ⇒
    // additive mode; the plain tier is treated as `+tier`. Both v1 and v3
    // end up present (atop the defaults).
    let names: Vec<String> = vec!["+v1".into(), "v3".into()];
    let tiers = resolve_tiers(&names, proc_macro2::Span::call_site(), true).unwrap();
    let suffixes: Vec<&str> = tiers.iter().map(|t| t.tier.suffix).collect();
    assert!(suffixes.contains(&"v1"));
    assert!(suffixes.contains(&"v3"));
}

#[test]
fn resolve_underscore_tier_name() {
    let tiers = resolve_tier_names(&["_v3", "_neon", "_scalar"], false);
    assert!(tiers.contains(&"v3".to_string()));
    assert!(tiers.contains(&"neon".to_string()));
    assert!(tiers.contains(&"scalar".to_string()));
}

// =========================================================================
// autoversion — argument parsing
// =========================================================================

#[test]
fn autoversion_args_empty() {
    let args: AutoversionArgs = syn::parse_str("").unwrap();
    assert!(args.self_type.is_none());
    assert!(args.tiers.is_none());
}

#[test]
fn autoversion_args_single_tier() {
    let args: AutoversionArgs = syn::parse_str("v3").unwrap();
    assert!(args.self_type.is_none());
    assert_eq!(args.tiers.as_ref().unwrap(), &["v3"]);
}

#[test]
fn autoversion_args_tiers_only() {
    let args: AutoversionArgs = syn::parse_str("v3, v4, neon").unwrap();
    assert!(args.self_type.is_none());
    let tiers = args.tiers.unwrap();
    assert_eq!(tiers, vec!["v3", "v4", "neon"]);
}

#[test]
fn autoversion_args_many_tiers() {
    let args: AutoversionArgs =
        syn::parse_str("v1, v2, v3, v4, v4x, neon, arm_v2, wasm128").unwrap();
    assert_eq!(
        args.tiers.unwrap(),
        vec!["v1", "v2", "v3", "v4", "v4x", "neon", "arm_v2", "wasm128"]
    );
}

#[test]
fn autoversion_args_trailing_comma() {
    let args: AutoversionArgs = syn::parse_str("v3, v4,").unwrap();
    assert_eq!(args.tiers.as_ref().unwrap(), &["v3", "v4"]);
}

#[test]
fn autoversion_args_self_only() {
    let args: AutoversionArgs = syn::parse_str("_self = MyType").unwrap();
    assert!(args.self_type.is_some());
    assert!(args.tiers.is_none());
}

#[test]
fn autoversion_args_self_and_tiers() {
    let args: AutoversionArgs = syn::parse_str("_self = MyType, v3, neon").unwrap();
    assert!(args.self_type.is_some());
    let tiers = args.tiers.unwrap();
    assert_eq!(tiers, vec!["v3", "neon"]);
}

#[test]
fn autoversion_args_tiers_then_self() {
    // _self can appear after tier names
    let args: AutoversionArgs = syn::parse_str("v3, neon, _self = MyType").unwrap();
    assert!(args.self_type.is_some());
    let tiers = args.tiers.unwrap();
    assert_eq!(tiers, vec!["v3", "neon"]);
}

#[test]
fn autoversion_args_self_with_path_type() {
    let args: AutoversionArgs = syn::parse_str("_self = crate::MyType").unwrap();
    assert!(args.self_type.is_some());
    assert!(args.tiers.is_none());
}

#[test]
fn autoversion_args_self_with_generic_type() {
    let args: AutoversionArgs = syn::parse_str("_self = Vec<u8>").unwrap();
    assert!(args.self_type.is_some());
    let ty_str = args.self_type.unwrap().to_token_stream().to_string();
    assert!(ty_str.contains("Vec"), "Expected Vec<u8>, got: {}", ty_str);
}

#[test]
fn autoversion_args_self_trailing_comma() {
    let args: AutoversionArgs = syn::parse_str("_self = MyType,").unwrap();
    assert!(args.self_type.is_some());
    assert!(args.tiers.is_none());
}

// =========================================================================
// autoversion — find_autoversion_token_param
// =========================================================================

#[test]
fn find_autoversion_token_param_simdtoken_first() {
    let f: ItemFn = syn::parse_str("fn process(token: SimdToken, data: &[f32]) -> f32 {}").unwrap();
    let param = find_autoversion_token_param(&f.sig).unwrap().unwrap();
    assert_eq!(param.index, 0);
    assert_eq!(param.ident, "token");
    assert_eq!(param.kind, AutoversionTokenKind::SimdToken);
}

#[test]
fn find_autoversion_token_param_simdtoken_second() {
    let f: ItemFn = syn::parse_str("fn process(data: &[f32], token: SimdToken) -> f32 {}").unwrap();
    let param = find_autoversion_token_param(&f.sig).unwrap().unwrap();
    assert_eq!(param.index, 1);
    assert_eq!(param.kind, AutoversionTokenKind::SimdToken);
}

#[test]
fn find_autoversion_token_param_underscore_prefix() {
    let f: ItemFn =
        syn::parse_str("fn process(_token: SimdToken, data: &[f32]) -> f32 {}").unwrap();
    let param = find_autoversion_token_param(&f.sig).unwrap().unwrap();
    assert_eq!(param.index, 0);
    assert_eq!(param.ident, "_token");
}

#[test]
fn find_autoversion_token_param_wildcard() {
    let f: ItemFn = syn::parse_str("fn process(_: SimdToken, data: &[f32]) -> f32 {}").unwrap();
    let param = find_autoversion_token_param(&f.sig).unwrap().unwrap();
    assert_eq!(param.index, 0);
    assert_eq!(param.ident, "__autoversion_token");
}

#[test]
fn find_autoversion_token_param_scalar_token() {
    let f: ItemFn =
        syn::parse_str("fn process_scalar(_: ScalarToken, data: &[f32]) -> f32 {}").unwrap();
    let param = find_autoversion_token_param(&f.sig).unwrap().unwrap();
    assert_eq!(param.index, 0);
    assert_eq!(param.kind, AutoversionTokenKind::ScalarToken);
}

#[test]
fn find_autoversion_token_param_not_found() {
    let f: ItemFn = syn::parse_str("fn process(data: &[f32]) -> f32 {}").unwrap();
    assert!(find_autoversion_token_param(&f.sig).unwrap().is_none());
}

#[test]
fn find_autoversion_token_param_no_params() {
    let f: ItemFn = syn::parse_str("fn process() {}").unwrap();
    assert!(find_autoversion_token_param(&f.sig).unwrap().is_none());
}

#[test]
fn find_autoversion_token_param_concrete_token_errors() {
    let f: ItemFn =
        syn::parse_str("fn process(token: X64V3Token, data: &[f32]) -> f32 {}").unwrap();
    let err = find_autoversion_token_param(&f.sig).unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("concrete token"),
        "error should mention concrete token: {msg}"
    );
    assert!(
        msg.contains("#[arcane]"),
        "error should suggest #[arcane]: {msg}"
    );
}

#[test]
fn find_autoversion_token_param_neon_token_errors() {
    let f: ItemFn = syn::parse_str("fn process(token: NeonToken, data: &[f32]) -> f32 {}").unwrap();
    assert!(find_autoversion_token_param(&f.sig).is_err());
}

#[test]
fn find_autoversion_token_param_unknown_type_ignored() {
    // Random types are just regular params, not token params
    let f: ItemFn = syn::parse_str("fn process(data: &[f32], scale: f32) -> f32 {}").unwrap();
    assert!(find_autoversion_token_param(&f.sig).unwrap().is_none());
}

#[test]
fn find_autoversion_token_param_among_many() {
    let f: ItemFn =
        syn::parse_str("fn process(a: i32, b: f64, token: SimdToken, c: &str, d: bool) -> f32 {}")
            .unwrap();
    let param = find_autoversion_token_param(&f.sig).unwrap().unwrap();
    assert_eq!(param.index, 2);
    assert_eq!(param.ident, "token");
}

#[test]
fn find_autoversion_token_param_with_generics() {
    let f: ItemFn =
        syn::parse_str("fn process<T: Clone>(token: SimdToken, data: &[T]) -> T {}").unwrap();
    let param = find_autoversion_token_param(&f.sig).unwrap().unwrap();
    assert_eq!(param.index, 0);
}

#[test]
fn find_autoversion_token_param_with_where_clause() {
    let f: ItemFn = syn::parse_str(
        "fn process<T>(token: SimdToken, data: &[T]) -> T where T: Copy + Default {}",
    )
    .unwrap();
    let param = find_autoversion_token_param(&f.sig).unwrap().unwrap();
    assert_eq!(param.index, 0);
}

#[test]
fn find_autoversion_token_param_with_lifetime() {
    let f: ItemFn =
        syn::parse_str("fn process<'a>(token: SimdToken, data: &'a [f32]) -> &'a f32 {}").unwrap();
    let param = find_autoversion_token_param(&f.sig).unwrap().unwrap();
    assert_eq!(param.index, 0);
}

// =========================================================================
// autoversion — tier resolution
// =========================================================================

#[test]
fn autoversion_default_tiers_all_resolve() {
    let names: Vec<String> = DEFAULT_TIER_NAMES.iter().map(|s| s.to_string()).collect();
    let tiers = resolve_tiers(&names, proc_macro2::Span::call_site(), false).unwrap();
    assert!(!tiers.is_empty());
    // scalar should be present
    assert!(tiers.iter().any(|t| t.name == "scalar"));
}

#[test]
fn autoversion_scalar_always_appended() {
    let names = vec!["v3".to_string(), "neon".to_string()];
    let tiers = resolve_tiers(&names, proc_macro2::Span::call_site(), false).unwrap();
    assert!(
        tiers.iter().any(|t| t.name == "scalar"),
        "scalar must be auto-appended"
    );
}

#[test]
fn autoversion_scalar_not_duplicated() {
    let names = vec!["v3".to_string(), "scalar".to_string()];
    let tiers = resolve_tiers(&names, proc_macro2::Span::call_site(), false).unwrap();
    let scalar_count = tiers.iter().filter(|t| t.name == "scalar").count();
    assert_eq!(scalar_count, 1, "scalar must not be duplicated");
}

#[test]
fn autoversion_tiers_sorted_by_priority() {
    let names = vec!["neon".to_string(), "v4".to_string(), "v3".to_string()];
    let tiers = resolve_tiers(&names, proc_macro2::Span::call_site(), false).unwrap();
    // v4 (priority 40) > v3 (30) > neon (20) > scalar (0)
    let priorities: Vec<u32> = tiers.iter().map(|t| t.priority).collect();
    for window in priorities.windows(2) {
        assert!(
            window[0] >= window[1],
            "Tiers not sorted by priority: {:?}",
            priorities
        );
    }
}

#[test]
fn autoversion_unknown_tier_errors() {
    let names = vec!["v3".to_string(), "avx9000".to_string()];
    let result = resolve_tiers(&names, proc_macro2::Span::call_site(), false);
    match result {
        Ok(_) => panic!("Expected error for unknown tier 'avx9000'"),
        Err(e) => {
            let err_msg = e.to_string();
            assert!(
                err_msg.contains("avx9000"),
                "Error should mention unknown tier: {}",
                err_msg
            );
        }
    }
}

#[test]
fn autoversion_all_known_tiers_resolve() {
    // Every tier in ALL_TIERS should be findable
    for tier in ALL_TIERS {
        assert!(
            find_tier(tier.name).is_some(),
            "Tier '{}' should be findable by name",
            tier.name
        );
    }
}

#[test]
fn autoversion_default_tier_list_is_sensible() {
    // Defaults should cover x86, ARM, WASM, and scalar
    let names: Vec<String> = DEFAULT_TIER_NAMES.iter().map(|s| s.to_string()).collect();
    let tiers = resolve_tiers(&names, proc_macro2::Span::call_site(), false).unwrap();

    let has_x86 = tiers.iter().any(|t| t.target_arch == Some("x86_64"));
    let has_arm = tiers.iter().any(|t| t.target_arch == Some("aarch64"));
    let has_wasm = tiers.iter().any(|t| t.target_arch == Some("wasm32"));
    let has_scalar = tiers.iter().any(|t| t.name == "scalar");

    assert!(has_x86, "Default tiers should include an x86_64 tier");
    assert!(has_arm, "Default tiers should include an aarch64 tier");
    assert!(has_wasm, "Default tiers should include a wasm32 tier");
    assert!(has_scalar, "Default tiers should include scalar");
}

// =========================================================================
// autoversion — variant replacement (AST manipulation)
// =========================================================================

/// Mirrors what `autoversion_impl` does for a single variant: parse an
/// ItemFn (for test convenience), rename it, swap the SimdToken param
/// type, optionally inject the `_self` preamble for scalar+self.
fn do_variant_replacement(func: &str, tier_name: &str, has_self: bool) -> ItemFn {
    let mut f: ItemFn = syn::parse_str(func).unwrap();
    let fn_name = f.sig.ident.to_string();

    let tier = find_tier(tier_name).unwrap();

    // Rename
    f.sig.ident = format_ident!("{}_{}", fn_name, tier.suffix);

    // Find and replace SimdToken param type (skip for "default" — tokenless)
    let token_idx = find_autoversion_token_param(&f.sig)
        .expect("should not error on SimdToken")
        .unwrap_or_else(|| panic!("No SimdToken param in: {}", func))
        .index;
    if tier_name == "default" {
        // Remove the token param for default tier
        let stmts = f.block.stmts.clone();
        let mut inputs: Vec<FnArg> = f.sig.inputs.iter().cloned().collect();
        inputs.remove(token_idx);
        f.sig.inputs = inputs.into_iter().collect();
        f.block.stmts = stmts;
    } else {
        let concrete_type: Type = syn::parse_str(tier.token_path).unwrap();
        if let FnArg::Typed(pt) = &mut f.sig.inputs[token_idx] {
            *pt.ty = concrete_type;
        }
    }

    // Fallback (scalar/default) + self: inject preamble
    if (tier_name == "scalar" || tier_name == "default") && has_self {
        let preamble: syn::Stmt = syn::parse_quote!(let _self = self;);
        f.block.stmts.insert(0, preamble);
    }

    f
}

#[test]
fn variant_replacement_v3_renames_function() {
    let f = do_variant_replacement(
        "fn process(token: SimdToken, data: &[f32]) -> f32 { 0.0 }",
        "v3",
        false,
    );
    assert_eq!(f.sig.ident, "process_v3");
}

#[test]
fn variant_replacement_v3_replaces_token_type() {
    let f = do_variant_replacement(
        "fn process(token: SimdToken, data: &[f32]) -> f32 { 0.0 }",
        "v3",
        false,
    );
    let first_param_ty = match &f.sig.inputs[0] {
        FnArg::Typed(pt) => pt.ty.to_token_stream().to_string(),
        _ => panic!("Expected typed param"),
    };
    assert!(
        first_param_ty.contains("X64V3Token"),
        "Expected X64V3Token, got: {}",
        first_param_ty
    );
}

#[test]
fn variant_replacement_neon_produces_valid_fn() {
    let f = do_variant_replacement(
        "fn compute(token: SimdToken, data: &[f32]) -> f32 { 0.0 }",
        "neon",
        false,
    );
    assert_eq!(f.sig.ident, "compute_neon");
    let first_param_ty = match &f.sig.inputs[0] {
        FnArg::Typed(pt) => pt.ty.to_token_stream().to_string(),
        _ => panic!("Expected typed param"),
    };
    assert!(
        first_param_ty.contains("NeonToken"),
        "Expected NeonToken, got: {}",
        first_param_ty
    );
}

#[test]
fn variant_replacement_wasm128_produces_valid_fn() {
    let f = do_variant_replacement(
        "fn compute(_t: SimdToken, data: &[f32]) -> f32 { 0.0 }",
        "wasm128",
        false,
    );
    assert_eq!(f.sig.ident, "compute_wasm128");
}

#[test]
fn variant_replacement_scalar_produces_valid_fn() {
    let f = do_variant_replacement(
        "fn compute(token: SimdToken, data: &[f32]) -> f32 { 0.0 }",
        "scalar",
        false,
    );
    assert_eq!(f.sig.ident, "compute_scalar");
    let first_param_ty = match &f.sig.inputs[0] {
        FnArg::Typed(pt) => pt.ty.to_token_stream().to_string(),
        _ => panic!("Expected typed param"),
    };
    assert!(
        first_param_ty.contains("ScalarToken"),
        "Expected ScalarToken, got: {}",
        first_param_ty
    );
}

#[test]
fn variant_replacement_v4_produces_valid_fn() {
    let f = do_variant_replacement(
        "fn transform(token: SimdToken, data: &mut [f32]) { }",
        "v4",
        false,
    );
    assert_eq!(f.sig.ident, "transform_v4");
    let first_param_ty = match &f.sig.inputs[0] {
        FnArg::Typed(pt) => pt.ty.to_token_stream().to_string(),
        _ => panic!("Expected typed param"),
    };
    assert!(
        first_param_ty.contains("X64V4Token"),
        "Expected X64V4Token, got: {}",
        first_param_ty
    );
}

#[test]
fn variant_replacement_v4x_produces_valid_fn() {
    let f = do_variant_replacement(
        "fn transform(token: SimdToken, data: &mut [f32]) { }",
        "v4x",
        false,
    );
    assert_eq!(f.sig.ident, "transform_v4x");
}

#[test]
fn variant_replacement_arm_v2_produces_valid_fn() {
    let f = do_variant_replacement(
        "fn transform(token: SimdToken, data: &mut [f32]) { }",
        "arm_v2",
        false,
    );
    assert_eq!(f.sig.ident, "transform_arm_v2");
}

#[test]
fn variant_replacement_preserves_generics() {
    let f = do_variant_replacement(
        "fn process<T: Copy + Default>(token: SimdToken, data: &[T]) -> T { T::default() }",
        "v3",
        false,
    );
    assert_eq!(f.sig.ident, "process_v3");
    // Generic params should still be present
    assert!(
        !f.sig.generics.params.is_empty(),
        "Generics should be preserved"
    );
}

#[test]
fn variant_replacement_preserves_where_clause() {
    let f = do_variant_replacement(
        "fn process<T>(token: SimdToken, data: &[T]) -> T where T: Copy + Default { T::default() }",
        "v3",
        false,
    );
    assert!(
        f.sig.generics.where_clause.is_some(),
        "Where clause should be preserved"
    );
}

#[test]
fn variant_replacement_preserves_return_type() {
    let f = do_variant_replacement(
        "fn process(token: SimdToken, data: &[f32]) -> Vec<f32> { vec![] }",
        "neon",
        false,
    );
    let ret = f.sig.output.to_token_stream().to_string();
    assert!(
        ret.contains("Vec"),
        "Return type should be preserved, got: {}",
        ret
    );
}

#[test]
fn variant_replacement_preserves_multiple_params() {
    let f = do_variant_replacement(
        "fn process(token: SimdToken, a: &[f32], b: &[f32], scale: f32) -> f32 { 0.0 }",
        "v3",
        false,
    );
    // SimdToken → X64V3Token, plus the 3 other params
    assert_eq!(f.sig.inputs.len(), 4);
}

#[test]
fn variant_replacement_preserves_no_return_type() {
    let f = do_variant_replacement(
        "fn transform(token: SimdToken, data: &mut [f32]) { }",
        "v3",
        false,
    );
    assert!(
        matches!(f.sig.output, ReturnType::Default),
        "No return type should remain as Default"
    );
}

#[test]
fn variant_replacement_preserves_lifetime_params() {
    let f = do_variant_replacement(
        "fn process<'a>(token: SimdToken, data: &'a [f32]) -> &'a [f32] { data }",
        "v3",
        false,
    );
    assert!(!f.sig.generics.params.is_empty());
}

#[test]
fn variant_replacement_scalar_self_injects_preamble() {
    let f = do_variant_replacement(
        "fn method(token: SimdToken, data: &[f32]) -> f32 { 0.0 }",
        "scalar",
        true, // has_self
    );
    assert_eq!(f.sig.ident, "method_scalar");

    // First statement should be `let _self = self;`
    let body_str = f.block.to_token_stream().to_string();
    assert!(
        body_str.contains("let _self = self"),
        "Scalar+self variant should have _self preamble, got: {}",
        body_str
    );
}

#[test]
fn variant_replacement_all_default_tiers_produce_valid_fns() {
    let names: Vec<String> = DEFAULT_TIER_NAMES.iter().map(|s| s.to_string()).collect();
    let tiers = resolve_tiers(&names, proc_macro2::Span::call_site(), false).unwrap();

    for tier in &tiers {
        let f = do_variant_replacement(
            "fn process(token: SimdToken, data: &[f32]) -> f32 { 0.0 }",
            tier.name,
            false,
        );
        let expected_name = format!("process_{}", tier.suffix);
        assert_eq!(
            f.sig.ident.to_string(),
            expected_name,
            "Tier '{}' should produce function '{}'",
            tier.name,
            expected_name
        );
    }
}

#[test]
fn variant_replacement_all_known_tiers_produce_valid_fns() {
    for tier in ALL_TIERS {
        let f = do_variant_replacement(
            "fn compute(token: SimdToken, data: &[f32]) -> f32 { 0.0 }",
            tier.name,
            false,
        );
        let expected_name = format!("compute_{}", tier.suffix);
        assert_eq!(
            f.sig.ident.to_string(),
            expected_name,
            "Tier '{}' should produce function '{}'",
            tier.name,
            expected_name
        );
    }
}

#[test]
fn variant_replacement_no_simdtoken_remains() {
    for tier in ALL_TIERS {
        let f = do_variant_replacement(
            "fn compute(token: SimdToken, data: &[f32]) -> f32 { 0.0 }",
            tier.name,
            false,
        );
        let full_str = f.to_token_stream().to_string();
        assert!(
            !full_str.contains("SimdToken"),
            "Tier '{}' variant still contains 'SimdToken': {}",
            tier.name,
            full_str
        );
    }
}

// =========================================================================
// autoversion — cfg guard and tier descriptor properties
// =========================================================================

#[test]
fn tier_v3_targets_x86_64() {
    let tier = find_tier("v3").unwrap();
    assert_eq!(tier.target_arch, Some("x86_64"));
}

#[test]
fn tier_v4_targets_x86_64() {
    let tier = find_tier("v4").unwrap();
    assert_eq!(tier.target_arch, Some("x86_64"));
}

#[test]
fn tier_v4x_targets_x86_64() {
    let tier = find_tier("v4x").unwrap();
    assert_eq!(tier.target_arch, Some("x86_64"));
}

#[test]
fn tier_neon_targets_aarch64() {
    let tier = find_tier("neon").unwrap();
    assert_eq!(tier.target_arch, Some("aarch64"));
}

#[test]
fn tier_wasm128_targets_wasm32() {
    let tier = find_tier("wasm128").unwrap();
    assert_eq!(tier.target_arch, Some("wasm32"));
}

#[test]
fn tier_scalar_has_no_guards() {
    let tier = find_tier("scalar").unwrap();
    assert_eq!(tier.target_arch, None);
    assert_eq!(tier.priority, 0);
}

#[test]
fn tier_priorities_are_consistent() {
    // Higher-capability tiers within the same arch should have higher priority
    let v2 = find_tier("v2").unwrap();
    let v3 = find_tier("v3").unwrap();
    let v4 = find_tier("v4").unwrap();
    assert!(v4.priority > v3.priority);
    assert!(v3.priority > v2.priority);

    let neon = find_tier("neon").unwrap();
    let arm_v2 = find_tier("arm_v2").unwrap();
    let arm_v3 = find_tier("arm_v3").unwrap();
    assert!(arm_v3.priority > arm_v2.priority);
    assert!(arm_v2.priority > neon.priority);

    // scalar is lowest
    let scalar = find_tier("scalar").unwrap();
    assert!(neon.priority > scalar.priority);
    assert!(v2.priority > scalar.priority);
}

// =========================================================================
// autoversion — dispatcher structure
// =========================================================================

#[test]
fn dispatcher_param_removal_free_fn() {
    // Simulate what autoversion_impl does: remove the SimdToken param
    let f: ItemFn =
        syn::parse_str("fn process(token: SimdToken, data: &[f32], scale: f32) -> f32 { 0.0 }")
            .unwrap();

    let token_param = find_autoversion_token_param(&f.sig).unwrap().unwrap();
    // SimdToken → strip from dispatcher
    assert_eq!(token_param.kind, AutoversionTokenKind::SimdToken);
    let mut dispatcher_inputs: Vec<FnArg> = f.sig.inputs.iter().cloned().collect();
    dispatcher_inputs.remove(token_param.index);
    assert_eq!(dispatcher_inputs.len(), 2);
}

#[test]
fn dispatcher_param_removal_token_only() {
    let f: ItemFn = syn::parse_str("fn process(token: SimdToken) -> f32 { 0.0 }").unwrap();
    let token_param = find_autoversion_token_param(&f.sig).unwrap().unwrap();
    let mut dispatcher_inputs: Vec<FnArg> = f.sig.inputs.iter().cloned().collect();
    dispatcher_inputs.remove(token_param.index);
    assert_eq!(dispatcher_inputs.len(), 0);
}

#[test]
fn dispatcher_param_removal_token_last() {
    let f: ItemFn =
        syn::parse_str("fn process(data: &[f32], scale: f32, token: SimdToken) -> f32 { 0.0 }")
            .unwrap();
    let token_param = find_autoversion_token_param(&f.sig).unwrap().unwrap();
    assert_eq!(token_param.index, 2);
    let mut dispatcher_inputs: Vec<FnArg> = f.sig.inputs.iter().cloned().collect();
    dispatcher_inputs.remove(token_param.index);
    assert_eq!(dispatcher_inputs.len(), 2);
}

#[test]
fn dispatcher_scalar_token_kept() {
    // ScalarToken is a real type — kept in dispatcher for incant! compatibility
    let f: ItemFn =
        syn::parse_str("fn process_scalar(_: ScalarToken, data: &[f32]) -> f32 { 0.0 }").unwrap();
    let token_param = find_autoversion_token_param(&f.sig).unwrap().unwrap();
    assert_eq!(token_param.kind, AutoversionTokenKind::ScalarToken);
    // Should NOT be removed — dispatcher keeps it
    assert_eq!(f.sig.inputs.len(), 2);
}

#[test]
fn dispatcher_dispatch_args_extraction() {
    // Test that we correctly extract idents for the dispatch call
    let f: ItemFn = syn::parse_str("fn process(data: &[f32], scale: f32) -> f32 { 0.0 }").unwrap();

    let dispatch_args: Vec<String> = f
        .sig
        .inputs
        .iter()
        .filter_map(|arg| {
            if let FnArg::Typed(PatType { pat, .. }) = arg
                && let syn::Pat::Ident(pi) = pat.as_ref()
            {
                return Some(pi.ident.to_string());
            }
            None
        })
        .collect();

    assert_eq!(dispatch_args, vec!["data", "scale"]);
}

#[test]
fn dispatcher_wildcard_params_get_renamed() {
    let f: ItemFn = syn::parse_str("fn process(_: &[f32], _: f32) -> f32 { 0.0 }").unwrap();

    let mut dispatcher_inputs: Vec<FnArg> = f.sig.inputs.iter().cloned().collect();

    let mut wild_counter = 0u32;
    for arg in &mut dispatcher_inputs {
        if let FnArg::Typed(pat_type) = arg
            && matches!(pat_type.pat.as_ref(), syn::Pat::Wild(_))
        {
            let ident = format_ident!("__autoversion_wild_{}", wild_counter);
            wild_counter += 1;
            *pat_type.pat = syn::Pat::Ident(syn::PatIdent {
                attrs: vec![],
                by_ref: None,
                mutability: None,
                ident,
                subpat: None,
            });
        }
    }

    // Both wildcards should be renamed
    assert_eq!(wild_counter, 2);

    let names: Vec<String> = dispatcher_inputs
        .iter()
        .filter_map(|arg| {
            if let FnArg::Typed(PatType { pat, .. }) = arg
                && let syn::Pat::Ident(pi) = pat.as_ref()
            {
                return Some(pi.ident.to_string());
            }
            None
        })
        .collect();

    assert_eq!(names, vec!["__autoversion_wild_0", "__autoversion_wild_1"]);
}

// =========================================================================
// autoversion — suffix_path (reused in dispatch)
// =========================================================================

#[test]
fn suffix_path_simple() {
    let path: syn::Path = syn::parse_str("process").unwrap();
    let suffixed = suffix_path(&path, "v3");
    assert_eq!(suffixed.to_token_stream().to_string(), "process_v3");
}

#[test]
fn suffix_path_qualified() {
    let path: syn::Path = syn::parse_str("module::process").unwrap();
    let suffixed = suffix_path(&path, "neon");
    let s = suffixed.to_token_stream().to_string();
    assert!(
        s.contains("process_neon"),
        "Expected process_neon, got: {}",
        s
    );
}
