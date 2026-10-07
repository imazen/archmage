//! Checks the emitted signatures and forwarding calls of legacy families.
use super::*;

// =========================================================================
// autoversion — variant replacement (AST manipulation)
// =========================================================================

/// Mirrors what `autoversion_impl` does for a single variant: parse an
/// ItemFn (for test convenience), rename it, swap the SimdToken param
/// type, optionally inject the `_self` preamble for scalar+self.
/// Run the real `#[autoversion]` expansion and return it as a string, so the
/// assertions below test production output rather than a copy of its logic.
fn autoversion_expansion(attr: &str, item: &str) -> String {
    let attr: proc_macro2::TokenStream = attr.parse().unwrap();
    let item: proc_macro2::TokenStream = item.parse().unwrap();
    let out = super::expansion_tests::expand("autoversion", attr, item).unwrap();
    syn::parse2::<syn::File>(out.clone()).expect("expansion parses as items");
    out.to_string()
}

/// One assertion per tier: the variant is named `<fn>_<suffix>` and takes
/// the tier's token type at the token parameter's position.
fn assert_variant(expansion: &str, variant_signature: &str) {
    let normalized = expansion.replace(' ', "");
    let wanted = variant_signature.replace(' ', "");
    assert!(
        normalized.contains(&wanted),
        "expected `{variant_signature}` in:\n{expansion}"
    );
}

#[test]
fn variant_replacement_renames_and_retypes_per_tier() {
    let out = autoversion_expansion(
        "v3, neon, wasm128, scalar",
        "fn process(token: SimdToken, data: &[f32]) -> f32 { 0.0 }",
    );
    assert_variant(
        &out,
        "fn process_v3(token: archmage::X64V3Token, data: &[f32]) -> f32",
    );
    assert_variant(
        &out,
        "fn process_neon(token: archmage::NeonToken, data: &[f32]) -> f32",
    );
    assert_variant(
        &out,
        "fn process_wasm128(token: archmage::Wasm128Token, data: &[f32]) -> f32",
    );
    assert_variant(
        &out,
        "fn process_scalar(token: archmage::ScalarToken, data: &[f32]) -> f32",
    );
}

#[test]
fn variant_replacement_default_tier_drops_the_token() {
    let out = autoversion_expansion(
        "v3, default",
        "fn compute(token: SimdToken, data: &[f32]) -> f32 { 0.0 }",
    );
    assert_variant(&out, "fn compute_default(data: &[f32]) -> f32");
    assert_variant(&out, "compute_default(data)");
}

#[test]
fn variant_replacement_keeps_the_token_position() {
    // The token is not the first parameter: variants and dispatch calls both
    // keep it where the user put it.
    let out = autoversion_expansion("v3, scalar", "fn sum(x: u32, _: ScalarToken) -> u32 { x }");
    assert_variant(
        &out,
        "fn sum_v3(x: u32, __archmage_arg_0: archmage::X64V3Token) -> u32",
    );
    assert_variant(
        &out,
        "fn __arcane_sum_v3(x: u32, __archmage_arg_0: archmage::X64V3Token) -> u32",
    );
    assert_variant(&out, "__arcane_sum_v3(x, __archmage_arg_0)");
    assert_variant(&out, "sum_v3(x, __t)");
    assert_variant(&out, "sum_scalar(x, archmage::ScalarToken)");
}

#[test]
fn variant_replacement_covers_every_known_tier() {
    // Every tier the registry knows produces a `<fn>_<suffix>` variant with
    // the tier's token type, and the placeholder never survives.
    for tier in ALL_TIERS {
        let out = autoversion_expansion(
            tier.name,
            "fn compute(token: SimdToken, data: &mut [f32]) { }",
        );
        if tier.name == "default" {
            assert_variant(&out, "fn compute_default(data: &mut [f32])");
        } else {
            let token = tier.token_path;
            assert_variant(
                &out,
                &format!(
                    "fn compute_{}(token: {token}, data: &mut [f32])",
                    tier.suffix
                ),
            );
        }
        assert!(
            // The dispatcher's deprecation note names the placeholder; no
            // signature may still carry it as a type.
            !out.replace(' ', "").contains("token:SimdToken"),
            "tier {} left the SimdToken placeholder in:\n{out}",
            tier.name
        );
    }
}

#[test]
fn variant_replacement_preserves_the_rest_of_the_signature() {
    let out = autoversion_expansion(
        "v3, scalar",
        "fn process<'a, T: Copy + Default>(token: SimdToken, data: &'a [T], scale: f32) -> Vec<T> \
         where T: core::fmt::Debug { vec![] }",
    );
    assert_variant(
        &out,
        "fn process_v3<'a, T: Copy + Default>(token: archmage::X64V3Token, data: &'a [T], scale: f32) \
         -> Vec<T> where T: core::fmt::Debug",
    );
    assert_variant(&out, "process_v3::<T>(__t, data, scale)");
}

#[test]
fn variant_replacement_scalar_self_injects_preamble() {
    // With `_self = Type` outside a trait, the scalar variant stays a method
    // and binds `_self` itself, since it has no #[arcane] inner function.
    let out = autoversion_expansion(
        "v3, scalar, _self = S",
        "fn method(&self, token: SimdToken, data: &[f32]) -> f32 { _self.k }",
    );
    assert_variant(
        &out,
        "fn method_scalar(&self, token: archmage::ScalarToken, data: &[f32]) -> f32 { let _self = self;",
    );
}

#[test]
fn dispatcher_wildcard_params_get_renamed() {
    // The dispatcher names wildcard parameters so it can forward them; the
    // variants keep the user's patterns.
    let out = autoversion_expansion("v3, scalar", "fn process(_: &[f32], _: f32) -> f32 { 0.0 }");
    assert_variant(
        &out,
        "fn process(__archmage_arg_0: &[f32], __archmage_arg_1: f32) -> f32",
    );
    assert_variant(&out, "process_v3(__t, __archmage_arg_0, __archmage_arg_1)");
    assert_variant(
        &out,
        "fn process_scalar(_token: archmage::ScalarToken, _: &[f32], _: f32) -> f32",
    );
}
