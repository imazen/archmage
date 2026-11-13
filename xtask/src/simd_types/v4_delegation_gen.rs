//! Declarative parent-backend trait delegation generator.
//!
//! Derived directly from the generated `backends/*.rs` trait definitions, so
//! every method a backend trait declares is explicitly forwarded to a parent
//! token's implementation (or routed to an audited override helper) and a newly
//! added trait method can never silently fall back to a scalar default body
//! (issue #60).
//!
//! Generates four files:
//! - `impls/x86_v4_f32_delegated.rs`: `X64V4Token` / `X64V4xToken` /
//!   `Avx512Fp16Token` W128/W256 `F32x4Backend` + `F32x8Backend` -> `X64V3Token`
//!   (`self.v3()`), with AVX-512VL pixel-packing overrides.
//! - `impls/x86_v4_narrow_delegated.rs`: `X64V4Token` / `X64V4xToken` /
//!   `Avx512Fp16Token` remaining W128/W256 backends + `*Convert` / `*Bitcast`
//!   -> `X64V3Token` (`self.v3()`).
//! - `impls/x86_v4_w512_delegated.rs`: `X64V4xToken` W512 backends +
//!   `F32x16Convert` -> `X64V4Token` (`self.v4()`), eliminating duplicate
//!   `#[arcane]` AVX-512 bodies in `x86_v4.rs`.
//! - `impls/wasm128_fma_delegated.rs`: `Wasm128RelaxedFmaToken` all 43 W128 /
//!   W256 / W512 backends + `*Convert` / `*Bitcast` -> `Wasm128Token`
//!   (`self.wasm128()`), overriding the 24 float `mul_add*` / `mul_sub*`
//!   methods via `wasm128_fma_overrides.rs`.

use std::collections::HashSet;

use indoc::formatdoc;
use quote::{ToTokens, format_ident, quote};
use syn::{FnArg, Item, Pat, TraitItem};

/// Declarative parent-delegation specification for one `(child -> parent)` pair.
struct DelegationRule<'a> {
    /// Concrete child token path implementing the trait (e.g. `"archmage::X64V4xToken"`).
    child_token: &'a str,
    /// Parent token path whose trait impl is forwarded to (e.g. `"archmage::X64V4Token"`).
    parent_token: &'a str,
    /// Expression extracting `parent_token` from `self` (e.g. `"self.v4()"`).
    parent_expr: &'a str,
    /// Expression extracting the proof token expected by `overrides::<helper>` from `self`.
    override_proof_expr: &'a str,
    /// `(trait_name, method_name, override_fn_name)` triples routed to `overrides::<helper>`.
    overrides: &'a [(&'a str, &'a str, &'a str)],
}

/// AVX-512 tokens delegating W128/W256 traits to `X64V3Token`, paired with the
/// expression yielding `X64V4Token` for AVX-512VL pixel-packing overrides.
const V4_TO_V3_TOKENS: &[(&str, &str)] = &[
    ("archmage::X64V4Token", "self"),
    ("archmage::X64V4xToken", "self.v4()"),
    ("archmage::Avx512Fp16Token", "self.v4()"),
];

/// Methods that take the AVX-512VL path in `x86_v4_f32_overrides.rs`:
/// `(trait, method, helper)`.
const V4_F32_OVERRIDES: &[(&str, &str, &str)] = &[
    ("F32x4Backend", "to_u8_bytes", "to_u8_bytes_x4_v4"),
    ("F32x4Backend", "store_rgba_bytes", "store_rgba_bytes_x4_v4"),
    ("F32x8Backend", "to_u8_bytes", "to_u8_bytes_x8_v4"),
];

/// Methods on `Wasm128RelaxedFmaToken` routed to hardware `+relaxed-simd`
/// helpers in `wasm128_fma_overrides.rs` (single-rounding FMA, relaxed min/max,
/// lane-indexed FMA, relaxed float-to-int truncation, and Q15 rounding mul).
const WASM_FMA_OVERRIDES: &[(&str, &str, &str)] = &[
    ("F32x4Backend", "mul_add", "mul_add_f32x4"),
    ("F32x4Backend", "mul_sub", "mul_sub_f32x4"),
    ("F32x4Backend", "mul_add_portable", "mul_add_f32x4"),
    ("F32x4Backend", "mul_sub_portable", "mul_sub_f32x4"),
    ("F32x4Backend", "min_fast", "min_fast_f32x4"),
    ("F32x4Backend", "max_fast", "max_fast_f32x4"),
    ("F32x4Backend", "mul_add_lane_relaxed", "mul_add_lane_f32x4"),
    ("F32x8Backend", "mul_add", "mul_add_f32x8"),
    ("F32x8Backend", "mul_sub", "mul_sub_f32x8"),
    ("F32x8Backend", "mul_add_portable", "mul_add_f32x8"),
    ("F32x8Backend", "mul_sub_portable", "mul_sub_f32x8"),
    ("F32x8Backend", "min_fast", "min_fast_f32x8"),
    ("F32x8Backend", "max_fast", "max_fast_f32x8"),
    ("F32x8Backend", "mul_add_lane_relaxed", "mul_add_lane_f32x8"),
    ("F32x16Backend", "mul_add", "mul_add_f32x16"),
    ("F32x16Backend", "mul_sub", "mul_sub_f32x16"),
    ("F32x16Backend", "mul_add_portable", "mul_add_f32x16"),
    ("F32x16Backend", "mul_sub_portable", "mul_sub_f32x16"),
    ("F32x16Backend", "min_fast", "min_fast_f32x16"),
    ("F32x16Backend", "max_fast", "max_fast_f32x16"),
    ("F64x2Backend", "mul_add", "mul_add_f64x2"),
    ("F64x2Backend", "mul_sub", "mul_sub_f64x2"),
    ("F64x2Backend", "mul_add_portable", "mul_add_f64x2"),
    ("F64x2Backend", "mul_sub_portable", "mul_sub_f64x2"),
    ("F64x2Backend", "min_fast", "min_fast_f64x2"),
    ("F64x2Backend", "max_fast", "max_fast_f64x2"),
    ("F64x4Backend", "mul_add", "mul_add_f64x4"),
    ("F64x4Backend", "mul_sub", "mul_sub_f64x4"),
    ("F64x4Backend", "mul_add_portable", "mul_add_f64x4"),
    ("F64x4Backend", "mul_sub_portable", "mul_sub_f64x4"),
    ("F64x4Backend", "min_fast", "min_fast_f64x4"),
    ("F64x4Backend", "max_fast", "max_fast_f64x4"),
    ("F64x8Backend", "mul_add", "mul_add_f64x8"),
    ("F64x8Backend", "mul_sub", "mul_sub_f64x8"),
    ("F64x8Backend", "mul_add_portable", "mul_add_f64x8"),
    ("F64x8Backend", "mul_sub_portable", "mul_sub_f64x8"),
    ("F64x8Backend", "min_fast", "min_fast_f64x8"),
    ("F64x8Backend", "max_fast", "max_fast_f64x8"),
    (
        "F32x4Convert",
        "convert_f32_to_i32_fast",
        "convert_f32_to_i32_fast_x4",
    ),
    (
        "F32x8Convert",
        "convert_f32_to_i32_fast",
        "convert_f32_to_i32_fast_x8",
    ),
    (
        "F32x16Convert",
        "convert_f32_to_i32_fast",
        "convert_f32_to_i32_fast_x16",
    ),
    ("I16x8Backend", "mul_hrs_q15", "mul_hrs_q15_x8"),
    ("I16x16Backend", "mul_hrs_q15", "mul_hrs_q15_x16"),
    ("I16x32Backend", "mul_hrs_q15", "mul_hrs_q15_x32"),
];

/// Generate `impls/x86_v4_f32_delegated.rs` from the `F32x4Backend` and
/// `F32x8Backend` trait sources.
pub(crate) fn generate(f32x4_trait_src: &str, f32x8_trait_src: &str) -> String {
    let mut body = String::new();
    let mut matched = HashSet::new();
    for (trait_name, src) in [
        ("F32x4Backend", f32x4_trait_src),
        ("F32x8Backend", f32x8_trait_src),
    ] {
        let trait_item = find_trait(src, trait_name);
        for &(child_token, v4_expr) in V4_TO_V3_TOKENS {
            let rule = DelegationRule {
                child_token,
                parent_token: "archmage::X64V3Token",
                parent_expr: "self.v3()",
                override_proof_expr: v4_expr,
                overrides: V4_F32_OVERRIDES,
            };
            body.push_str(&delegating_impl(
                trait_name,
                &trait_item,
                None,
                &rule,
                &mut matched,
            ));
        }
    }
    assert_all_overrides_matched(&matched, V4_F32_OVERRIDES);

    formatdoc! {r#"
        //! W128 / W256 f32 backend delegation for `X64V4Token` / `X64V4xToken` /
        //! `Avx512Fp16Token`. Generated by xtask from the `F32x4Backend` and
        //! `F32x8Backend` definitions; do not edit.
        //!
        //! ## Why this file exists
        //!
        //! The AVX-512 tokens advertise W512 support but have no implementations
        //! of their own for the narrower [`F32x4Backend`] / [`F32x8Backend`]
        //! traits. Without these, generic code parameterized on `T: F32x8Backend`
        //! cannot accept a V4 token, and the cross-width
        //! [`F32x16FromHalves`](crate::simd::generic::F32x16FromHalves) primitive
        //! cannot either, though V4's native `_mm512_insertf32x8` is its best path.
        //!
        //! ## Deductive safety proof
        //!
        //! `X64V3Token`'s feature set is a strict subset of `X64V4Token`'s (the
        //! registry adds AVX-512 F/BW/CD/DQ/VL, PCLMULQDQ and AES on top of V3), so
        //! every instruction that is safe to execute under V3's proof is safe under
        //! V4's. [`F32x4Backend`] / [`F32x8Backend`] for `X64V3Token` use only V3
        //! intrinsics, so forwarding V4's methods to them is sound; `Self::Repr`
        //! stays `__m128` / `__m256`, so callers see no representation change.
        //! Each method forwards with the `.v3()` extractor, so the proof chain is
        //! explicit at every call. `X64V4xToken` and `Avx512Fp16Token` are V4 plus
        //! further AVX-512 sub-features, supersets of V3 by the same argument.
        //!
        //! ## The exception: pixel packing
        //!
        //! `to_u8_bytes` (both widths) and `f32x4::store_rgba_bytes` call the
        //! hand-written AVX-512VL helpers in `x86_v4_f32_overrides.rs`, which take
        //! a concrete `X64V4Token` (`self` for V4, `self.v4()` for V4x and FP16).

        #![cfg(all(target_arch = "x86_64", feature = "avx512"))]

        use crate::simd::backends::{{F32x4Backend, F32x8Backend}};
        use super::x86_v4_f32_overrides as overrides;

        {body}
    "#}
}

/// `impls/x86_v4_narrow_delegated.rs`: every other W128/W256 backend the
/// AVX-512 tokens lack — f64x2/f64x4, the i8/i16/i32/i64 and
/// u8/u16/u32/u64 families — plus the float↔int `*Convert` and `*Bitcast`
/// traits. Same delegation proof as the f32 file; no overrides — every
/// method forwards to `X64V3Token`.
pub(crate) fn generate_narrow(specs: &[(String, String)]) -> String {
    let mut body = String::new();
    let mut matched = HashSet::new();
    for (trait_name, src) in specs {
        let trait_item = find_trait(src, trait_name);
        for &(child_token, v4_expr) in V4_TO_V3_TOKENS {
            let rule = DelegationRule {
                child_token,
                parent_token: "archmage::X64V3Token",
                parent_expr: "self.v3()",
                override_proof_expr: v4_expr,
                overrides: &[],
            };
            body.push_str(&delegating_impl(
                trait_name,
                &trait_item,
                None,
                &rule,
                &mut matched,
            ));
        }
    }

    formatdoc! {r#"
        //! W128 / W256 f64 and integer backends plus the float↔int
        //! conversion traits for `X64V4Token` / `X64V4xToken` /
        //! `Avx512Fp16Token`, delegated to `X64V3Token`'s. Generated by
        //! xtask from the backend trait definitions; do not edit.
        //!
        //! ## Why this file exists
        //!
        //! The AVX-512 tokens implement the W512 backends natively and
        //! delegate the W128/W256 f32 backends (`x86_v4_f32_delegated.rs`),
        //! but the matching f64 and integer traits — and the `*Convert` /
        //! `*Bitcast` traits built on them — were missing, so generic code
        //! parameterized on (say) `T: F64x4Backend` or `T: F32x8Convert`
        //! could not accept a V4 token at all.
        //!
        //! ## Deductive safety proof
        //!
        //! `X64V3Token`'s feature set is a strict subset of `X64V4Token`'s,
        //! so every instruction the V3 impls execute is safe under V4's
        //! proof; `Self::Repr` stays the V3 representation, so callers see
        //! no change. `X64V4xToken` / `Avx512Fp16Token` are V4 supersets.
        //! The `*Convert` / `*Bitcast` impls forward to the V3 impls of the
        //! same traits, whose `Repr` types are likewise unchanged.
        //!
        //! Popcnt extension backends are NOT delegated: they exist only
        //! for `X64V4xToken` (native AVX-512 VPOPCNTDQ) and have no V3
        //! impls to forward to.

        #![cfg(all(target_arch = "x86_64", feature = "avx512"))]

        // Wildcard: the impls' signatures name supertraits of the
        // implemented traits (`<Self as F32x8Backend>::Repr` inside a
        // `F32x8Convert` impl, `<Self as U8x16Backend>::Repr` inside an
        // `I8x16Bitcast` impl, …) — every sibling backend trait must be
        // in scope, not just the implemented set.
        use crate::simd::backends::*;

        {body}
    "#}
}

/// `impls/x86_v4_w512_delegated.rs`: `X64V4xToken`'s 10 W512 `*Backend`
/// traits plus `F32x16Convert`, forwarded to `X64V4Token` via `self.v4()`.
///
/// `X64V4xToken`'s Modern-only `*PopcntBackend` extension traits remain in
/// `impls/x86_v4.rs`.
pub(crate) fn generate_v4x_w512(w512_specs: &[(String, String)]) -> String {
    let mut body = String::new();
    let mut matched = HashSet::new();
    let rule = DelegationRule {
        child_token: "archmage::X64V4xToken",
        parent_token: "archmage::X64V4Token",
        parent_expr: "self.v4()",
        override_proof_expr: "self",
        overrides: &[],
    };
    for (trait_name, src) in w512_specs {
        let trait_item = find_trait(src, trait_name);
        body.push_str(&delegating_impl(
            trait_name,
            &trait_item,
            None,
            &rule,
            &mut matched,
        ));
    }

    formatdoc! {r#"
        //! W512 backend and conversion delegation for `X64V4xToken` -> `X64V4Token`.
        //! Generated by xtask from the backend trait definitions; do not edit.
        //!
        //! ## Why this file exists
        //!
        //! Base W512 vector operations (`F32x16Backend`, `F64x8Backend`, the eight
        //! 512-bit integer backends, and `F32x16Convert`) emit identical AVX-512
        //! F/BW/CD/DQ/VL instructions on `X64V4Token` and `X64V4xToken`. Forwarding
        //! `X64V4xToken` to `X64V4Token` via `self.v4()` avoids ~250 duplicate
        //! `#[arcane]` proc-macro expansions in `x86_v4.rs` while keeping
        //! `X64V4xToken`'s native VPOPCNTDQ/BITALG `*PopcntBackend` impls in
        //! `x86_v4.rs`.

        #![cfg(all(target_arch = "x86_64", feature = "avx512", feature = "w512"))]

        use crate::simd::backends::*;

        {body}
    "#}
}

/// `impls/wasm128_fma_delegated.rs`: `Wasm128RelaxedFmaToken` backend and
/// conversion delegation to `Wasm128Token` (`self.wasm128()`), overriding the
/// 24 float `mul_add*` / `mul_sub*` methods via `wasm128_fma_overrides.rs`.
pub(crate) fn generate_wasm_fma(
    narrow_specs: &[(String, String)],
    w512_specs: &[(String, String)],
) -> String {
    let mut body = String::new();
    let mut matched = HashSet::new();
    let rule = DelegationRule {
        child_token: "archmage::Wasm128RelaxedFmaToken",
        parent_token: "archmage::Wasm128Token",
        parent_expr: "self.wasm128()",
        override_proof_expr: "self",
        overrides: WASM_FMA_OVERRIDES,
    };
    for (trait_name, src) in narrow_specs {
        let trait_item = find_trait(src, trait_name);
        body.push_str(&delegating_impl(
            trait_name,
            &trait_item,
            None,
            &rule,
            &mut matched,
        ));
    }
    for (trait_name, src) in w512_specs {
        let trait_item = find_trait(src, trait_name);
        body.push_str(&delegating_impl(
            trait_name,
            &trait_item,
            Some("feature = \"w512\""),
            &rule,
            &mut matched,
        ));
    }
    assert_all_overrides_matched(&matched, WASM_FMA_OVERRIDES);

    formatdoc! {r#"
        //! Backend and conversion delegation for [`archmage::Wasm128RelaxedFmaToken`],
        //! forwarded to [`archmage::Wasm128Token`] (`self.wasm128()`) except for the
        //! 24 float `mul_add` / `mul_sub` / `mul_add_portable` / `mul_sub_portable`
        //! methods routed to `wasm128_fma_overrides.rs`. Generated by xtask from the
        //! backend trait definitions; do not edit.
        //!
        //! ## Deductive safety proof
        //!
        //! `Wasm128RelaxedFmaToken` proves `+simd128,+relaxed-simd` plus runtime-probed
        //! single-rounding fusion of all lanes of `f32x4.relaxed_madd` and
        //! `f64x2.relaxed_madd`. Every `Wasm128Token` instruction (`+simd128`) is a
        //! strict subset of that capability, and `Self::Repr` is identical (`v128` /
        //! `[v128; 2]` / `[v128; 4]`). The 24 overridden float FMA methods pass `self`
        //! directly to `wasm128_fma_overrides`, eliminating per-op `AtomicU8` probe
        //! loads and software-FMA fallback branches.

        #![cfg(all(
            target_arch = "wasm32",
            target_feature = "simd128",
            target_feature = "relaxed-simd"
        ))]

        use crate::simd::backends::*;
        use super::wasm128_fma_overrides as overrides;

        {body}
    "#}
}

fn assert_all_overrides_matched(
    matched: &HashSet<(String, String)>,
    overrides: &[(&str, &str, &str)],
) {
    for &(trait_name, method_name, helper) in overrides {
        assert!(
            matched.contains(&(trait_name.to_string(), method_name.to_string())),
            "delegation override ({trait_name}::{method_name} -> {helper}) matched no trait method"
        );
    }
}

fn find_trait(src: &str, name: &str) -> syn::ItemTrait {
    let file = syn::parse_file(src).expect("generated backend trait source must parse");
    file.items
        .into_iter()
        .find_map(|item| match item {
            Item::Trait(t) if t.ident == name => Some(t),
            _ => None,
        })
        .unwrap_or_else(|| panic!("trait {name} not found in generated source"))
}

fn delegating_impl(
    trait_name: &str,
    trait_item: &syn::ItemTrait,
    item_cfg: Option<&str>,
    rule: &DelegationRule<'_>,
    matched_overrides: &mut HashSet<(String, String)>,
) -> String {
    let trait_ident = format_ident!("{trait_name}");
    let child_path: syn::Path = syn::parse_str(rule.child_token).expect("child token path");
    let parent_path: syn::Path = syn::parse_str(rule.parent_token).expect("parent token path");
    let parent_expr: syn::Expr = syn::parse_str(rule.parent_expr).expect("parent expression");
    let override_proof: syn::Expr =
        syn::parse_str(rule.override_proof_expr).expect("override proof expression");
    let mut methods = Vec::new();
    for item in &trait_item.items {
        let TraitItem::Fn(method) = item else {
            continue;
        };
        // `convert_int.rs` qualifies supertraits with `super::` (its own
        // module is `backends`); copied into `impls/` that path would
        // resolve to `crate::simd::impls` — retarget to the trait home.
        let sig: syn::Signature = syn::parse_str(
            &method
                .sig
                .to_token_stream()
                .to_string()
                .replace("super ::", "crate :: simd :: backends ::"),
        )
        .expect("method signature re-parse");
        let sig = &sig;
        let name = &sig.ident;
        assert!(
            matches!(sig.inputs.first(), Some(FnArg::Receiver(_))),
            "{trait_name}::{name} has no receiver; the delegation needs `self`"
        );
        let args: Vec<syn::Ident> = sig
            .inputs
            .iter()
            .skip(1)
            .map(|arg| match arg {
                FnArg::Typed(pat) => match pat.pat.as_ref() {
                    Pat::Ident(p) => p.ident.clone(),
                    other => panic!(
                        "{trait_name}::{name}: parameter pattern `{}` is not an identifier",
                        other.to_token_stream()
                    ),
                },
                FnArg::Receiver(_) => unreachable!("receiver after the first parameter"),
            })
            .collect();
        let turbofish = {
            let params: Vec<_> = sig
                .generics
                .params
                .iter()
                .map(|p| match p {
                    syn::GenericParam::Type(t) => t.ident.to_token_stream(),
                    syn::GenericParam::Const(c) => c.ident.to_token_stream(),
                    syn::GenericParam::Lifetime(l) => l.lifetime.to_token_stream(),
                })
                .collect();
            if params.is_empty() {
                quote! {}
            } else {
                quote! { ::<#(#params),*> }
            }
        };
        let override_helper = rule
            .overrides
            .iter()
            .find(|(t, m, _)| *t == trait_name && name == m)
            .map(|(_, _, helper)| {
                matched_overrides.insert((trait_name.to_string(), name.to_string()));
                format_ident!("{helper}")
            });
        let body = match override_helper {
            Some(helper) => quote! { overrides::#helper #turbofish(#override_proof, #(#args),*) },
            None => {
                quote! { <#parent_path as #trait_ident>::#name #turbofish(#parent_expr, #(#args),*) }
            }
        };
        methods.push(quote! {
            #[inline(always)]
            #sig { #body }
        });
    }
    // `type Repr` exists on the backend traits but not on the
    // float↔int conversion traits (whose methods name supertrait Reprs
    // in signature position) — emit it only when the trait declares it.
    let has_repr = trait_item
        .items
        .iter()
        .any(|item| matches!(item, TraitItem::Type(_)));
    let repr_impl = has_repr.then(|| {
        quote! { type Repr = <#parent_path as #trait_ident>::Repr; }
    });
    let cfg_attr = item_cfg.map(|cfg_str| {
        let cfg_meta: syn::Meta = syn::parse_str(cfg_str).expect("valid item cfg");
        quote! { #[cfg(#cfg_meta)] }
    });
    let tokens = quote! {
        #cfg_attr
        impl #trait_ident for #child_path {
            #repr_impl
            #(#methods)*
        }
    };
    format!("{tokens}\n\n")
}
