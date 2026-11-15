//! Unified canonical-backend code generation (`issue #175`).
//!
//! Collapses all per-token backend delegation into:
//! - `backends/ops.rs`: crate-private `*Ops` mirror traits implemented only by
//!   the canonical base tokens (`ScalarToken`, `X64V3Token`, `X64V4Token`,
//!   `NeonToken`, `Wasm128Token`), plus `NarrowOps` and `W512Ops`.
//! - `backends/sealed.rs`: `Sealed` supertrait mapping all 21 tokens (all 20
//!   registry tokens on their native architectures + `ScalarToken`) to their
//!   `Narrow` / `W512` canonical bases and hosting override hooks for AVX-512VL
//!   pixel packing, WASM `relaxed-simd` FMA/min/max/cvt/q15/dot, fast
//!   `to_u32_fast`, `dot_i8_i7_add`, `F32x16FromHalves`, and `F16Convert`.
//! - `impls/delegated.rs`: single blanket `impl<T: SimdToken + Sealed + Copy + 'static>`
//!   blocks for all 30 `*Backend`, 3 `*Convert`, and 10 `*Bitcast` traits.

use std::collections::{BTreeMap, HashSet};

use indoc::formatdoc;
use proc_macro2::{Group, Ident, Spacing, TokenStream, TokenTree};
use quote::{ToTokens, format_ident, quote};
use syn::{FnArg, Item, Pat, TraitItem};

/// Exact list of the 43 public backend/convert/bitcast trait names unified
/// through `*Ops` + `Sealed` + blanket impls (`PopcntBackend` stays on `X64V4xToken`).
pub(crate) fn is_unified_trait(name: &str) -> bool {
    matches!(
        name,
        "F32x4Backend"
            | "F32x8Backend"
            | "F32x16Backend"
            | "F64x2Backend"
            | "F64x4Backend"
            | "F64x8Backend"
            | "I8x16Backend"
            | "I8x32Backend"
            | "I8x64Backend"
            | "U8x16Backend"
            | "U8x32Backend"
            | "U8x64Backend"
            | "I16x8Backend"
            | "I16x16Backend"
            | "I16x32Backend"
            | "U16x8Backend"
            | "U16x16Backend"
            | "U16x32Backend"
            | "I32x4Backend"
            | "I32x8Backend"
            | "I32x16Backend"
            | "U32x4Backend"
            | "U32x8Backend"
            | "U32x16Backend"
            | "I64x2Backend"
            | "I64x4Backend"
            | "I64x8Backend"
            | "U64x2Backend"
            | "U64x4Backend"
            | "U64x8Backend"
            | "F32x4Convert"
            | "F32x8Convert"
            | "F32x16Convert"
            | "F64x2Convert"
            | "F64x4Convert"
            | "U32x4Bitcast"
            | "U32x8Bitcast"
            | "I64x2Bitcast"
            | "I64x4Bitcast"
            | "I8x16Bitcast"
            | "I8x32Bitcast"
            | "I16x8Bitcast"
            | "I16x16Bitcast"
            | "U64x2Bitcast"
            | "U64x4Bitcast"
    )
}

/// Map a public backend/convert/bitcast trait name to its internal `*Ops` counterpart.
pub(crate) fn ops_trait_name(public_name: &str) -> String {
    if let Some(stem) = public_name.strip_suffix("Backend") {
        format!("{stem}Ops")
    } else if public_name.ends_with("Convert") || public_name.ends_with("Bitcast") {
        format!("{public_name}Ops")
    } else {
        panic!("unexpected unified trait name: {public_name}");
    }
}

/// AVX-512VL pixel-packing overrides: `(trait_name, method_name, override_fn)`.
const V4_F32_OVERRIDES: &[(&str, &str, &str)] = &[
    ("F32x4Backend", "to_u8_bytes", "to_u8_bytes_x4_v4"),
    ("F32x4Backend", "store_rgba_bytes", "store_rgba_bytes_x4_v4"),
    ("F32x4Backend", "store_rgb_bytes", "store_rgb_bytes_x4_v4"),
    ("F32x8Backend", "to_u8_bytes", "to_u8_bytes_x8_v4"),
];

/// WASM `relaxed-simd` overrides: `(trait_name, method_name, override_fn)`.
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

/// Generate `(backends/ops.rs, backends/sealed.rs, impls/delegated.rs)` from
/// `(trait_name, trait_file_src, is_w512)` tuples for all 45 unified traits.
pub(crate) fn generate_ops_and_sealed_and_delegated(
    specs: &[(String, String, bool)],
) -> (String, String, String) {
    assert_eq!(specs.len(), 45, "expected all 45 unified traits");

    let mut parsed_files: BTreeMap<String, syn::File> = BTreeMap::new();
    for (_, src, _) in specs {
        parsed_files
            .entry(src.clone())
            .or_insert_with(|| syn::parse_file(src).expect("valid generated backend trait file"));
    }

    let traits: Vec<(String, syn::ItemTrait, bool)> = specs
        .iter()
        .map(|(name, src, is_w512)| {
            let tr = find_trait(&parsed_files[src], name);
            (name.clone(), tr, *is_w512)
        })
        .collect();

    let ops_rs = generate_ops_rs(&traits);
    let (sealed_rs, v4_matched, wasm_matched) = generate_sealed_rs(&traits);
    assert_all_overrides_matched(&v4_matched, V4_F32_OVERRIDES);
    assert_all_overrides_matched(&wasm_matched, WASM_FMA_OVERRIDES);
    let delegated_rs = generate_delegated_rs(&traits);

    (ops_rs, sealed_rs, delegated_rs)
}

/// Convert a concrete base implementation file (`scalar.rs`, `x86_v3.rs`,
/// `x86_v4.rs`, `arm_neon.rs`, `wasm128.rs`) from public `*Backend`/`*Convert`/
/// `*Bitcast` traits to crate-private `*Ops` traits while leaving `*PopcntBackend`
/// on `X64V4xToken` untouched.
pub(crate) fn to_ops_impls(src: &str) -> String {
    let keep_popcnt = src.contains("PopcntBackend");
    let mut out = String::with_capacity(src.len());
    for line in src.split_inclusive('\n') {
        if let Some(use_idx) = line.find("use crate::simd::backends::*;") {
            let prefix = &line[..use_idx];
            if prefix.trim_start().is_empty() {
                if keep_popcnt {
                    out.push_str(prefix);
                    out.push_str("use crate::simd::backends::*;\n");
                }
                out.push_str(prefix);
                out.push_str("use crate::simd::backends::ops::*;\n");
                continue;
            }
        }
        // Split at line comments so doc comments & safety justifications are untouched.
        let (code, comment) = match line.find("//") {
            Some(pos) => (&line[..pos], &line[pos..]),
            None => (line, ""),
        };
        replace_unified_idents(code, &mut out);
        out.push_str(comment);
    }
    out
}

fn replace_unified_idents(code: &str, out: &mut String) {
    let bytes = code.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if is_ident_start(bytes[i]) {
            let start = i;
            i += 1;
            while i < bytes.len() && is_ident_continue(bytes[i]) {
                i += 1;
            }
            let ident = &code[start..i];
            if is_unified_trait(ident) {
                out.push_str(&ops_trait_name(ident));
            } else {
                out.push_str(ident);
            }
        } else {
            let ch = code[i..].chars().next().unwrap();
            out.push(ch);
            i += ch.len_utf8();
        }
    }
}

fn is_ident_start(b: u8) -> bool {
    b.is_ascii_alphabetic() || b == b'_'
}

fn is_ident_continue(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b == b'_'
}

fn find_trait(file: &syn::File, name: &str) -> syn::ItemTrait {
    for item in &file.items {
        if let Item::Trait(tr) = item
            && tr.ident == name
        {
            return tr.clone();
        }
    }
    panic!("trait `{name}` not found in parsed backend file");
}

fn rewrite_unified_stream(tokens: TokenStream, map_to_ops: bool) -> TokenStream {
    let tts: Vec<TokenTree> = tokens.into_iter().collect();
    let mut out: Vec<TokenTree> = Vec::with_capacity(tts.len());
    let mut i = 0;
    while i < tts.len() {
        if i + 3 < tts.len()
            && let TokenTree::Ident(sup) = &tts[i]
            && sup == "super"
            && let TokenTree::Punct(p1) = &tts[i + 1]
            && p1.as_char() == ':'
            && p1.spacing() == Spacing::Joint
            && let TokenTree::Punct(p2) = &tts[i + 2]
            && p2.as_char() == ':'
            && let TokenTree::Ident(id) = &tts[i + 3]
            && is_unified_trait(&id.to_string())
        {
            i += 3;
            continue;
        }
        match &tts[i] {
            TokenTree::Group(g) => {
                let mut g2 = Group::new(
                    g.delimiter(),
                    rewrite_unified_stream(g.stream(), map_to_ops),
                );
                g2.set_span(g.span());
                out.push(TokenTree::Group(g2));
            }
            TokenTree::Ident(id) if map_to_ops && is_unified_trait(&id.to_string()) => {
                out.push(TokenTree::Ident(Ident::new(
                    &ops_trait_name(&id.to_string()),
                    id.span(),
                )));
            }
            other => out.push(other.clone()),
        }
        i += 1;
    }
    out.into_iter().collect()
}

fn map_tokens_to_ops(tokens: TokenStream) -> TokenStream {
    rewrite_unified_stream(tokens, true)
}

fn strip_super_on_unified(tokens: TokenStream) -> TokenStream {
    rewrite_unified_stream(tokens, false)
}

fn generate_ops_rs(traits: &[(String, syn::ItemTrait, bool)]) -> String {
    let mut blocks = Vec::new();
    let mut narrow_ops_bounds = Vec::new();
    let mut w512_ops_bounds = Vec::new();

    for (name, tr, is_w512) in traits {
        let ops_name = ops_trait_name(name);
        let ops_ident = format_ident!("{ops_name}");
        if *is_w512 {
            w512_ops_bounds.push(quote!(#ops_ident));
        } else {
            narrow_ops_bounds.push(quote!(#ops_ident));
        }

        let mut super_tokens: Vec<TokenStream> = Vec::new();
        for bound in &tr.supertraits {
            match bound {
                syn::TypeParamBound::Lifetime(lt) => super_tokens.push(quote!(#lt)),
                syn::TypeParamBound::Trait(tb) => {
                    let last = tb.path.segments.last().map(|s| s.ident.to_string());
                    match last.as_deref() {
                        Some("SimdToken") | Some("Sealed") | Some("Copy") => {}
                        Some(other) if is_unified_trait(other) => {
                            let mapped = format_ident!("{}", ops_trait_name(other));
                            super_tokens.push(quote!(#mapped));
                        }
                        _ => super_tokens.push(quote!(#tb)),
                    }
                }
                other => super_tokens.push(quote!(#other)),
            }
        }
        super_tokens.push(quote!(archmage::SimdToken));
        super_tokens.push(quote!(Copy));

        let mut item_streams: Vec<TokenStream> = Vec::new();
        for item in &tr.items {
            match item {
                TraitItem::Type(ty) => {
                    let attrs: Vec<_> = ty
                        .attrs
                        .iter()
                        .filter(|a| !a.path().is_ident("doc"))
                        .collect();
                    let ident = &ty.ident;
                    let bounds = &ty.bounds;
                    item_streams.push(quote! { #(#attrs)* type #ident: #bounds; });
                }
                TraitItem::Fn(m) => {
                    let attrs: Vec<_> = m
                        .attrs
                        .iter()
                        .filter(|a| !a.path().is_ident("doc") && !a.path().is_ident("inline"))
                        .collect();
                    let sig_ts = map_tokens_to_ops(m.sig.to_token_stream());
                    if let Some(default) = &m.default {
                        let default_ts = map_tokens_to_ops(default.to_token_stream());
                        item_streams.push(quote! {
                            #(#attrs)*
                            #[inline(always)]
                            #sig_ts #default_ts
                        });
                    } else {
                        item_streams.push(quote! {
                            #(#attrs)*
                            #sig_ts;
                        });
                    }
                }
                _ => panic!("unexpected trait item in {name}"),
            }
        }

        let cfg_attr = if *is_w512 {
            quote!(#[cfg(feature = "w512")])
        } else {
            quote!()
        };

        blocks.push(
            quote! {
                #cfg_attr
                pub trait #ops_ident: #(#super_tokens)+* {
                    #(#item_streams)*
                }
            }
            .to_string(),
        );
    }

    let narrow_bundle = quote! {
        pub trait NarrowOps:
            #(#narrow_ops_bounds)+*
            + crate::simd::generic::cross_width::NarrowCrossWidthOps
            + crate::simd::generic::fast_ops::NarrowToU32Ops
            + crate::simd::generic::quant_ops::NarrowDotI8Ops
            + archmage::SimdToken
            + Copy
            + 'static
        {}
        impl<T> NarrowOps for T where
            T: #(#narrow_ops_bounds)+*
                + crate::simd::generic::cross_width::NarrowCrossWidthOps
                + crate::simd::generic::fast_ops::NarrowToU32Ops
                + crate::simd::generic::quant_ops::NarrowDotI8Ops
                + archmage::SimdToken
                + Copy
                + 'static
        {}
    }
    .to_string();

    let w512_bundle = quote! {
        #[cfg(feature = "w512")]
        pub trait W512Ops:
            #(#w512_ops_bounds)+*
            + crate::simd::generic::fast_ops::W512ToU32Ops
            + crate::simd::generic::quant_ops::W512DotI8Ops
            + archmage::SimdToken
            + Copy
            + 'static
        {}
        #[cfg(feature = "w512")]
        impl<T> W512Ops for T where
            T: #(#w512_ops_bounds)+*
                + crate::simd::generic::fast_ops::W512ToU32Ops
                + crate::simd::generic::quant_ops::W512DotI8Ops
                + archmage::SimdToken
                + Copy
                + 'static
        {}
    }
    .to_string();

    let body = blocks.join("\n\n");
    formatdoc! {r#"
        //! Crate-private canonical backend operation traits.
        //!
        //! Implemented only by the five canonical base tokens (`ScalarToken`,
        //! `X64V3Token`, `X64V4Token`, `NeonToken`, `Wasm128Token`). Public
        //! `*Backend` / `*Convert` / `*Bitcast` traits delegate to these via
        //! blanket impls over `SimdToken + Sealed`.
        //!
        //! **Auto-generated** by `cargo xtask generate` - do not edit manually.

        #![allow(dead_code)]

        {narrow_bundle}

        {w512_bundle}

        {body}
    "#}
}

/// Rewrite a trait method signature for use as a hook method on `Sealed`.
/// Replaces `Self::Repr` with `<Self::{proj} as ops::{ops_trait}>::Repr`
/// and `<Self as {OtherTrait}>` with `<Self::{proj} as ops::{OtherTraitOps}>`.
fn map_sig_for_sealed(tokens: TokenStream, proj: &Ident, default_ops: &Ident) -> TokenStream {
    let tts: Vec<TokenTree> = tokens.into_iter().collect();
    let mut out = Vec::with_capacity(tts.len());
    let mut i = 0;
    while i < tts.len() {
        // Detect `Self :: Repr`
        if i + 2 < tts.len()
            && let TokenTree::Ident(id) = &tts[i]
            && id == "Self"
            && let TokenTree::Punct(p1) = &tts[i + 1]
            && p1.as_char() == ':'
            && p1.spacing() == Spacing::Joint
            && let TokenTree::Punct(p2) = &tts[i + 2]
            && p2.as_char() == ':'
            && i + 3 < tts.len()
            && let TokenTree::Ident(rhs) = &tts[i + 3]
            && rhs == "Repr"
        {
            let span = id.span();
            let q = quote::quote_spanned!(span => <Self::#proj as crate::simd::backends::ops::#default_ops>::Repr);
            out.extend(q);
            i += 4;
            continue;
        }
        // Detect `< Self as TraitName >` (inside `TokenStream` directly)
        if let TokenTree::Punct(lt) = &tts[i]
            && lt.as_char() == '<'
            && i + 4 < tts.len()
            && let TokenTree::Ident(self_id) = &tts[i + 1]
            && self_id == "Self"
            && let TokenTree::Ident(as_id) = &tts[i + 2]
            && as_id == "as"
            && let TokenTree::Ident(tr_id) = &tts[i + 3]
            && is_unified_trait(&tr_id.to_string())
            && let TokenTree::Punct(gt) = &tts[i + 4]
            && gt.as_char() == '>'
        {
            let mapped = format_ident!("{}", ops_trait_name(&tr_id.to_string()));
            let q = quote! { <Self::#proj as crate::simd::backends::ops::#mapped> };
            out.extend(q);
            i += 5;
            continue;
        }
        match &tts[i] {
            TokenTree::Group(g) => {
                let mut g2 = Group::new(
                    g.delimiter(),
                    map_sig_for_sealed(g.stream(), proj, default_ops),
                );
                g2.set_span(g.span());
                out.push(TokenTree::Group(g2));
            }
            other => out.push(other.clone()),
        }
        i += 1;
    }
    out.into_iter().collect()
}

fn hook_method_sig_and_call(
    m: &syn::TraitItemFn,
    hook_name: &str,
    proj: &Ident,
    default_ops: &Ident,
) -> (TokenStream, TokenStream) {
    let mut sig = m.sig.clone();
    sig.ident = format_ident!("{hook_name}");
    let sig_tokens = map_sig_for_sealed(sig.to_token_stream(), proj, default_ops);

    let call_args: Vec<TokenStream> = m
        .sig
        .inputs
        .iter()
        .filter_map(|arg| match arg {
            FnArg::Receiver(_) => None,
            FnArg::Typed(pat_ty) => match &*pat_ty.pat {
                Pat::Ident(pi) => {
                    let id = &pi.ident;
                    Some(quote!(#id))
                }
                _ => panic!("expected simple ident arg in `{}`", m.sig.ident),
            },
        })
        .collect();
    let generic_args = generic_turbofish(&m.sig.generics);
    let hook_ident = format_ident!("{hook_name}");
    let call_expr = quote! { self.#hook_ident #generic_args (#(#call_args),*) };
    (sig_tokens, call_expr)
}

fn generic_turbofish(generics: &syn::Generics) -> TokenStream {
    let params: Vec<TokenStream> = generics
        .params
        .iter()
        .filter_map(|p| match p {
            syn::GenericParam::Const(c) => {
                let id = &c.ident;
                Some(quote!(#id))
            }
            syn::GenericParam::Type(t) => {
                let id = &t.ident;
                Some(quote!(#id))
            }
            syn::GenericParam::Lifetime(_) => None,
        })
        .collect();
    if params.is_empty() {
        quote!()
    } else {
        quote!(::<#(#params),*>)
    }
}

fn v4_hook_ident(trait_name: &str, method_name: &str) -> String {
    let tag = match trait_name {
        "F32x4Backend" => "f32x4",
        "F32x8Backend" => "f32x8",
        _ => panic!("unexpected V4 override trait `{trait_name}`"),
    };
    format!("v4_{tag}_{method_name}")
}

fn wasm_hook_ident(trait_name: &str, method_name: &str) -> String {
    let tag = match trait_name {
        "F32x4Backend" => "f32x4",
        "F32x8Backend" => "f32x8",
        "F32x16Backend" => "f32x16",
        "F64x2Backend" => "f64x2",
        "F64x4Backend" => "f64x4",
        "F64x8Backend" => "f64x8",
        "F32x4Convert" => "f32x4_cvt",
        "F32x8Convert" => "f32x8_cvt",
        "F32x16Convert" => "f32x16_cvt",
        "I16x8Backend" => "i16x8",
        "I16x16Backend" => "i16x16",
        "I16x32Backend" => "i16x32",
        _ => panic!("unexpected WASM FMA override trait `{trait_name}`"),
    };
    format!("wasm_fma_{tag}_{method_name}")
}

#[allow(clippy::type_complexity)]
fn generate_sealed_rs(
    traits: &[(String, syn::ItemTrait, bool)],
) -> (String, HashSet<(String, String)>, HashSet<(String, String)>) {
    let mut v4_matched = HashSet::new();
    let mut wasm_matched = HashSet::new();

    let mut trait_hooks: Vec<TokenStream> = Vec::new();
    let mut v4_override_self_methods: Vec<TokenStream> = Vec::new();
    let mut v4_override_v4_methods: Vec<TokenStream> = Vec::new();
    let mut wasm_override_methods: Vec<TokenStream> = Vec::new();

    let trait_map: BTreeMap<&str, (&syn::ItemTrait, bool)> = traits
        .iter()
        .map(|(n, tr, w)| (n.as_str(), (tr, *w)))
        .collect();

    for &(tname, mname, helper) in V4_F32_OVERRIDES {
        let (tr, is_w512) = trait_map[tname];
        let m = find_fn(tr, mname);
        v4_matched.insert((tname.to_string(), mname.to_string()));
        let proj = format_ident!("{}", if is_w512 { "W512" } else { "Narrow" });
        let proj_fn = format_ident!("{}", if is_w512 { "w512" } else { "narrow" });
        let ops_id = format_ident!("{}", ops_trait_name(tname));
        let hook = v4_hook_ident(tname, mname);
        let (sig_ts, _) = hook_method_sig_and_call(m, &hook, &proj, &ops_id);
        let args = extract_arg_idents(&m.sig);
        let orig_fn = &m.sig.ident;
        let tf = generic_turbofish(&m.sig.generics);
        let helper_ident = format_ident!("{helper}");

        trait_hooks.push(quote! {
            #[inline(always)]
            #sig_ts {
                <Self::#proj as crate::simd::backends::ops::#ops_id>::#orig_fn #tf (self.#proj_fn(), #(#args),*)
            }
        });
        v4_override_self_methods.push(quote! {
            #[inline(always)]
            #sig_ts {
                crate::simd::impls::x86_v4_f32_overrides::#helper_ident #tf (self, #(#args),*)
            }
        });
        v4_override_v4_methods.push(quote! {
            #[inline(always)]
            #sig_ts {
                crate::simd::impls::x86_v4_f32_overrides::#helper_ident #tf (self.v4(), #(#args),*)
            }
        });
    }

    for &(tname, mname, helper) in WASM_FMA_OVERRIDES {
        let (tr, is_w512) = trait_map[tname];
        let m = find_fn(tr, mname);
        wasm_matched.insert((tname.to_string(), mname.to_string()));
        let proj = format_ident!("{}", if is_w512 { "W512" } else { "Narrow" });
        let proj_fn = format_ident!("{}", if is_w512 { "w512" } else { "narrow" });
        let ops_id = format_ident!("{}", ops_trait_name(tname));
        let hook = wasm_hook_ident(tname, mname);
        let (sig_ts, _) = hook_method_sig_and_call(m, &hook, &proj, &ops_id);
        let args = extract_arg_idents(&m.sig);
        let orig_fn = &m.sig.ident;
        let tf = generic_turbofish(&m.sig.generics);
        let helper_ident = format_ident!("{helper}");
        let cfg_w512 = if is_w512 {
            quote!(#[cfg(feature = "w512")])
        } else {
            quote!()
        };

        trait_hooks.push(quote! {
            #cfg_w512
            #[inline(always)]
            #sig_ts {
                <Self::#proj as crate::simd::backends::ops::#ops_id>::#orig_fn #tf (self.#proj_fn(), #(#args),*)
            }
        });
        wasm_override_methods.push(quote! {
            #cfg_w512
            #[inline(always)]
            #sig_ts {
                crate::simd::impls::wasm128_fma_overrides::#helper_ident #tf (self, #(#args),*)
            }
        });
    }

    let trait_hooks_str: String = trait_hooks.into_iter().map(|t| t.to_string()).collect();
    let v4_overrides_self_str: String = v4_override_self_methods
        .into_iter()
        .map(|t| t.to_string())
        .collect();
    let v4_overrides_v4_str: String = v4_override_v4_methods
        .into_iter()
        .map(|t| t.to_string())
        .collect();
    let wasm_overrides_str: String = wasm_override_methods
        .into_iter()
        .map(|t| t.to_string())
        .collect();

    // Token list and concrete Sealed impl blocks.
    let impls = build_sealed_token_impls(
        &v4_overrides_self_str,
        &v4_overrides_v4_str,
        &wasm_overrides_str,
    );

    let out = formatdoc! {r#"
        //! Marker trait sealing all SIMD backend traits to validated archmage tokens
        //! and routing all 21 tokens through their canonical base `*Ops` traits.
        //!
        //! **Auto-generated** by `cargo xtask generate` - do not edit manually.

        #[allow(unused_imports)]
        use archmage::SimdToken as _;

        /// Seals backend traits so only validated archmage tokens can implement them.
        ///
        /// Also supplies the canonical `Narrow` / `W512` projections and optional
        /// hardware override hooks (AVX-512VL pixel packing, WASM `relaxed-simd`
        /// single-rounding ops, AVX-512F `to_u32_fast` / `dot_i8` / `f16` slice,
        /// `F32x16FromHalves`).
        pub trait Sealed: Copy + 'static {{
            /// Canonical 128/256-bit backend token implementing [`super::ops::NarrowOps`].
            type Narrow: super::ops::NarrowOps;

            /// Canonical 512-bit backend token implementing [`super::ops::W512Ops`].
            #[cfg(feature = "w512")]
            type W512: super::ops::W512Ops;

            /// Project `self` to its canonical narrow backend proof.
            fn narrow(self) -> Self::Narrow;

            /// Project `self` to its canonical 512-bit backend proof.
            #[cfg(feature = "w512")]
            fn w512(self) -> Self::W512;

            {trait_hooks_str}

            // Extension-trait override hooks (used by `simd::generic::*` blanket impls).

            #[inline(always)]
            fn f32x8_from_halves(
                self,
                lo: <Self::Narrow as super::ops::F32x4Ops>::Repr,
                hi: <Self::Narrow as super::ops::F32x4Ops>::Repr,
            ) -> <Self::Narrow as super::ops::F32x8Ops>::Repr {{
                <Self::Narrow as crate::simd::generic::cross_width::NarrowCrossWidthOps>::from_halves_x8(
                    self.narrow(),
                    lo,
                    hi,
                )
            }}

            #[inline(always)]
            fn f32x8_low(
                self,
                wide: <Self::Narrow as super::ops::F32x8Ops>::Repr,
            ) -> <Self::Narrow as super::ops::F32x4Ops>::Repr {{
                <Self::Narrow as crate::simd::generic::cross_width::NarrowCrossWidthOps>::low_x8(
                    self.narrow(),
                    wide,
                )
            }}

            #[inline(always)]
            fn f32x8_high(
                self,
                wide: <Self::Narrow as super::ops::F32x8Ops>::Repr,
            ) -> <Self::Narrow as super::ops::F32x4Ops>::Repr {{
                <Self::Narrow as crate::simd::generic::cross_width::NarrowCrossWidthOps>::high_x8(
                    self.narrow(),
                    wide,
                )
            }}

            #[inline(always)]
            fn mul_add_lane4_relaxed_x8<const LANE: i32>(
                self,
                a: <Self::Narrow as super::ops::F32x8Ops>::Repr,
                v_in: <Self::Narrow as super::ops::F32x4Ops>::Repr,
                acc: <Self::Narrow as super::ops::F32x8Ops>::Repr,
            ) -> <Self::Narrow as super::ops::F32x8Ops>::Repr {{
                <Self::Narrow as crate::simd::generic::cross_width::NarrowCrossWidthOps>::mul_add_lane4_relaxed_x8::<LANE>(
                    self.narrow(),
                    a,
                    v_in,
                    acc,
                )
            }}

            #[cfg(feature = "w512")]
            fn f32x16_from_halves(
                self,
                lo: <Self::Narrow as super::ops::F32x8Ops>::Repr,
                hi: <Self::Narrow as super::ops::F32x8Ops>::Repr,
            ) -> <Self::W512 as super::ops::F32x16Ops>::Repr;

            #[cfg(feature = "w512")]
            fn f32x16_low(
                self,
                wide: <Self::W512 as super::ops::F32x16Ops>::Repr,
            ) -> <Self::Narrow as super::ops::F32x8Ops>::Repr;

            #[cfg(feature = "w512")]
            fn f32x16_high(
                self,
                wide: <Self::W512 as super::ops::F32x16Ops>::Repr,
            ) -> <Self::Narrow as super::ops::F32x8Ops>::Repr;

            #[inline(always)]
            fn f32x4_to_u32_fast(
                self,
                a: <Self::Narrow as super::ops::F32x4Ops>::Repr,
            ) -> <Self::Narrow as super::ops::U32x4Ops>::Repr {{
                <Self::Narrow as crate::simd::generic::fast_ops::NarrowToU32Ops>::f32x4_to_u32_fast(
                    self.narrow(),
                    a,
                )
            }}

            #[inline(always)]
            fn f32x8_to_u32_fast(
                self,
                a: <Self::Narrow as super::ops::F32x8Ops>::Repr,
            ) -> <Self::Narrow as super::ops::U32x8Ops>::Repr {{
                <Self::Narrow as crate::simd::generic::fast_ops::NarrowToU32Ops>::f32x8_to_u32_fast(
                    self.narrow(),
                    a,
                )
            }}

            #[cfg(feature = "w512")]
            #[inline(always)]
            fn f32x16_to_u32_fast(
                self,
                a: <Self::W512 as super::ops::F32x16Ops>::Repr,
            ) -> <Self::W512 as super::ops::U32x16Ops>::Repr {{
                <Self::W512 as crate::simd::generic::fast_ops::W512ToU32Ops>::f32x16_to_u32_fast(
                    self.w512(),
                    a,
                )
            }}

            #[inline(always)]
            fn i32x4_dot_i8_i7_add(
                self,
                acc: <Self::Narrow as super::ops::I32x4Ops>::Repr,
                a: <Self::Narrow as super::ops::I8x16Ops>::Repr,
                b: <Self::Narrow as super::ops::I8x16Ops>::Repr,
            ) -> <Self::Narrow as super::ops::I32x4Ops>::Repr {{
                <Self::Narrow as crate::simd::generic::quant_ops::NarrowDotI8Ops>::dot_i8_i7_add_x4(
                    self.narrow(),
                    acc,
                    a,
                    b,
                )
            }}

            #[inline(always)]
            fn i32x8_dot_i8_i7_add(
                self,
                acc: <Self::Narrow as super::ops::I32x8Ops>::Repr,
                a: <Self::Narrow as super::ops::I8x32Ops>::Repr,
                b: <Self::Narrow as super::ops::I8x32Ops>::Repr,
            ) -> <Self::Narrow as super::ops::I32x8Ops>::Repr {{
                <Self::Narrow as crate::simd::generic::quant_ops::NarrowDotI8Ops>::dot_i8_i7_add_x8(
                    self.narrow(),
                    acc,
                    a,
                    b,
                )
            }}

            #[cfg(feature = "w512")]
            #[inline(always)]
            fn i32x16_dot_i8_i7_add(
                self,
                acc: <Self::W512 as super::ops::I32x16Ops>::Repr,
                a: <Self::W512 as super::ops::I8x64Ops>::Repr,
                b: <Self::W512 as super::ops::I8x64Ops>::Repr,
            ) -> <Self::W512 as super::ops::I32x16Ops>::Repr {{
                <Self::W512 as crate::simd::generic::quant_ops::W512DotI8Ops>::dot_i8_i7_add_x16(
                    self.w512(),
                    acc,
                    a,
                    b,
                )
            }}

            fn f16_to_f32_into_ops(self, input: &[u16], output: &mut [f32]);

            fn f32_to_f16_into_ops(self, input: &[f32], output: &mut [u16]);

            #[cfg(feature = "w512")]
            fn f64x8_convert_to_i32(
                self,
                a: <Self::W512 as super::ops::F64x8Ops>::Repr,
            ) -> <Self::Narrow as super::ops::I32x8Ops>::Repr;

            #[cfg(feature = "w512")]
            fn f64x8_convert_to_i32_saturating(
                self,
                a: <Self::W512 as super::ops::F64x8Ops>::Repr,
            ) -> <Self::Narrow as super::ops::I32x8Ops>::Repr;

            #[cfg(feature = "w512")]
            fn f64x8_convert_to_i32_round(
                self,
                a: <Self::W512 as super::ops::F64x8Ops>::Repr,
            ) -> <Self::Narrow as super::ops::I32x8Ops>::Repr;

            #[cfg(feature = "w512")]
            fn f64x8_convert_from_i32(
                self,
                a: <Self::Narrow as super::ops::I32x8Ops>::Repr,
            ) -> <Self::W512 as super::ops::F64x8Ops>::Repr;
        }}

        {impls}
    "#};

    (out, v4_matched, wasm_matched)
}

#[allow(clippy::too_many_lines)]
fn build_sealed_token_impls(
    v4_overrides_self: &str,
    v4_overrides_v4: &str,
    wasm_overrides: &str,
) -> String {
    #[derive(Clone, Copy)]
    enum W512Halves {
        Scalar,
        Pair,
        Cascade4Arm,
        Cascade4Wasm,
        V4Self,
        V4Extract,
    }

    struct SimpleImplSpec<'a> {
        cfg: Option<&'a str>,
        token: &'a str,
        narrow_ty: &'a str,
        narrow_expr: &'a str,
        w512_ty: &'a str,
        w512_expr: &'a str,
        halves: W512Halves,
        extra: &'a str,
    }

    fn emit_simple_impl(spec: SimpleImplSpec<'_>) -> String {
        let SimpleImplSpec {
            cfg,
            token,
            narrow_ty,
            narrow_expr,
            w512_ty,
            w512_expr,
            halves,
            extra,
        } = spec;
        let cfg_attr = cfg.map(|c| format!("#[cfg({c})]\n")).unwrap_or_default();
        let halves_body = match halves {
            W512Halves::Scalar => r#"
    #[cfg(feature = "w512")]
    #[inline(always)]
    fn f32x16_from_halves(self, lo: [f32; 8], hi: [f32; 8]) -> [f32; 16] {
        crate::simd::generic::cross_width::scalar_f32x16_from_halves(lo, hi)
    }
    #[cfg(feature = "w512")]
    #[inline(always)]
    fn f32x16_low(self, wide: [f32; 16]) -> [f32; 8] {
        crate::simd::generic::cross_width::scalar_f32x16_low(wide)
    }
    #[cfg(feature = "w512")]
    #[inline(always)]
    fn f32x16_high(self, wide: [f32; 16]) -> [f32; 8] {
        crate::simd::generic::cross_width::scalar_f32x16_high(wide)
    }
    #[cfg(feature = "w512")]
    #[inline(always)]
    fn f64x8_convert_to_i32(self, a: [f64; 8]) -> [i32; 8] {
        crate::simd::impls::scalar::scalar_f64x8_to_i32(a)
    }
    #[cfg(feature = "w512")]
    #[inline(always)]
    fn f64x8_convert_to_i32_saturating(self, a: [f64; 8]) -> [i32; 8] {
        crate::simd::impls::scalar::scalar_f64x8_to_i32(a)
    }
    #[cfg(feature = "w512")]
    #[inline(always)]
    fn f64x8_convert_to_i32_round(self, a: [f64; 8]) -> [i32; 8] {
        crate::simd::impls::scalar::scalar_f64x8_to_i32_round(a)
    }
    #[cfg(feature = "w512")]
    #[inline(always)]
    fn f64x8_convert_from_i32(self, a: [i32; 8]) -> [f64; 8] {
        crate::simd::impls::scalar::scalar_f64x8_from_i32(a)
    }"#
            .to_string(),
            W512Halves::Pair => r#"
    #[cfg(feature = "w512")]
    #[inline(always)]
    fn f32x16_from_halves(
        self,
        lo: <Self::Narrow as super::ops::F32x8Ops>::Repr,
        hi: <Self::Narrow as super::ops::F32x8Ops>::Repr,
    ) -> [<Self::Narrow as super::ops::F32x8Ops>::Repr; 2] {
        [lo, hi]
    }
    #[cfg(feature = "w512")]
    #[inline(always)]
    fn f32x16_low(
        self,
        wide: [<Self::Narrow as super::ops::F32x8Ops>::Repr; 2],
    ) -> <Self::Narrow as super::ops::F32x8Ops>::Repr {
        wide[0]
    }
    #[cfg(feature = "w512")]
    #[inline(always)]
    fn f32x16_high(
        self,
        wide: [<Self::Narrow as super::ops::F32x8Ops>::Repr; 2],
    ) -> <Self::Narrow as super::ops::F32x8Ops>::Repr {
        wide[1]
    }
    #[cfg(feature = "w512")]
    #[inline(always)]
    fn f64x8_convert_to_i32(
        self,
        a: [core::arch::x86_64::__m256d; 2],
    ) -> core::arch::x86_64::__m256i {
        crate::simd::impls::x86_v3::v3_f64x8_to_i32(self.narrow(), a)
    }
    #[cfg(feature = "w512")]
    #[inline(always)]
    fn f64x8_convert_to_i32_saturating(
        self,
        a: [core::arch::x86_64::__m256d; 2],
    ) -> core::arch::x86_64::__m256i {
        crate::simd::impls::x86_v3::v3_f64x8_to_i32_saturating(self.narrow(), a)
    }
    #[cfg(feature = "w512")]
    #[inline(always)]
    fn f64x8_convert_to_i32_round(
        self,
        a: [core::arch::x86_64::__m256d; 2],
    ) -> core::arch::x86_64::__m256i {
        crate::simd::impls::x86_v3::v3_f64x8_to_i32_round(self.narrow(), a)
    }
    #[cfg(feature = "w512")]
    #[inline(always)]
    fn f64x8_convert_from_i32(
        self,
        a: core::arch::x86_64::__m256i,
    ) -> [core::arch::x86_64::__m256d; 2] {
        crate::simd::impls::x86_v3::v3_f64x8_from_i32(self.narrow(), a)
    }"#
            .to_string(),
            W512Halves::Cascade4Arm => r#"
    #[cfg(feature = "w512")]
    #[inline(always)]
    fn f32x16_from_halves(
        self,
        lo: [<Self::Narrow as super::ops::F32x4Ops>::Repr; 2],
        hi: [<Self::Narrow as super::ops::F32x4Ops>::Repr; 2],
    ) -> [<Self::Narrow as super::ops::F32x4Ops>::Repr; 4] {
        [lo[0], lo[1], hi[0], hi[1]]
    }
    #[cfg(feature = "w512")]
    #[inline(always)]
    fn f32x16_low(
        self,
        wide: [<Self::Narrow as super::ops::F32x4Ops>::Repr; 4],
    ) -> [<Self::Narrow as super::ops::F32x4Ops>::Repr; 2] {
        [wide[0], wide[1]]
    }
    #[cfg(feature = "w512")]
    #[inline(always)]
    fn f32x16_high(
        self,
        wide: [<Self::Narrow as super::ops::F32x4Ops>::Repr; 4],
    ) -> [<Self::Narrow as super::ops::F32x4Ops>::Repr; 2] {
        [wide[2], wide[3]]
    }
    #[cfg(feature = "w512")]
    #[inline(always)]
    fn f64x8_convert_to_i32(
        self,
        a: [core::arch::aarch64::float64x2_t; 4],
    ) -> [core::arch::aarch64::int32x4_t; 2] {
        crate::simd::impls::arm_neon::neon_f64x8_to_i32(self.narrow(), a)
    }
    #[cfg(feature = "w512")]
    #[inline(always)]
    fn f64x8_convert_to_i32_saturating(
        self,
        a: [core::arch::aarch64::float64x2_t; 4],
    ) -> [core::arch::aarch64::int32x4_t; 2] {
        crate::simd::impls::arm_neon::neon_f64x8_to_i32(self.narrow(), a)
    }
    #[cfg(feature = "w512")]
    #[inline(always)]
    fn f64x8_convert_to_i32_round(
        self,
        a: [core::arch::aarch64::float64x2_t; 4],
    ) -> [core::arch::aarch64::int32x4_t; 2] {
        crate::simd::impls::arm_neon::neon_f64x8_to_i32_round(self.narrow(), a)
    }
    #[cfg(feature = "w512")]
    #[inline(always)]
    fn f64x8_convert_from_i32(
        self,
        a: [core::arch::aarch64::int32x4_t; 2],
    ) -> [core::arch::aarch64::float64x2_t; 4] {
        crate::simd::impls::arm_neon::neon_f64x8_from_i32(self.narrow(), a)
    }"#
            .to_string(),
            W512Halves::Cascade4Wasm => r#"
    #[cfg(feature = "w512")]
    #[inline(always)]
    fn f32x16_from_halves(
        self,
        lo: [<Self::Narrow as super::ops::F32x4Ops>::Repr; 2],
        hi: [<Self::Narrow as super::ops::F32x4Ops>::Repr; 2],
    ) -> [<Self::Narrow as super::ops::F32x4Ops>::Repr; 4] {
        [lo[0], lo[1], hi[0], hi[1]]
    }
    #[cfg(feature = "w512")]
    #[inline(always)]
    fn f32x16_low(
        self,
        wide: [<Self::Narrow as super::ops::F32x4Ops>::Repr; 4],
    ) -> [<Self::Narrow as super::ops::F32x4Ops>::Repr; 2] {
        [wide[0], wide[1]]
    }
    #[cfg(feature = "w512")]
    #[inline(always)]
    fn f32x16_high(
        self,
        wide: [<Self::Narrow as super::ops::F32x4Ops>::Repr; 4],
    ) -> [<Self::Narrow as super::ops::F32x4Ops>::Repr; 2] {
        [wide[2], wide[3]]
    }
    #[cfg(feature = "w512")]
    #[inline(always)]
    fn f64x8_convert_to_i32(
        self,
        a: [core::arch::wasm32::v128; 4],
    ) -> [core::arch::wasm32::v128; 2] {
        crate::simd::impls::wasm128::wasm_f64x8_to_i32(self.narrow(), a)
    }
    #[cfg(feature = "w512")]
    #[inline(always)]
    fn f64x8_convert_to_i32_saturating(
        self,
        a: [core::arch::wasm32::v128; 4],
    ) -> [core::arch::wasm32::v128; 2] {
        crate::simd::impls::wasm128::wasm_f64x8_to_i32(self.narrow(), a)
    }
    #[cfg(feature = "w512")]
    #[inline(always)]
    fn f64x8_convert_to_i32_round(
        self,
        a: [core::arch::wasm32::v128; 4],
    ) -> [core::arch::wasm32::v128; 2] {
        crate::simd::impls::wasm128::wasm_f64x8_to_i32_round(self.narrow(), a)
    }
    #[cfg(feature = "w512")]
    #[inline(always)]
    fn f64x8_convert_from_i32(
        self,
        a: [core::arch::wasm32::v128; 2],
    ) -> [core::arch::wasm32::v128; 4] {
        crate::simd::impls::wasm128::wasm_f64x8_from_i32(self.narrow(), a)
    }"#
            .to_string(),
            W512Halves::V4Self | W512Halves::V4Extract => {
                let proof = if matches!(halves, W512Halves::V4Self) {
                    "self"
                } else {
                    "self.v4()"
                };
                format!(
                    r#"
    #[cfg(feature = "w512")]
    #[inline(always)]
    fn f32x16_from_halves(
        self,
        lo: core::arch::x86_64::__m256,
        hi: core::arch::x86_64::__m256,
    ) -> core::arch::x86_64::__m512 {{
        crate::simd::generic::cross_width::v4_f32x16_from_halves({proof}, lo, hi)
    }}
    #[cfg(feature = "w512")]
    #[inline(always)]
    fn f32x16_low(self, wide: core::arch::x86_64::__m512) -> core::arch::x86_64::__m256 {{
        crate::simd::generic::cross_width::v4_f32x16_low({proof}, wide)
    }}
    #[cfg(feature = "w512")]
    #[inline(always)]
    fn f32x16_high(self, wide: core::arch::x86_64::__m512) -> core::arch::x86_64::__m256 {{
        crate::simd::generic::cross_width::v4_f32x16_high({proof}, wide)
    }}
    #[cfg(feature = "w512")]
    #[inline(always)]
    fn f64x8_convert_to_i32(self, a: core::arch::x86_64::__m512d) -> core::arch::x86_64::__m256i {{
        crate::simd::impls::x86_v4::v4_f64x8_to_i32({proof}, a)
    }}
    #[cfg(feature = "w512")]
    #[inline(always)]
    fn f64x8_convert_to_i32_saturating(self, a: core::arch::x86_64::__m512d) -> core::arch::x86_64::__m256i {{
        crate::simd::impls::x86_v4::v4_f64x8_to_i32_saturating({proof}, a)
    }}
    #[cfg(feature = "w512")]
    #[inline(always)]
    fn f64x8_convert_to_i32_round(self, a: core::arch::x86_64::__m512d) -> core::arch::x86_64::__m256i {{
        crate::simd::impls::x86_v4::v4_f64x8_to_i32_round({proof}, a)
    }}
    #[cfg(feature = "w512")]
    #[inline(always)]
    fn f64x8_convert_from_i32(self, a: core::arch::x86_64::__m256i) -> core::arch::x86_64::__m512d {{
        crate::simd::impls::x86_v4::v4_f64x8_from_i32({proof}, a)
    }}"#
                )
            }
        };

        formatdoc! {r#"
            {cfg_attr}impl Sealed for {token} {{
                type Narrow = {narrow_ty};
                #[cfg(feature = "w512")]
                type W512 = {w512_ty};

                #[inline(always)]
                fn narrow(self) -> Self::Narrow {{
                    {narrow_expr}
                }}

                #[cfg(feature = "w512")]
                #[inline(always)]
                fn w512(self) -> Self::W512 {{
                    {w512_expr}
                }}
            {halves_body}
            {extra}
            }}
        "#}
    }

    let mut parts = Vec::new();

    let soft_f16_extra = r#"
    #[inline(always)]
    fn f16_to_f32_into_ops(self, input: &[u16], output: &mut [f32]) {
        crate::simd::generic::convert_f16::f16_to_f32_slice_soft(self.narrow(), input, output);
    }
    #[inline(always)]
    fn f32_to_f16_into_ops(self, input: &[f32], output: &mut [u16]) {
        crate::simd::generic::convert_f16::f32_to_f16_slice_soft(self.narrow(), input, output);
    }"#;

    // ScalarToken (always present on every target)
    parts.push(emit_simple_impl(SimpleImplSpec {
        cfg: None,
        token: "archmage::ScalarToken",
        narrow_ty: "archmage::ScalarToken",
        narrow_expr: "self",
        w512_ty: "archmage::ScalarToken",
        w512_expr: "self",
        halves: W512Halves::Scalar,
        extra: soft_f16_extra,
    }));

    // x86_64 tokens (11 total)
    for token in [
        "archmage::X64V1Token",
        "archmage::X64V2Token",
        "archmage::X64CryptoToken",
    ] {
        parts.push(emit_simple_impl(SimpleImplSpec {
            cfg: Some(r#"target_arch = "x86_64""#),
            token,
            narrow_ty: "archmage::ScalarToken",
            narrow_expr: "archmage::ScalarToken",
            w512_ty: "archmage::ScalarToken",
            w512_expr: "archmage::ScalarToken",
            halves: W512Halves::Scalar,
            extra: soft_f16_extra,
        }));
    }

    let v3_f16_extra = r#"
    #[inline(always)]
    fn f16_to_f32_into_ops(self, input: &[u16], output: &mut [f32]) {
        crate::simd::generic::convert_f16::f16c_decode_select(self, input, output);
    }
    #[inline(always)]
    fn f32_to_f16_into_ops(self, input: &[f32], output: &mut [u16]) {
        crate::simd::generic::convert_f16::f16c_encode_select(self, input, output);
    }"#;
    let v3_sub_f16_extra = r#"
    #[inline(always)]
    fn f16_to_f32_into_ops(self, input: &[u16], output: &mut [f32]) {
        crate::simd::generic::convert_f16::f16c_decode_select(self.v3(), input, output);
    }
    #[inline(always)]
    fn f32_to_f16_into_ops(self, input: &[f32], output: &mut [u16]) {
        crate::simd::generic::convert_f16::f16c_encode_select(self.v3(), input, output);
    }"#;
    let neon_self_f16_extra = r#"
    #[inline(always)]
    fn f16_to_f32_into_ops(self, input: &[u16], output: &mut [f32]) {
        crate::simd::generic::convert_f16::neon_f16_decode(self, input, output);
    }
    #[inline(always)]
    fn f32_to_f16_into_ops(self, input: &[f32], output: &mut [u16]) {
        crate::simd::generic::convert_f16::neon_f16_encode(self, input, output);
    }"#;
    let neon_sub_f16_extra = r#"
    #[inline(always)]
    fn f16_to_f32_into_ops(self, input: &[u16], output: &mut [f32]) {
        crate::simd::generic::convert_f16::neon_f16_decode(self.neon(), input, output);
    }
    #[inline(always)]
    fn f32_to_f16_into_ops(self, input: &[f32], output: &mut [u16]) {
        crate::simd::generic::convert_f16::neon_f16_encode(self.neon(), input, output);
    }"#;

    parts.push(emit_simple_impl(SimpleImplSpec {
        cfg: Some(r#"target_arch = "x86_64""#),
        token: "archmage::X64V3Token",
        narrow_ty: "archmage::X64V3Token",
        narrow_expr: "self",
        w512_ty: "archmage::X64V3Token",
        w512_expr: "self",
        halves: W512Halves::Pair,
        extra: v3_f16_extra,
    }));

    for token in [
        "archmage::X64V3CryptoToken",
        "archmage::X64V3GfniCryptoToken",
        "archmage::X64V3VnniToken",
        "archmage::X64V3Vnni2Token",
    ] {
        parts.push(emit_simple_impl(SimpleImplSpec {
            cfg: Some(r#"target_arch = "x86_64""#),
            token,
            narrow_ty: "archmage::X64V3Token",
            narrow_expr: "self.v3()",
            w512_ty: "archmage::X64V3Token",
            w512_expr: "self.v3()",
            halves: W512Halves::Pair,
            extra: v3_sub_f16_extra,
        }));
    }

    // AVX-512 tokens with feature = "avx512"
    let v4_common_extra_self = formatdoc! {r#"
        {v4_overrides_self}
        #[inline(always)]
        fn f32x4_to_u32_fast(self, a: core::arch::x86_64::__m128) -> core::arch::x86_64::__m128i {{
            crate::simd::generic::fast_ops::v4_f32x4_to_u32_fast(self, a)
        }}
        #[inline(always)]
        fn f32x8_to_u32_fast(self, a: core::arch::x86_64::__m256) -> core::arch::x86_64::__m256i {{
            crate::simd::generic::fast_ops::v4_f32x8_to_u32_fast(self, a)
        }}
        #[inline(always)]
        fn f16_to_f32_into_ops(self, input: &[u16], output: &mut [f32]) {{
            crate::simd::generic::convert_f16::v4_f16_to_f32_into(self, input, output);
        }}
        #[inline(always)]
        fn f32_to_f16_into_ops(self, input: &[f32], output: &mut [u16]) {{
            crate::simd::generic::convert_f16::v4_f32_to_f16_into(self, input, output);
        }}
    "#};
    let v4_common_extra_v4 = formatdoc! {r#"
        {v4_overrides_v4}
        #[inline(always)]
        fn f32x4_to_u32_fast(self, a: core::arch::x86_64::__m128) -> core::arch::x86_64::__m128i {{
            crate::simd::generic::fast_ops::v4_f32x4_to_u32_fast(self.v4(), a)
        }}
        #[inline(always)]
        fn f32x8_to_u32_fast(self, a: core::arch::x86_64::__m256) -> core::arch::x86_64::__m256i {{
            crate::simd::generic::fast_ops::v4_f32x8_to_u32_fast(self.v4(), a)
        }}
        #[inline(always)]
        fn f16_to_f32_into_ops(self, input: &[u16], output: &mut [f32]) {{
            crate::simd::generic::convert_f16::v4_f16_to_f32_into(self.v4(), input, output);
        }}
        #[inline(always)]
        fn f32_to_f16_into_ops(self, input: &[f32], output: &mut [u16]) {{
            crate::simd::generic::convert_f16::v4_f32_to_f16_into(self.v4(), input, output);
        }}
    "#};
    let v4x_extra = formatdoc! {r#"
        {v4_common_extra_v4}
        #[inline(always)]
        fn i32x4_dot_i8_i7_add(
            self,
            acc: core::arch::x86_64::__m128i,
            a: core::arch::x86_64::__m128i,
            b: core::arch::x86_64::__m128i,
        ) -> core::arch::x86_64::__m128i {{
            crate::simd::generic::quant_ops::v4x_i32x4_dot_i8(self, acc, a, b)
        }}
        #[inline(always)]
        fn i32x8_dot_i8_i7_add(
            self,
            acc: core::arch::x86_64::__m256i,
            a: core::arch::x86_64::__m256i,
            b: core::arch::x86_64::__m256i,
        ) -> core::arch::x86_64::__m256i {{
            crate::simd::generic::quant_ops::v4x_i32x8_dot_i8(self, acc, a, b)
        }}
        #[cfg(feature = "w512")]
        #[inline(always)]
        fn i32x16_dot_i8_i7_add(
            self,
            acc: core::arch::x86_64::__m512i,
            a: core::arch::x86_64::__m512i,
            b: core::arch::x86_64::__m512i,
        ) -> core::arch::x86_64::__m512i {{
            crate::simd::generic::quant_ops::v4x_i32x16_dot_i8(self, acc, a, b)
        }}
    "#};

    parts.push(emit_simple_impl(SimpleImplSpec {
        cfg: Some(r#"all(target_arch = "x86_64", feature = "avx512")"#),
        token: "archmage::X64V4Token",
        narrow_ty: "archmage::X64V3Token",
        narrow_expr: "self.v3()",
        w512_ty: "archmage::X64V4Token",
        w512_expr: "self",
        halves: W512Halves::V4Self,
        extra: &v4_common_extra_self,
    }));
    parts.push(emit_simple_impl(SimpleImplSpec {
        cfg: Some(r#"all(target_arch = "x86_64", feature = "avx512")"#),
        token: "archmage::X64V4xToken",
        narrow_ty: "archmage::X64V3Token",
        narrow_expr: "self.v3()",
        w512_ty: "archmage::X64V4Token",
        w512_expr: "self.v4()",
        halves: W512Halves::V4Extract,
        extra: &v4x_extra,
    }));
    parts.push(emit_simple_impl(SimpleImplSpec {
        cfg: Some(r#"all(target_arch = "x86_64", feature = "avx512")"#),
        token: "archmage::Avx512Fp16Token",
        narrow_ty: "archmage::X64V3Token",
        narrow_expr: "self.v3()",
        w512_ty: "archmage::X64V4Token",
        w512_expr: "self.v4()",
        halves: W512Halves::V4Extract,
        extra: &v4_common_extra_v4,
    }));

    // AVX-512 tokens without feature = "avx512": fallback seamlessly to V3
    for token in [
        "archmage::X64V4Token",
        "archmage::X64V4xToken",
        "archmage::Avx512Fp16Token",
    ] {
        parts.push(emit_simple_impl(SimpleImplSpec {
            cfg: Some(r#"all(target_arch = "x86_64", not(feature = "avx512"))"#),
            token,
            narrow_ty: "archmage::X64V3Token",
            narrow_expr: "self.v3()",
            w512_ty: "archmage::X64V3Token",
            w512_expr: "self.v3()",
            halves: W512Halves::Pair,
            extra: v3_sub_f16_extra,
        }));
    }

    // aarch64 tokens (6 total)
    parts.push(emit_simple_impl(SimpleImplSpec {
        cfg: Some(r#"target_arch = "aarch64""#),
        token: "archmage::NeonToken",
        narrow_ty: "archmage::NeonToken",
        narrow_expr: "self",
        w512_ty: "archmage::NeonToken",
        w512_expr: "self",
        halves: W512Halves::Cascade4Arm,
        extra: neon_self_f16_extra,
    }));
    for token in [
        "archmage::NeonAesToken",
        "archmage::NeonSha3Token",
        "archmage::NeonCrcToken",
        "archmage::Arm64V2Token",
        "archmage::Arm64V3Token",
    ] {
        parts.push(emit_simple_impl(SimpleImplSpec {
            cfg: Some(r#"target_arch = "aarch64""#),
            token,
            narrow_ty: "archmage::NeonToken",
            narrow_expr: "self.neon()",
            w512_ty: "archmage::NeonToken",
            w512_expr: "self.neon()",
            halves: W512Halves::Cascade4Arm,
            extra: neon_sub_f16_extra,
        }));
    }

    // wasm32 tokens (3 total)
    parts.push(emit_simple_impl(SimpleImplSpec {
        cfg: Some(r#"target_arch = "wasm32""#),
        token: "archmage::Wasm128Token",
        narrow_ty: "archmage::Wasm128Token",
        narrow_expr: "self",
        w512_ty: "archmage::Wasm128Token",
        w512_expr: "self",
        halves: W512Halves::Cascade4Wasm,
        extra: soft_f16_extra,
    }));
    parts.push(emit_simple_impl(SimpleImplSpec {
        cfg: Some(r#"target_arch = "wasm32""#),
        token: "archmage::Wasm128RelaxedToken",
        narrow_ty: "archmage::Wasm128Token",
        narrow_expr: "self.wasm128()",
        w512_ty: "archmage::Wasm128Token",
        w512_expr: "self.wasm128()",
        halves: W512Halves::Cascade4Wasm,
        extra: soft_f16_extra,
    }));

    let wasm_fma_extra = formatdoc! {r#"
        {wasm_overrides}
        #[inline(always)]
        fn mul_add_lane4_relaxed_x8<const LANE: i32>(
            self,
            a: [core::arch::wasm32::v128; 2],
            v_in: core::arch::wasm32::v128,
            acc: [core::arch::wasm32::v128; 2],
        ) -> [core::arch::wasm32::v128; 2] {{
            crate::simd::generic::cross_width::wasm_fma_mul_add_lane4_relaxed::<LANE>(
                self, a, v_in, acc,
            )
        }}
        #[inline(always)]
        fn f32x4_to_u32_fast(self, a: core::arch::wasm32::v128) -> core::arch::wasm32::v128 {{
            crate::simd::generic::fast_ops::wasm_fma_f32x4_to_u32_fast(self, a)
        }}
        #[inline(always)]
        fn f32x8_to_u32_fast(
            self,
            a: [core::arch::wasm32::v128; 2],
        ) -> [core::arch::wasm32::v128; 2] {{
            crate::simd::generic::fast_ops::wasm_fma_f32x8_to_u32_fast(self, a)
        }}
        #[cfg(feature = "w512")]
        #[inline(always)]
        fn f32x16_to_u32_fast(
            self,
            a: [core::arch::wasm32::v128; 4],
        ) -> [core::arch::wasm32::v128; 4] {{
            crate::simd::generic::fast_ops::wasm_fma_f32x16_to_u32_fast(self, a)
        }}
        #[inline(always)]
        fn i32x4_dot_i8_i7_add(
            self,
            acc: core::arch::wasm32::v128,
            a: core::arch::wasm32::v128,
            b: core::arch::wasm32::v128,
        ) -> core::arch::wasm32::v128 {{
            crate::simd::generic::quant_ops::wasm_fma_i32x4_dot_i8(self, acc, a, b)
        }}
        #[inline(always)]
        fn i32x8_dot_i8_i7_add(
            self,
            acc: [core::arch::wasm32::v128; 2],
            a: [core::arch::wasm32::v128; 2],
            b: [core::arch::wasm32::v128; 2],
        ) -> [core::arch::wasm32::v128; 2] {{
            crate::simd::generic::quant_ops::wasm_fma_i32x8_dot_i8(self, acc, a, b)
        }}
        #[cfg(feature = "w512")]
        #[inline(always)]
        fn i32x16_dot_i8_i7_add(
            self,
            acc: [core::arch::wasm32::v128; 4],
            a: [core::arch::wasm32::v128; 4],
            b: [core::arch::wasm32::v128; 4],
        ) -> [core::arch::wasm32::v128; 4] {{
            crate::simd::generic::quant_ops::wasm_fma_i32x16_dot_i8(self, acc, a, b)
        }}
        {soft_f16_extra}
    "#};
    parts.push(emit_simple_impl(SimpleImplSpec {
        cfg: Some(
            r#"all(target_arch = "wasm32", target_feature = "simd128", target_feature = "relaxed-simd")"#,
        ),
        token: "archmage::Wasm128RelaxedFmaToken",
        narrow_ty: "archmage::Wasm128Token",
        narrow_expr: "self.wasm128()",
        w512_ty: "archmage::Wasm128Token",
        w512_expr: "self.wasm128()",
        halves: W512Halves::Cascade4Wasm,
        extra: &wasm_fma_extra,
    }));
    parts.push(emit_simple_impl(SimpleImplSpec {
        cfg: Some(
            r#"all(target_arch = "wasm32", not(all(target_feature = "simd128", target_feature = "relaxed-simd")))"#,
        ),
        token: "archmage::Wasm128RelaxedFmaToken",
        narrow_ty: "archmage::Wasm128Token",
        narrow_expr: "self.wasm128()",
        w512_ty: "archmage::Wasm128Token",
        w512_expr: "self.wasm128()",
        halves: W512Halves::Cascade4Wasm,
        extra: soft_f16_extra,
    }));

    parts.join("\n")
}

fn find_fn<'a>(tr: &'a syn::ItemTrait, name: &str) -> &'a syn::TraitItemFn {
    for item in &tr.items {
        if let TraitItem::Fn(m) = item
            && m.sig.ident == name
        {
            return m;
        }
    }
    panic!("method `{name}` not found in trait `{}`", tr.ident);
}

fn extract_arg_idents(sig: &syn::Signature) -> Vec<Ident> {
    sig.inputs
        .iter()
        .filter_map(|arg| match arg {
            FnArg::Receiver(_) => None,
            FnArg::Typed(pat_ty) => match &*pat_ty.pat {
                Pat::Ident(pi) => Some(pi.ident.clone()),
                _ => panic!("expected simple ident arg in `{}`", sig.ident),
            },
        })
        .collect()
}

fn generate_delegated_rs(traits: &[(String, syn::ItemTrait, bool)]) -> String {
    let mut blocks = Vec::new();
    let v4_map: BTreeMap<(&str, &str), &str> = V4_F32_OVERRIDES
        .iter()
        .map(|&(t, m, h)| ((t, m), h))
        .collect();
    let wasm_map: BTreeMap<(&str, &str), &str> = WASM_FMA_OVERRIDES
        .iter()
        .map(|&(t, m, h)| ((t, m), h))
        .collect();

    for (name, tr, is_w512) in traits {
        let trait_ident = format_ident!("{name}");
        let ops_ident = format_ident!("{}", ops_trait_name(name));
        let proj = format_ident!("{}", if *is_w512 { "W512" } else { "Narrow" });
        let proj_fn = format_ident!("{}", if *is_w512 { "w512" } else { "narrow" });

        let mut items: Vec<TokenStream> = Vec::new();
        for item in &tr.items {
            match item {
                TraitItem::Type(ty) => {
                    assert_eq!(ty.ident, "Repr", "unexpected associated type in {name}");
                    items.push(quote! {
                        type Repr = <T::#proj as crate::simd::backends::ops::#ops_ident>::Repr;
                    });
                }
                TraitItem::Fn(m) => {
                    let raw_sig = &m.sig;
                    let mname = raw_sig.ident.to_string();
                    let args = extract_arg_idents(raw_sig);
                    let tf = generic_turbofish(&raw_sig.generics);
                    // Drop method-level `where Self: super::<Sibling>Backend` bounds in
                    // delegated.rs so rustc #152409 does not shadow `<T::Narrow as *Ops>::Repr`;
                    // all sibling *Backend traits already hold unconditionally on `T: SimdToken + Sealed`.
                    let mut clean_sig = raw_sig.clone();
                    clean_sig.generics.where_clause = None;
                    let sig = strip_super_on_unified(clean_sig.to_token_stream());
                    let body = if v4_map.contains_key(&(name.as_str(), mname.as_str())) {
                        let hook = format_ident!("{}", v4_hook_ident(name, &mname));
                        quote! { self.#hook #tf (#(#args),*) }
                    } else if wasm_map.contains_key(&(name.as_str(), mname.as_str())) {
                        let hook = format_ident!("{}", wasm_hook_ident(name, &mname));
                        quote! { self.#hook #tf (#(#args),*) }
                    } else {
                        let orig = &raw_sig.ident;
                        quote! {
                            <T::#proj as crate::simd::backends::ops::#ops_ident>::#orig #tf (
                                self.#proj_fn(),
                                #(#args),*
                            )
                        }
                    };
                    items.push(quote! {
                        #[inline(always)]
                        #sig {
                            #body
                        }
                    });
                }
                _ => panic!("unexpected trait item in {name}"),
            }
        }

        let cfg_attr = if *is_w512 {
            quote!(#[cfg(feature = "w512")])
        } else {
            quote!()
        };

        blocks.push(
            quote! {
                #cfg_attr
                impl<T: archmage::SimdToken + crate::simd::backends::sealed::Sealed + Copy + 'static>
                    crate::simd::backends::#trait_ident for T
                {
                    #(#items)*
                }
            }
            .to_string(),
        );
    }

    // `F64x8Convert` bridges W512 (`F64x8Backend`) and Narrow (`I32x8Backend`),
    // so it routes through `Sealed::f64x8_convert_*` hooks rather than a single
    // `*Ops` projection.
    blocks.push(
        quote! {
            #[cfg(feature = "w512")]
            impl<T: archmage::SimdToken + crate::simd::backends::sealed::Sealed + Copy + 'static>
                crate::simd::backends::F64x8Convert for T
            {
                #[inline(always)]
                fn convert_f64_to_i32(
                    self,
                    a: <Self as crate::simd::backends::F64x8Backend>::Repr,
                ) -> <Self as crate::simd::backends::I32x8Backend>::Repr {
                    self.f64x8_convert_to_i32(a)
                }

                #[inline(always)]
                fn convert_f64_to_i32_saturating(
                    self,
                    a: <Self as crate::simd::backends::F64x8Backend>::Repr,
                ) -> <Self as crate::simd::backends::I32x8Backend>::Repr {
                    self.f64x8_convert_to_i32_saturating(a)
                }

                #[inline(always)]
                fn convert_f64_to_i32_round(
                    self,
                    a: <Self as crate::simd::backends::F64x8Backend>::Repr,
                ) -> <Self as crate::simd::backends::I32x8Backend>::Repr {
                    self.f64x8_convert_to_i32_round(a)
                }

                #[inline(always)]
                fn convert_i32_to_f64(
                    self,
                    a: <Self as crate::simd::backends::I32x8Backend>::Repr,
                ) -> <Self as crate::simd::backends::F64x8Backend>::Repr {
                    self.f64x8_convert_from_i32(a)
                }
            }
        }
        .to_string(),
    );

    let body = blocks.join("\n\n");
    formatdoc! {r#"
        //! Blanket implementations of all 45 public backend, conversion, and bitcast
        //! traits (plus `F64x8Convert`) over every sealed archmage token.
        //!
        //! Routes through `T::Narrow` / `T::W512` (`backends/ops.rs`) and optional
        //! hardware overrides on `Sealed`.
        //!
        //! **Auto-generated** by `cargo xtask generate` - do not edit manually.
        //!
        //! # Safety (audit contract — checked backend boundaries)
        //!
        //! `Sealed` is crate-private and implemented only for archmage tokens whose
        //! `Narrow` / `W512` projections invoke `.v3()` / `.v4()` / `.neon()` /
        //! `.wasm128()` safe extractors or `ScalarToken`. Hardware override hooks
        //! forward to concrete `#[arcane]` / target-feature-proven functions.
        //! Every method delegates safely without `unsafe` or raw pointers.

        #![allow(clippy::too_many_arguments, unused_imports)]

        use crate::simd::backends::*;
        use crate::simd::backends::sealed::Sealed;

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
            "override ({trait_name}, {method_name} -> {helper}) did not match any parsed method; \
             method renamed or removed?"
        );
    }
}
