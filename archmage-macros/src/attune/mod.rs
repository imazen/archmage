//! Unified definition and call model. Legacy frontends retain their own syntax
//! contracts; target-feature bodies and proof boundaries share the emitter.
pub(crate) mod call;
mod syntax;

use proc_macro2::TokenStream;
use quote::{ToTokens, format_ident, quote};
use syn::{FnArg, parse_quote};

use crate::{
    common::*,
    tiers::{TierDescriptor, find_tier},
};
use syntax::{Args, Form, Inline, Selection};

pub(crate) fn expand(attr: TokenStream, item: TokenStream) -> syn::Result<TokenStream> {
    let args: Args = syn::parse2(attr)?;
    let input: LightFn = syn::parse2(item)?;
    if args.family {
        family(input, args)
    } else if args.wrap {
        wrap(input, args)
    } else {
        let tier = args
            .tier
            .or_else(|| {
                let name = input.sig.ident.to_string();
                crate::tiers::ALL_TIERS
                    .iter()
                    .filter(|t| t.name != "default" && name.ends_with(&format!("_{}", t.suffix)))
                    .max_by_key(|t| t.suffix.len())
            })
            .ok_or_else(|| {
                syn::Error::new_spanned(
                    &input.sig.ident,
                    "attune needs a tier, a registered tier suffix, wrap, or make(...)",
                )
            })?;
        if args.in_trait || args.self_type.is_some() {
            return Err(syn::Error::new_spanned(
                &input.sig,
                "direct target features cannot be placed on a safe trait method; move this kernel outside of the impl, or use attune(wrap) with a proof parameter",
            ));
        }
        direct(input, tier, &args, None, None)
    }
}

fn inline_attribute(policy: Inline) -> syn::Attribute {
    match policy {
        Inline::Hint => parse_quote!(#[inline]),
        Inline::Always => parse_quote!(#[inline(always)]),
        Inline::Never => parse_quote!(#[inline(never)]),
    }
}

fn direct(
    mut input: LightFn,
    tier: &TierDescriptor,
    args: &Args,
    gate: Option<&str>,
    inline: Option<Inline>,
) -> syn::Result<TokenStream> {
    let token = crate::generated::tier_to_canonical_token(tier.name).expect("registered tier");
    let features = crate::generated::token_to_features(token).expect("registered token");
    if let Some(error) =
        avx512_import_error(&input.sig, args.imports.import_intrinsics, features, token)
    {
        return Ok(error);
    }
    if inline == Some(Inline::Always) && !features.is_empty() {
        return Err(syn::Error::new_spanned(
            &input.sig.ident,
            "inline(always) on target-feature bodies requires nightly; use inline(hint) or inline(never)",
        ));
    }
    if let Some(policy) = inline {
        input.attrs.retain(|attr| !attr.path().is_ident("inline"));
        input.attrs.push(inline_attribute(policy));
    } else if !input
        .attrs
        .iter()
        .any(|attr| attr.path().is_ident("inline"))
    {
        input.attrs.push(inline_attribute(Inline::Hint));
    }
    if !features.is_empty() {
        let csv = crate::token_discovery::features_csv(Some(token), features);
        input
            .attrs
            .push(parse_quote!(#[target_feature(enable = #csv)]));
    }
    let token_path: syn::Path = syn::parse_str(tier.token_path)?;
    input.body = call::rewrite(input.body, tier);
    let imports = generate_imports(
        tier.target_arch,
        crate::generated::token_to_magetypes_namespace(token),
        args.imports.import_intrinsics,
        args.imports.import_magetypes,
    );
    let defines = args.defines.iter().map(|name| {
        quote! {
            #[allow(non_camel_case_types, dead_code)]
            type #name = ::magetypes::simd::generic::#name<#token_path>;
        }
    });
    prepend_to_body(&mut input.body, quote!(#imports #(#defines)*));
    let guard = gen_cfg_guard(
        tier.target_arch,
        gate.or(args.imports.cfg_feature.as_deref()),
    );
    // Token placeholders are discrete identifiers. No string substitutions.
    let function = replace_ident_in_tokens(input.to_token_stream(), "Token", &quote!(#token_path));
    Ok(quote!(#guard #function))
}

fn output_name(
    base: &syn::Ident,
    tier: &TierDescriptor,
    form: Form,
    args: &Args,
) -> syn::Result<syn::Ident> {
    if let Some(rename) = args
        .names
        .iter()
        .find(|r| r.tier.name == tier.name && r.form == form)
    {
        if rename.path.leading_colon.is_some()
            || rename.path.segments.len() != 1
            || !matches!(rename.path.segments[0].arguments, syn::PathArguments::None)
        {
            return Err(syn::Error::new_spanned(
                &rename.path,
                "definition names must be identifiers; call-site names may be paths",
            ));
        }
        Ok(rename.path.segments[0].ident.clone())
    } else if form == Form::Proof {
        Ok(format_ident!("{}_{}_t", base, tier.suffix))
    } else {
        Ok(format_ident!("{}_{}", base, tier.suffix))
    }
}

fn forward(
    sig: &syn::Signature,
    target: &syn::Ident,
    in_impl: bool,
    proof: Option<TokenStream>,
) -> TokenStream {
    let turbofish = build_turbofish(&sig.generics);
    let arguments = sig.inputs.iter().filter_map(|arg| match arg {
        FnArg::Typed(p) => match p.pat.as_ref() {
            syn::Pat::Ident(p) => {
                let name = &p.ident;
                Some(quote!(#name))
            }
            _ => unreachable!("normalized forwarding pattern"),
        },
        _ => None,
    });
    let proof = proof.map(|p| quote!(#p,));
    if sig
        .inputs
        .iter()
        .any(|arg| matches!(arg, FnArg::Receiver(_)))
    {
        quote!(self.#target #turbofish(#proof #(#arguments),*))
    } else if in_impl {
        quote!(Self::#target #turbofish(#proof #(#arguments),*))
    } else {
        quote!(#target #turbofish(#proof #(#arguments),*))
    }
}

fn family(mut input: LightFn, args: Args) -> syn::Result<TokenStream> {
    if args.in_trait || args.self_type.is_some() {
        return Err(syn::Error::new_spanned(
            &input.sig,
            "family generation adds sibling functions; move this kernel outside of the impl for a trait method",
        ));
    }
    if args.dispatcher.is_some()
        && let syn::ReturnType::Type(_, ty) = &input.sig.output
        && matches!(**ty, syn::Type::ImplTrait(_))
    {
        return Err(syn::Error::new_spanned(
            ty,
            "a dispatcher cannot combine distinct opaque return types; return a named type or generate direct variants only",
        ));
    }
    let rebinds = rename_non_ident_params(&mut input.sig);
    prepend_to_body(&mut input.body, quote!(#(#rebinds)*));
    let base = input.sig.ident.clone();
    let mut tiers: Vec<_> = args.selections.iter().map(|s| s.tier).collect();
    if args.dispatcher.is_some() && !tiers.iter().any(|t| t.name == "scalar") {
        tiers.push(find_tier("scalar").unwrap());
    }
    tiers.sort_by_key(|t| std::cmp::Reverse(t.priority));
    tiers.dedup_by_key(|t| t.name);
    let mut output = TokenStream::new();
    let mut dispatch_arms = Vec::new();
    for tier in tiers {
        let direct_selection = args
            .selections
            .iter()
            .find(|s| s.tier.name == tier.name && s.form == Form::Direct);
        let proof_selection = args
            .selections
            .iter()
            .find(|s| s.tier.name == tier.name && s.form == Form::Proof);
        if let (Some(a), Some(b)) = (direct_selection, proof_selection)
            && a.gate != b.gate
        {
            return Err(syn::Error::new_spanned(
                &base,
                "a tier's direct and proof outputs must have the same feature gate",
            ));
        }
        let gate = direct_selection
            .or(proof_selection)
            .and_then(|s| s.gate.as_deref());
        let direct_name = if direct_selection.is_some() {
            output_name(&base, tier, Form::Direct, &args)?
        } else {
            format_ident!("__attune_{}_{}", base, tier.suffix)
        };
        let mut body = input.clone();
        body.sig.ident = direct_name.clone();
        body.vis = direct_selection
            .map(|s| s.visibility.clone().unwrap_or_else(|| input.vis.clone()))
            .unwrap_or(syn::Visibility::Inherited);
        output.extend(direct(
            body,
            tier,
            &args,
            gate,
            direct_selection.and_then(|s| s.inline),
        )?);
        if proof_selection.is_some() || args.dispatcher.is_some() {
            let proof_name = if proof_selection.is_some() {
                output_name(&base, tier, Form::Proof, &args)?
            } else {
                format_ident!("__attune_{}_{}_t", base, tier.suffix)
            };
            output.extend(proof_entry(
                &input,
                tier,
                &direct_name,
                &proof_name,
                proof_selection,
                gate,
                &args,
            )?);
            let token: syn::Path = syn::parse_str(tier.token_path)?;
            let invocation = forward(
                &input.sig,
                &proof_name,
                args.in_impl,
                Some(quote!(__attune_proof)),
            );
            let guard = gen_cfg_guard(tier.target_arch, gate);
            dispatch_arms.push(if tier.name == "scalar" {
                let invocation = forward(&input.sig, &proof_name, args.in_impl, Some(quote!(::archmage::ScalarToken)));
                quote!(#guard { break '__attune_dispatch #invocation; })
            } else { quote!(#guard { if let Some(__attune_proof) = <#token as ::archmage::SimdToken>::summon() { break '__attune_dispatch #invocation; } }) });
        }
    }
    if let Some((visibility, inline)) = &args.dispatcher {
        let mut dispatcher = input;
        if let Some(vis) = visibility {
            dispatcher.vis = vis.clone();
        }
        if let Some(policy) = inline {
            dispatcher
                .attrs
                .retain(|attr| !attr.path().is_ident("inline"));
            dispatcher.attrs.push(inline_attribute(*policy));
        }
        dispatcher.body = quote!('__attune_dispatch: { #(#dispatch_arms)* });
        output.extend(dispatcher.to_token_stream());
    }
    Ok(output)
}

fn proof_entry(
    input: &LightFn,
    tier: &TierDescriptor,
    direct_name: &syn::Ident,
    name: &syn::Ident,
    selection: Option<&Selection>,
    gate: Option<&str>,
    args: &Args,
) -> syn::Result<TokenStream> {
    let mut wrapper = input.clone();
    let invocation = forward(&wrapper.sig, direct_name, args.in_impl, None);
    let token: syn::Path = syn::parse_str(tier.token_path)?;
    let position = usize::from(matches!(
        wrapper.sig.inputs.first(),
        Some(FnArg::Receiver(_))
    ));
    wrapper
        .sig
        .inputs
        .insert(position, parse_quote!(__attune_token: #token));
    wrapper.sig.ident = name.clone();
    wrapper.vis = selection
        .map(|s| s.visibility.clone().unwrap_or_else(|| input.vis.clone()))
        .unwrap_or(syn::Visibility::Inherited);
    wrapper.attrs.retain(|attr| !attr.path().is_ident("inline"));
    wrapper.attrs.push(inline_attribute(
        selection.and_then(|s| s.inline).unwrap_or(Inline::Always),
    ));
    wrapper.body = if tier.name == "scalar" || tier.target_arch == Some("wasm32") {
        invocation
    } else {
        // SAFETY: this signature accepts the registry's concrete sealed
        // proof, and the sibling exists in the same definition scope.
        quote!(unsafe { #invocation })
    };
    let guard = gen_cfg_guard(
        tier.target_arch,
        gate.or(args.imports.cfg_feature.as_deref()),
    );
    Ok(quote!(#guard #wrapper))
}

fn wrap(input: LightFn, args: Args) -> syn::Result<TokenStream> {
    // The compatibility boundary is moved to the shared emitter in the next
    // lowering step; retain its sealed generic proof checks while doing so.
    let in_impl = args.in_impl.then(|| quote!(in_impl,));
    let nested = args.in_trait.then(|| quote!(in_trait,));
    let self_type = args.self_type.as_ref().map(|ty| quote!(_self = #ty,));
    let intrinsics = args
        .imports
        .import_intrinsics
        .then(|| quote!(import_intrinsics,));
    let magetypes = args
        .imports
        .import_magetypes
        .then(|| quote!(import_magetypes,));
    let gate = args.imports.cfg_feature.as_ref().map(|feature| {
        let feature = format_ident!("{feature}");
        quote!(cfg(#feature),)
    });
    let options = syn::parse2(quote!(#in_impl #nested #self_type #intrinsics #magetypes #gate))?;
    Ok(crate::arcane::arcane_impl(input, "attune", options))
}
