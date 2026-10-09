//! Unified definition and call model. Legacy frontends retain their own syntax
//! contracts; target-feature bodies and proof boundaries share the emitter.
pub(crate) mod call;
#[cfg(test)]
mod convention_tests;
#[cfg(test)]
mod inline_tests;
mod parent;
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
    let mut input: LightFn = syn::parse2(item)?;
    if args.family {
        family(input, args)
    } else if args.wrap {
        wrap(input, args)
    } else {
        let inferred = if args.tier.is_none() {
            infer_suffix(&input.sig.ident)
        } else {
            None
        };
        if let Some((tier, Form::Proof)) = inferred {
            if !args.defines.is_empty() {
                return Err(syn::Error::new_spanned(
                    &input.sig,
                    "define(...) on a proof wrapper requires a generated family",
                ));
            }
            // A suffix is an exact context declaration, not permission to mint
            // proof. Add the concrete proof parameter when none was written;
            // otherwise ensure the existing proof declares the named context.
            let placeholder = input.sig.inputs.iter().any(placeholder_parameter);
            if let Some(proof) = crate::token_discovery::find_token_param(&input.sig) {
                let expected = crate::generated::tier_to_canonical_token(tier.name)
                    .and_then(crate::generated::token_to_features)
                    .expect("registered tier");
                if proof.target_arch != tier.target_arch
                    || proof.features.len() != expected.len()
                    || !expected.iter().all(|f| proof.features.contains(f))
                {
                    return Err(syn::Error::new_spanned(
                        &input.sig,
                        "proof parameter does not match the inferred tier suffix; use the matching token or explicit wrap",
                    ));
                }
            } else {
                let token: syn::Path = syn::parse_str(&format!("::{}", tier.token_path))?;
                if placeholder {
                    specialize_syntax(&mut input.sig, &quote!(#token))?;
                } else {
                    let position =
                        usize::from(matches!(input.sig.inputs.first(), Some(FnArg::Receiver(_))));
                    input
                        .sig
                        .inputs
                        .insert(position, parse_quote!(__attune_token: #token));
                }
            }
            let output = wrap(input, args)?;
            return Ok(if placeholder {
                let token: syn::Path = syn::parse_str(&format!("::{}", tier.token_path))?;
                replace_ident_in_tokens(output, "Token", &quote!(#token))
            } else {
                output
            });
        }
        let tier = args.tier.or_else(|| inferred.map(|(tier, _)| tier)).ok_or_else(|| {
            syn::Error::new_spanned(&input.sig.ident,
                "attune needs a tier, a registered tier suffix (optionally _t), wrap, or make(...)")
        })?;
        if args.in_trait || args.self_type.is_some() {
            return Err(syn::Error::new_spanned(
                &input.sig,
                "direct target features cannot be placed on a safe trait method; move this kernel outside of the impl, or use attune(wrap) with a proof parameter",
            ));
        }
        direct(input, tier, &args, None, args.body_inline)
    }
}

fn infer_suffix(name: &syn::Ident) -> Option<(&'static TierDescriptor, Form)> {
    let name = name.to_string();
    let (stem, form) = name
        .strip_suffix("_t")
        .map_or((name.as_str(), Form::Direct), |stem| (stem, Form::Proof));
    crate::tiers::ALL_TIERS
        .iter()
        .filter(|tier| {
            tier.name != "default"
                && stem
                    .strip_suffix(tier.suffix)
                    .is_some_and(|prefix| prefix.is_empty() || prefix.ends_with('_'))
        })
        .max_by_key(|tier| tier.suffix.len())
        .map(|tier| (tier, form))
}

fn direct(
    mut input: LightFn,
    tier: &TierDescriptor,
    args: &Args,
    gate: Option<&str>,
    inline: Option<Inline>,
) -> syn::Result<TokenStream> {
    let token = crate::generated::tier_to_canonical_token(tier.name).expect("registered tier");
    let context =
        crate::engine::feature::FeatureContext::from_tier_token(token).expect("registered context");
    if inline == Some(Inline::Always) && !context.features.is_empty() {
        return Err(syn::Error::new_spanned(
            &input.sig.ident,
            "inline(always) on target-feature bodies requires nightly; use inline(hint) or inline(never)",
        ));
    }
    // Raw functions without aliases need no token path or substitution pass.
    let token_path = if args.family || !args.defines.is_empty() {
        Some(tier.token_path.parse::<TokenStream>()?)
    } else {
        None
    };
    let defines = args.defines.iter().map(|name| {
        quote! {
            #[allow(non_camel_case_types, dead_code)]
            type #name = ::magetypes::simd::generic::#name<::#token_path>;
        }
    });
    prepend_to_body(&mut input.body, quote!(#(#defines)*));
    let options = SharedOptions {
        import_intrinsics: args.imports.import_intrinsics,
        import_magetypes: args.imports.import_magetypes,
        cfg_feature: gate
            .map(str::to_string)
            .or_else(|| args.imports.cfg_feature.clone()),
    };
    // Signature proof discovery must see each family's concrete parameter;
    // body Token markers remain intact until contextual calls are rewritten.
    if args.family
        && input
            .sig
            .inputs
            .pairs()
            .any(|arg| placeholder_parameter(arg.value()))
    {
        specialize_syntax(&mut input.sig, &quote!(::#token_path))?;
    }
    let function = crate::engine::feature::emit(input, &options, inline, true, context)
        .unwrap_or_else(|diagnostic| diagnostic);
    // A single raw context preserves the written signature and identifiers.
    // Token is a substitution placeholder only for generated families.
    Ok(
        if args.family && tokens_contain_ident(&function, &["Token"]) {
            replace_ident_in_tokens(function, "Token", &quote!(::#token_path))
        } else {
            function
        },
    )
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

fn placeholder_parameter(arg: &FnArg) -> bool {
    matches!(arg, FnArg::Typed(param) if matches!(param.ty.as_ref(), syn::Type::Path(path)
        if path.qself.is_none() && path.path.is_ident("Token")))
}

fn forward(
    sig: &syn::Signature,
    target: &syn::Ident,
    in_impl: bool,
    proof: Option<TokenStream>,
) -> TokenStream {
    let turbofish = build_turbofish(&sig.generics);
    let has_placeholder = sig.inputs.iter().any(placeholder_parameter);
    let arguments = sig.inputs.iter().filter_map(|arg| match arg {
        FnArg::Typed(p) => match p.pat.as_ref() {
            syn::Pat::Ident(p) => {
                let name = &p.ident;
                Some(if placeholder_parameter(arg) {
                    proof.clone().unwrap_or_else(|| quote!(#name))
                } else {
                    quote!(#name)
                })
            }
            _ => unreachable!("normalized forwarding pattern"),
        },
        _ => None,
    });
    let leading = proof
        .as_ref()
        .filter(|_| !has_placeholder)
        .map(|p| quote!(#p,));
    if sig
        .inputs
        .iter()
        .any(|arg| matches!(arg, FnArg::Receiver(_)))
    {
        quote!(self.#target #turbofish(#leading #(#arguments),*))
    } else if in_impl {
        quote!(Self::#target #turbofish(#leading #(#arguments),*))
    } else {
        quote!(#target #turbofish(#leading #(#arguments),*))
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
    tiers.sort_by(|a, b| b.priority.cmp(&a.priority).then_with(|| a.name.cmp(b.name)));
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
        let implementation = direct_selection.or(proof_selection).or_else(|| {
            args.selections
                .iter()
                .find(|s| s.tier.name == tier.name && s.form == Form::Hidden)
        });
        let gate = implementation.and_then(|s| s.gate.as_deref());
        if args
            .selections
            .iter()
            .any(|s| s.tier.name == tier.name && s.gate.as_deref() != gate)
        {
            return Err(syn::Error::new_spanned(
                &base,
                "a tier's direct, proof and dispatcher implementations must have the same feature gate",
            ));
        }
        let direct_name = if direct_selection.is_some() {
            output_name(&base, tier, Form::Direct, &args)?
        } else {
            format_ident!("__attune_{}_{}", base, tier.suffix)
        };
        let mut body = input.clone();
        body.sig.ident = direct_name.clone();
        // Direct outputs use their visibility override; hidden implementations
        // retain the source operation's policy before becoming private. Wrapper
        // visibility and inline overrides apply independently to wrappers.
        let operation_vis = direct_selection
            .and_then(|s| s.visibility.as_ref())
            .unwrap_or(&input.vis);
        let body_inline = direct_selection
            .and_then(|s| s.inline)
            .or(args.body_inline)
            .map(|policy| policy.resolve(operation_vis));
        body.vis = if direct_selection.is_some() {
            operation_vis.clone()
        } else {
            syn::Visibility::Inherited
        };
        output.extend(direct(body, tier, &args, gate, body_inline)?);
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
            let token: syn::Path = syn::parse_str(&format!("::{}", tier.token_path))?;
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
        // Expectations belong to the user-written operation. Forwarding layers
        // do not repeat that operation and cannot fulfill its expectations.
        dispatcher
            .attrs
            .retain(|attr| !attr.path().is_ident("expect"));
        if let Some(vis) = visibility {
            dispatcher.vis = vis.clone();
        }
        if let Some(policy) = inline {
            dispatcher
                .attrs
                .retain(|attr| !attr.path().is_ident("inline"));
            dispatcher.attrs.extend(policy.attribute(&dispatcher.vis));
        }
        let proofs: Vec<_> = dispatcher
            .sig
            .inputs
            .iter()
            .filter(|arg| placeholder_parameter(arg))
            .filter_map(|arg| {
                if let FnArg::Typed(param) = arg {
                    Some(&param.pat)
                } else {
                    None
                }
            })
            .collect();
        dispatcher.body =
            quote! { #(let _ = #proofs;)* '__attune_dispatch: { #(#dispatch_arms)* } };
        specialize_syntax(&mut dispatcher.sig, &quote!(::archmage::ScalarToken))?;
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
    if let Some(error) = crate::engine::boundary::reserved_param_error(
        &wrapper.sig,
        &direct_name.to_string(),
        "attune",
    ) {
        return Ok(error);
    }
    let invocation = forward(&wrapper.sig, direct_name, args.in_impl, None);
    let token: syn::Path = syn::parse_str(&format!("::{}", tier.token_path))?;
    let position = usize::from(matches!(
        wrapper.sig.inputs.first(),
        Some(FnArg::Receiver(_))
    ));
    if !wrapper.sig.inputs.iter().any(placeholder_parameter) {
        wrapper
            .sig
            .inputs
            .insert(position, parse_quote!(__attune_token: #token));
    }
    specialize_syntax(&mut wrapper.sig, &quote!(#token))?;
    wrapper.sig.ident = name.clone();
    wrapper.vis = selection
        .map(|s| s.visibility.clone().unwrap_or_else(|| input.vis.clone()))
        .unwrap_or(syn::Visibility::Inherited);
    wrapper
        .attrs
        .retain(|attr| !attr.path().is_ident("inline") && !attr.path().is_ident("expect"));
    wrapper.attrs.extend(
        selection
            .and_then(|s| s.inline)
            .unwrap_or(Inline::Always)
            .attribute(&wrapper.vis),
    );
    wrapper.body = if tier.name == "scalar" || tier.target_arch == Some("wasm32") {
        invocation
    } else {
        // SAFETY: this signature accepts the registry's concrete sealed
        // proof, and the sibling exists in the same definition scope.
        crate::engine::boundary::proof_call(invocation)
    };
    let guard = gen_cfg_guard(
        tier.target_arch,
        gate.or(args.imports.cfg_feature.as_deref()),
    );
    Ok(quote!(#guard #wrapper))
}

fn wrap(input: LightFn, args: Args) -> syn::Result<TokenStream> {
    let options = crate::engine::boundary::BoundaryOptions {
        body_inline: args.body_inline,
        nested: args.in_trait || args.self_type.is_some(),
        in_impl: args.in_impl,
        self_type: args.self_type,
        shared: args.imports,
        ..Default::default()
    };
    Ok(crate::engine::boundary::expand(input, "attune", options))
}
