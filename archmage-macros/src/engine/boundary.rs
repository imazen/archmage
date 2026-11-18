//! Feature-proof boundaries shared by all attribute frontends.
//! Parsing syntax is kept out of this module; the proof is authenticated before
//! emitting the single unsafe call. Body tokens are never parsed as expressions.
use super::inline::InlinePolicy;
use crate::common::*;
use crate::token_discovery::*;
use proc_macro2::TokenStream;
use quote::{ToTokens, format_ident, quote};
use syn::{Attribute, FnArg, Type, parse_quote};

/// Validate a concrete token using the tier-specific constant supplied by archmage.
///
/// The signature's full type path preserves renamed dependencies and re-exports.
/// A weaker tier lacks the expected constant, so accidental aliasing is rejected.
/// The initializer is checked once in archmage; each expansion only references it.
/// This public constant is an accidental-misuse check, not an anti-forgery boundary.
/// The exact archmage-macros dependency pin keeps this internal protocol in sync.
fn gen_token_assertion(
    token_type_name: &Option<String>,
    token_type: &Option<Type>,
    suppress_const_test: bool,
) -> TokenStream {
    if suppress_const_test {
        return quote! {};
    }
    if let (Some(name), Some(ty)) = (token_type_name, token_type)
        && let Some(expected_tag) = crate::generated::expected_tier_tag(name)
    {
        let assertion = format_ident!("__ARCHMAGE_ASSERT_TIER_{:08X}", expected_tag);
        return quote! { let _: () = <#ty>::#assertion; };
    }
    quote! {}
}

#[derive(Default)]
pub(crate) struct BoundaryOptions {
    /// Attune body policy; legacy frontends leave this unset.
    pub(crate) body_inline: Option<InlinePolicy>,
    /// Options spelled the same way in `#[rite]`: imports and `cfg(feature)`.
    pub(crate) shared: SharedOptions,
    /// Trusted generators may omit the accidental token-name mismatch check.
    /// Intrinsic feature checking remains enabled.
    pub(crate) suppress_const_test: bool,
    /// Use `#[inline(always)]` instead of `#[inline]` for the inner function.
    /// Requires nightly Rust with `#![feature(target_feature_inline_always)]`.
    pub(crate) inline_always: bool,
    /// The concrete type to use for `self` receiver.
    /// When specified, `self`/`&self`/`&mut self` is transformed to `_self: Type`/`&Type`/`&mut Type`.
    /// Implies `nested = true`.
    pub(crate) self_type: Option<Type>,
    /// Use a nested inner function instead of a sibling function. Spelled
    /// `nested` or `in_trait`; implied by `_self = Type`. Required in trait
    /// impls, which cannot take the extra sibling item.
    pub(crate) nested: bool,
    /// The function is a receiver-less associated function in an inherent
    /// impl, so the wrapper calls its sibling as `Self::__arcane_fn`. An
    /// attribute macro cannot see the enclosing impl, so this is spelled out.
    pub(crate) in_impl: bool,
    /// Attributes for the forwarding wrapper only. In sibling mode the
    /// wrapper holds no author code, so lint levels here never reach the
    /// author's body; on wasm32 (one function) and in nested mode they do.
    pub(crate) wrapper_attrs: Vec<Attribute>,
}

/// Shared implementation for arcane/arcane macros.
pub(crate) fn expand(
    mut input_fn: LightFn,
    macro_name: &str,
    args: BoundaryOptions,
) -> TokenStream {
    // Check for self receiver
    let has_self_receiver = input_fn
        .sig
        .inputs
        .first()
        .map(|arg| matches!(arg, FnArg::Receiver(_)))
        .unwrap_or(false);

    // Nested mode is required when _self = Type is used (for Self replacement in nested fn).
    // In sibling mode, self/Self work naturally since both fns live in the same impl scope.
    // However, if there's a self receiver in nested mode, we still need _self = Type.
    if has_self_receiver && args.nested && args.self_type.is_none() {
        let msg = format!(
            "{} with self receiver in nested mode requires `_self = Type` argument.\n\
             Example: #[{}(in_trait, _self = MyType)]\n\
             The body may keep using `self`; it is renamed to `_self` for the inner function.\n\
             \n\
             Alternatively, remove `nested`/`in_trait` to use sibling expansion (default), \
             which handles self/Self naturally in inherent impls.",
            macro_name, macro_name
        );
        return syn::Error::new_spanned(&input_fn.sig, msg).to_compile_error();
    }

    if let Some(err) = mixed_arch_error(&input_fn.sig, args.self_type.as_ref(), macro_name) {
        return err;
    }

    // Find the token parameter, its features, target arch, and token type name
    let mut from_self_receiver = false;
    let mut receiver_ref_depth = 0usize;
    let TokenParamInfo {
        ident: mut _token_ident,
        ref_depth,
        features,
        target_arch,
        token_type_name,
        magetypes_namespace,
        token_type,
        tier_traits,
    } = match find_token_param(&input_fn.sig).or_else(|| {
        // A concrete or trait-bounded token receiver can itself prove the required features.
        // Reuse normal discovery so token validation and feature lookup agree.
        let self_ty = args.self_type.as_ref()?;
        let Some(FnArg::Receiver(receiver)) = input_fn.sig.inputs.first() else {
            return None;
        };
        let mut receiver_sig = input_fn.sig.clone();
        receiver_sig.inputs.clear();
        receiver_sig.inputs.push(syn::parse_quote!(_self: #self_ty));
        let found = find_token_param(&receiver_sig)?;
        from_self_receiver = true;
        receiver_ref_depth = receiver_deref_depth(receiver) + found.ref_depth;
        Some(found)
    }) {
        Some(result) => result,
        None => {
            return missing_token_error(
                &input_fn.sig,
                args.self_type.as_ref(),
                macro_name,
                &format!(
                    ". Supported forms:\n\
                     - Concrete: `token: X64V3Token`\n\
                     - impl Trait: `token: impl HasX64V2`\n\
                     - Generic: `fn foo<T: HasX64V2>(token: T, ...)`\n\
                     - With self: `#[{macro_name}(_self = Type)] fn method(&self, token: impl HasNeon, ...)`"
                ),
            );
        }
    };

    // One token decides the features. Two would have been resolved by position,
    // which is the surprising rule issue #122 reports.
    let token_params = token_param_idents(&input_fn.sig);
    if token_params.len() > 1 {
        return multiple_tokens_error(&input_fn.sig, macro_name, &token_params);
    }

    // import_intrinsics with AVX-512 features needs archmage's avx512 feature
    // (propagated to archmage-macros) for the 512-bit safe memory wrappers.
    if let Some(err) = avx512_import_error(
        &input_fn.sig,
        args.shared.import_intrinsics,
        &features,
        token_type_name.as_deref().unwrap_or("an AVX-512 token"),
    ) {
        return err;
    }

    // Prepend import statements to body if requested
    prepend_to_body(
        &mut input_fn.body,
        generate_imports(
            target_arch,
            magetypes_namespace,
            args.shared.import_intrinsics,
            args.shared.import_magetypes,
        ),
    );

    // Only the forwarding wrapper needs identifiers for every argument. Keep
    // the user's patterns on the feature body; name wildcard proof parameters
    // so contextual dispatch can use them.
    let wrapper_sig = {
        let mut sig = input_fn.sig.clone();
        rename_non_ident_params(&mut sig);
        crate::common::rename_underscore_params(&mut sig);
        strip_forwarding_mutability(&mut sig);
        sig
    };
    let (wrapper_token_ident, wrapper_ref_depth) = if from_self_receiver {
        (format_ident!("self"), receiver_ref_depth)
    } else if let Some(info) = find_token_param(&wrapper_sig) {
        (info.ident, info.ref_depth)
    } else {
        (_token_ident.clone(), ref_depth)
    };
    if let Some(ident) = rename_wildcard_token(&mut input_fn.sig) {
        _token_ident = ident;
    }

    // Rewrite incant!() calls in the body to direct tier calls.
    // Only for concrete tokens where we can determine the tier suffix.
    if let Some(ref type_name) = token_type_name
        && let Some(tier_suffix) = crate::generated::canonical_token_to_tier_suffix(type_name)
        && let Some(tier) = crate::tiers::find_tier(tier_suffix)
    {
        let ctx = crate::rewrite::CallerContext {
            tier_suffix,
            target_arch: tier.target_arch,
            token_ident: _token_ident.clone(),
            token_ref_depth: ref_depth,
            has_token: true,
            derive_token: false,
        };
        input_fn.body =
            crate::rewrite::rewrite_incant_in_function(input_fn.body, &ctx, Some(&input_fn.sig));
    }

    if token_type_name.is_none() {
        input_fn.body = crate::attune::call::rewrite_context(
            input_fn.body,
            crate::attune::call::Context {
                features: &features,
                target_arch,
            },
            Some(&input_fn.sig),
        );
    }

    // Build a single target_feature attribute with all features comma-joined
    let features_csv = features_csv(token_type_name.as_deref(), &features);
    if args.body_inline == Some(InlinePolicy::Always) && !features.is_empty() {
        return syn::Error::new_spanned(
            &input_fn.sig.ident,
            "inline(always) on target-feature bodies requires nightly; use inline(hint) or inline(never)",
        ).to_compile_error();
    }
    // The operation's visibility determines the body policy. Native lowering
    // makes the implementation private without changing its inlining needs.
    let inline_attr = args
        .body_inline
        .unwrap_or(if args.inline_always {
            InlinePolicy::Always
        } else {
            InlinePolicy::Hint
        })
        .attribute(&input_fn.vis);

    let token_assertion =
        gen_token_assertion(&token_type_name, &token_type, args.suppress_const_test);

    // Scalar has no instruction-set boundary. Keep it directly in `impl`/item
    // scope (even when `_self = Type` is given), preserving outer `impl<T>`
    // generics, receiver lifetime elision, `const`/`async`, and `Self` while
    // binding `_self` with an explicit type annotation to enforce `Self == Type`.
    if features_csv.is_empty() {
        let self_binding = prepare_in_scope_self(&mut input_fn, args.self_type.as_ref());
        let (assert_ident, assert_depth) = if from_self_receiver {
            (format_ident!("_self"), receiver_ref_depth)
        } else {
            (_token_ident, ref_depth)
        };
        let direct_tier_trait_assertion =
            gen_tier_trait_assertion(&tier_traits, &assert_ident, assert_depth);
        mark_emitted_fn_header(&mut input_fn.vis, &mut input_fn.sig);
        let vis = &input_fn.vis;
        let sig = &input_fn.sig;
        let attrs = filter_inline_attrs(&input_fn.attrs);
        let body = &input_fn.body;
        let cfg = gen_cfg_guard(None, args.shared.cfg_feature.as_deref());
        let body = braced_body(
            &input_fn.brace_token,
            quote! { #self_binding #token_assertion #direct_tier_trait_assertion #body },
        );
        return quote! { #cfg #(#attrs)* #inline_attr #vis #sig #body };
    }
    let target_feature_attrs: Vec<Attribute> = if features_csv.is_empty() {
        Vec::new()
    } else {
        vec![parse_quote!(#[target_feature(enable = #features_csv)])]
    };

    // On wasm32, #[target_feature(enable = "simd128")] is safe even on trait
    // methods, so keep single-function emission in `impl`/item scope and bind
    // `_self` with a type check when `_self = Type` is present.
    if target_arch == Some("wasm32") {
        let self_binding = prepare_in_scope_self(&mut input_fn, args.self_type.as_ref());
        let (assert_ident, assert_depth) = if from_self_receiver {
            (format_ident!("_self"), receiver_ref_depth)
        } else {
            (_token_ident, ref_depth)
        };
        let direct_tier_trait_assertion =
            gen_tier_trait_assertion(&tier_traits, &assert_ident, assert_depth);
        return arcane_impl_wasm_safe(
            input_fn,
            &args,
            target_feature_attrs,
            inline_attr,
            quote! { #self_binding #token_assertion #direct_tier_trait_assertion },
        );
    }

    let safe_callee = features_csv.is_empty() || target_arch == Some("wasm32");
    let cfg_guard = gen_cfg_guard(target_arch, args.shared.cfg_feature.as_deref());
    drop_attrs_equal_to(&mut input_fn.attrs, &cfg_guard);
    let parts = BoundaryParts {
        cfg_guard,
        target_feature_attrs,
        inline_attr,
        token_assertion,
        tier_trait_assertion: gen_tier_trait_assertion(
            &tier_traits,
            &wrapper_token_ident,
            wrapper_ref_depth,
        ),
        wrapper_sig,
        safe_callee,
    };
    let callee = if args.nested {
        format!("__simd_inner_{}", input_fn.sig.ident)
    } else {
        format!("__arcane_{}", input_fn.sig.ident)
    };
    if let Some(err) = reserved_param_error(&input_fn.sig, &callee, macro_name) {
        return err;
    }
    if let Some(err) = reserved_param_error(&parts.wrapper_sig, &callee, macro_name) {
        return err;
    }
    emit_boundary(input_fn, &args, parts)
}

fn prepare_in_scope_self(input_fn: &mut LightFn, self_type: Option<&Type>) -> TokenStream {
    let Some(self_ty) = self_type else {
        return quote! {};
    };
    let Some(FnArg::Receiver(receiver)) = input_fn.sig.inputs.first_mut() else {
        return quote! {};
    };
    let binding = inline_self_binding(receiver, self_ty);
    receiver.mutability = None;
    input_fn.body =
        replace_self_value_in_tokens(std::mem::take(&mut input_fn.body), &format_ident!("_self"));
    binding
}

/// Reject any parameter binding (including `@` subpatterns, destructuring, and
/// `r#` raw identifiers) named like the generated callee, so nothing a caller
/// writes can stand between the wrapper's `unsafe` call and that function.
pub(crate) fn reserved_param_error(
    sig: &syn::Signature,
    callee: &str,
    macro_name: &str,
) -> Option<TokenStream> {
    use syn::ext::IdentExt as _;
    let target = callee.strip_prefix("r#").unwrap_or(callee);
    sig.inputs.iter().find_map(|arg| match arg {
        FnArg::Typed(pat_type) => {
            let mut bound = Vec::new();
            collect_pat_idents(&pat_type.pat, &mut bound);
            bound
                .into_iter()
                .find(|ident| ident.unraw() == target)
                .map(|ident| {
                    syn::Error::new_spanned(
                        ident,
                        format!(
                            "parameter `{target}` shadows the function #[{macro_name}] generates for \
                             `{}` and would receive its `unsafe` call; rename the parameter",
                            sig.ident
                        ),
                    )
                    .to_compile_error()
                })
        }
        FnArg::Receiver(_) => None,
    })
}

/// The pieces every boundary expansion emits: the cfg guard on both halves,
/// the attributes of the feature-enabled half, and the two assertions that
/// authenticate the token before the one `unsafe` call.
struct BoundaryParts {
    cfg_guard: TokenStream,
    target_feature_attrs: Vec<Attribute>,
    inline_attr: Option<Attribute>,
    token_assertion: TokenStream,
    tier_trait_assertion: TokenStream,
    wrapper_sig: syn::Signature,
    safe_callee: bool,
}

/// WASM-safe expansion: emits rite-style output (no unsafe wrapper) when `_self` is absent.
///
/// On wasm32, `#[target_feature(enable = "simd128")]` is safe — the wasm validation
/// model traps deterministically on unsupported instructions, so there's no UB.
/// We emit the function directly with `#[target_feature]` + `#[inline]`, like `#[rite]`,
/// while still running the token/trait identity assertions at the top of the body.
pub(crate) fn arcane_impl_wasm_safe(
    mut input_fn: LightFn,
    args: &BoundaryOptions,
    target_feature_attrs: Vec<Attribute>,
    inline_attr: Option<Attribute>,
    assertions: TokenStream,
) -> TokenStream {
    drop_attrs_equal_to(
        &mut input_fn.attrs,
        &gen_cfg_guard(Some("wasm32"), args.shared.cfg_feature.as_deref()),
    );
    mark_emitted_fn_header(&mut input_fn.vis, &mut input_fn.sig);
    let vis = &input_fn.vis;
    let sig = &input_fn.sig;
    let attrs = &input_fn.attrs;

    let original_body = &input_fn.body;
    let body = quote! {
        #assertions
        #original_body
    };

    // Prepend target_feature + inline attrs, filtering user #[inline] to avoid duplicates
    let mut new_attrs = target_feature_attrs;
    new_attrs.extend(inline_attr);
    // One function is both wrapper and body here.
    new_attrs.extend(args.wrapper_attrs.iter().cloned());
    for attr in filter_inline_attrs(attrs) {
        new_attrs.push(attr.clone());
    }

    let cfg_guard = gen_cfg_guard(Some("wasm32"), args.shared.cfg_feature.as_deref());
    let body = braced_body(&input_fn.brace_token, body);
    quote! {
        #cfg_guard
        #(#new_attrs)*
        #vis #sig #body
    }
}

/// One boundary emitter for sibling and nested placement. Placement changes
/// name resolution and the receiver representation, never the proof obligation.
fn emit_boundary(input: LightFn, options: &BoundaryOptions, parts: BoundaryParts) -> TokenStream {
    let BoundaryParts {
        cfg_guard,
        target_feature_attrs,
        inline_attr,
        token_assertion,
        tier_trait_assertion,
        wrapper_sig,
        safe_callee,
    } = parts;
    let sig = &input.sig;
    let generics = &sig.generics;
    let vis = &input.vis;
    let name = if options.nested {
        format_ident!("__simd_inner_{}", sig.ident)
    } else {
        format_ident!("__arcane_{}", sig.ident)
    };
    let mut unsafety = sig.safety.clone();
    if let syn::Safety::Unsafe(tok) = &mut unsafety {
        tok.span = proc_macro2::Span::call_site();
    }
    let attrs = if options.nested {
        filter_inline_attrs(&input.attrs)
            .into_iter()
            .cloned()
            .collect()
    } else {
        expect_as_allow(filter_inline_attrs(&input.attrs))
    };
    let body_attrs = body_attrs(&input.attrs, !options.nested);
    let turbofish = build_turbofish(generics);
    let has_self = sig
        .inputs
        .iter()
        .any(|arg| matches!(arg, FnArg::Receiver(_)));
    let self_ident = format_ident!("self");
    let forwarded: Vec<_> = wrapper_sig
        .inputs
        .iter()
        .filter_map(|arg| match arg {
            FnArg::Receiver(_) => options.nested.then_some(&self_ident),
            FnArg::Typed(param) => match param.pat.as_ref() {
                syn::Pat::Ident(pat) => Some(&pat.ident),
                _ => None,
            },
        })
        .collect();
    let (override_recv_lt, inner_generics_syn, resolved_output_syn) = if options.nested {
        resolve_nested_receiver_lifetime(sig)
    } else {
        (None, generics.clone(), sig.output.clone())
    };
    let params = if options.nested {
        let params: Vec<_> = sig
            .inputs
            .iter()
            .map(|arg| match arg {
                FnArg::Receiver(receiver) => {
                    let ty = options
                        .self_type
                        .as_ref()
                        .expect("nested receiver validated before emission");
                    nested_self_param_with_lifetime(receiver, ty, override_recv_lt.as_ref())
                        .to_token_stream()
                }
                FnArg::Typed(param) => options.self_type.as_ref().map_or_else(
                    || quote!(#param),
                    |ty| replace_self_in_tokens(quote!(#param), ty),
                ),
            })
            .collect();
        quote!(#(#params),*)
    } else {
        sig.inputs.to_token_stream()
    };
    let mut inner_generics = inner_generics_syn.to_token_stream();
    let mut output = resolved_output_syn.to_token_stream();
    let mut where_clause = inner_generics_syn.where_clause.to_token_stream();
    let mut body = input.body;
    if options.nested
        && let Some(ty) = &options.self_type
    {
        inner_generics = replace_self_in_tokens(inner_generics, ty);
        output = replace_self_in_tokens(output, ty);
        where_clause = replace_self_in_tokens(where_clause, ty);
        body =
            replace_self_value_in_tokens(replace_self_in_tokens(body, ty), &format_ident!("_self"));
    }
    let call = if options.nested {
        quote!(#name #turbofish(#(#forwarded),*))
    } else if has_self {
        quote!(self.#name #turbofish(#(#forwarded),*))
    } else if options.in_impl {
        quote!(Self::#name #turbofish(#(#forwarded),*))
    } else {
        quote!(#name #turbofish(#(#forwarded),*))
    };
    // Keep legacy attribute order, including lint expectations. The body is
    // emitted once; only its lexical placement differs between these policies.
    let before = if options.nested {
        quote!()
    } else {
        quote!(#[doc(hidden)] #(#body_attrs)*)
    };
    let after = if options.nested {
        quote!(#(#body_attrs)*)
    } else {
        quote!()
    };
    let body = braced_body(&input.brace_token, body);
    let inner = quote! {
        #before #(#target_feature_attrs)* #inline_attr #after
        #unsafety fn #name #inner_generics (#params) #output #where_clause #body
    };
    let proof_call = if safe_callee && matches!(unsafety, syn::Safety::Default) {
        call
    } else {
        proof_call(call)
    };
    let sibling = (!options.nested).then_some(&inner);
    let sibling_cfg = (!options.nested).then_some(&cfg_guard);
    let nested = options.nested.then_some(&inner);
    // The wrapper's parameters carry reserved names for any `_name` the author
    // wrote (`rename_underscore_params`), so forwarding them trips no lint; in
    // nested mode the author's body (the inner fn) keeps its own names. Clippy
    // does not report `inline_always` on this macro-spanned wrapper, so it
    // carries no lint levels of its own (measured 2026-10-24).
    let wrapper_attrs = &options.wrapper_attrs;
    quote! {
        #sibling_cfg #sibling
        #cfg_guard #(#attrs)* #(#wrapper_attrs)* #[inline(always)]
        #vis #wrapper_sig {
            #nested #token_assertion #tier_trait_assertion #proof_call
        }
    }
}

/// The only feature-boundary call emitted for native proof entries. Callers
/// must authenticate the supplied proof and resolve a body generated in the
/// same scope before reaching this function.
pub(crate) fn proof_call(call: TokenStream) -> TokenStream {
    // SAFETY: the enclosing proof entry establishes the callee's feature set.
    quote!(unsafe { #call })
}

/// Preserve Rust's receiver lifetime elision rule when lowering `&self` / `&mut self`
/// into an ordinary `_self: &Type` parameter on a nested inner function.
///
/// Only activates when a reference receiver coexists with at least one other
/// reference-bearing parameter AND an elided lifetime in the return type — the
/// exact configuration where Rust's single-input function lifetime elision no
/// longer assigns `_self`'s lifetime to the return type.
fn resolve_nested_receiver_lifetime(
    sig: &syn::Signature,
) -> (Option<syn::Lifetime>, syn::Generics, syn::ReturnType) {
    let mut generics = sig.generics.clone();
    let mut output = sig.output.clone();
    let Some(FnArg::Receiver(recv)) = sig.inputs.first() else {
        return (None, generics, output);
    };
    let raw_recv_lt = match &recv.kind {
        syn::ReceiverKind::Reference(_, lt, _) => lt.as_ref(),
        syn::ReceiverKind::Typed(_, ty) => {
            let mut cur: &Type = ty;
            while let Type::Paren(p) = cur {
                cur = &p.elem;
            }
            while let Type::Group(g) = cur {
                cur = &g.elem;
            }
            match cur {
                Type::Reference(tr) => tr.lifetime.as_ref(),
                _ => return (None, generics, output),
            }
        }
        _ => return (None, generics, output),
    };
    // Treat an explicit `'_` placeholder identically to an omitted lifetime:
    // leaving `'_` on a free inner function's `_self` parameter loses Rust's
    // receiver elision rule when another borrowed parameter is present.
    let recv_lt = raw_recv_lt.filter(|lt| lt.ident != "_");
    if !output_has_elided_lifetime(&output) {
        return (None, generics, output);
    }
    let other_input_has_ref = sig.inputs.iter().skip(1).any(arg_has_reference_or_lifetime);
    if !other_input_has_ref {
        return (None, generics, output);
    }
    let (override_lt, target_lt) = match recv_lt {
        Some(lt) => (None, lt.clone()),
        None => {
            let lt = syn::Lifetime::new("'__archmage_self", proc_macro2::Span::call_site());
            generics.params.insert(
                0,
                syn::GenericParam::Lifetime(syn::LifetimeParam::new(lt.clone())),
            );
            (Some(lt.clone()), lt)
        }
    };
    fill_elided_output_lifetimes(&mut output, &target_lt);
    (override_lt, generics, output)
}

fn output_has_elided_lifetime(output: &syn::ReturnType) -> bool {
    match output {
        syn::ReturnType::Default => false,
        syn::ReturnType::Type(_, ty) => type_has_elided_lifetime(ty),
    }
}

fn arg_has_reference_or_lifetime(arg: &FnArg) -> bool {
    match arg {
        FnArg::Receiver(r) => matches!(r.kind, syn::ReceiverKind::Reference(..)),
        FnArg::Typed(pt) => type_has_reference_or_lifetime(&pt.ty),
    }
}

fn type_has_reference_or_lifetime(ty: &Type) -> bool {
    match ty {
        Type::Reference(_) => true,
        Type::Paren(p) => type_has_reference_or_lifetime(&p.elem),
        Type::Group(g) => type_has_reference_or_lifetime(&g.elem),
        Type::Tuple(t) => t.elems.iter().any(type_has_reference_or_lifetime),
        Type::Slice(s) => type_has_reference_or_lifetime(&s.elem),
        Type::Array(a) => type_has_reference_or_lifetime(&a.elem),
        Type::Path(tp) => tp.path.segments.iter().any(|seg| match &seg.arguments {
            syn::PathArguments::AngleBracketed(ab) => ab.args.iter().any(|ga| match ga {
                syn::GenericArgument::Lifetime(_) => true,
                syn::GenericArgument::Type(inner) => type_has_reference_or_lifetime(inner),
                _ => false,
            }),
            _ => false,
        }),
        _ => false,
    }
}

fn type_has_elided_lifetime(ty: &Type) -> bool {
    match ty {
        Type::Reference(r) => {
            r.lifetime.as_ref().is_none_or(|lt| lt.ident == "_")
                || type_has_elided_lifetime(&r.elem)
        }
        Type::Paren(p) => type_has_elided_lifetime(&p.elem),
        Type::Group(g) => type_has_elided_lifetime(&g.elem),
        Type::Tuple(t) => t.elems.iter().any(type_has_elided_lifetime),
        Type::Slice(s) => type_has_elided_lifetime(&s.elem),
        Type::Array(a) => type_has_elided_lifetime(&a.elem),
        Type::Path(tp) => tp.path.segments.iter().any(|seg| match &seg.arguments {
            syn::PathArguments::AngleBracketed(ab) => ab.args.iter().any(|ga| match ga {
                syn::GenericArgument::Lifetime(lt) => lt.ident == "_",
                syn::GenericArgument::Type(inner) => type_has_elided_lifetime(inner),
                _ => false,
            }),
            _ => false,
        }),
        _ => false,
    }
}

fn fill_elided_output_lifetimes(output: &mut syn::ReturnType, target: &syn::Lifetime) {
    if let syn::ReturnType::Type(_, ty) = output {
        fill_elided_type_lifetimes(ty, target);
    }
}

fn fill_elided_type_lifetimes(ty: &mut Type, target: &syn::Lifetime) {
    match ty {
        Type::Reference(r) => {
            if r.lifetime.as_ref().is_none_or(|lt| lt.ident == "_") {
                r.lifetime = Some(target.clone());
            }
            fill_elided_type_lifetimes(&mut r.elem, target);
        }
        Type::Paren(p) => fill_elided_type_lifetimes(&mut p.elem, target),
        Type::Group(g) => fill_elided_type_lifetimes(&mut g.elem, target),
        Type::Tuple(t) => {
            for elem in &mut t.elems {
                fill_elided_type_lifetimes(elem, target);
            }
        }
        Type::Slice(s) => fill_elided_type_lifetimes(&mut s.elem, target),
        Type::Array(a) => fill_elided_type_lifetimes(&mut a.elem, target),
        Type::Path(tp) => {
            for seg in &mut tp.path.segments {
                if let syn::PathArguments::AngleBracketed(ab) = &mut seg.arguments {
                    for ga in &mut ab.args {
                        match ga {
                            syn::GenericArgument::Lifetime(lt) if lt.ident == "_" => {
                                *lt = target.clone();
                            }
                            syn::GenericArgument::Type(inner) => {
                                fill_elided_type_lifetimes(inner, target);
                            }
                            _ => {}
                        }
                    }
                }
            }
        }
        _ => {}
    }
}

#[cfg(test)]
mod assertion_tests {
    use super::*;
    use crate::arcane::ArcaneArgs;

    #[test]
    fn concrete_checks_are_a_single_shared_lookup() {
        let ty = Some(parse_quote!(renamed::X64V3Token));
        let check = gen_token_assertion(&Some("X64V3Token".into()), &ty, false);
        assert_eq!(
            check.to_string(),
            quote!(let _: () = <renamed::X64V3Token>::__ARCHMAGE_ASSERT_TIER_F38B284B;).to_string()
        );
        // No fresh const initializer, comparison, indexing, or branch per call.
        assert!(gen_token_assertion(&None, &None, false).is_empty());
        assert!(gen_token_assertion(&Some("X64V3Token".into()), &ty, true).is_empty());
    }

    #[test]
    fn no_shared_opt_in() {
        assert!(syn::parse_str::<ArcaneArgs>("shared").is_err());
    }

    #[test]
    fn in_impl_and_nested_are_exclusive() {
        assert!(syn::parse_str::<ArcaneArgs>("in_impl, nested").is_err());
        assert!(syn::parse_str::<ArcaneArgs>("in_impl").unwrap().in_impl);
        assert!(syn::parse_str::<ArcaneArgs>("in_trait").unwrap().nested);
    }
}
