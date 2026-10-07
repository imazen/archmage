//! Shared utilities for all proc-macros.

use proc_macro2::Ident;
use quote::{ToTokens, format_ident, quote};
use syn::{Attribute, GenericParam, Signature, Type, parse::ParseStream, token};

/// A function parsed with the body left as an opaque TokenStream.
///
/// Only the signature is fully parsed into an AST — the body tokens are collected
/// without building any AST nodes (no expressions, statements, or patterns parsed).
/// Parsing cost therefore scales with the signature rather than a body AST.
#[derive(Clone)]
pub(crate) struct LightFn {
    pub attrs: Vec<Attribute>,
    pub vis: syn::Visibility,
    pub sig: Signature,
    pub brace_token: token::Brace,
    pub body: proc_macro2::TokenStream,
}

impl syn::parse::Parse for LightFn {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let attrs = input.call(Attribute::parse_outer)?;
        let vis: syn::Visibility = input.parse()?;
        let sig: Signature = input.parse()?;
        let content;
        let brace_token = syn::braced!(content in input);
        let body: proc_macro2::TokenStream = content.parse()?;
        Ok(LightFn {
            attrs,
            vis,
            sig,
            brace_token,
            body,
        })
    }
}

impl ToTokens for LightFn {
    fn to_tokens(&self, tokens: &mut proc_macro2::TokenStream) {
        for attr in &self.attrs {
            attr.to_tokens(tokens);
        }
        self.vis.to_tokens(tokens);
        self.sig.to_tokens(tokens);
        self.brace_token.surround(tokens, |tokens| {
            self.body.to_tokens(tokens);
        });
    }
}

/// Filter out `#[inline]`, `#[inline(always)]`, `#[inline(never)]` from attributes.
pub(crate) fn filter_inline_attrs(attrs: &[Attribute]) -> Vec<&Attribute> {
    attrs
        .iter()
        .filter(|attr| !attr.path().is_ident("inline"))
        .collect()
}

/// Check if an attribute is a lint-control attribute.
pub(crate) fn is_lint_attr(attr: &Attribute) -> bool {
    let path = attr.path();
    path.is_ident("allow")
        || path.is_ident("expect")
        || path.is_ident("deny")
        || path.is_ident("warn")
        || path.is_ident("forbid")
}

/// Extract lint-control attributes from a list of attributes.
pub(crate) fn filter_lint_attrs(attrs: &[Attribute]) -> Vec<&Attribute> {
    attrs.iter().filter(|attr| is_lint_attr(attr)).collect()
}

/// Generate a cfg guard combining target_arch and an optional feature gate.
pub(crate) fn gen_cfg_guard(
    target_arch: Option<&str>,
    cfg_feature: Option<&str>,
) -> proc_macro2::TokenStream {
    match (target_arch, cfg_feature) {
        (Some(arch), Some(feat)) => {
            quote! { #[cfg(all(target_arch = #arch, feature = #feat))] }
        }
        (Some(arch), None) => quote! { #[cfg(target_arch = #arch)] },
        (None, Some(feat)) => quote! { #[cfg(feature = #feat)] },
        (None, None) => quote! {},
    }
}

/// Build a turbofish token stream from a function's generics.
pub(crate) fn build_turbofish(generics: &syn::Generics) -> proc_macro2::TokenStream {
    let params: Vec<proc_macro2::TokenStream> = generics
        .params
        .iter()
        .filter_map(|param| match param {
            GenericParam::Type(tp) => {
                let ident = &tp.ident;
                Some(quote! { #ident })
            }
            GenericParam::Const(cp) => {
                let ident = &cp.ident;
                Some(quote! { #ident })
            }
            GenericParam::Lifetime(_) => None,
        })
        .collect();
    if params.is_empty() {
        quote! {}
    } else {
        quote! { ::<#(#params),*> }
    }
}

/// Conservative token-level presence check. Literals and partial identifier
/// matches do not count. A false positive only costs a rewrite pass; a false
/// negative would change expansion, so all delimiter kinds are traversed.
pub(crate) fn tokens_contain_ident(tokens: &proc_macro2::TokenStream, names: &[&str]) -> bool {
    tokens.clone().into_iter().any(|token| match token {
        proc_macro2::TokenTree::Ident(id) => names.iter().any(|name| id == *name),
        proc_macro2::TokenTree::Group(group) => tokens_contain_ident(&group.stream(), names),
        _ => false,
    })
}

/// Replace all occurrences of a named identifier in a token stream.
///
/// Recurses into groups (braces, parens, brackets). Each matching `Ident` is
/// replaced with `replacement` — which can be multiple tokens (e.g., a path
/// like `archmage::X64V3Token`). Non-matching tokens pass through unchanged.
pub(crate) fn replace_ident_in_tokens(
    tokens: proc_macro2::TokenStream,
    target: &str,
    replacement: &proc_macro2::TokenStream,
) -> proc_macro2::TokenStream {
    let mut result = proc_macro2::TokenStream::new();
    let mut tokens = tokens.into_iter().peekable();
    while let Some(tt) = tokens.next() {
        // A bare Token in an incant argument list is syntax, not a type.
        // Preserve only that marker; substitute Token in generic arguments,
        // type paths, and nested expressions normally.
        if target == "Token"
            && matches!(&tt, proc_macro2::TokenTree::Ident(id) if id == "incant" || id == "dispatch_variant")
            && matches!(tokens.peek(), Some(proc_macro2::TokenTree::Punct(p)) if p.as_char() == '!')
        {
            result.extend([tt, tokens.next().unwrap()]);
            if let Some(proc_macro2::TokenTree::Group(group)) = tokens.peek() {
                let group = group.clone();
                tokens.next();
                let parser =
                    |input: syn::parse::ParseStream| -> syn::Result<proc_macro2::TokenStream> {
                        let path: syn::Path = input.parse()?;
                        let args;
                        syn::parenthesized!(args in input);
                        let args = args.parse_terminated(syn::Expr::parse, syn::Token![,])?;
                        let rest: proc_macro2::TokenStream = input.parse()?;
                        let path =
                            replace_ident_in_tokens(path.to_token_stream(), target, replacement);
                        let args = args.iter().map(|arg| {
                            if matches!(arg, syn::Expr::Path(p) if p.path.is_ident("Token")) {
                                arg.to_token_stream()
                            } else {
                                replace_ident_in_tokens(arg.to_token_stream(), target, replacement)
                            }
                        });
                        let rest = replace_ident_in_tokens(rest, target, replacement);
                        Ok(quote! { #path(#(#args),*) #rest })
                    };
                use syn::parse::{Parse, Parser};
                let replaced = parser
                    .parse2(group.stream())
                    .unwrap_or_else(|_| group.stream());
                let mut new_group = proc_macro2::Group::new(group.delimiter(), replaced);
                new_group.set_span(group.span());
                result.extend([proc_macro2::TokenTree::Group(new_group)]);
            }
            continue;
        }
        match tt {
            proc_macro2::TokenTree::Ident(ref ident) if *ident == target => {
                result.extend(replacement.clone());
            }
            proc_macro2::TokenTree::Group(group) => {
                let stream = group.stream();
                if !tokens_contain_ident(&stream, &[target]) {
                    result.extend(std::iter::once(proc_macro2::TokenTree::Group(group)));
                    continue;
                }
                let new_stream = replace_ident_in_tokens(stream, target, replacement);
                let mut new_group = proc_macro2::Group::new(group.delimiter(), new_stream);
                new_group.set_span(group.span());
                result.extend(std::iter::once(proc_macro2::TokenTree::Group(new_group)));
            }
            other => {
                result.extend(std::iter::once(other));
            }
        }
    }
    result
}

/// Replace all `Self` identifier tokens with a concrete type in a token stream.
pub(crate) fn replace_self_in_tokens(
    tokens: proc_macro2::TokenStream,
    replacement: &Type,
) -> proc_macro2::TokenStream {
    replace_ident_in_tokens(tokens, "Self", &replacement.to_token_stream())
}

/// Generate import statements for intrinsics and/or magetypes.
pub(crate) fn generate_imports(
    target_arch: Option<&str>,
    magetypes_namespace: Option<&str>,
    import_intrinsics: bool,
    import_magetypes: bool,
) -> proc_macro2::TokenStream {
    let mut imports = proc_macro2::TokenStream::new();

    if import_intrinsics && let Some(arch) = target_arch {
        let arch_ident = quote::format_ident!("{}", arch);
        imports.extend(quote! {
            #[allow(unused_imports)]
            use archmage::intrinsics::#arch_ident::*;
        });
    }

    if import_magetypes && let Some(ns) = magetypes_namespace {
        let ns_ident = quote::format_ident!("{}", ns);
        imports.extend(quote! {
            #[allow(unused_imports)]
            use magetypes::simd::#ns_ident::*;
            #[allow(unused_imports)]
            use magetypes::simd::backends::*;
        });
    }

    imports
}

/// Check if any argument expression contains the `Token` identifier.
/// Check if an expression is a bare ident matching a given name.
pub(crate) fn is_bare_ident_pub(expr: &syn::Expr, name: &str) -> bool {
    is_bare_ident(expr, name)
}

fn is_bare_ident(expr: &syn::Expr, name: &str) -> bool {
    match expr {
        syn::Expr::Path(p) => {
            p.qself.is_none() && p.path.segments.len() == 1 && p.path.segments[0].ident == name
        }
        _ => false,
    }
}

/// Detect how the token is placed in incant! arguments.
///
/// Returns the kind of token placement found:
/// - `TokenPlacement::Explicit(index)`: arg at `index` is the `Token` marker
/// - `TokenPlacement::Variable(index)`: arg at `index` is the caller's token variable
/// - `TokenPlacement::None`: no token in args (will be prepended, deprecated)
pub(crate) enum TokenPlacement {
    /// `Token` marker found in args
    Explicit,
    /// Caller's token variable name found at this arg index
    Variable(usize),
    /// No token in args — will prepend (deprecated)
    None,
}

/// Find where the token is placed in incant! arguments.
///
/// Checks for:
/// 1. `Token` marker (explicit placeholder for summon mode)
/// 2. A bare ident matching `caller_token_ident` (the actual variable)
///
/// Falls back to `None` (token will be prepended).
pub(crate) fn find_token_placement(
    args: &[syn::Expr],
    caller_token_ident: Option<&str>,
) -> TokenPlacement {
    for arg in args.iter() {
        if is_bare_ident(arg, "Token") {
            return TokenPlacement::Explicit;
        }
    }
    if let Some(ident_name) = caller_token_ident {
        for (i, arg) in args.iter().enumerate() {
            if is_bare_ident(arg, ident_name) {
                return TokenPlacement::Variable(i);
            }
        }
    }
    TokenPlacement::None
}

/// Build call arguments with the token in the correct position.
///
/// Handles three cases:
/// - `Token` marker in args: replace with `token_expr`
/// - Caller's token variable in args: replace with `token_expr` (may downcast)
/// - Neither: prepend `token_expr` (backward compat, deprecated)
pub(crate) fn build_call_args(
    args: &[syn::Expr],
    token_expr: &proc_macro2::TokenStream,
) -> proc_macro2::TokenStream {
    build_call_args_with_ident(args, token_expr, None)
}

/// Build call arguments, also checking for the caller's token variable name.
pub(crate) fn build_call_args_with_ident(
    args: &[syn::Expr],
    token_expr: &proc_macro2::TokenStream,
    caller_token_ident: Option<&str>,
) -> proc_macro2::TokenStream {
    match find_token_placement(args, caller_token_ident) {
        TokenPlacement::Explicit => {
            // Replace Token marker with token expression
            let replaced = args
                .iter()
                .map(|arg| replace_ident_in_tokens(arg.to_token_stream(), "Token", token_expr));
            quote! { #(#replaced),* }
        }
        TokenPlacement::Variable(idx) => {
            // Replace the caller's token variable with the target token expression
            let replaced = args.iter().enumerate().map(|(i, arg)| {
                if i == idx {
                    token_expr.clone()
                } else {
                    arg.to_token_stream()
                }
            });
            quote! { #(#replaced),* }
        }
        TokenPlacement::None => {
            // No token in args — prepend (backward compat)
            quote! { #token_expr, #(#args),* }
        }
    }
}

/// Build call arguments for scalar fallback.
///
/// If `Token` marker or caller's token variable is in args, replace with `ScalarToken`.
/// If neither, prepend `ScalarToken`.
pub(crate) fn build_scalar_call_args(args: &[syn::Expr]) -> proc_macro2::TokenStream {
    let scalar = quote! { archmage::ScalarToken };
    build_call_args(args, &scalar)
}

/// Suffix the last segment of a path: `process` → `process_v3`.
pub(crate) fn suffix_path(path: &syn::Path, suffix: &str) -> syn::Path {
    let mut suffixed = path.clone();
    if let Some(last) = suffixed.segments.last_mut() {
        last.ident = quote::format_ident!("{}_{}", last.ident, suffix);
    }
    suffixed
}

/// Emit the tier-trait identity assertion for a trait or generic token bound.
///
/// `#[arcane]`/`#[rite]` resolve a tier trait **by name**: seeing `HasX64V2` in
/// the signature is what selects the SSE4.2 `#[target_feature]` list. On its own
/// that proves nothing, because a downstream crate can declare
///
/// ```ignore
/// pub trait HasX64V2 {}
/// impl HasX64V2 for NotAToken {}
/// ```
///
/// and receive the whole feature set with no token, no sealing and no runtime
/// detection — in ordinary safe code, with no diagnostic. On a CPU without the
/// features that is SIGILL.
///
/// A concrete token is already covered: the wrapper asserts its generated
/// `__ARCHMAGE_ASSERT_TIER_<tag>` const, which a same-named local struct does
/// not have. This is the equivalent for the bound case. It re-states each tier
/// trait through an **absolute** `::archmage::` path, which a local trait cannot
/// shadow, and requires the token value's type to satisfy it:
///
/// ```ignore
/// const fn __archmage_assert_tier_trait<__T: ?Sized + ::archmage::HasX64V2>(_: &__T) {}
/// __archmage_assert_tier_trait(&token);
/// ```
///
/// Because archmage's tier traits are sealed through `SimdToken`, satisfying the
/// bound is only possible for a genuine token of that tier or stronger. A
/// same-named local trait fails with rustc's own diagnostic, which names both:
/// "`impl HasX64V2` implements similarly named trait `HasX64V2`, but not
/// `archmage::HasX64V2`". A real but *weaker* token fails too, closing the
/// trait-bound analogue of the token-aliasing hole.
///
/// It is a `const fn` so it provably has no runtime body, and its parameter is
/// taken by reference so the form works for `impl Trait` in argument position,
/// where the type cannot be named. All bounds are emitted on one helper, so a
/// multi-trait bound (`impl HasNeon + HasNeonAes`) is checked as the same union
/// of tiers the `#[target_feature]` list was built from.
///
/// Returns nothing for a concrete token (`tier_traits` empty) — that path keeps
/// its cheaper const assertion.
pub(crate) fn gen_tier_trait_assertion(
    tier_traits: &[String],
    token_ident: &Ident,
) -> proc_macro2::TokenStream {
    if tier_traits.is_empty() {
        return quote! {};
    }
    let bounds = tier_traits.iter().map(|name| {
        let ident = format_ident!("{}", name);
        quote! { + ::archmage::#ident }
    });
    quote! {
        {
            #[inline(always)]
            const fn __archmage_assert_tier_trait<__T: ?Sized #(#bounds)*>(_: &__T) {}
            __archmage_assert_tier_trait(&#token_ident);
        }
    }
}

#[cfg(test)]
mod dispatch_marker_tests {
    use super::*;

    #[test]
    fn substitutes_types_but_preserves_dispatch_markers() {
        let source = quote! {
            fn f(t: Token) {
                archmage::incant!(callee::<Token>(Token, Token::from_context(), wrap::<Token>()), [scalar]);
                dispatch_variant!(other(value, Token) with t);
            }
        };
        let expected = quote! {
            fn f(t: archmage::ScalarToken) {
                archmage::incant!(callee::<archmage::ScalarToken>(Token, archmage::ScalarToken::from_context(), wrap::<archmage::ScalarToken>()), [scalar]);
                dispatch_variant!(other(value, Token) with t);
            }
        };
        assert_eq!(
            replace_ident_in_tokens(source, "Token", &quote!(archmage::ScalarToken)).to_string(),
            expected.to_string()
        );
    }
}

// ============================================================================
// Helpers shared by #[arcane] and #[rite]
// ============================================================================

/// Options that `#[arcane]` and `#[rite]` spell the same way.
///
/// Each macro's parser calls [`parse_shared_option`] first and handles only
/// its own options itself, so the two accept identical spellings for the
/// shared ones and give the same diagnostics.
#[derive(Default)]
pub(crate) struct SharedOptions {
    /// Inject `use archmage::intrinsics::{arch}::*;` (includes safe memory ops).
    pub(crate) import_intrinsics: bool,
    /// Inject `use magetypes::simd::{ns}::*;`, `use magetypes::simd::generic::*;`,
    /// and `use magetypes::simd::backends::*;`.
    pub(crate) import_magetypes: bool,
    /// Additional cargo feature gate: `cfg(avx512)` adds `feature = "avx512"`
    /// to the generated `#[cfg(...)]`.
    pub(crate) cfg_feature: Option<String>,
}

/// Parse one of the shared options. Returns `Ok(true)` when `ident` named one.
pub(crate) fn parse_shared_option(
    ident: &Ident,
    input: syn::parse::ParseStream,
    opts: &mut SharedOptions,
) -> syn::Result<bool> {
    match ident.to_string().as_str() {
        "import_intrinsics" => opts.import_intrinsics = true,
        "import_magetypes" => opts.import_magetypes = true,
        "cfg" => {
            let content;
            syn::parenthesized!(content in input);
            let feat: Ident = content.parse()?;
            opts.cfg_feature = Some(feat.to_string());
        }
        "stub" => {
            return Err(syn::Error::new(
                ident.span(),
                "`stub` has been removed. Use `incant!` for cross-arch dispatch \
                 instead — it cfg-gates each architecture automatically.\n\
                 \n\
                 Before: #[arcane(stub)] fn process(token: X64V3Token, ...) { ... }\n\
                 After:  #[arcane] fn process_v3(token: X64V3Token, ...) { ... }\n\
                 \x20       fn dispatch(...) { incant!(process(...)) }",
            ));
        }
        _ => return Ok(false),
    }
    Ok(true)
}

/// The error for `import_intrinsics` with AVX-512 features when archmage was
/// built without its `avx512` feature: the 512-bit safe memory wrappers are
/// missing, so `_mm512_loadu_ps` would resolve to the pointer-taking intrinsic.
#[cfg(not(feature = "avx512"))]
pub(crate) fn avx512_import_error(
    sig: &syn::Signature,
    import_intrinsics: bool,
    features: &[&str],
    token_desc: &str,
) -> Option<proc_macro2::TokenStream> {
    if !import_intrinsics || !features.iter().any(|f| f.starts_with("avx512")) {
        return None;
    }
    let msg = format!(
        "Using {token_desc} with `import_intrinsics` requires the `avx512` feature.\n\
         \n\
         Add to your Cargo.toml:\n\
         \x20 archmage = {{ version = \"...\", features = [\"avx512\"] }}\n\
         \n\
         Without it, 512-bit safe memory ops (_mm512_loadu_ps etc.) are not available.\n\
         If you only need value intrinsics (no memory ops), remove `import_intrinsics`."
    );
    Some(syn::Error::new_spanned(sig, msg).to_compile_error())
}

#[cfg(feature = "avx512")]
pub(crate) fn avx512_import_error(
    _sig: &syn::Signature,
    _import_intrinsics: bool,
    _features: &[&str],
    _token_desc: &str,
) -> Option<proc_macro2::TokenStream> {
    None
}

/// The error when a signature has no usable token parameter. A featureless
/// bound such as `SimdToken` gets its own explanation; `forms` lists the
/// macro's accepted spellings.
pub(crate) fn missing_token_error(
    sig: &syn::Signature,
    macro_name: &str,
    forms: &str,
) -> proc_macro2::TokenStream {
    if let Some(trait_name) = crate::token_discovery::diagnose_featureless_token(sig) {
        let msg = format!(
            "`{trait_name}` cannot be used as a token bound in #[{macro_name}] \
             because it doesn't specify any CPU features.\n\
             \n\
             #[{macro_name}] needs concrete features to generate #[target_feature]. \
             Use a concrete token or a feature trait:\n\
             \n\
             Concrete tokens: X64V3Token, Desktop64, NeonToken, Arm64V2Token, ...\n\
             Feature traits:  impl HasX64V2, impl HasNeon, impl HasArm64V3, ...{}",
            if macro_name == "rite" {
                "\nTier names:      #[rite(v3)], #[rite(neon)], #[rite(v4)], ..."
            } else {
                ""
            }
        );
        return syn::Error::new_spanned(sig, msg).to_compile_error();
    }
    let msg = format!("{macro_name} requires a token parameter{forms}");
    syn::Error::new_spanned(sig, msg).to_compile_error()
}

/// Rename every non-identifier parameter pattern to a generated identifier,
/// so the parameter can be forwarded by name, and return the statements that
/// re-bind the original pattern inside the body.
///
/// `_: T` becomes `__archmage_arg_0: T` with no re-binding (a wildcard binds
/// nothing). `(a, b): (T, U)` becomes `__archmage_arg_1: (T, U)` plus
/// `let (a, b): (T, U) = __archmage_arg_1;`. The type annotation is left off
/// when the type is `impl Trait`, which a `let` cannot name.
pub(crate) fn rename_non_ident_params(sig: &mut syn::Signature) -> Vec<proc_macro2::TokenStream> {
    let mut rebinds = Vec::new();
    let mut counter = 0u32;
    for arg in &mut sig.inputs {
        let syn::FnArg::Typed(pat_type) = arg else {
            continue;
        };
        if matches!(pat_type.pat.as_ref(), syn::Pat::Ident(_)) {
            continue;
        }
        let generated = format_ident!("__archmage_arg_{}", counter);
        counter += 1;
        let original_pat = pat_type.pat.clone();
        let ty = &pat_type.ty;
        if !matches!(original_pat.as_ref(), syn::Pat::Wild(_)) {
            rebinds.push(if matches!(ty.as_ref(), syn::Type::ImplTrait(_)) {
                quote! { let #original_pat = #generated; }
            } else {
                quote! { let #original_pat: #ty = #generated; }
            });
        }
        *pat_type.pat = syn::Pat::Ident(syn::PatIdent {
            attrs: vec![],
            by_ref: None,
            mutability: None,
            ident: generated,
            subpat: None,
        });
    }
    rebinds
}

/// Prepend statements to a function body.
pub(crate) fn prepend_to_body(
    body: &mut proc_macro2::TokenStream,
    prefix: proc_macro2::TokenStream,
) {
    if prefix.is_empty() {
        return;
    }
    let original = std::mem::take(body);
    *body = quote! { #prefix #original };
}

/// The `_self` parameter a nested function takes in place of a receiver,
/// with the receiver's own shape kept: `&'a self` becomes `_self: &'a Type`,
/// `&mut self` becomes `_self: &mut Type`, `self: Box<Self>` becomes
/// `_self: Box<Type>`, and `self` / `mut self` become `_self: Type`.
pub(crate) fn nested_self_param(receiver: &syn::Receiver, self_ty: &syn::Type) -> syn::FnArg {
    let ty: proc_macro2::TokenStream = match &receiver.kind {
        syn::ReceiverKind::Value => quote!(#self_ty),
        syn::ReceiverKind::Reference(_, lifetime, mutability) => {
            quote!(& #lifetime #mutability #self_ty)
        }
        syn::ReceiverKind::Typed(_, ty) => replace_self_in_tokens(ty.to_token_stream(), self_ty),
        // `#[non_exhaustive]`: a receiver shape syn adds later is passed by value.
        _ => quote!(#self_ty),
    };
    syn::parse_quote!(_self: #ty)
}

/// Replace the value `self` with another identifier, leaving `self::` paths
/// alone. Used when a method body moves into a nested function that receives
/// the receiver as `_self`.
pub(crate) fn replace_self_value_in_tokens(
    tokens: proc_macro2::TokenStream,
    replacement: &Ident,
) -> proc_macro2::TokenStream {
    let mut result = proc_macro2::TokenStream::new();
    let mut tokens = tokens.into_iter().peekable();
    // A nested `impl` or `trait` item has receivers of its own: once one of
    // those keywords is seen, the next brace group is that item's body and is
    // copied untouched. Closures and blocks still refer to the outer `self`
    // and are rewritten.
    let mut in_item_header = false;
    while let Some(tt) = tokens.next() {
        match tt {
            proc_macro2::TokenTree::Ident(ref ident) if *ident == "impl" || *ident == "trait" => {
                in_item_header = true;
                result.extend([tt]);
            }
            proc_macro2::TokenTree::Ident(ref ident) if *ident == "self" => {
                let is_path = matches!(tokens.peek(), Some(proc_macro2::TokenTree::Punct(p)) if p.as_char() == ':');
                if is_path {
                    result.extend([tt]);
                } else {
                    let mut replaced = replacement.clone();
                    replaced.set_span(ident.span());
                    result.extend([proc_macro2::TokenTree::Ident(replaced)]);
                }
            }
            proc_macro2::TokenTree::Group(group)
                if in_item_header && group.delimiter() == proc_macro2::Delimiter::Brace =>
            {
                in_item_header = false;
                result.extend([proc_macro2::TokenTree::Group(group)]);
            }
            proc_macro2::TokenTree::Group(group) => {
                let inner = replace_self_value_in_tokens(group.stream(), replacement);
                let mut new_group = proc_macro2::Group::new(group.delimiter(), inner);
                new_group.set_span(group.span());
                result.extend([proc_macro2::TokenTree::Group(new_group)]);
            }
            other => result.extend([other]),
        }
    }
    result
}

/// The identifiers of every parameter whose type is a token (concrete, trait
/// bound or bounded generic), in signature order.
pub(crate) fn token_param_idents(sig: &syn::Signature) -> Vec<String> {
    let mut out = Vec::new();
    for arg in &sig.inputs {
        let syn::FnArg::Typed(syn::PatType { pat, ty, .. }) = arg else {
            continue;
        };
        let Some(info) = crate::token_discovery::extract_token_type_info(ty) else {
            continue;
        };
        let is_token = match &info {
            crate::token_discovery::TokenTypeInfo::Concrete(_) => true,
            crate::token_discovery::TokenTypeInfo::ImplTrait(names) => {
                crate::token_discovery::traits_to_features(names).is_some()
            }
            crate::token_discovery::TokenTypeInfo::Generic(name) => {
                crate::token_discovery::find_generic_bounds(sig, name)
                    .and_then(|b| crate::token_discovery::traits_to_features(&b))
                    .is_some()
            }
        };
        if is_token {
            out.push(match pat.as_ref() {
                syn::Pat::Ident(p) => p.ident.to_string(),
                _ => "_".to_string(),
            });
        }
    }
    out
}

/// The error for a signature with more than one token parameter: the macros
/// take their features from one token, and picking the first silently was
/// the surprising rule that issue #122 reports.
pub(crate) fn multiple_tokens_error(
    sig: &syn::Signature,
    macro_name: &str,
    idents: &[String],
) -> proc_macro2::TokenStream {
    let list = idents
        .iter()
        .map(|i| format!("`{i}`"))
        .collect::<Vec<_>>()
        .join(", ");
    let hint = if macro_name == "rite" {
        "name the tier instead: `#[rite(v3)]`, or keep one token parameter"
    } else {
        "keep the strongest token as the parameter and derive the weaker ones inside \
         the body with its extractors (`token.v3()`, `token.v2()`), or split the \
         function so each part takes one token"
    };
    let msg = format!(
        "#[{macro_name}] found {} token parameters ({list}) and takes its CPU features \
         from only one. {hint}.",
        idents.len()
    );
    syn::Error::new_spanned(sig, msg).to_compile_error()
}
