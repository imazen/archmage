//! Shared utilities for all proc-macros.

use std::fmt::Write as _;

use proc_macro2::{Ident, Punct, Spacing, Span, TokenStream, TokenTree};
use quote::{ToTokens, format_ident, quote};
use syn::{Attribute, GenericParam, Signature, Type, parse::ParseStream, token};

/// Format `ident` into a stack `[u8; 64]` buffer when possible, avoiding a heap
/// `String` allocation on every identifier keyword/registry lookup.
#[inline]
pub(crate) fn with_ident_str<R>(ident: &Ident, f: impl FnOnce(&str) -> R) -> R {
    struct Buf {
        bytes: [u8; 64],
        len: usize,
        overflow: bool,
    }
    impl std::fmt::Write for Buf {
        #[inline]
        fn write_str(&mut self, s: &str) -> std::fmt::Result {
            let next = self.len + s.len();
            if next <= self.bytes.len() {
                self.bytes[self.len..next].copy_from_slice(s.as_bytes());
                self.len = next;
            } else {
                self.overflow = true;
            }
            Ok(())
        }
    }
    let mut buf = Buf {
        bytes: [0u8; 64],
        len: 0,
        overflow: false,
    };
    let _ = write!(buf, "{ident}");
    if !buf.overflow
        && let Ok(s) = std::str::from_utf8(&buf.bytes[..buf.len])
    {
        return f(s);
    }
    let owned = ident.to_string();
    f(&owned)
}

/// Safe, stack-first small buffer (`0` heap allocations up to `N` items).
#[derive(Clone, Debug)]
pub(crate) enum SmallVec<T, const N: usize> {
    Inline { buf: [Option<T>; N], len: usize },
    Heap(Vec<T>),
}

impl<T, const N: usize> Default for SmallVec<T, N> {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl<T, const N: usize> SmallVec<T, N> {
    #[inline]
    pub(crate) const fn new() -> Self {
        Self::Inline {
            buf: [const { None }; N],
            len: 0,
        }
    }

    #[inline]
    pub(crate) fn is_empty(&self) -> bool {
        self.len() == 0
    }

    #[inline]
    pub(crate) fn len(&self) -> usize {
        match self {
            Self::Inline { len, .. } => *len,
            Self::Heap(v) => v.len(),
        }
    }

    #[inline]
    pub(crate) fn push(&mut self, value: T) {
        match self {
            Self::Inline { buf, len } => {
                if *len < N {
                    buf[*len] = Some(value);
                    *len += 1;
                } else {
                    let mut heap = Vec::with_capacity(N + 2);
                    for slot in buf.iter_mut() {
                        if let Some(item) = slot.take() {
                            heap.push(item);
                        }
                    }
                    heap.push(value);
                    *self = Self::Heap(heap);
                }
            }
            Self::Heap(v) => v.push(value),
        }
    }

    #[inline]
    pub(crate) fn iter(&self) -> SmallVecIter<'_, T> {
        match self {
            Self::Inline { buf, len } => SmallVecIter::Inline(buf[..*len].iter()),
            Self::Heap(v) => SmallVecIter::Heap(v.iter()),
        }
    }

    #[inline]
    pub(crate) fn contains(&self, target: &T) -> bool
    where
        T: PartialEq,
    {
        self.iter().any(|item| item == target)
    }
}

impl<T, const N: usize> Extend<T> for SmallVec<T, N> {
    #[inline]
    fn extend<I: IntoIterator<Item = T>>(&mut self, iter: I) {
        for item in iter {
            self.push(item);
        }
    }
}

pub(crate) enum SmallVecIter<'a, T> {
    Inline(std::slice::Iter<'a, Option<T>>),
    Heap(std::slice::Iter<'a, T>),
}

impl<'a, T> Iterator for SmallVecIter<'a, T> {
    type Item = &'a T;
    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        match self {
            Self::Inline(it) => it.next().and_then(Option::as_ref),
            Self::Heap(it) => it.next(),
        }
    }
}

impl<'a, T, const N: usize> IntoIterator for &'a SmallVec<T, N> {
    type Item = &'a T;
    type IntoIter = SmallVecIter<'a, T>;
    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

/// Emit a `::`-separated identifier path (such as `"archmage::X64V3Token"`) directly
/// into a `TokenStream` without formatting a `String` or running `syn`'s parser.
pub(crate) fn colon_path_tokens(path: &str, rooted: bool) -> TokenStream {
    let mut out = TokenStream::new();
    let span = Span::call_site();
    if rooted {
        out.extend([
            TokenTree::Punct(Punct::new(':', Spacing::Joint)),
            TokenTree::Punct(Punct::new(':', Spacing::Alone)),
        ]);
    }
    for (i, seg) in path.split("::").enumerate() {
        if i > 0 {
            out.extend([
                TokenTree::Punct(Punct::new(':', Spacing::Joint)),
                TokenTree::Punct(Punct::new(':', Spacing::Alone)),
            ]);
        }
        out.extend([TokenTree::Ident(Ident::new(seg, span))]);
    }
    out
}

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
    fn parse(input: ParseStream<'_>) -> syn::Result<Self> {
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

impl LightFn {
    /// Substitute the family placeholder without printing and reparsing the
    /// entire function. Only syntax that contains `Token` needs to be parsed
    /// again; the body stays opaque, including macro argument markers.
    pub(crate) fn specialize_token(&mut self, token: &proc_macro2::TokenStream) -> syn::Result<()> {
        for attr in &mut self.attrs {
            specialize_syntax(&mut attr.meta, token)?;
        }
        specialize_syntax(&mut self.vis, token)?;
        specialize_syntax(&mut self.sig, token)?;
        self.body = replace_ident_in_tokens(std::mem::take(&mut self.body), "Token", token);
        Ok(())
    }
}

/// Preserve the parsed node when it has no placeholder. This also covers
/// `Token` in generic bounds, return types, attributes and visibility paths.
pub(crate) fn specialize_syntax<T: syn::parse::Parse + ToTokens>(
    syntax: &mut T,
    token: &proc_macro2::TokenStream,
) -> syn::Result<()> {
    let original = syntax.to_token_stream();
    if tokens_contain_ident(&original, &["Token"]) {
        *syntax = syn::parse2(replace_ident_in_tokens(original, "Token", token))?;
    }
    Ok(())
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

/// The attributes for the half of a boundary expansion that holds the user's
/// body (the `__arcane_` sibling, or the nested inner fn): the lint levels,
/// which must govern the body's code (#17), and `#[track_caller]`, which must
/// sit on every function between a panic in the body and the caller it names.
/// In nested mode the inner fn sits inside the wrapper, so the wrapper's lint
/// levels already cover it and `lint_levels` is false.
pub(crate) fn body_attrs(attrs: &[Attribute], lint_levels: bool) -> Vec<Attribute> {
    attrs
        .iter()
        .filter(|attr| (lint_levels && is_lint_attr(attr)) || attr.path().is_ident("track_caller"))
        .cloned()
        .collect()
}

/// The attributes for the wrapper or dispatcher that only forwards its
/// arguments, with each `#[expect(..)]` turned into `#[allow(..)]`. The
/// forwarding function uses every parameter and has no body of its own, so a
/// copy of the expectation there would be unfulfilled; the body half keeps
/// the `#[expect]`, which is fulfilled or reported there.
pub(crate) fn expect_as_allow<'a>(
    attrs: impl IntoIterator<Item = &'a Attribute>,
) -> Vec<Attribute> {
    attrs
        .into_iter()
        .map(|attr| {
            let mut attr = attr.clone();
            if attr.path().is_ident("expect")
                && let syn::Meta::List(list) = &mut attr.meta
                && let Some(last) = list.path.segments.last_mut()
            {
                last.ident = Ident::new("allow", last.ident.span());
            }
            attr
        })
        .collect()
}

/// Generate a cfg guard combining target_arch and an optional feature gate.
pub(crate) fn gen_cfg_guard(
    target_arch: Option<&str>,
    cfg_feature: Option<&str>,
) -> proc_macro2::TokenStream {
    gen_target_cfg_guard::<&str>(target_arch, cfg_feature, &[])
}

/// Generate a cfg guard combining `target_arch`, WASM compile-time
/// `target_feature` predicates (since WebAssembly validates the whole module at
/// load time and has no runtime instruction-set detection), and an optional
/// Cargo `feature` gate.
pub(crate) fn gen_target_cfg_guard<S: AsRef<str>>(
    target_arch: Option<&str>,
    cfg_feature: Option<&str>,
    target_features: &[S],
) -> proc_macro2::TokenStream {
    let mut preds = SmallVec::<proc_macro2::TokenStream, 5>::new();
    if let Some(arch) = target_arch {
        preds.push(quote! { target_arch = #arch });
        if arch == "wasm32" {
            for tf in target_features {
                let tf = tf.as_ref();
                preds.push(quote! { target_feature = #tf });
            }
        }
    }
    if let Some(feat) = cfg_feature {
        preds.push(quote! { feature = #feat });
    }
    match preds.len() {
        0 => quote! {},
        1 => {
            let only = preds.iter().next().unwrap();
            quote! { #[cfg(#only)] }
        }
        _ => {
            let iter = preds.iter();
            quote! { #[cfg(all(#(#iter),*))] }
        }
    }
}

/// Required target features for a registered tier name (used for WASM compile-time
/// `cfg(target_feature = ...)` gating).
pub(crate) fn tier_wasm_features(tier_name: &str) -> &'static [&'static str] {
    crate::generated::tier_to_canonical_token(tier_name)
        .and_then(crate::generated::token_to_features)
        .unwrap_or(&[])
}

/// Drop from `attrs` any `#[cfg]` equal to the guard the macro emits on both
/// halves of its output (a `#[magetypes]` variant arrives with its
/// `#[cfg(target_arch = ...)]` already on it), so the wrapper, which copies
/// the input's attributes, is not gated twice.
pub(crate) fn drop_attrs_equal_to(attrs: &mut Vec<Attribute>, guard: &proc_macro2::TokenStream) {
    if guard.is_empty() || !attrs.iter().any(|a| a.path().is_ident("cfg")) {
        return;
    }
    let wanted = guard.to_string();
    if wanted.is_empty() {
        return;
    }
    attrs.retain(|a| !a.path().is_ident("cfg") || a.to_token_stream().to_string() != wanted);
}

/// Build a turbofish token stream from a function's generics.
pub(crate) fn build_turbofish(generics: &syn::Generics) -> proc_macro2::TokenStream {
    if !generics
        .params
        .iter()
        .any(|p| matches!(p, GenericParam::Type(_) | GenericParam::Const(_)))
    {
        return quote! {};
    }
    let params = generics.params.iter().filter_map(|param| match param {
        GenericParam::Type(tp) => Some(&tp.ident),
        GenericParam::Const(cp) => Some(&cp.ident),
        GenericParam::Lifetime(_) => None,
    });
    quote! { ::<#(#params),*> }
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

/// Marker for a dispatch-macro invocation that a body rewriter already
/// expanded. Rewriters emit `<the caller's macro path>!(@__archmage_expanded
/// tokens)` instead of splicing `tokens` directly, so the caller's import of
/// that macro is used (no `unused_imports`) and the result keeps bang-macro
/// expression parsing (`incant!(f(x)) + 1` at statement start). A path cannot
/// begin with `@`, so the marker never collides with user input.
const EXPANDED_MARKER: &str = "__archmage_expanded";

/// The already-expanded tokens of a marked invocation, if `input` is one.
pub(crate) fn expanded_passthrough(
    input: &proc_macro2::TokenStream,
) -> Option<proc_macro2::TokenStream> {
    let mut tokens = input.clone().into_iter();
    match (tokens.next(), tokens.next()) {
        (Some(proc_macro2::TokenTree::Punct(at)), Some(proc_macro2::TokenTree::Ident(marker)))
            if at.as_char() == '@' && marker == EXPANDED_MARKER =>
        {
            Some(tokens.collect())
        }
        _ => None,
    }
}

/// Re-emit a rewritten invocation through the caller's own macro path and
/// bang (`path_and_bang`, copied verbatim) and the caller's delimiter: see
/// [`EXPANDED_MARKER`]. Keeping the delimiter keeps statement parsing as
/// written (`attuned!{ f(x) }` needs no trailing `;`, `attuned!(f(x))` does).
pub(crate) fn wrap_expanded(
    path_and_bang: impl IntoIterator<Item = proc_macro2::TokenTree>,
    original: &proc_macro2::Group,
    expanded: proc_macro2::TokenStream,
) -> proc_macro2::TokenStream {
    let span = original.span();
    let marker = Ident::new(EXPANDED_MARKER, span);
    let mut group = proc_macro2::Group::new(original.delimiter(), quote!(@ #marker #expanded));
    group.set_span(span);
    let mut out: proc_macro2::TokenStream = path_and_bang.into_iter().collect();
    out.extend([proc_macro2::TokenTree::Group(group)]);
    out
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
                let parser = |input: ParseStream<'_>| -> syn::Result<proc_macro2::TokenStream> {
                    let path: syn::Path = input.parse()?;
                    let args;
                    syn::parenthesized!(args in input);
                    let args = args.parse_terminated(syn::Expr::parse, syn::Token![,])?;
                    let rest: proc_macro2::TokenStream = input.parse()?;
                    let path = replace_ident_in_tokens(path.to_token_stream(), target, replacement);
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

pub(crate) fn is_bare_ident(expr: &syn::Expr, name: &str) -> bool {
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
pub(crate) fn gen_tier_trait_assertion<const N: usize>(
    tier_traits: &SmallVec<&'static str, N>,
    token_ident: &Ident,
) -> proc_macro2::TokenStream {
    if tier_traits.is_empty() {
        return quote! {};
    }
    let bounds = tier_traits.iter().map(|name| {
        let ident = Ident::new(name, Span::call_site());
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
    input: ParseStream<'_>,
    opts: &mut SharedOptions,
) -> syn::Result<bool> {
    with_ident_str(ident, |s| {
        match s {
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
    })
}

/// The error for `import_intrinsics` with AVX-512 features when archmage was
/// built without its `avx512` feature: the 512-bit safe memory wrappers are
/// missing, so `_mm512_loadu_ps` would resolve to the pointer-taking intrinsic.
#[cfg(not(feature = "avx512"))]
pub(crate) fn avx512_import_error(
    sig: &Signature,
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
    _sig: &Signature,
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
    sig: &Signature,
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
pub(crate) fn rename_non_ident_params(sig: &mut Signature) -> Vec<proc_macro2::TokenStream> {
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
            rebinds.push(if matches!(ty.as_ref(), Type::ImplTrait(_)) {
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

/// Forwarding layers (wrappers, proof wrappers, dispatchers) use every
/// parameter, so an author's `_name` (conventionally "unused") would trip
/// `clippy::used_underscore_binding` in them. Their bodies are generated, so
/// such a parameter takes a reserved name instead (`_t` becomes
/// `__archmage_t`; clippy exempts `__`-prefixed names). No rebinding: only
/// the forwarding call reads it. Feature bodies keep the author's names.
pub(crate) fn rename_underscore_params(sig: &mut Signature) {
    for arg in &mut sig.inputs {
        if let syn::FnArg::Typed(pat_type) = arg
            && let syn::Pat::Ident(pat) = pat_type.pat.as_mut()
            && let Some(renamed) = with_ident_str(&pat.ident, |name| {
                if name.starts_with('_') && !name.starts_with("__") && name.len() > 1 {
                    Some(format_ident!("__archmage{}", name, span = pat.ident.span()))
                } else {
                    None
                }
            })
        {
            pat.ident = renamed;
        }
    }
}

/// Apply the same `_x -> __archmage_x` forwarding rename to a single identifier.
pub(crate) fn forwarded_param_ident(ident: &Ident) -> Ident {
    with_ident_str(ident, |name| {
        if name.starts_with('_') && !name.starts_with("__") && name.len() > 1 {
            format_ident!("__archmage{}", name, span = ident.span())
        } else {
            ident.clone()
        }
    })
}

/// Give a wildcard token parameter (`_: X64V3Token`) the name
/// `find_token_param` synthesizes for it, so the nested dispatch rewrite and
/// the tier assertions can refer to it; every other pattern is left as
/// written. Returns the name when something was renamed.
pub(crate) fn rename_wildcard_token(sig: &mut Signature) -> Option<Ident> {
    let mut found = None;
    for (idx, arg) in sig.inputs.iter().enumerate() {
        let syn::FnArg::Typed(pat_type) = arg else {
            continue;
        };
        let syn::Pat::Wild(w) = pat_type.pat.as_ref() else {
            continue;
        };
        if crate::token_discovery::is_token_type_in_sig(sig, &pat_type.ty) {
            found = Some((idx, Ident::new("__archmage_token", w.underscore_token.span)));
            break;
        }
    }
    let (idx, ident) = found?;
    if let syn::FnArg::Typed(pat_type) = &mut sig.inputs[idx] {
        *pat_type.pat = syn::Pat::Ident(syn::PatIdent {
            attrs: vec![],
            by_ref: None,
            mutability: None,
            ident: ident.clone(),
            subpat: None,
        });
    }
    Some(ident)
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
pub(crate) fn nested_self_param(receiver: &syn::Receiver, self_ty: &Type) -> syn::FnArg {
    let ty: proc_macro2::TokenStream = match &receiver.kind {
        syn::ReceiverKind::Value => quote!(#self_ty),
        syn::ReceiverKind::Reference(_, lifetime, mutability) => {
            quote!(& #lifetime #mutability #self_ty)
        }
        syn::ReceiverKind::Typed(_, ty) => {
            let ts = ty.to_token_stream();
            if tokens_contain_ident(&ts, &["Self"]) {
                replace_self_in_tokens(ts, self_ty)
            } else {
                ts
            }
        }
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
                let stream = group.stream();
                if !tokens_contain_ident(&stream, &["self"]) {
                    result.extend([proc_macro2::TokenTree::Group(group)]);
                    continue;
                }
                let inner = replace_self_value_in_tokens(stream, replacement);
                let mut new_group = proc_macro2::Group::new(group.delimiter(), inner);
                new_group.set_span(group.span());
                result.extend([proc_macro2::TokenTree::Group(new_group)]);
            }
            other => result.extend([other]),
        }
    }
    result
}

/// Check that at most one parameter in `sig` is a token parameter; returns `None`
/// in the valid `<= 1` fast path without allocating, and builds the diagnostic
/// error only when 2+ token parameters are present.
pub(crate) fn check_single_token_param(
    sig: &Signature,
    macro_name: &str,
) -> Option<proc_macro2::TokenStream> {
    let mut count = 0usize;
    for arg in &sig.inputs {
        if let syn::FnArg::Typed(syn::PatType { ty, .. }) = arg
            && crate::token_discovery::is_token_type_in_sig(sig, ty)
        {
            count += 1;
        }
    }
    if count <= 1 {
        return None;
    }
    let mut idents = Vec::with_capacity(count);
    for arg in &sig.inputs {
        if let syn::FnArg::Typed(syn::PatType { pat, ty, .. }) = arg
            && crate::token_discovery::is_token_type_in_sig(sig, ty)
        {
            idents.push(match pat.as_ref() {
                syn::Pat::Ident(p) => p.ident.to_string(),
                _ => "_".to_string(),
            });
        }
    }
    Some(multiple_tokens_error(sig, macro_name, &idents))
}

/// The error for a signature with more than one token parameter: the macros
/// take their features from one token, and picking the first silently was
/// the surprising rule that issue #122 reports.
pub(crate) fn multiple_tokens_error(
    sig: &Signature,
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
