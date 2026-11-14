//! Lightweight `syn/derive`-only function signature and call-argument parsers.
//!
//! Replaces `syn/full`'s `Signature`, `FnArg`, `Pat`, and `Expr` ASTs so
//! `archmage-macros` compiles without `syn`'s `full` or `visit-mut` features
//! (~40 KLOC of unused Rust item/statement/expression/pattern AST code).

use proc_macro2::{Delimiter, TokenStream, TokenTree};
use quote::ToTokens;
use syn::{
    Attribute, Generics, Ident, Lifetime, LitStr, ReturnType, Token, Type,
    parse::{Parse, ParseStream, discouraged::Speculative},
    punctuated::Punctuated,
    token,
};

/// Matches `syn::Safety` (`Unsafe` vs `Default`) without `syn/full`.
#[derive(Clone)]
pub(crate) enum Safety {
    Unsafe(Token![unsafe]),
    Default,
}

impl ToTokens for Safety {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        if let Self::Unsafe(tok) = self {
            tok.to_tokens(tokens);
        }
    }
}

/// Optional `extern "C"` ABI qualifier on a function signature.
#[derive(Clone)]
pub(crate) struct FnAbi {
    pub extern_token: Token![extern],
    pub name: Option<LitStr>,
}

impl Parse for FnAbi {
    fn parse(input: ParseStream<'_>) -> syn::Result<Self> {
        Ok(Self {
            extern_token: input.parse()?,
            name: input.parse()?,
        })
    }
}

impl ToTokens for FnAbi {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        self.extern_token.to_tokens(tokens);
        self.name.to_tokens(tokens);
    }
}

/// Lightweight replacement for `syn::Signature` using only `syn/derive` types.
#[derive(Clone)]
pub(crate) struct Signature {
    pub constness: Option<Token![const]>,
    pub asyncness: Option<Token![async]>,
    pub safety: Safety,
    pub abi: Option<FnAbi>,
    pub fn_token: Token![fn],
    pub ident: Ident,
    pub generics: Generics,
    pub paren_token: token::Paren,
    pub inputs: Punctuated<FnArg, Token![,]>,
    pub output: ReturnType,
}

impl Parse for Signature {
    fn parse(input: ParseStream<'_>) -> syn::Result<Self> {
        let constness: Option<Token![const]> = input.parse()?;
        let asyncness: Option<Token![async]> = input.parse()?;
        let safety = if input.peek(Token![unsafe]) {
            Safety::Unsafe(input.parse()?)
        } else {
            Safety::Default
        };
        let abi = if input.peek(Token![extern]) {
            Some(input.parse()?)
        } else {
            None
        };
        let fn_token: Token![fn] = input.parse()?;
        let ident: Ident = input.parse()?;
        let mut generics: Generics = input.parse()?;
        let content;
        let paren_token = syn::parenthesized!(content in input);
        let inputs = content.parse_terminated(FnArg::parse, Token![,])?;
        let output: ReturnType = input.parse()?;
        generics.where_clause = input.parse()?;
        Ok(Self {
            constness,
            asyncness,
            safety,
            abi,
            fn_token,
            ident,
            generics,
            paren_token,
            inputs,
            output,
        })
    }
}

impl ToTokens for Signature {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        self.constness.to_tokens(tokens);
        self.asyncness.to_tokens(tokens);
        self.safety.to_tokens(tokens);
        self.abi.to_tokens(tokens);
        self.fn_token.to_tokens(tokens);
        self.ident.to_tokens(tokens);
        self.generics.to_tokens(tokens);
        self.paren_token.surround(tokens, |tokens| {
            self.inputs.to_tokens(tokens);
        });
        self.output.to_tokens(tokens);
        self.generics.where_clause.to_tokens(tokens);
    }
}

#[derive(Clone)]
pub(crate) enum FnArg {
    Receiver(Receiver),
    Typed(PatType),
}

#[derive(Clone)]
pub(crate) struct Receiver {
    pub attrs: Vec<Attribute>,
    pub mutability: Option<Token![mut]>,
    pub self_token: Token![self],
    pub kind: ReceiverKind,
}

#[derive(Clone)]
pub(crate) enum ReceiverKind {
    Value,
    Reference(Token![&], Option<Lifetime>, Option<Token![mut]>),
    Typed(Token![:], Box<Type>),
}

impl ToTokens for Receiver {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        for attr in &self.attrs {
            attr.to_tokens(tokens);
        }
        match &self.kind {
            ReceiverKind::Reference(and_tok, lt, mut_tok) => {
                and_tok.to_tokens(tokens);
                lt.to_tokens(tokens);
                mut_tok.to_tokens(tokens);
                self.self_token.to_tokens(tokens);
            }
            ReceiverKind::Value => {
                self.mutability.to_tokens(tokens);
                self.self_token.to_tokens(tokens);
            }
            ReceiverKind::Typed(colon, ty) => {
                self.mutability.to_tokens(tokens);
                self.self_token.to_tokens(tokens);
                colon.to_tokens(tokens);
                ty.to_tokens(tokens);
            }
        }
    }
}

#[derive(Clone)]
pub(crate) struct PatType {
    pub attrs: Vec<Attribute>,
    pub pat: Box<Pat>,
    pub colon_token: Token![:],
    pub ty: Box<Type>,
}

impl ToTokens for PatType {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        for attr in &self.attrs {
            attr.to_tokens(tokens);
        }
        self.pat.to_tokens(tokens);
        self.colon_token.to_tokens(tokens);
        self.ty.to_tokens(tokens);
    }
}

#[derive(Clone)]
pub(crate) enum Pat {
    Ident(PatIdent),
    Wild(PatWild),
    Other(TokenStream),
}

#[derive(Clone)]
pub(crate) struct PatIdent {
    pub attrs: Vec<Attribute>,
    pub by_ref: Option<Token![ref]>,
    pub mutability: Option<Token![mut]>,
    pub ident: Ident,
}

#[derive(Clone)]
pub(crate) struct PatWild {
    pub attrs: Vec<Attribute>,
    pub underscore_token: Token![_],
}

impl ToTokens for Pat {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        match self {
            Self::Ident(pi) => {
                for attr in &pi.attrs {
                    attr.to_tokens(tokens);
                }
                pi.by_ref.to_tokens(tokens);
                pi.mutability.to_tokens(tokens);
                pi.ident.to_tokens(tokens);
            }
            Self::Wild(pw) => {
                for attr in &pw.attrs {
                    attr.to_tokens(tokens);
                }
                pw.underscore_token.to_tokens(tokens);
            }
            Self::Other(ts) => ts.to_tokens(tokens),
        }
    }
}

impl ToTokens for FnArg {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        match self {
            Self::Receiver(r) => r.to_tokens(tokens),
            Self::Typed(pt) => pt.to_tokens(tokens),
        }
    }
}

impl Parse for FnArg {
    fn parse(input: ParseStream<'_>) -> syn::Result<Self> {
        let attrs = input.call(Attribute::parse_outer)?;

        // 1. `& ['a] [mut] self`
        {
            let fork = input.fork();
            if fork.peek(Token![&]) {
                let and_token: Token![&] = fork.parse()?;
                let lifetime: Option<Lifetime> = fork.parse()?;
                let mutability: Option<Token![mut]> = fork.parse()?;
                if fork.peek(Token![self]) {
                    let self_token: Token![self] = fork.parse()?;
                    if fork.is_empty() || fork.peek(Token![,]) {
                        input.advance_to(&fork);
                        return Ok(Self::Receiver(Receiver {
                            attrs,
                            mutability,
                            self_token,
                            kind: ReceiverKind::Reference(and_token, lifetime, mutability),
                        }));
                    }
                }
            }
        }

        // 2. `[mut] self` or `[mut] self: Type`
        {
            let fork = input.fork();
            let mutability: Option<Token![mut]> = fork.parse()?;
            if fork.peek(Token![self]) {
                let self_token: Token![self] = fork.parse()?;
                if fork.is_empty() || fork.peek(Token![,]) {
                    input.advance_to(&fork);
                    return Ok(Self::Receiver(Receiver {
                        attrs,
                        mutability,
                        self_token,
                        kind: ReceiverKind::Value,
                    }));
                }
                if fork.peek(Token![:]) && !fork.peek(Token![::]) {
                    let colon_token: Token![:] = fork.parse()?;
                    input.advance_to(&fork);
                    let ty: Type = input.parse()?;
                    return Ok(Self::Receiver(Receiver {
                        attrs,
                        mutability,
                        self_token,
                        kind: ReceiverKind::Typed(colon_token, Box::new(ty)),
                    }));
                }
            }
        }

        // 3. Fast path: `[ref] [mut] <ident>: Type`
        {
            let fork = input.fork();
            let by_ref: Option<Token![ref]> = fork.parse()?;
            let mutability: Option<Token![mut]> = fork.parse()?;
            if fork.peek(Ident) {
                let ident: Ident = fork.parse()?;
                if fork.peek(Token![:]) && !fork.peek(Token![::]) {
                    let colon_token: Token![:] = fork.parse()?;
                    input.advance_to(&fork);
                    let ty: Type = input.parse()?;
                    return Ok(Self::Typed(PatType {
                        attrs,
                        pat: Box::new(Pat::Ident(PatIdent {
                            attrs: Vec::new(),
                            by_ref,
                            mutability,
                            ident,
                        })),
                        colon_token,
                        ty: Box::new(ty),
                    }));
                }
            }
        }

        // 4. Wildcard: `_: Type`
        {
            let fork = input.fork();
            if fork.peek(Token![_]) {
                let underscore_token: Token![_] = fork.parse()?;
                if fork.peek(Token![:]) && !fork.peek(Token![::]) {
                    let colon_token: Token![:] = fork.parse()?;
                    input.advance_to(&fork);
                    let ty: Type = input.parse()?;
                    return Ok(Self::Typed(PatType {
                        attrs,
                        pat: Box::new(Pat::Wild(PatWild {
                            attrs: Vec::new(),
                            underscore_token,
                        })),
                        colon_token,
                        ty: Box::new(ty),
                    }));
                }
            }
        }

        // 5. General pattern `<pat>: Type` (tuples, slices, structs, subpatterns, etc.)
        let mut pat_tokens = TokenStream::new();
        let mut angle_depth = 0usize;
        while !input.is_empty() {
            if angle_depth == 0 {
                if input.peek(Token![:]) && !input.peek(Token![::]) {
                    break;
                }
                if input.peek(Token![,]) {
                    return Err(input.error("expected `:`"));
                }
            }
            if input.peek(Token![::]) {
                let colons: Token![::] = input.parse()?;
                colons.to_tokens(&mut pat_tokens);
                if input.peek(Token![<]) {
                    let lt: Token![<] = input.parse()?;
                    lt.to_tokens(&mut pat_tokens);
                    angle_depth += 1;
                }
                continue;
            }
            let tt: TokenTree = input.parse()?;
            if let TokenTree::Punct(ref p) = tt {
                if p.as_char() == '<' && (angle_depth > 0 || pat_tokens.is_empty()) {
                    angle_depth += 1;
                } else if p.as_char() == '>' && angle_depth > 0 {
                    angle_depth -= 1;
                }
            }
            pat_tokens.extend([tt]);
        }
        if pat_tokens.is_empty() {
            return Err(input.error("expected pattern"));
        }
        let colon_token: Token![:] = input.parse()?;
        let ty: Type = input.parse()?;
        Ok(Self::Typed(PatType {
            attrs,
            pat: Box::new(Pat::Other(pat_tokens)),
            colon_token,
            ty: Box::new(ty),
        }))
    }
}

/// Construct `ident: ty` directly without `parse_quote!` when `ty` is already a `Type`,
/// or parse `ty` from tokens.
pub(crate) fn make_typed_arg(ident: Ident, ty: Type) -> FnArg {
    let span = ident.span();
    FnArg::Typed(PatType {
        attrs: Vec::new(),
        pat: Box::new(Pat::Ident(PatIdent {
            attrs: Vec::new(),
            by_ref: None,
            mutability: None,
            ident,
        })),
        colon_token: Token![:](span),
        ty: Box::new(ty),
    })
}

/// Parse one expression-like argument slice from `input` up to a top-level `,` or end of stream,
/// tracking `::<...>` / `<Type as Trait>` angle brackets and `|a, b|` closure parameters.
pub(crate) fn parse_expr_until_top_comma(input: ParseStream<'_>) -> syn::Result<TokenStream> {
    let mut out = TokenStream::new();
    let mut angle_depth = 0usize;
    let mut in_closure_params = false;
    let mut at_expr_prefix = true;

    while !input.is_empty() {
        if angle_depth == 0 && !in_closure_params && input.peek(Token![,]) {
            break;
        }
        // Turbofish `::<`
        if input.peek(Token![::]) {
            let colons: Token![::] = input.parse()?;
            colons.to_tokens(&mut out);
            at_expr_prefix = false;
            if input.peek(Token![<]) {
                let lt: Token![<] = input.parse()?;
                lt.to_tokens(&mut out);
                angle_depth += 1;
            }
            continue;
        }
        // Arrow `->` or fat arrow `=>` (must not decrement `angle_depth`)
        if input.peek(Token![->]) {
            let arrow: Token![->] = input.parse()?;
            arrow.to_tokens(&mut out);
            at_expr_prefix = false;
            continue;
        }
        if input.peek(Token![=>]) {
            let arrow: Token![=>] = input.parse()?;
            arrow.to_tokens(&mut out);
            at_expr_prefix = false;
            continue;
        }
        // Logical `||` (empty closure params or logical OR — never opens multi-param closure list)
        if input.peek(Token![||]) {
            let or_or: Token![||] = input.parse()?;
            or_or.to_tokens(&mut out);
            at_expr_prefix = false;
            continue;
        }

        let tt: TokenTree = input.parse()?;
        match &tt {
            TokenTree::Ident(id)
                if at_expr_prefix && (*id == "move" || *id == "async" || *id == "mut") =>
            {
                // Still before potential closure `|a, b|` or `&mut <T as Trait>::X`
            }
            TokenTree::Punct(p) => {
                let ch = p.as_char();
                if ch == '|' && angle_depth == 0 {
                    if in_closure_params {
                        in_closure_params = false;
                    } else if at_expr_prefix {
                        in_closure_params = true;
                    }
                    at_expr_prefix = false;
                } else if ch == '<' && (angle_depth > 0 || at_expr_prefix) {
                    angle_depth += 1;
                    at_expr_prefix = false;
                } else if ch == '>' && angle_depth > 0 {
                    angle_depth -= 1;
                    at_expr_prefix = false;
                } else if ch == ';' {
                    angle_depth = 0;
                    in_closure_params = false;
                    at_expr_prefix = true;
                } else if !at_expr_prefix || !matches!(ch, '&' | '*' | '!' | '-') {
                    at_expr_prefix = false;
                }
            }
            _ => {
                at_expr_prefix = false;
            }
        }
        out.extend([tt]);
    }

    if out.is_empty() {
        return Err(input.error("expected expression"));
    }
    Ok(out)
}

/// Split parenthesized call arguments `(a, b, c)` into `Vec<TokenStream>` without `syn::Expr`.
pub(crate) fn parse_comma_args(content: ParseStream<'_>) -> syn::Result<Vec<TokenStream>> {
    let mut args = Vec::new();
    while !content.is_empty() {
        args.push(parse_expr_until_top_comma(content)?);
        if !content.is_empty() {
            let _: Token![,] = content.parse()?;
        }
    }
    Ok(args)
}

/// Check whether `tokens` is a single bare identifier equal to `name` (unwrapping
/// transparent `Delimiter::None` groups if present).
pub(crate) fn is_bare_ident_tokens(tokens: &TokenStream, name: &str) -> bool {
    let mut iter = tokens.clone().into_iter();
    let Some(first) = iter.next() else {
        return false;
    };
    if iter.next().is_some() {
        return false;
    }
    match first {
        TokenTree::Ident(id) => id == name,
        TokenTree::Group(g) if g.delimiter() == Delimiter::None => {
            is_bare_ident_tokens(&g.stream(), name)
        }
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use syn::parse::Parser;

    #[test]
    fn parse_signature_qualifiers_generics_and_receivers() {
        let sig: Signature = syn::parse_str(
            "const unsafe extern \"C\" fn f<'a, T: Copy>(&'a mut self, (a, b): (T, i32), _: u8) -> T where T: Default",
        )
        .unwrap();
        assert!(sig.constness.is_some());
        assert!(matches!(sig.safety, Safety::Unsafe(_)));
        assert!(sig.abi.is_some());
        assert_eq!(sig.ident, "f");
        assert_eq!(sig.inputs.len(), 3);
        assert!(matches!(sig.inputs[0], FnArg::Receiver(_)));
        assert!(matches!(
            &sig.inputs[1],
            FnArg::Typed(pt) if matches!(pt.pat.as_ref(), Pat::Other(_))
        ));
        assert!(matches!(
            &sig.inputs[2],
            FnArg::Typed(pt) if matches!(pt.pat.as_ref(), Pat::Wild(_))
        ));
        assert!(sig.generics.where_clause.is_some());
    }

    #[test]
    fn parse_typed_receiver_and_mut_ident() {
        let sig: Signature =
            syn::parse_str("fn into_inner(self: Box<Self>, ref mut out: i32)").unwrap();
        assert!(matches!(
            &sig.inputs[0],
            FnArg::Receiver(r) if matches!(r.kind, ReceiverKind::Typed(_, _))
        ));
        assert!(matches!(
            &sig.inputs[1],
            FnArg::Typed(pt) if matches!(
                pt.pat.as_ref(),
                Pat::Ident(pi) if pi.by_ref.is_some() && pi.mutability.is_some() && pi.ident == "out"
            )
        ));
    }

    #[test]
    fn parse_comma_args_handles_turbofish_qualified_closures_and_comparisons() {
        let parsed = parse_comma_args
            .parse_str(
                "foo::<HashMap<K, V>, B>(x, y), &<T as Into<(A, B)>>::into(v), move |a: i32, b: i32| -> i32 { a + b }, a < b, c > d",
            )
            .unwrap();
        assert_eq!(parsed.len(), 5);
        assert!(parsed[0].to_string().starts_with("foo"));
        assert!(parsed[1].to_string().starts_with('&'));
        assert!(parsed[2].to_string().starts_with("move"));
        assert_eq!(parsed[3].to_string(), "a < b");
        assert_eq!(parsed[4].to_string(), "c > d");
    }
}
