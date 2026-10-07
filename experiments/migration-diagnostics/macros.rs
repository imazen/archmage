extern crate proc_macro;

use proc_macro::{Delimiter, Group, TokenStream, TokenTree};

fn warning(note: &str) -> TokenStream {
    format!(
        "#[deprecated(note = {note:?})] const __ARCHMAGE_MIGRATION: () = ();\n\
         let _ = __ARCHMAGE_MIGRATION;"
    )
    .parse()
    .unwrap()
}

// This probe emits replacement text for a trivial identity attribute, not attune.
#[proc_macro_attribute]
pub fn legacy_attr(attr: TokenStream, item: TokenStream) -> TokenStream {
    let replacement = if attr.to_string() == "always" {
        "#[inline(always)]"
    } else {
        "#[inline]"
    };
    let note =
        format!("ARCHMAGE-MIGRATE probe: replace this legacy_attr attribute with:\n{replacement}");
    let mut tokens: Vec<_> = item.into_iter().collect();
    let index = tokens
        .iter()
        .rposition(|t| matches!(t, TokenTree::Group(g) if g.delimiter() == Delimiter::Brace))
        .expect("probe needs a function body");
    let TokenTree::Group(body) = &tokens[index] else {
        unreachable!()
    };
    let mut stream = warning(&note);
    stream.extend(body.stream());
    let mut group = Group::new(Delimiter::Brace, stream);
    group.set_span(body.span());
    tokens[index] = TokenTree::Group(group);
    tokens.into_iter().collect()
}

#[proc_macro]
pub fn legacy_expr(input: TokenStream) -> TokenStream {
    let note =
        format!("ARCHMAGE-MIGRATE probe: replace this legacy_expr invocation with:\n({input})");
    let mut stream = warning(&note);
    stream.extend([TokenTree::Group(Group::new(Delimiter::Parenthesis, input))]);
    TokenTree::Group(Group::new(Delimiter::Brace, stream)).into()
}

#[deprecated(note = "Static macro-level note: this cannot describe each invocation's arguments")]
#[proc_macro_attribute]
pub fn statically_deprecated(_: TokenStream, item: TokenStream) -> TokenStream {
    item
}
