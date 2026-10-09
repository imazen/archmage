//! Diagnostic-only instrumentation copied into a disposable experiment stack.
//! It is never compiled into the normal archmage macro crate.
use quote::ToTokens;

pub(crate) struct Event<'a> {
    pub macro_name: &'a str,
    pub ident: &'a syn::Ident,
    pub body: &'a proc_macro2::TokenStream,
    pub source_vis: &'a syn::Visibility,
    pub body_vis: &'a syn::Visibility,
    pub kind: &'a str,
    pub tier: &'a str,
    pub arch: Option<&'a str>,
    pub gate: Option<&'a str>,
    pub selected: &'a str,
    pub policy: &'a str,
}

pub(crate) fn record(event: Event<'_>) {
    let span = event.ident.span().unwrap();
    let body_span = event
        .body
        .clone()
        .into_iter()
        .next()
        .map_or(span, |t| t.span().unwrap());
    let visibility = |vis: &syn::Visibility| match vis {
        syn::Visibility::Inherited => "inherited".to_owned(),
        _ => vis.to_token_stream().to_string(),
    };
    let fields = [
        std::env::var("CARGO_PKG_NAME").expect("Cargo package name"),
        std::env::var("CARGO_PKG_VERSION").expect("Cargo package version"),
        event.macro_name.to_owned(),
        span.file(),
        span.line().to_string(),
        span.column().to_string(),
        body_span.file(),
        body_span.line().to_string(),
        event.ident.to_string(),
        visibility(event.source_vis),
        if event.kind.starts_with("hidden_") {
            "private".to_owned()
        } else {
            visibility(event.body_vis)
        },
        event.kind.to_owned(),
        event.tier.to_owned(),
        event.arch.unwrap_or("any").to_owned(),
        event.gate.unwrap_or("").to_owned(),
        event.selected.to_owned(),
        event.policy.to_owned(),
    ];
    assert!(fields.iter().all(|s| !s.contains(['\t', '\n', '\r'])));
    eprintln!("ARCHMAGE_INLINE_INVENTORY\t{}", fields.join("\t"));
}
