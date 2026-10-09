use proc_macro2::TokenStream;
use quote::{ToTokens, quote};

fn functions(args: TokenStream, item: TokenStream) -> Vec<syn::ItemFn> {
    syn::parse2::<syn::File>(super::expand(args, item).unwrap())
        .unwrap()
        .items
        .into_iter()
        .map(|item| match item {
            syn::Item::Fn(function) => function,
            other => panic!("expected function, got {}", other.to_token_stream()),
        })
        .collect()
}

fn policy(function: &syn::ItemFn) -> Option<String> {
    let attrs: Vec<_> = function
        .attrs
        .iter()
        .filter(|a| a.path().is_ident("inline"))
        .collect();
    assert!(attrs.len() <= 1, "duplicate inline attributes");
    attrs.first().map(|a| a.meta.to_token_stream().to_string())
}

#[test]
fn default_uses_syntactic_visibility_on_every_architecture() {
    for tier in [quote!(scalar), quote!(v3), quote!(neon), quote!(wasm128)] {
        for (visibility, expected) in [
            (quote!(pub), Some("inline")),
            (quote!(pub(crate)), None),
            (quote!(pub(super)), None),
            (quote!(pub(in crate::kernels)), None),
            (quote!(), None),
        ] {
            let output = functions(
                quote!(#tier, inline(default)),
                quote!(#visibility fn kernel<T: Copy>(x: T) -> T { x }),
            );
            assert_eq!(policy(&output[0]).as_deref(), expected);
            assert_eq!(output[0].sig.generics.params.len(), 1);
            assert_eq!(
                output[0]
                    .attrs
                    .iter()
                    .any(|a| a.path().is_ident("target_feature")),
                tier.to_string() != "scalar"
            );
        }
    }
}

#[test]
fn family_policy_uses_output_visibility_and_keeps_proof_defaults() {
    let output = functions(
        quote!(inline(default), make(pub(crate) _v3, _v3_t, pub _scalar, _scalar_t, _)),
        quote!(
            pub fn kernel(x: u32) -> u32 {
                x + 1
            }
        ),
    );
    for (name, expected) in [
        ("kernel_v3", None),
        ("kernel_v3_t", Some("inline (always)")),
        ("kernel_scalar", Some("inline")),
        ("kernel_scalar_t", Some("inline (always)")),
        ("kernel", None),
    ] {
        let function = output.iter().find(|f| f.sig.ident == name).unwrap();
        assert_eq!(policy(function).as_deref(), expected, "{name}");
    }
    let output = functions(
        quote!(inline(default), make(pub _scalar)),
        quote!(
            fn kernel() {}
        ),
    );
    assert_eq!(policy(&output[0]).as_deref(), Some("inline"));
}

#[test]
fn explicit_output_overrides_body_policy_and_source_attribute() {
    let output = functions(
        quote!(inline(default), make(inline(never) _v3, inline(none) _scalar, inline(default) _scalar_t, inline(default) _)),
        quote!(
            #[inline(always)]
            pub fn kernel(x: u32) -> u32 {
                x
            }
        ),
    );
    for (name, expected) in [
        ("kernel_v3", Some("inline (never)")),
        ("kernel_scalar", None),
        ("kernel_scalar_t", Some("inline")),
        ("kernel", Some("inline")),
    ] {
        let function = output.iter().find(|f| f.sig.ident == name).unwrap();
        assert_eq!(policy(function).as_deref(), expected, "{name}");
    }
    let output = functions(
        quote!(make(inline(default) _scalar_t, inline(default) _)),
        quote!(
            fn kernel() {}
        ),
    );
    for function in output
        .iter()
        .filter(|f| !f.sig.ident.to_string().starts_with("__attune"))
    {
        assert_eq!(policy(function), None);
    }
}

#[test]
fn hidden_bodies_and_wrappers_have_separate_policies() {
    for args in [quote!(wrap, inline(default)), quote!(wrap, inline(none))] {
        let output = functions(
            args,
            quote!(
                pub fn kernel(token: X64V3Token) {}
            ),
        );
        assert_eq!(policy(&output[0]), None);
        assert_eq!(policy(&output[1]).as_deref(), Some("inline (always)"));
    }
    let output = functions(
        quote!(inline(default), make(_v3_t)),
        quote!(
            pub fn kernel() {}
        ),
    );
    assert_eq!(policy(&output[0]), None);
    assert_eq!(policy(&output[1]).as_deref(), Some("inline (always)"));
    for token in [quote!(ScalarToken), quote!(Wasm128Token)] {
        let output = functions(
            quote!(wrap, inline(default)),
            quote!(pub fn kernel(token: #token) {}),
        );
        assert_eq!(output.len(), 1);
        assert_eq!(policy(&output[0]).as_deref(), Some("inline"));
    }
}

#[test]
fn unambiguous_body_choices_and_existing_defaults() {
    for (args, expected) in [
        (quote!(scalar, inline(hint)), Some("inline")),
        (quote!(scalar, inline(none)), None),
        (quote!(scalar, inline(always)), Some("inline (always)")),
        (quote!(scalar, inline(never)), Some("inline (never)")),
        (quote!(scalar), Some("inline (never)")),
    ] {
        let output = functions(
            args,
            quote!(
                #[inline(never)]
                fn kernel() {}
            ),
        );
        assert_eq!(policy(&output[0]).as_deref(), expected);
    }
    let output = functions(
        quote!(v3),
        quote!(
            fn kernel() {}
        ),
    );
    assert_eq!(policy(&output[0]).as_deref(), Some("inline"));
}

#[test]
fn invalid_or_ambiguous_policies_are_rejected() {
    for (args, message) in [
        (
            quote!(v3, inline(default), inline(hint)),
            "duplicate body inline policy",
        ),
        (
            quote!(v3, inline(unknown)),
            "expected default, none, hint, always, or never",
        ),
        (
            quote!(v3, inline(default, hint)),
            "expected one inline policy",
        ),
        (
            quote!(v3, inline(always)),
            "inline(always) on target-feature bodies",
        ),
        (
            quote!(wrap, in_trait, inline(default)),
            "cannot infer trait visibility",
        ),
    ] {
        let error = super::expand(
            args,
            quote!(
                fn kernel(token: X64V3Token) {}
            ),
        )
        .unwrap_err();
        assert!(error.to_string().contains(message), "{error}");
    }
    let output = super::expand(
        quote!(wrap, inline(always)),
        quote!(
            fn kernel(token: X64V3Token) {}
        ),
    )
    .unwrap()
    .to_string();
    assert!(output.contains("compile_error"));
    assert!(output.contains("inline(always) on target-feature bodies"));
}
