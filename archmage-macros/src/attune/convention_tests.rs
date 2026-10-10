use proc_macro2::TokenStream;
use quote::{ToTokens, quote};

fn expand(args: TokenStream, item: TokenStream) -> Vec<syn::ItemFn> {
    syn::parse2::<syn::File>(super::expand(args, item).unwrap())
        .unwrap()
        .items
        .into_iter()
        .map(|item| {
            let syn::Item::Fn(function) = item else {
                panic!("expected function")
            };
            function
        })
        .collect()
}

#[test]
fn proof_suffix_adds_a_real_proof_parameter_and_keeps_boundary_policy() {
    let functions = expand(
        quote!(),
        quote!(
            pub fn work_v3_t<T: Copy>(value: T) -> T {
                value
            }
        ),
    );
    let wrapper = functions
        .iter()
        .find(|f| f.sig.ident == "work_v3_t")
        .unwrap();
    assert_eq!(wrapper.sig.inputs.len(), 2);
    assert!(
        wrapper.sig.inputs[0]
            .to_token_stream()
            .to_string()
            .contains("X64V3Token")
    );
    assert!(
        wrapper
            .attrs
            .iter()
            .any(|a| a.to_token_stream().to_string() == "# [inline (always)]")
    );
    assert!(
        !wrapper
            .attrs
            .iter()
            .any(|a| a.path().is_ident("target_feature"))
    );
    let body = functions
        .iter()
        .find(|f| f.sig.ident != "work_v3_t")
        .unwrap();
    assert!(matches!(body.vis, syn::Visibility::Inherited));
    assert!(
        body.attrs
            .iter()
            .any(|a| a.path().is_ident("target_feature"))
    );
    assert!(body.attrs.iter().any(|a| a.path().is_ident("inline")));
}

#[test]
fn proof_suffix_preserves_written_position_and_specializes_placeholder() {
    for ty in [quote!(X64V3Token), quote!(Token)] {
        let functions = expand(
            quote!(),
            quote!(fn work_v3_t(x: u32, proof: #ty) { let _ = (x, proof); }),
        );
        let wrapper = functions
            .iter()
            .find(|f| f.sig.ident == "work_v3_t")
            .unwrap();
        assert_eq!(wrapper.sig.inputs.len(), 2);
        assert!(
            wrapper.sig.inputs[1]
                .to_token_stream()
                .to_string()
                .starts_with("proof :")
        );
        assert!(
            wrapper.sig.inputs[1]
                .to_token_stream()
                .to_string()
                .contains("X64V3Token")
        );
    }
    let error = super::expand(
        quote!(),
        quote!(
            fn work_v3_t(proof: X64V2Token) {}
        ),
    )
    .unwrap_err();
    assert!(error.to_string().contains("does not match"));
    assert_eq!(
        super::infer_suffix(&syn::parse_quote!(work_v3_gfni_crypto_t))
            .unwrap()
            .0
            .name,
        "v3_gfni_crypto"
    );
    assert_eq!(
        super::infer_suffix(&syn::parse_quote!(v3_t))
            .unwrap()
            .0
            .name,
        "v3"
    );
}

#[test]
fn dispatcher_alias_and_modifiers_preserve_private_implementations() {
    let item = quote!(
        pub fn work(x: u32) -> u32 {
            x
        }
    );
    assert_eq!(
        super::expand(quote!(make(_)), item.clone())
            .unwrap()
            .to_string(),
        super::expand(quote!(make(dispatch)), item.clone())
            .unwrap()
            .to_string(),
    );
    for args in [
        quote!(make(dispatch, +v4(avx512), -neon)),
        quote!(make(+v4(avx512), -neon, _)),
    ] {
        let functions = expand(args, item.clone());
        let public: Vec<_> = functions
            .iter()
            .filter(|f| matches!(f.vis, syn::Visibility::Public(_)))
            .collect();
        assert_eq!(public.len(), 1);
        assert_eq!(public[0].sig.ident, "work");
        for tier in ["v3", "v4", "wasm128", "scalar"] {
            let name = format!("__attune_work_{tier}");
            assert!(functions.iter().any(|f| f.sig.ident == name), "{tier}");
        }
        assert!(
            !functions
                .iter()
                .any(|f| f.sig.ident.to_string().contains("neon"))
        );
        let v4 = functions
            .iter()
            .find(|f| f.sig.ident == "__attune_work_v4")
            .unwrap();
        assert!(
            v4.attrs
                .iter()
                .any(|a| a.to_token_stream().to_string().contains("avx512"))
        );
    }
    let functions = expand(quote!(make(dispatch, -v3, -v4)), item);
    assert!(
        !functions
            .iter()
            .any(|f| f.sig.ident.to_string().contains("_v3"))
    );
}

#[test]
fn dispatcher_rejects_duplicates_and_missing_portable_fallback() {
    for args in [
        quote!(make(_, dispatch)),
        quote!(make(dispatch, all)),
        quote!(make(dispatch, -scalar)),
        quote!(make(dispatch, _scalar(optional))),
    ] {
        assert!(
            super::expand(
                args,
                quote!(
                    fn work() {}
                )
            )
            .is_err()
        );
    }
}

#[test]
fn inherited_proof_never_probes_and_keeps_covered_calls_direct() {
    let text = super::expand(
        quote!(scalar),
        quote!(
            fn caller<T: IntoConcreteToken>(proof: T) -> u32 {
                attuned!(work(), [_v3, _scalar])
            }
        ),
    )
    .unwrap()
    .to_string();
    assert!(text.contains("IntoConcreteToken :: as_x64v3"), "{text}");
    assert!(text.contains("let __attune_parent_proof = proof"), "{text}");
    assert!(!text.contains("summon"), "{text}");
    assert!(text.contains("work_v3_t"), "{text}");
    assert!(text.contains("work_scalar ()"), "{text}");

    let text = super::expand(
        quote!(v3),
        quote!(
            fn caller(proof: X64V3Token) -> u32 {
                attuned!(work(), [_v3])
            }
        ),
    )
    .unwrap()
    .to_string();
    assert!(text.contains("work_v3 ()"), "{text}");
    assert!(!text.contains("work_v3_t"), "{text}");
    assert!(!text.contains("__attune_parent_proof"), "{text}");
    assert!(!text.contains("summon"), "{text}");
}

#[test]
fn concrete_parent_proves_a_registry_ancestor_without_runtime_selection() {
    let text = super::expand(
        quote!(scalar),
        quote!(
            fn caller(proof: X64V3Token) -> u32 {
                attuned!(work(), [_v2])
            }
        ),
    )
    .unwrap()
    .to_string();
    assert!(text.contains("__attune_parent_proof . v2 ()"), "{text}");
    assert!(text.contains("work_v2_t"), "{text}");
    assert!(!text.contains("summon"), "{text}");
    assert!(!text.contains("IntoConcreteToken"), "{text}");
}

#[test]
fn ambiguous_parent_requires_explicit_choice_only_when_needed() {
    let item = quote!(
        fn caller(a: X64V2Token, b: X64V3Token) -> u32 {
            attuned!(work(), [_v3, _scalar])
        }
    );
    let text = super::expand(quote!(scalar), item.clone())
        .unwrap()
        .to_string();
    assert!(text.contains("multiple parent proof parameters"), "{text}");
    let covered = super::expand(quote!(v3), item).unwrap().to_string();
    assert!(!covered.contains("compile_error"), "{covered}");
    let explicit = super::expand(
        quote!(scalar),
        quote!(
            fn caller(a: X64V2Token, b: X64V3Token) -> u32 {
                attuned!(work(), [_v3, _scalar], using(b))
            }
        ),
    )
    .unwrap()
    .to_string();
    assert!(!explicit.contains("compile_error"), "{explicit}");
    assert!(!explicit.contains("__attune_parent_proof"), "{explicit}");
}

#[test]
fn reattune_still_probes_and_nested_functions_do_not_inherit_parent_proof() {
    let text = super::expand(
        quote!(scalar),
        quote!(
            fn caller(proof: impl IntoConcreteToken) -> u32 {
                fn nested() -> u32 {
                    attuned!(work(), [_v3, _scalar])
                }
                reattune!(work(), [_v3, _scalar])
            }
        ),
    )
    .unwrap()
    .to_string();
    assert!(text.contains("summon"), "{text}");
    assert!(text.contains("attuned !"), "{text}");
    assert!(!text.contains("__attune_parent_proof"), "{text}");
}
