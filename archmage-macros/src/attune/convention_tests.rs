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

#[test]
fn suppress_const_test_controls_token_assertion_on_proof_boundaries() {
    let checked_wrap = super::expand(
        quote!(wrap),
        quote!(
            fn work(token: X64V3Token, x: u32) -> u32 {
                let _ = token;
                x
            }
        ),
    )
    .unwrap()
    .to_string();
    assert!(
        checked_wrap.contains("__ARCHMAGE_ASSERT_TIER_F38B284B"),
        "{checked_wrap}"
    );

    let suppressed_wrap = super::expand(
        quote!(wrap, suppress_const_test),
        quote!(
            fn work(token: X64V3Token, x: u32) -> u32 {
                let _ = token;
                x
            }
        ),
    )
    .unwrap()
    .to_string();
    assert!(
        !suppressed_wrap.contains("__ARCHMAGE_ASSERT_TIER_"),
        "{suppressed_wrap}"
    );

    let suppressed_self = super::expand(
        quote!(wrap, suppress_const_test, _self = X64V3Token),
        quote!(
            fn work(self, x: u32) -> u32 {
                let _ = self;
                x
            }
        ),
    )
    .unwrap()
    .to_string();
    assert!(
        !suppressed_self.contains("__ARCHMAGE_ASSERT_TIER_"),
        "{suppressed_self}"
    );

    // Explicit user-written proof type on a `_t` function checks by default,
    // and omits the check when `suppress_const_test` is passed.
    let checked_suffix = super::expand(
        quote!(),
        quote!(
            fn work_v3_t(token: X64V3Token, x: u32) -> u32 {
                let _ = token;
                x
            }
        ),
    )
    .unwrap()
    .to_string();
    assert!(
        checked_suffix.contains("__ARCHMAGE_ASSERT_TIER_F38B284B"),
        "{checked_suffix}"
    );

    let suppressed_suffix = super::expand(
        quote!(suppress_const_test),
        quote!(
            fn work_v3_t(token: X64V3Token, x: u32) -> u32 {
                let _ = token;
                x
            }
        ),
    )
    .unwrap()
    .to_string();
    assert!(
        !suppressed_suffix.contains("__ARCHMAGE_ASSERT_TIER_"),
        "{suppressed_suffix}"
    );

    // Macro-synthesized `::archmage::...` parameters cannot be shadowed, so
    // they automatically omit the redundant const assertion.
    for sig in [
        quote!(
            fn work_v3_t(x: u32) -> u32 {
                x
            }
        ),
        quote!(
            fn work_v3_t(token: Token, x: u32) -> u32 {
                let _ = token;
                x
            }
        ),
    ] {
        let synthesized = super::expand(quote!(), sig).unwrap().to_string();
        assert!(
            !synthesized.contains("__ARCHMAGE_ASSERT_TIER_"),
            "{synthesized}"
        );
    }

    // Direct bodies emit no wrapper and reject `suppress_const_test`.
    let direct_err = super::expand(
        quote!(suppress_const_test),
        quote!(
            fn work_v3(x: u32) -> u32 {
                x
            }
        ),
    )
    .unwrap_err();
    assert!(direct_err.to_string().contains("requires a proof wrapper"));
}

#[test]
fn define_maps_subtiers_to_representation_backend_tokens() {
    for (attr, body, expected_backend, expected_feature) in [
        (
            quote!(v3_vnni, define(f32x8)),
            quote!(
                fn kernel(x: f32) -> f32 {
                    x
                }
            ),
            "X64V3Token",
            "avxvnni",
        ),
        (
            quote!(v3_crypto, define(f32x8)),
            quote!(
                fn kernel(x: f32) -> f32 {
                    x
                }
            ),
            "X64V3Token",
            "vaes",
        ),
        (
            quote!(arm_v2, define(f32x4)),
            quote!(
                fn kernel(x: f32) -> f32 {
                    x
                }
            ),
            "NeonToken",
            "dotprod",
        ),
        (
            quote!(arm_v3, define(f32x4)),
            quote!(
                fn kernel(x: f32) -> f32 {
                    x
                }
            ),
            "NeonToken",
            "i8mm",
        ),
        (
            quote!(wasm128_relaxed, define(f32x4)),
            quote!(
                fn kernel(x: f32) -> f32 {
                    x
                }
            ),
            "Wasm128Token",
            "relaxed-simd",
        ),
        (
            // Direct tokenless `wasm128_relaxed_fma` requires `unsafe fn` because
            // single-rounding FMA needs the runtime probe beyond `#[target_feature]`.
            quote!(wasm128_relaxed_fma, define(f32x4)),
            quote!(
                unsafe fn kernel(x: f32) -> f32 {
                    x
                }
            ),
            "Wasm128RelaxedFmaToken",
            "relaxed-simd",
        ),
        (
            quote!(fp16, define(f32x16)),
            quote!(
                fn kernel(x: f32) -> f32 {
                    x
                }
            ),
            "X64V4Token",
            "avx512fp16",
        ),
        (
            quote!(v4x, define(f32x16)),
            quote!(
                fn kernel(x: f32) -> f32 {
                    x
                }
            ),
            "X64V4xToken",
            "avx512vnni",
        ),
    ] {
        let text = super::expand(attr, body).unwrap().to_string();
        assert!(
            text.contains(expected_backend),
            "expected {expected_backend} in {text}"
        );
        assert!(
            text.contains(expected_feature),
            "expected {expected_feature} in {text}"
        );
    }

    // Tiers below `v3` (`v2` lacks AVX2) must NOT resolve to `X64V3Token`.
    let v2 = super::expand(
        quote!(v2, define(f32x8)),
        quote!(
            fn kernel(x: f32) -> f32 {
                x
            }
        ),
    )
    .unwrap()
    .to_string();
    assert!(
        v2.contains("X64V2Token") && !v2.contains("X64V3Token"),
        "{v2}"
    );
}

#[test]
fn runtime_probed_wasm_fma_is_not_covered_by_relaxed_feature_context() {
    // Same compile-time features (`simd128,relaxed-simd`), but `Wasm128RelaxedToken`
    // does not prove single-rounding FMA and must be rejected as proof for a
    // `_wasm128_relaxed_fma_t` wrapper.
    let mismatch = super::expand(
        quote!(),
        quote!(
            fn work_wasm128_relaxed_fma_t(proof: Wasm128RelaxedToken, x: f32) -> f32 {
                let _ = proof;
                x
            }
        ),
    )
    .unwrap_err();
    assert!(
        mismatch
            .to_string()
            .contains("proof parameter does not match the inferred tier suffix"),
        "{mismatch}"
    );

    // Inside `#[attune(wasm128_relaxed)]`, contextual `attuned!` must NOT treat
    // `wasm128_relaxed_fma` as covered by the enclosing feature set alone.
    let text = super::expand(
        quote!(wasm128_relaxed),
        quote!(
            fn caller(x: f32) -> f32 {
                attuned!(callee(x), [wasm128_relaxed_fma, wasm128_relaxed, scalar])
            }
        ),
    )
    .unwrap()
    .to_string();
    assert!(text.contains("callee_wasm128_relaxed"), "{text}");
    assert!(!text.contains("callee_wasm128_relaxed_fma"), "{text}");

    // Safe tokenless `#[attune(wasm128_relaxed_fma)]` is rejected because
    // single-rounding FMA cannot be verified from compile-time #[target_feature].
    assert!(
        super::expand(
            quote!(wasm128_relaxed_fma),
            quote!(
                fn caller(x: f32) -> f32 {
                    attuned!(callee(x), [wasm128_relaxed_fma, wasm128_relaxed, scalar])
                }
            ),
        )
        .is_err()
    );

    // Inside tokenless `#[attune(wasm128_relaxed_fma)] unsafe fn`, contextual
    // `attuned!` still refuses static promotion to `wasm128_relaxed_fma`
    // without a token proof, falling back to statically-known `wasm128_relaxed`.
    let unsafe_tokenless_text = super::expand(
        quote!(wasm128_relaxed_fma),
        quote!(
            unsafe fn caller(x: f32) -> f32 {
                attuned!(callee(x), [wasm128_relaxed_fma, wasm128_relaxed, scalar])
            }
        ),
    )
    .unwrap()
    .to_string();
    assert!(
        unsafe_tokenless_text.contains("callee_wasm128_relaxed"),
        "{unsafe_tokenless_text}"
    );
    assert!(
        !unsafe_tokenless_text.contains("callee_wasm128_relaxed_fma"),
        "{unsafe_tokenless_text}"
    );

    // Holding a summoned concrete `Wasm128RelaxedFmaToken` proof in `#[attune(wrap)]`
    // forwards the runtime-probed proof without re-summoning or calling from_context().
    let wrapped_fma_text = super::expand(
        quote!(wrap),
        quote!(
            fn caller(tok: Wasm128RelaxedFmaToken, x: f32) -> f32 {
                attuned!(callee(x), [wasm128_relaxed_fma, wasm128_relaxed, scalar])
            }
        ),
    )
    .unwrap()
    .to_string();
    assert!(
        wrapped_fma_text.contains("callee_wasm128_relaxed_fma_t"),
        "{wrapped_fma_text}"
    );
    assert!(!wrapped_fma_text.contains("summon"), "{wrapped_fma_text}");
    assert!(
        !wrapped_fma_text.contains("from_context"),
        "{wrapped_fma_text}"
    );
    assert!(
        wrapped_fma_text.contains("target_feature = \"simd128\"")
            && wrapped_fma_text.contains("target_feature = \"relaxed-simd\""),
        "{wrapped_fma_text}"
    );
}

#[test]
fn wasm_items_and_dispatch_arms_gate_on_compile_time_target_features_while_scalar_always_exists() {
    // Requesting a safe tokenless `Form::Direct` for `wasm128_relaxed_fma` is rejected.
    assert!(
        super::expand(
            quote!(
                _wasm128,
                _wasm128_relaxed,
                _wasm128_relaxed_fma,
                _scalar,
                dispatch
            ),
            quote!(
                pub fn kernel(x: f32) -> f32 {
                    x
                }
            ),
        )
        .is_err()
    );

    // Proof-based `_wasm128_relaxed_fma_t` + `dispatch` injects a hidden
    // `Wasm128RelaxedFmaToken` parameter into the inner implementation so it stays
    // completely safe (compatible with `#![forbid(unsafe_code)]`).
    let family = super::expand(
        quote!(
            _wasm128,
            _wasm128_relaxed,
            _wasm128_relaxed_fma_t,
            _scalar,
            dispatch
        ),
        quote!(
            pub fn kernel(x: f32) -> f32 {
                x
            }
        ),
    )
    .unwrap()
    .to_string();
    assert!(
        !family.contains("unsafe"),
        "expected safe expansion: {family}"
    );
    let fns = syn::parse2::<syn::File>(family.parse().unwrap())
        .unwrap()
        .items;
    let find_fn = |name: &str| -> &syn::ItemFn {
        fns.iter()
            .find_map(|item| match item {
                syn::Item::Fn(f) if f.sig.ident == name => Some(f),
                _ => None,
            })
            .unwrap_or_else(|| panic!("missing {name} in {family}"))
    };
    let wasm128_attrs = find_fn("kernel_wasm128")
        .attrs
        .iter()
        .map(|a| a.to_token_stream().to_string())
        .collect::<Vec<_>>()
        .join(" ");
    assert!(
        wasm128_attrs.contains("target_feature = \"simd128\""),
        "{wasm128_attrs}"
    );
    assert!(!wasm128_attrs.contains("relaxed-simd"), "{wasm128_attrs}");

    for relaxed_name in [
        "kernel_wasm128_relaxed",
        "__attune_kernel_wasm128_relaxed_fma",
        "kernel_wasm128_relaxed_fma_t",
    ] {
        let attrs = find_fn(relaxed_name)
            .attrs
            .iter()
            .map(|a| a.to_token_stream().to_string())
            .collect::<Vec<_>>()
            .join(" ");
        assert!(attrs.contains("target_feature = \"simd128\""), "{attrs}");
        assert!(
            attrs.contains("target_feature = \"relaxed-simd\""),
            "{attrs}"
        );
    }

    let scalar_attrs = find_fn("kernel_scalar")
        .attrs
        .iter()
        .map(|a| a.to_token_stream().to_string())
        .collect::<Vec<_>>()
        .join(" ");
    assert!(
        !scalar_attrs.contains("cfg"),
        "scalar must always exist: {scalar_attrs}"
    );

    // An `unsafe fn` family allows tokenless `_wasm128_relaxed_fma` and wraps
    // wrapper/dispatcher forwarding calls in `unsafe { ... }`.
    let unsafe_family = super::expand(
        quote!(
            _wasm128,
            _wasm128_relaxed,
            _wasm128_relaxed_fma,
            _scalar,
            dispatch
        ),
        quote!(
            pub unsafe fn kernel(x: f32) -> f32 {
                x
            }
        ),
    )
    .unwrap()
    .to_string();
    assert!(
        unsafe_family.contains("kernel_wasm128_relaxed_fma"),
        "{unsafe_family}"
    );
    let dispatcher_tokens = unsafe_family
        .rsplit("pub unsafe fn kernel (x : f32)")
        .next()
        .unwrap_or(&unsafe_family);
    assert!(
        dispatcher_tokens.contains("unsafe {"),
        "unsafe dispatcher must wrap unsafe proof wrapper calls: {unsafe_family}"
    );
}
