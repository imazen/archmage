//! Exercise the implementation without rustc's proc-macro bridge. The existing
//! tests/expand corpus remains the reviewed, compiled input/output oracle.
//! These tests add cross-architecture and error-path coverage on every host.
use super::*;
use proc_macro2::{Delimiter, Group, Spacing, TokenStream as Tokens, TokenTree};
use quote::{ToTokens, quote};
use syn::parse::Parser;

fn expand(name: &str, args: Tokens, item: Tokens) -> syn::Result<Tokens> {
    Ok(match name {
        "arcane" | "simd_fn" | "token_target_features_boundary" => {
            arcane_impl(syn::parse2(item)?, name, syn::parse2(args)?)
        }
        "rite" | "token_target_features" => rite_impl(syn::parse2(item)?, syn::parse2(args)?),
        "autoversion" => autoversion_impl(syn::parse2(item)?, syn::parse2(args)?),
        "magetypes" => {
            let (rite, defines, names) = parse_magetypes_attr.parse2(args)?;
            let tiers = if names.is_empty() {
                default_tiers(true)
            } else {
                resolve_tiers(&names, proc_macro2::Span::call_site(), true)?
            };
            magetypes::magetypes_impl(syn::parse2(item)?, &tiers, rite, &defines)
        }
        _ => panic!("unrecognized macro {name}"),
    })
}

fn check_item(item: Tokens, count: &mut usize) {
    let Ok(mut fun) = syn::parse2::<LightFn>(item) else {
        return;
    };
    let Some(index) = fun.attrs.iter().position(|a| {
        a.path().segments.last().is_some_and(|s| {
            matches!(
                s.ident.to_string().as_str(),
                "arcane"
                    | "rite"
                    | "magetypes"
                    | "autoversion"
                    | "simd_fn"
                    | "token_target_features"
                    | "token_target_features_boundary"
            )
        })
    }) else {
        return;
    };
    let attr = fun.attrs.remove(index);
    let args = match &attr.meta {
        syn::Meta::List(l) => l.tokens.clone(),
        _ => Tokens::new(),
    };
    let name = attr.path().segments.last().unwrap().ident.to_string();
    let output = expand(&name, args, fun.to_token_stream()).unwrap();
    assert!(!output.is_empty(), "{name}");
    // Both successful expansions and intentional compile_error! outputs must
    // remain valid Rust syntax. Compiler acceptance is checked by macro_expand.
    syn::parse2::<syn::File>(output.clone()).unwrap_or_else(|e| panic!("{name}: {e}\n{output}"));
    *count += 1;
}

fn visit(items: Vec<syn::Item>, count: &mut usize) {
    for item in items {
        match item {
            syn::Item::Fn(f) => check_item(f.into_token_stream(), count),
            syn::Item::Mod(m) => {
                if let Some((_, items)) = m.content {
                    visit(items, count);
                }
            }
            syn::Item::Impl(i) => {
                for item in i.items {
                    if let syn::ImplItem::Fn(f) = item {
                        check_item(f.into_token_stream(), count);
                    }
                }
            }
            _ => {}
        }
    }
}

#[test]
fn existing_input_corpus_exercises_internal_expansion() {
    fn walk(dir: &std::path::Path, count: &mut usize) {
        for entry in std::fs::read_dir(dir).unwrap() {
            let path = entry.unwrap().path();
            if path.is_dir() {
                walk(&path, count);
            } else if path.extension().is_some_and(|e| e == "rs")
                && !path.to_string_lossy().ends_with(".expanded.rs")
            {
                let file = syn::parse_file(&std::fs::read_to_string(&path).unwrap()).unwrap();
                visit(file.items, count);
            }
        }
    }
    let mut count = 0;
    walk(
        &std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../tests/expand"),
        &mut count,
    );
    assert!(count >= 100, "vacuous fixture scan: {count}");
}

#[test]
fn every_registered_token_and_trait_expands_on_every_host() {
    for name in generated::ALL_CONCRETE_TOKENS
        .iter()
        .chain(generated::ALL_TRAIT_NAMES)
    {
        let ty: syn::Type = syn::parse_str(name).unwrap();
        for macro_name in ["arcane", "rite"] {
            if !generated::ALL_CONCRETE_TOKENS.contains(name) {
                continue;
            }
            for args in [
                quote!(),
                quote!(cfg(custom)),
                quote!(import_magetypes),
                quote!(import_intrinsics),
            ] {
                let input = quote!(#[allow(unused)] pub fn kernel<'a, const N: usize>(token: archmage::#ty, data: &'a [u8; N]) -> &'a [u8; N] { opaque!(Token, "Token", 'x'); data });
                let out = expand(macro_name, args, input).unwrap();
                let text = out.to_string();
                if !cfg!(feature = "avx512") && text.contains("requires the `avx512` feature") {
                    continue;
                }
                assert!(
                    !text.contains("compile_error"),
                    "{name} {macro_name}: {text}"
                );
                assert!(
                    text.contains("opaque ! (Token , \"Token\" , 'x')"),
                    "{text}"
                );
                assert!(text.contains("target_feature"), "{name}");
                syn::parse2::<syn::File>(out).unwrap();
            }
        }
        // Explicit paths and trait bounds must retain their original spelling.
        for signature in [
            quote!(fn kernel<T: archmage::#ty + Copy>(token: T) {}),
            quote!(fn kernel<T>(token: T) where T: archmage::#ty {}),
            quote!(fn kernel(token: impl archmage::#ty + Copy) {}),
        ] {
            let out = expand("arcane", quote!(), signature).unwrap();
            assert!(!out.to_string().contains("compile_error"), "{name}: {out}");
        }
    }
}

#[test]
fn expansion_modes_preserve_body_and_generated_contracts() {
    for (name, args, input, required) in [
        (
            "arcane",
            quote!(nested, _self = Container),
            quote!(
                fn kernel(&mut self, token: X64V3Token) -> Self {
                    opaque!(_self);
                    todo!()
                }
            ),
            "__simd_inner_kernel",
        ),
        (
            "arcane",
            quote!(_self = X64V3Token),
            quote!(
                fn kernel(self) {
                    opaque!(_self);
                }
            ),
            "__ARCHMAGE_ASSERT_TIER_",
        ),
        (
            "arcane",
            quote!(nested),
            quote!(
                fn kernel(token: X64V3Token, (x, y): (u8, u8), _: u32) {
                    opaque!(x, y);
                }
            ),
            "__archmage_arg_",
        ),
        (
            "arcane",
            quote!(suppress_const_test),
            quote!(
                unsafe fn kernel(token: X64V3Token) {
                    opaque!();
                }
            ),
            "target_feature",
        ),
        (
            "rite",
            quote!(scalar),
            quote!(
                fn kernel() {
                    opaque!();
                }
            ),
            "inline",
        ),
        (
            "rite",
            quote!(v3, neon, wasm128, scalar, cfg(custom)),
            quote!(
                fn kernel() {
                    opaque!();
                }
            ),
            "kernel_neon",
        ),
        (
            "autoversion",
            quote!(v3, neon, default, cfg(custom)),
            quote!(
                pub fn kernel<const N: usize>(data: &[u8; N]) {
                    opaque!(data);
                }
            ),
            "kernel_default",
        ),
        (
            "autoversion",
            quote!(v3, neon, scalar, _self = Container),
            quote!(
                fn kernel(&self, token: SimdToken) {
                    opaque!(_self);
                }
            ),
            "SIMDTOKEN_DEPRECATED",
        ),
        (
            "autoversion",
            quote!(scalar),
            quote!(
                fn kernel(token: ScalarToken) {
                    opaque!();
                }
            ),
            "kernel_scalar",
        ),
        (
            "magetypes",
            quote!(rite, define(f32x8, u8x16), v3, neon, scalar),
            quote!(
                fn kernel(token: Token) {
                    opaque!(Token, "Token");
                }
            ),
            "type f32x8",
        ),
        (
            "magetypes",
            quote!(v3(cfg(custom)), default),
            quote!(
                fn kernel() {
                    opaque!();
                }
            ),
            "kernel_default",
        ),
    ] {
        let out = expand(name, args, input).unwrap();
        let text = out.to_string();
        assert!(!text.contains("compile_error"), "{text}");
        assert!(text.contains(required), "{name} missing {required}: {text}");
        assert!(text.contains("opaque !"), "{name}: {text}");
        syn::parse2::<syn::File>(out).unwrap();
    }
}

#[test]
fn rejected_inputs_keep_actionable_diagnostics() {
    for (name, args, input, expected) in [
        (
            "arcane",
            quote!(stub),
            quote!(
                fn f(t: X64V3Token) {}
            ),
            "`stub` has been removed",
        ),
        (
            "rite",
            quote!(stub),
            quote!(
                fn f(t: X64V3Token) {}
            ),
            "`stub` has been removed",
        ),
        (
            "arcane",
            quote!(nonsense),
            quote!(
                fn f(t: X64V3Token) {}
            ),
            "unknown arcane argument",
        ),
        (
            "arcane",
            quote!(nested),
            quote!(
                fn f(&self, t: X64V3Token) {}
            ),
            "requires `_self = Type`",
        ),
        (
            "arcane",
            quote!(),
            quote!(
                fn f(x: u8) {}
            ),
            "requires a token parameter",
        ),
        (
            "rite",
            quote!(),
            quote!(
                fn f(x: u8) {}
            ),
            "requires a token parameter or a tier name",
        ),
        (
            "arcane",
            quote!(),
            quote!(
                fn f(t: impl SimdToken) {}
            ),
            "doesn't specify any CPU features",
        ),
        (
            "rite",
            quote!(),
            quote!(
                fn f<T: IntoConcreteToken>(t: T) {}
            ),
            "doesn't specify any CPU features",
        ),
        (
            "autoversion",
            quote!(),
            quote!(
                fn f(t: X64V3Token) {}
            ),
            "can't take a concrete token",
        ),
        (
            "magetypes",
            quote!(define("NotAVector")),
            quote!(
                fn f(t: Token) {}
            ),
            "expected identifier",
        ),
    ] {
        let text = match expand(name, args, input) {
            Ok(t) => t.to_string(),
            Err(e) => e.to_string(),
        };
        assert!(
            text.contains(expected),
            "{name}: expected {expected:?}, got {text}"
        );
    }
    for (input, expected) in [
        (
            quote!(f(x) without thing),
            "expected `token` after `without`",
        ),
        (quote!(f(x) without token, [v3]), "takes no tier list"),
        (quote!(f(x) nonsense token), "expected `with <token>`"),
    ] {
        let Err(error) = syn::parse2::<IncantInput>(input) else {
            panic!("accepted invalid input")
        };
        assert!(error.to_string().contains(expected));
    }
    let rejected = incant_impl(syn::parse2(quote!(f(x) without token)).unwrap()).to_string();
    assert!(rejected.contains("only valid inside a tier-macro body"));
}

#[test]
fn dispatch_entry_passthrough_and_cfg_outputs() {
    for input in [
        quote!(module::f::<T, 4>(Token, x), [v3, neon, wasm128, scalar]),
        quote!(f(x, Token), [v4(cfg(custom)), v3, default]),
        quote!(f(x) with token, [v3, neon, scalar]),
        quote!(f(Token, x) with token, [default]),
        quote!(f(x)),
        quote!(f(x), [v3]),
        quote!(f(x), [-scalar, v3]),
        quote!(f(x), [+default]),
    ] {
        let out = incant_impl(syn::parse2(input).unwrap());
        assert!(!out.to_string().contains("compile_error"), "{out}");
        syn::parse2::<syn::Expr>(out).unwrap();
    }
}

#[test]
fn token_transform_preserves_literals_delimiters_and_punctuation() {
    let input = quote!(Token::new([Token], "Token", r#"Token"#, 'T', -1.0); OtherToken; r#Token);
    let out = replace_ident_in_tokens(input, "Token", &quote!(archmage::X64V3Token));
    assert_eq!(out.to_string(), quote!(archmage::X64V3Token::new([archmage::X64V3Token], "Token", r#"Token"#, 'T', -1.0); OtherToken; r#Token).to_string());
    let mut input = Tokens::new();
    input.extend([TokenTree::Group(Group::new(Delimiter::None, quote!(Token)))]);
    let out = replace_ident_in_tokens(input, "Token", &quote!(Replacement));
    let TokenTree::Group(g) = out.into_iter().next().unwrap() else {
        panic!()
    };
    assert_eq!(g.delimiter(), Delimiter::None);
    assert_eq!(g.stream().to_string(), "Replacement");
    let path = replace_ident_in_tokens(quote!(Other::method), "Token", &quote!(Replacement));
    let punctuation: Vec<_> = path
        .into_iter()
        .filter_map(|t| {
            if let TokenTree::Punct(p) = t {
                Some(p.spacing())
            } else {
                None
            }
        })
        .collect();
    assert_eq!(punctuation, [Spacing::Joint, Spacing::Alone]);
}

// Allocation measurements are isolated to the current test thread. Nothing in
// this allocator is compiled into the distributed proc-macro library.
thread_local! {
    static ALLOCS: std::cell::Cell<Option<(usize, usize)>> = const { std::cell::Cell::new(None) };
}
struct CountAlloc;
#[global_allocator]
static ALLOCATOR: CountAlloc = CountAlloc;
// SAFETY: allocation and deallocation are forwarded unchanged to System.
unsafe impl std::alloc::GlobalAlloc for CountAlloc {
    unsafe fn alloc(&self, layout: std::alloc::Layout) -> *mut u8 {
        let _ = ALLOCS.try_with(|c| {
            if let Some((n, bytes)) = c.get() {
                c.set(Some((n + 1, bytes + layout.size())));
            }
        });
        unsafe { std::alloc::System.alloc(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: std::alloc::Layout) {
        unsafe { std::alloc::System.dealloc(ptr, layout) }
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: std::alloc::Layout, size: usize) -> *mut u8 {
        let _ = ALLOCS.try_with(|c| {
            if let Some((n, bytes)) = c.get() {
                c.set(Some((n + 1, bytes + size)));
            }
        });
        unsafe { std::alloc::System.realloc(ptr, layout, size) }
    }
}

/// Run with `cargo test -p archmage-macros --lib profile_allocations -- --ignored --nocapture`.
/// Counts signature/argument parsing and expansion, but not input lexing. Uses
/// proc_macro2's standalone backend; confirm wall-time wins in actual consumers.
#[test]
#[ignore = "measurement, not a correctness test"]
fn profile_allocations() {
    let simple = quote!(
        fn kernel(token: X64V3Token, data: &mut [f32], factor: f32) {
            for chunk in data.as_chunks_mut::<8>().0 {
                let a = f32x8::load(token, chunk);
                (a * f32x8::splat(token, factor)).store(chunk);
            }
        }
    );
    let dispatch = quote!(
        fn kernel(token: X64V3Token, data: &mut [f32]) {
            incant!(helper(data), [v4, v3, neon, scalar]);
        }
    );
    for (name, args, input) in [
        ("arcane", quote!(), simple.clone()),
        ("arcane-dispatch", quote!(), dispatch),
        ("rite", quote!(), simple),
        (
            "magetypes",
            quote!(define(f32x8)),
            quote!(
                fn kernel(token: Token, data: &mut [f32; 8]) {
                    f32x8::load(token, data).sqrt().store(data);
                }
            ),
        ),
        (
            "autoversion",
            quote!(),
            quote!(
                fn kernel(data: &mut [f32]) {
                    for x in data {
                        *x *= 2.0;
                    }
                }
            ),
        ),
    ] {
        let macro_name = if name == "arcane-dispatch" {
            "arcane"
        } else {
            name
        };
        let iterations = 1000;
        let start = std::time::Instant::now();
        ALLOCS.with(|c| c.set(Some((0, 0))));
        for _ in 0..iterations {
            std::hint::black_box(expand(macro_name, args.clone(), input.clone()).unwrap());
        }
        let (allocations, bytes) = ALLOCS.with(|c| c.take().unwrap());
        println!(
            "{name}: allocations={} bytes={} ns={}",
            allocations / iterations,
            bytes / iterations,
            start.elapsed().as_nanos() / iterations as u128
        );
    }
}

#[test]
fn default_tiers_are_sorted_unique_and_match_general_resolution() {
    let names = DEFAULT_TIER_NAMES
        .iter()
        .map(|n| n.to_string())
        .collect::<Vec<_>>();
    for gates in [false, true] {
        let general = resolve_tiers(&names, proc_macro2::Span::call_site(), gates).unwrap();
        let defaults = default_tiers(gates);
        assert_eq!(defaults.len(), general.len());
        assert!(defaults.windows(2).all(|t| t[0].priority >= t[1].priority));
        for (i, (a, b)) in defaults.iter().zip(&general).enumerate() {
            assert_eq!(
                (a.name, &a.feature_gate, a.allow_unexpected_cfg),
                (b.name, &b.feature_gate, b.allow_unexpected_cfg)
            );
            assert!(defaults[..i].iter().all(|t| t.name != a.name));
        }
        assert_eq!(defaults.last().unwrap().name, "scalar");
    }
    // Explicit equal-priority entries retain caller order, including duplicates.
    let names = ["wasm128", "neon", "v3", "neon", "scalar"].map(String::from);
    let resolved = resolve_tiers(&names, proc_macro2::Span::call_site(), false).unwrap();
    assert_eq!(
        resolved.iter().map(|t| t.name).collect::<Vec<_>>(),
        ["v3", "wasm128", "neon", "neon", "scalar"]
    );
}

#[test]
fn presence_scan_is_conservative_across_all_group_kinds() {
    assert!(!tokens_contain_ident(
        &quote!("incant Token"; r#"dispatch_variant"#; SomeToken; incantation),
        &["incant", "dispatch_variant", "Token"]
    ));
    for delimiter in [
        Delimiter::None,
        Delimiter::Brace,
        Delimiter::Bracket,
        Delimiter::Parenthesis,
    ] {
        for name in ["incant", "dispatch_variant", "Token"] {
            let id = proc_macro2::Ident::new(name, proc_macro2::Span::call_site());
            let group = Group::new(delimiter, quote!(before #id after));
            let tokens: Tokens = std::iter::once(TokenTree::Group(group)).collect();
            assert!(tokens_contain_ident(&tokens, &[name]));
        }
    }
}

#[test]
fn light_parser_preserves_opaque_body_and_rejects_broken_signatures() {
    let input = quote!(#[allow(dead_code)] pub(crate) unsafe extern "C" fn kernel<'a, T: Copy, const N: usize>(token: X64V3Token, data: &'a [T; N]) -> &'a [T; N] where T: 'a {
        // Deliberately not a Rust expression: LightFn must not parse the body.
        opaque DSL => [Token :: { arbitrary : tokens }];
    });
    let parsed: LightFn = syn::parse2(input.clone()).unwrap();
    assert_eq!(parsed.to_token_stream().to_string(), input.to_string());
    for invalid in [
        quote!(fn f(token X64V3Token) {}),
        quote!(fn f<T(token: T) {}),
        quote!(
            fn f();
        ),
        quote!(fn f() {}, trailing),
    ] {
        assert!(syn::parse2::<LightFn>(invalid).is_err());
    }
}

#[test]
fn nested_dispatch_rewrites_have_exact_outputs() {
    let ctx = rewrite::CallerContext {
        tier_suffix: "v3".into(),
        target_arch: Some("x86_64"),
        token_ident: quote::format_ident!("token"),
        has_token: true,
        derive_token: false,
    };
    for (input, expected) in [
        (
            quote!(({ incant!(f(Token, x), [v3, scalar]) })),
            quote!(({ f_v3(token, x) })),
        ),
        (
            quote!([dispatch_variant!(f(x) without token)]),
            quote!([f_v3(x)]),
        ),
        (
            quote!(|| { incant!(f(x), [v2, scalar]) }),
            quote!(|| { f_v2(token.v2(), x) }),
        ),
        (
            quote!(incant!(f(Token, x), [neon, default])),
            quote!(f_default(x)),
        ),
        (
            quote!(
                fn inner() {
                    incant!(f(x));
                }
            ),
            quote!(
                fn inner() {
                    incant!(f(x));
                }
            ),
        ),
        (quote!(incant!(invalid...)), quote!(incant!(invalid...))),
        (
            quote!(incant!(f(x) with held, [v3, scalar])),
            quote!(incant!(f(x) with held, [v3, scalar])),
        ),
    ] {
        assert_eq!(
            rewrite::rewrite_incant_in_body(input, &ctx).to_string(),
            expected.to_string()
        );
    }
    let ctx = rewrite::CallerContext {
        has_token: false,
        derive_token: true,
        ..ctx
    };
    let output =
        rewrite::rewrite_incant_in_body(quote!(incant!(f(Token, x), [neon, default])), &ctx);
    assert_eq!(output.to_string(), quote!(f_default(x)).to_string());
    let output = rewrite::rewrite_incant_in_body(quote!(incant!(f(x))), &ctx).to_string();
    assert!(output.contains("f_v3 (archmage :: X64V3Token :: from_context () , x)"));
}

#[test]
fn receivers_and_generic_modes_keep_feature_checks() {
    for (args, input) in [
        (
            quote!(),
            quote!(
                fn f(&mut self, token: X64V3Token) {
                    self.touch(token);
                }
            ),
        ),
        (
            quote!(inline_always),
            quote!(
                fn f(token: X64V3Token) {}
            ),
        ),
        (
            quote!(_self = Container),
            quote!(
                fn f(&self, token: Wasm128Token) {
                    _self.touch(token);
                }
            ),
        ),
        (
            quote!(nested),
            quote!(
                fn f<T: HasX64V2 + HasNeon>(token: T) {}
            ),
        ),
        (
            quote!(),
            quote!(
                fn f<T: HasX64V2 + HasNeon>(token: T) {}
            ),
        ),
    ] {
        let out = expand("arcane", args, input).unwrap();
        assert!(!out.to_string().contains("compile_error"));
        assert!(out.to_string().contains("target_feature"));
        syn::parse2::<syn::File>(out).unwrap();
    }
    let out = expand(
        "rite",
        quote!(v3, neon, import_magetypes),
        quote!(
            #[allow(dead_code)]
            fn f() {
                opaque!();
            }
        ),
    )
    .unwrap();
    assert!(out.to_string().contains("magetypes :: simd"));
    if !cfg!(feature = "avx512") {
        let out = expand(
            "rite",
            quote!(v4, v3, import_intrinsics),
            quote!(
                fn f() {}
            ),
        )
        .unwrap();
        assert!(out.to_string().contains("requires the `avx512` feature"));
    }
}
