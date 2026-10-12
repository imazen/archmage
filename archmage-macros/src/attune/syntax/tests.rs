use super::{grammar::*, *};
use quote::{ToTokens, quote};

fn plan(source: &str) -> Args {
    syn::parse_str(source).unwrap_or_else(|e| panic!("{source}: {e}"))
}

#[test]
fn grammar_retains_selectors_before_normalization() {
    let syntax: Definition = syn::parse_str(
        "_*(pub(crate)), _v4x_t(cfg(avx512), pub), +v2, -neon, dispatch, inline(hint)",
    )
    .unwrap();
    assert!(syntax.family);
    assert_eq!(syntax.outputs.len(), 5);
    assert!(matches!(
        syntax.outputs[0].target,
        Target::Wildcard(Form::Direct)
    ));
    assert!(matches!(syntax.outputs[1].target, Target::Tier(t, Form::Proof) if t.name == "v4x"));
    assert!(matches!(syntax.outputs[2].action, Action::Add));
    assert!(matches!(syntax.outputs[3].action, Action::Remove));
    assert!(matches!(syntax.outputs[4].target, Target::Dispatcher));
    assert_eq!(syntax.outputs[1].options.gate.as_deref(), Some("avx512"));
    assert!(syntax.options.body_inline == Some(Inline::Hint));
    // Parsing accepts a grammatical but incompatible declaration. Resolution is
    // a separately testable phase and is responsible for the cross-option error.
    let invalid: Definition = syn::parse_str("wrap, _v3").unwrap();
    assert!(
        invalid
            .resolve()
            .err()
            .unwrap()
            .to_string()
            .contains("cannot be combined")
    );
}

#[test]
fn context_tiers_and_output_suffixes_have_distinct_meanings() {
    let context = plan("v3");
    assert!(!context.family);
    assert_eq!(context.options.tier.unwrap().name, "v3");
    assert!(context.selections.is_empty());
    let generated = plan("_v3");
    assert!(generated.family);
    assert!(generated.options.tier.is_none());
    assert_eq!(generated.selections.len(), 1);
    assert_eq!(generated.selections[0].tier.name, "v3");
    assert!(generated.selections[0].form == Form::Direct);
}

fn emitted(source: &str) -> String {
    crate::attune::expand(
        source.parse().unwrap(),
        quote!(
            pub fn work<T: Copy>(x: T) -> T {
                x
            }
        ),
    )
    .unwrap()
    .to_string()
}

#[test]
fn structured_forms_preserve_existing_emission() {
    for (structured, grouped) in [
        ("dispatch", "make(_)"),
        ("_v4x_t(cfg(avx512), pub)", "make(pub _v4x_t(avx512))"),
        ("all", "make(all)"),
        ("_v3, _v3_t", "make(_v3, _v3_t)"),
        ("_*(pub(crate)), _*_t(pub)", "make(pub(crate) _*, pub _*_t)"),
        ("_v4x_t(cfg(avx512), pub)", "make(pub _v4x_t(cfg(avx512)))"),
        (
            "_v3(inline(never)), _scalar",
            "make(inline(never) _v3, _scalar)",
        ),
        (
            "dispatch, +v4(cfg(avx512)), -neon",
            "make(dispatch, +v4(avx512), -neon)",
        ),
        ("_*, +v4x(cfg(avx512))", "make(_*, +v4x(avx512))"),
        (
            "_scalar_t(pub(crate), inline(always)), inline(never)",
            "make(pub(crate) inline(always) _scalar_t), inline(never)",
        ),
        (
            "_v3, names(_v3 = renamed)",
            "make(_v3), names(_v3 = renamed)",
        ),
    ] {
        assert_eq!(emitted(structured), emitted(grouped), "{structured}");
    }
}

#[test]
fn selector_option_order_and_trailing_commas_are_irrelevant() {
    let reference = emitted("_v4x_t(pub(crate), cfg(avx512), inline(never))");
    for spelling in [
        "_v4x_t(cfg(avx512), inline(never), pub(crate))",
        "_v4x_t(inline(never), pub(crate), cfg(avx512),),",
        "make(_v4x_t(pub(crate), cfg(avx512), inline(never),),)",
    ] {
        assert_eq!(reference, emitted(spelling));
    }
}

#[test]
fn body_and_wrapper_policies_remain_independent() {
    let result = emitted("_scalar_t(pub, inline(always)), inline(never)");
    let file: syn::File = syn::parse_str(&result).unwrap();
    for item in &file.items {
        let syn::Item::Fn(f) = item else {
            panic!("function expected")
        };
        let attributes: Vec<_> = f
            .attrs
            .iter()
            .map(|a| a.to_token_stream().to_string())
            .collect();
        if f.sig.ident == "work_scalar_t" {
            assert!(matches!(f.vis, Visibility::Public(_)));
            assert!(attributes.contains(&"# [inline (always)]".to_string()));
            assert!(!attributes.contains(&"# [inline (never)]".to_string()));
        } else {
            assert_eq!(f.sig.ident, "__attune_work_scalar");
            assert!(matches!(f.vis, Visibility::Inherited));
            assert!(attributes.contains(&"# [inline (never)]".to_string()));
        }
    }
    assert_eq!(file.items.len(), 2);
}

#[test]
fn resolution_expands_wildcards_and_keeps_dispatch_additions_hidden() {
    let args = plan("dispatch, +v4x(cfg(avx512)), -neon, -v4");
    assert!(args.dispatcher.is_some());
    assert!(args.selections.iter().all(|s| s.form == Form::Hidden));
    assert!(
        args.selections
            .iter()
            .all(|s| s.tier.name != "neon" && s.tier.name != "v4")
    );
    assert_eq!(args.selections.len(), DEFAULTS.len());
    let added = args
        .selections
        .iter()
        .find(|s| s.tier.name == "v4x")
        .unwrap();
    assert_eq!(added.gate.as_deref(), Some("avx512"));
    let args = plan("_*(pub(crate)), _*_t(pub), +v4x(cfg(avx512))");
    let added: Vec<_> = args
        .selections
        .iter()
        .filter(|s| s.tier.name == "v4x")
        .collect();
    assert_eq!(added.len(), 2);
    assert!(matches!(
        added[0].visibility,
        Some(Visibility::Restricted(_))
    ));
    assert!(matches!(added[1].visibility, Some(Visibility::Public(_))));
    assert!(added.iter().all(|s| s.gate.as_deref() == Some("avx512")));
}

#[test]
fn duplicate_options_and_unknown_syntax_never_silently_overwrite() {
    for source in [
        "wrap, wrap",
        "in_trait, nested",
        "in_impl, in_impl",
        "inline(hint), inline(hint)",
        "define(f32x8), define(f32x4)",
        "names(_v3 = a), names(_v2 = b), _v3",
        "_self = A, _self = B",
        "import_intrinsics, import_intrinsics",
        "import_magetypes, import_magetypes",
        "v3, cfg(one), cfg(two)",
        "v3, v3",
        "make(_v3), make(_v2)",
        "_v3(pub, pub(crate))",
        "_v3(inline(hint), inline(never))",
        "_v3(cfg(one), cfg(two))",
        "make(pub _v3(pub))",
    ] {
        let error = syn::parse_str::<Args>(source)
            .err()
            .unwrap_or_else(|| panic!("accepted {source}"));
        assert!(error.to_string().contains("duplicate"), "{source}: {error}");
    }
    for source in [
        "__v3",
        "__v3_t",
        "_v3(magic)",
        "_v3(cfg(avx512, other))",
        "_v3(cfg())",
        "_v3(inline(maybe))",
        "_v3 _scalar",
        "_v3_t_t",
        "_**",
        "_v3(pub, garbage)",
        "pub(crate) _v3",
        "v3_t",
        "auto",
    ] {
        assert!(syn::parse_str::<Args>(source).is_err(), "accepted {source}");
    }
}

#[test]
fn incompatible_options_fail_before_emission() {
    for (source, message) in [
        ("wrap, v3", "wrap derives"),
        ("wrap, define(f32x8)", "define(...) requires"),
        ("v3, _v3", "cannot be combined"),
        ("wrap, _v3", "cannot be combined"),
        ("names(_v3 = a)", "requires generated outputs"),
        ("cfg(avx512), dispatch", "gate a whole family"),
        ("in_impl, in_trait, _v3", "different placements"),
        ("in_trait, inline(default)", "cannot infer trait visibility"),
        ("make(_v3), _scalar", "keep all outputs"),
        ("_scalar, make(_v3)", "keep all outputs"),
        ("dispatch, _", "duplicate dispatcher"),
        ("dispatch, all", "duplicate dispatcher"),
        ("dispatch, -scalar", "requires its scalar fallback"),
        (
            "dispatch, _scalar(cfg(optional))",
            "ungated scalar fallback",
        ),
        ("_v3(cfg(a)), _v3_t(cfg(b))", "same feature gate"),
        ("_v3(pub), _v3(pub(crate))", "conflicting policies"),
        (
            "_v3(inline(hint)), _v3(inline(never))",
            "conflicting policies",
        ),
        ("+v4", "requires a wildcard"),
        ("_*, +v4_t", "use +tier"),
        ("dispatch, +v4(pub)", "hidden dispatcher additions"),
        ("dispatch(cfg(optional))", "cfg belongs on a named tier"),
        ("_*(cfg(optional))", "cfg belongs on a named tier"),
        ("-v4(pub)", "removals do not accept options"),
        ("+all", "modifiers require a named tier"),
        ("-dispatch", "modifiers require a named tier"),
        ("make()", "selects no outputs"),
        ("_scalar, -scalar", "selects no outputs"),
    ] {
        let error = syn::parse_str::<Args>(source)
            .err()
            .unwrap_or_else(|| panic!("accepted {source}"));
        assert!(error.to_string().contains(message), "{source}: {error}");
    }
}

#[test]
fn equivalent_duplicates_coalesce_and_absent_removal_is_idempotent() {
    assert_eq!(
        emitted("_v3(pub(crate)), _v3(pub(crate)), -v4, -v4"),
        emitted("_v3(pub(crate))")
    );
    assert_eq!(
        emitted("+v4(cfg(avx512)), dispatch"),
        emitted("dispatch, +v4(cfg(avx512))")
    );
}

#[test]
fn registered_tiers_share_one_grammar() {
    for tier in crate::tiers::ALL_TIERS
        .iter()
        .filter(|t| t.name != "default")
    {
        let source = format!("_{}(pub), _{}_t(pub(crate))", tier.suffix, tier.suffix);
        let args = plan(&source);
        assert_eq!(args.selections.len(), 2, "{source}");
        assert!(args.selections.iter().all(|s| s.tier.name == tier.name));
        assert!(args.selections[0].form == Form::Direct);
        assert!(args.selections[1].form == Form::Proof);
    }
}
