use super::*;
use std::collections::HashSet;

#[test]
fn registry_is_acyclic_unique_and_architecture_preserving() {
    let mut names = HashSet::new();
    let mut proofs = HashSet::new();
    for tier in TIERS {
        assert!(names.insert(tier.name));
        assert!(proofs.insert(tier.proof));
        if !tier.parent.is_empty() {
            assert!(names.contains(tier.parent), "parent must precede child");
            let parent = TIERS.iter().find(|t| t.name == tier.parent).unwrap();
            assert_eq!(parent.arch, tier.arch);
        }
        let features = tier.features();
        assert_eq!(
            features.iter().collect::<HashSet<_>>().len(),
            features.len()
        );
    }
    let v3 = lookup(&format_ident!("v3")).unwrap().features();
    for f in ["sse2", "cmpxchg16b", "avx2", "fma", "f16c", "movbe"] {
        assert!(v3.contains(&f));
    }
    assert!(!v3.contains(&"avx512f"));
}

#[test]
fn dispatch_emits_each_branch_and_each_argument_exactly_once() {
    let output = select(syn::parse2(quote!(work(argument()); v4x, v4, v3, v2, neon, arm_v2, arm_v3, wasm128, wasm128_relaxed, scalar)).unwrap()).unwrap().to_string();
    assert_eq!(output.matches("summon").count(), 9);
    assert_eq!(output.matches("argument").count(), 1);
    for tier in [
        "v4x",
        "v4",
        "v3",
        "v2",
        "neon",
        "arm_v2",
        "arm_v3",
        "wasm128",
        "wasm128_relaxed",
        "scalar",
    ] {
        assert_eq!(output.matches(&format!("work_{tier}_t")).count(), 1);
    }
}

#[test]
fn feature_attributes_and_inline_choices_are_on_the_right_items() {
    let mode = syn::parse2(quote!(v3)).unwrap();
    let raw = syn::parse2(quote!(
        #[cfg(unix)]
        #[inline(never)]
        #[track_caller]
        pub fn work(x: u32) -> u32 {
            x
        }
    ))
    .unwrap();
    let out: syn::File = syn::parse2(emit(mode, raw, true).unwrap()).unwrap();
    assert_eq!(out.items.len(), 3);
    let fns: Vec<_> = out
        .items
        .iter()
        .filter_map(|item| {
            if let syn::Item::Fn(f) = item {
                Some(f)
            } else {
                None
            }
        })
        .collect();
    assert_eq!(fns.len(), 2);
    for f in &fns {
        assert!(matches!(f.vis, syn::Visibility::Public(_)));
        assert!(f.attrs.iter().any(|a| a.path().is_ident("track_caller")));
        assert_eq!(
            f.attrs.iter().filter(|a| a.path().is_ident("cfg")).count(),
            2
        );
    }
    assert!(
        fns[0]
            .attrs
            .iter()
            .any(|a| a.path().is_ident("target_feature"))
    );
    assert!(
        !fns[1]
            .attrs
            .iter()
            .any(|a| a.path().is_ident("target_feature"))
    );
    let inline = |f: &ItemFn| {
        f.attrs
            .iter()
            .find(|a| a.path().is_ident("inline"))
            .unwrap()
            .meta
            .require_list()
            .unwrap()
            .tokens
            .to_string()
    };
    assert_eq!(inline(fns[0]), "never");
    assert_eq!(inline(fns[1]), "always");
}
