//! # artisan-macros
//!
//! Auditable, convention-forward proc-macros for sound SIMD tier dispatch.
//!
//! See the companion markdown files for the full design and specs:
//!
//! - `DESIGN.md` — design rationale, trampoline-chain model
//! - `SPEC-CPU-TIER.md` — normative spec for `#[cpu_tier]`
//! - `SPEC-CHAIN.md` — normative spec for `#[chain]`
//! - `SPEC-TEST-HOOKS.md` — normative spec for the thread-local test hooks
//!
//! Two public macros, both attribute-form:
//!
//! - [`macro@cpu_tier`] — attaches `#[target_feature]` + `#[cfg(target_arch)]` + `#[inline]`
//!   to a function. User owns the feature string; arch inferred from feature names.
//! - [`macro@chain`] — declares a trampoline chain. Applied to an empty-body function;
//!   attribute args carry per-arch tier lists with their feature strings inline.
//!   Each tier's cache miss falls through to the next tier down; chain bottoms out
//!   at `default`.
//!
//! Status: **draft implementation, pending review.** Both macros expand end-to-end
//! on x86_64, aarch64, and wasm32. Test hooks (thread-local `force_max_tier`) are
//! emitted behind `#[cfg(any(test, feature = "artisan_test_hooks"))]`.

use proc_macro::TokenStream;
use proc_macro2::{Span, TokenStream as TokenStream2};
use quote::{format_ident, quote};
use syn::{
    Error, Expr, ExprLit, FnArg, Ident, ItemFn, Lit, LitStr, MetaNameValue, Pat, PatType, Result,
    Token,
    parse::{Parse, ParseStream},
    punctuated::Punctuated,
    spanned::Spanned,
};

// =============================================================================
// Arch inference table
// =============================================================================
//
// Feature names in this table unambiguously belong to exactly one target_arch.
// Features that appear on multiple arches (e.g. `aes`, `sha2`, `sha3`, `crc`,
// `fp16`) are intentionally omitted — they do not drive inference. If every
// feature in a user's `enable = "..."` list is ambiguous or unknown, inference
// fails with a compile error pointing at the explicit `arch = "..."` override.

const UNAMBIGUOUS_FEATURES: &[(&str, &str)] = &[
    // x86_64
    ("sse", "x86_64"),
    ("sse2", "x86_64"),
    ("sse3", "x86_64"),
    ("ssse3", "x86_64"),
    ("sse4.1", "x86_64"),
    ("sse4.2", "x86_64"),
    ("popcnt", "x86_64"),
    ("avx", "x86_64"),
    ("avx2", "x86_64"),
    ("fma", "x86_64"),
    ("bmi1", "x86_64"),
    ("bmi2", "x86_64"),
    ("lzcnt", "x86_64"),
    ("f16c", "x86_64"),
    ("movbe", "x86_64"),
    ("cmpxchg16b", "x86_64"),
    ("pclmulqdq", "x86_64"),
    ("avx512f", "x86_64"),
    ("avx512bw", "x86_64"),
    ("avx512cd", "x86_64"),
    ("avx512dq", "x86_64"),
    ("avx512vl", "x86_64"),
    ("avx512vnni", "x86_64"),
    ("avx512vbmi", "x86_64"),
    ("avx512vbmi2", "x86_64"),
    ("avx512ifma", "x86_64"),
    ("avx512bitalg", "x86_64"),
    ("avx512vpopcntdq", "x86_64"),
    ("avx512fp16", "x86_64"),
    ("avx512bf16", "x86_64"),
    ("vpclmulqdq", "x86_64"),
    ("vaes", "x86_64"),
    ("gfni", "x86_64"),
    // aarch64
    ("neon", "aarch64"),
    ("rdm", "aarch64"),
    ("dotprod", "aarch64"),
    ("fhm", "aarch64"),
    ("fcma", "aarch64"),
    ("i8mm", "aarch64"),
    ("bf16", "aarch64"),
    ("sve", "aarch64"),
    ("sve2", "aarch64"),
    ("pmull", "aarch64"),
    // wasm32
    ("simd128", "wasm32"),
    ("relaxed-simd", "wasm32"),
];

fn infer_arch(features: &str, span: Span) -> Result<&'static str> {
    for feat in features.split(',') {
        let feat = feat.trim();
        for (name, arch) in UNAMBIGUOUS_FEATURES {
            if *name == feat {
                return Ok(arch);
            }
        }
    }
    Err(Error::new(
        span,
        format!(
            "cannot infer target_arch from features `{features}`: all listed features \
             are ambiguous or unknown to artisan-macros. Add explicit \
             `arch = \"x86_64\" | \"aarch64\" | \"wasm32\"` to the `#[cpu_tier]` attribute."
        ),
    ))
}

fn split_features(features: &str) -> Vec<String> {
    features
        .split(',')
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect()
}

/// Canonicalise a feature string so two declarations with the same feature set
/// but different ordering / whitespace / duplicates produce the same normalized
/// form. Used by `#[cpu_tier]` (emits the normalized string as a const) and
/// `#[chain]` (compares against the const at compile time).
fn normalize_features(features: &str) -> String {
    let mut parts: Vec<String> = features
        .split(',')
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect();
    parts.sort();
    parts.dedup();
    parts.join(",")
}

/// Construct the hidden const ident that `#[cpu_tier]` emits alongside each
/// decorated function. `#[chain]` generates an assertion referencing this
/// ident, so both macros must produce the same mangled name for a given fn.
fn cpu_tier_feats_const_ident(fn_name: &Ident) -> Ident {
    format_ident!("__ARTISAN_CPU_TIER_FEATS_{}", fn_name)
}

// =============================================================================
// #[cpu_tier] — attach #[target_feature] + #[cfg(target_arch)] + #[inline]
// =============================================================================

/// Attach `#[target_feature(enable = "...")]` and the inferred `#[cfg(target_arch)]`
/// to a function.
///
/// ```ignore
/// #[cpu_tier(enable = "avx2,fma,bmi1,bmi2")]
/// fn compute_v3(data: &[f32]) -> f32 { /* AVX2+FMA body */ }
/// ```
///
/// Emits:
///
/// ```ignore
/// #[cfg(target_arch = "x86_64")]
/// #[target_feature(enable = "avx2,fma,bmi1,bmi2")]
/// #[inline]
/// fn compute_v3(data: &[f32]) -> f32 { /* ... */ }
/// ```
///
/// The generated function is `fn`, not `unsafe fn` (Rust 2024 edition rule). This
/// keeps `#![forbid(unsafe_code)]` working for downstream crates.
///
/// See `SPEC-CPU-TIER.md` for the full normative specification.
#[proc_macro_attribute]
pub fn cpu_tier(attr: TokenStream, item: TokenStream) -> TokenStream {
    match expand_cpu_tier(attr, item) {
        Ok(ts) => ts.into(),
        Err(e) => e.to_compile_error().into(),
    }
}

struct CpuTierArgs {
    enable: String,
    enable_span: Span,
    arch: Option<String>,
}

impl Parse for CpuTierArgs {
    fn parse(input: ParseStream) -> Result<Self> {
        let items: Punctuated<MetaNameValue, Token![,]> = Punctuated::parse_terminated(input)?;
        let mut enable: Option<(String, Span)> = None;
        let mut arch: Option<String> = None;
        for item in items {
            let name = item
                .path
                .get_ident()
                .ok_or_else(|| Error::new_spanned(&item.path, "expected identifier"))?
                .to_string();
            let value_span = item.value.span();
            let value = match &item.value {
                Expr::Lit(ExprLit {
                    lit: Lit::Str(s), ..
                }) => s.value(),
                other => {
                    return Err(Error::new_spanned(
                        other,
                        "expected string literal (e.g. \"avx2,fma\")",
                    ));
                }
            };
            match name.as_str() {
                "enable" => {
                    if enable.is_some() {
                        return Err(Error::new(value_span, "`enable` specified twice"));
                    }
                    enable = Some((value, value_span));
                }
                "arch" => {
                    if arch.is_some() {
                        return Err(Error::new(value_span, "`arch` specified twice"));
                    }
                    arch = Some(value);
                }
                other => {
                    return Err(Error::new_spanned(
                        &item.path,
                        format!("unknown argument `{other}`; expected `enable` or `arch`"),
                    ));
                }
            }
        }
        let (enable, enable_span) = enable.ok_or_else(|| {
            Error::new(
                Span::call_site(),
                "missing required `enable = \"...\"` argument",
            )
        })?;
        Ok(Self {
            enable,
            enable_span,
            arch,
        })
    }
}

fn expand_cpu_tier(attr: TokenStream, item: TokenStream) -> Result<TokenStream2> {
    let args: CpuTierArgs = syn::parse(attr)?;
    let item_fn: ItemFn = syn::parse(item)?;

    // Reject `unsafe fn` — would break #![forbid(unsafe_code)] for downstream crates.
    if let Some(u) = &item_fn.sig.unsafety {
        return Err(Error::new_spanned(
            u,
            "#[cpu_tier] expects a safe `fn`, not `unsafe fn`. Rust 2024 edition allows \
             `#[target_feature]` on safe fns; `unsafe fn` would break \
             `#![forbid(unsafe_code)]` for downstream crates.",
        ));
    }

    let arch = match &args.arch {
        Some(a) => a.clone(),
        None => infer_arch(&args.enable, args.enable_span)?.to_string(),
    };

    if !matches!(
        arch.as_str(),
        "x86_64" | "x86" | "aarch64" | "arm" | "wasm32"
    ) {
        return Err(Error::new(
            args.enable_span,
            format!(
                "arch `{arch}` is not one of: x86_64, x86, aarch64, arm, wasm32. \
                 (artisan-macros supports these five target_arch values.)"
            ),
        ));
    }

    let enable = &args.enable;
    let arch_lit = arch.as_str();
    let fn_name = &item_fn.sig.ident;
    let vis = &item_fn.vis;
    let feats_const_ident = cpu_tier_feats_const_ident(fn_name);
    let normalized = normalize_features(&args.enable);

    // Emit the decorated fn AND a public hidden const carrying the normalized
    // feature string. The const is cfg-gated to the same target_arch as the
    // fn, so references from `#[chain]` resolve only when both are present.
    //
    // Visibility: we mirror the fn's visibility so that `use crate::module::fn;`
    // also brings `__ARTISAN_CPU_TIER_FEATS_fn` into scope (Rust doesn't
    // auto-import consts with fns, but same-module references work without
    // explicit imports).
    Ok(quote! {
        #[cfg(target_arch = #arch_lit)]
        #[target_feature(enable = #enable)]
        #[inline]
        #item_fn

        #[cfg(target_arch = #arch_lit)]
        #[doc(hidden)]
        #[allow(non_upper_case_globals)]
        #vis const #feats_const_ident: &str = #normalized;
    })
}

// =============================================================================
// #[chain] — declare a trampoline chain
// =============================================================================

/// Declare a trampoline chain. Applied to an empty-body function; the macro
/// reads the signature and fills in the body with per-arch dispatch logic.
///
/// ```ignore
/// #[chain(
///     x86_64 = [
///         compute_v3 = "avx2,fma,bmi1,bmi2,f16c,lzcnt,popcnt,movbe",
///         compute_v2 = "sse4.2,popcnt",
///     ],
///     aarch64 = [
///         compute_neon = "neon",
///     ],
///     default = compute_scalar,
/// )]
/// pub fn compute(data: &[f32]) -> f32 {}
/// ```
///
/// Emits one trampoline per tier (each with its own `AtomicU8` cache), plus the
/// entry function body with compile-time arch dispatch and compile-time
/// feature-elision fast paths. Behind `#[cfg(any(test, feature = "artisan_test_hooks"))]`,
/// also emits a thread-local `force_max_tier` scope and per-chain tier enum.
///
/// Tier order in each arch list is highest-to-lowest. Feature strings inline at
/// the chain site must match the corresponding `#[cpu_tier(enable = "...")]` on
/// each tier function — see `SPEC-CHAIN.md` § "Feature-string duplication" for the
/// auditability rationale and failure modes.
///
/// See `SPEC-CHAIN.md` and `SPEC-TEST-HOOKS.md` for the full normative expansion.
#[proc_macro_attribute]
pub fn chain(attr: TokenStream, item: TokenStream) -> TokenStream {
    match expand_chain(attr, item) {
        Ok(ts) => ts.into(),
        Err(e) => e.to_compile_error().into(),
    }
}

struct ChainArgs {
    arches: Vec<ArchEntry>,
    default: Ident,
}

struct ArchEntry {
    arch: Ident,
    tiers: Vec<TierEntry>,
}

struct TierEntry {
    fn_name: Ident,
    features: String,
    features_span: Span,
}

impl Parse for ChainArgs {
    fn parse(input: ParseStream) -> Result<Self> {
        let mut arches: Vec<ArchEntry> = Vec::new();
        let mut default: Option<Ident> = None;

        while !input.is_empty() {
            let name: Ident = input.parse()?;
            input.parse::<Token![=]>()?;

            if name == "default" {
                let fn_name: Ident = input.parse()?;
                if default.is_some() {
                    return Err(Error::new(name.span(), "multiple `default = ...` entries"));
                }
                default = Some(fn_name);
            } else {
                let content;
                syn::bracketed!(content in input);
                let mut tiers = Vec::new();
                while !content.is_empty() {
                    let fn_name: Ident = content.parse()?;
                    content.parse::<Token![=]>()?;
                    let features_lit: LitStr = content.parse()?;
                    tiers.push(TierEntry {
                        fn_name,
                        features: features_lit.value(),
                        features_span: features_lit.span(),
                    });
                    if content.is_empty() {
                        break;
                    }
                    content.parse::<Token![,]>()?;
                }
                if tiers.is_empty() {
                    return Err(Error::new(
                        name.span(),
                        format!("arch `{name}` has an empty tier list; remove it or add tiers"),
                    ));
                }
                arches.push(ArchEntry { arch: name, tiers });
            }

            if input.is_empty() {
                break;
            }
            input.parse::<Token![,]>()?;
        }

        let default = default.ok_or_else(|| {
            Error::new(
                Span::call_site(),
                "missing required `default = <fn_name>` entry",
            )
        })?;

        Ok(Self { arches, default })
    }
}

fn detect_macro_for(arch: &str, span: Span) -> Result<TokenStream2> {
    match arch {
        "x86_64" | "x86" => Ok(quote! { ::std::is_x86_feature_detected }),
        "aarch64" => Ok(quote! { ::std::is_aarch64_feature_detected }),
        "arm" => Ok(quote! { ::std::is_arm_feature_detected }),
        "wasm32" => Err(Error::new(
            span,
            "wasm32 has no runtime feature detection; SIMD tiers on wasm must rely on \
             compile-time features only. For now, put the tier function directly in the \
             compile-time elision path via `#[cpu_tier]` and call it from `default` when \
             `simd128` is not enabled at compile time.",
        )),
        other => Err(Error::new(
            span,
            format!(
                "unsupported arch `{other}` in #[chain]: expected x86_64, x86, aarch64, or arm"
            ),
        )),
    }
}

fn extract_arg_idents(sig: &syn::Signature) -> Result<Vec<Ident>> {
    let mut out = Vec::with_capacity(sig.inputs.len());
    for arg in &sig.inputs {
        match arg {
            FnArg::Typed(PatType { pat, .. }) => {
                if let Pat::Ident(pi) = &**pat {
                    if pi.by_ref.is_some() || pi.mutability.is_some() || pi.subpat.is_some() {
                        return Err(Error::new_spanned(
                            pi,
                            "#[chain] expects simple `name: Type` parameters (no `ref`, no `mut` binding, no subpatterns)",
                        ));
                    }
                    out.push(pi.ident.clone());
                } else {
                    return Err(Error::new_spanned(
                        pat,
                        "#[chain] expects simple `name: Type` parameters (no destructuring)",
                    ));
                }
            }
            FnArg::Receiver(r) => {
                return Err(Error::new_spanned(
                    r,
                    "#[chain] cannot be applied to methods with `self` receivers; apply to a free function",
                ));
            }
        }
    }
    Ok(out)
}

fn mangle_chain_ident(fn_name: &Ident, arch: &str, tier_fn: &Ident) -> Ident {
    format_ident!("__artisan_{}__{}__{}__chain", fn_name, arch, tier_fn)
}

fn tier_variant_ident(tier_fn: &Ident, fn_name: &Ident) -> Ident {
    // Strip the chain's fn_name prefix from the tier fn name if present, uppercase
    // the rest. e.g. fn_name=compute, tier_fn=compute_v3 -> V3. If no common prefix,
    // use the whole tier fn name uppercased.
    let tier_str = tier_fn.to_string();
    let prefix = format!("{fn_name}_");
    let stripped = tier_str.strip_prefix(&prefix).unwrap_or(&tier_str);
    // Uppercase first letter, keep the rest. Avoid making this fancy.
    let mut chars = stripped.chars();
    let upper = match chars.next() {
        Some(c) => c.to_ascii_uppercase().to_string() + chars.as_str(),
        None => stripped.to_string(),
    };
    format_ident!("{}", upper)
}

fn expand_chain(attr: TokenStream, item: TokenStream) -> Result<TokenStream2> {
    let args: ChainArgs = syn::parse(attr)?;
    let item_fn: ItemFn = syn::parse(item)?;

    if let Some(u) = &item_fn.sig.unsafety {
        return Err(Error::new_spanned(
            u,
            "#[chain] expects a safe `fn`, not `unsafe fn`",
        ));
    }
    if item_fn.sig.asyncness.is_some() {
        return Err(Error::new_spanned(
            &item_fn.sig,
            "#[chain] does not support `async fn` in the current draft",
        ));
    }
    if !item_fn.sig.generics.params.is_empty() {
        return Err(Error::new_spanned(
            &item_fn.sig.generics,
            "#[chain] does not support generic functions in the current draft",
        ));
    }

    let attrs = &item_fn.attrs;
    let vis = &item_fn.vis;
    let sig = &item_fn.sig;
    let name = &sig.ident;
    let inputs = &sig.inputs;
    let output = &sig.output;
    let arg_idents = extract_arg_idents(sig)?;
    let default_fn = &args.default;

    // Validate: no duplicate arch entries, no duplicate tier fn names within an arch
    {
        let mut seen_arches = std::collections::HashSet::new();
        for a in &args.arches {
            let s = a.arch.to_string();
            if !seen_arches.insert(s.clone()) {
                return Err(Error::new(
                    a.arch.span(),
                    format!("arch `{s}` listed more than once in #[chain]"),
                ));
            }
            let mut seen_fns = std::collections::HashSet::new();
            for t in &a.tiers {
                let fs = t.fn_name.to_string();
                if !seen_fns.insert(fs.clone()) {
                    return Err(Error::new(
                        t.fn_name.span(),
                        format!(
                            "tier function `{fs}` listed more than once under arch `{}`",
                            a.arch
                        ),
                    ));
                }
            }
        }
    }

    // Build arch-covered cfg list for the fallback branch.
    let arch_strs: Vec<String> = args.arches.iter().map(|a| a.arch.to_string()).collect();
    let not_any_cfg = if arch_strs.is_empty() {
        quote! { all() } // true — no arches declared, default always
    } else {
        let arch_strs = arch_strs.iter();
        quote! { not(any(#(target_arch = #arch_strs),*)) }
    };

    // Per-arch dispatch blocks for the entry function.
    let mut arch_dispatch_blocks: Vec<TokenStream2> = Vec::new();
    // Per-arch chain trampolines (emitted outside the entry fn).
    let mut trampolines: Vec<TokenStream2> = Vec::new();
    // Test-hooks shared state (thread-local + enum + scope) for all tiers across all arches.
    let mut hook_variants: Vec<Ident> = Vec::new();
    // Per-arch compile-time equality assertion blocks. Each block verifies that
    // every tier's feature string in the chain matches the const emitted by
    // that tier's `#[cpu_tier]` decoration. Gated by target_arch so references
    // to the per-arch consts resolve only on matching arches.
    let mut feature_check_blocks: Vec<TokenStream2> = Vec::new();

    for arch_entry in &args.arches {
        let arch_name = arch_entry.arch.to_string();
        let arch_str = arch_name.as_str();
        let detect = detect_macro_for(arch_str, arch_entry.arch.span())?;

        // Build per-tier chain trampolines.
        for (i, tier) in arch_entry.tiers.iter().enumerate() {
            let this_fn = &tier.fn_name;
            let this_chain = mangle_chain_ident(name, arch_str, this_fn);
            let features = split_features(&tier.features);
            if features.is_empty() {
                return Err(Error::new(
                    tier.features_span,
                    format!("empty feature list for tier `{this_fn}`"),
                ));
            }
            let next_call: TokenStream2 = if i + 1 < arch_entry.tiers.len() {
                let next_fn = &arch_entry.tiers[i + 1].fn_name;
                let next_chain = mangle_chain_ident(name, arch_str, next_fn);
                quote! { #next_chain(#(#arg_idents),*) }
            } else {
                quote! { #default_fn(#(#arg_idents),*) }
            };
            let feature_checks = features.iter().map(|f| {
                quote! { #detect!(#f) }
            });

            // The tier variant name (for test-hooks enum)
            let variant = tier_variant_ident(this_fn, name);
            hook_variants.push(variant.clone());

            // Thread-local forced-max-tier override. Consulted BEFORE the atomic
            // cache. If override points at a tier lower than this one, fall
            // through to next_call without touching the cache.
            let variant_u8_index = (hook_variants.len() as u8) - 1;
            let cell_path = hook_cell_path(name);
            let forced_check = quote! {
                #[cfg(any(test, feature = "artisan_test_hooks"))]
                {
                    let forced: ::core::option::Option<u8> = #cell_path.with(|c| c.get());
                    if let ::core::option::Option::Some(max_idx) = forced {
                        if (#variant_u8_index) < max_idx {
                            return #next_call;
                        }
                    }
                }
            };

            trampolines.push(quote! {
                #[cfg(target_arch = #arch_str)]
                #[inline]
                #[allow(non_snake_case)]
                fn #this_chain(#inputs) #output {
                    #forced_check
                    use ::core::sync::atomic::{AtomicU8, Ordering};
                    static CACHE: AtomicU8 = AtomicU8::new(0);
                    match CACHE.load(Ordering::Relaxed) {
                        2u8 => unsafe { #this_fn(#(#arg_idents),*) },
                        1u8 => #next_call,
                        _ => {
                            let ok = #(#feature_checks)&&*;
                            CACHE.store(if ok { 2u8 } else { 1u8 }, Ordering::Relaxed);
                            if ok {
                                unsafe { #this_fn(#(#arg_idents),*) }
                            } else {
                                #next_call
                            }
                        }
                    }
                }
            });
        }

        // Entry arch block: compile-time elision for the top tier, fallback to
        // the top tier's chain trampoline.
        let top = &arch_entry.tiers[0];
        let top_fn = &top.fn_name;
        let top_features = split_features(&top.features);
        let top_feature_cfgs: Vec<TokenStream2> = top_features
            .iter()
            .map(|f| quote! { target_feature = #f })
            .collect();
        let top_chain = mangle_chain_ident(name, arch_str, top_fn);

        arch_dispatch_blocks.push(quote! {
            #[cfg(target_arch = #arch_str)]
            {
                #[cfg(all(#(#top_feature_cfgs),*))]
                { unsafe { #top_fn(#(#arg_idents),*) } }
                #[cfg(not(all(#(#top_feature_cfgs),*)))]
                { #top_chain(#(#arg_idents),*) }
            }
        });

        // Compile-time feature-string equality check for this arch.
        // Emits one assertion per tier comparing the chain-site normalized
        // feature string against the const emitted by `#[cpu_tier]` on the
        // tier function. Catches drift between the two declarations.
        let assertions: Vec<TokenStream2> = arch_entry
            .tiers
            .iter()
            .map(|tier| {
                let tier_fn = &tier.fn_name;
                let expected_const = cpu_tier_feats_const_ident(tier_fn);
                let normalized = normalize_features(&tier.features);
                let err_msg = format!(
                    "artisan-macros feature-string mismatch for tier `{tier_fn}` on arch `{arch_str}`:\n  \
                     chain site declares (normalized): {normalized}\n  \
                     cpu_tier declares (normalized):   (see const {expected_const})\n  \
                     Both strings must be equal after normalization (sort+dedupe+trim).\n  \
                     Fix: make the feature strings in #[cpu_tier(enable=\"...\")] and \
                     #[chain(... = \"...\")] list the same features."
                );
                quote! {
                    {
                        const CHAIN_SITE: &str = #normalized;
                        const fn __artisan_str_eq(a: &str, b: &str) -> bool {
                            let a = a.as_bytes();
                            let b = b.as_bytes();
                            if a.len() != b.len() { return false; }
                            let mut i = 0usize;
                            while i < a.len() {
                                if a[i] != b[i] { return false; }
                                i += 1;
                            }
                            true
                        }
                        assert!(__artisan_str_eq(#expected_const, CHAIN_SITE), #err_msg);
                    }
                }
            })
            .collect();

        feature_check_blocks.push(quote! {
            #[cfg(target_arch = #arch_str)]
            const _: () = {
                #(#assertions)*
            };
        });
    }

    // Test-hooks: emit once per chain. If no tiers at all, skip.
    let test_hooks = if hook_variants.is_empty() {
        quote! {}
    } else {
        build_test_hooks(name, &hook_variants)
    };

    let fallback_expr = quote! { #default_fn(#(#arg_idents),*) };

    // The function body is a sequence of cfg-gated block expressions where
    // exactly one compiles for any given target_arch. The compiled block's
    // value is the tail expression of the fn body.
    let expansion = quote! {
        #(#attrs)*
        #vis fn #name(#inputs) #output {
            #(#arch_dispatch_blocks)*
            #[cfg(#not_any_cfg)]
            { #fallback_expr }
        }

        #(#trampolines)*

        #(#feature_check_blocks)*

        #test_hooks
    };

    Ok(expansion)
}

// =============================================================================
// Test hooks (thread-local force_max_tier scope + per-chain tier enum)
// =============================================================================
//
// Gated behind `#[cfg(any(test, feature = "artisan_test_hooks"))]`. Users who
// want to invoke force_max_tier from tests in downstream crates need to enable
// the `artisan_test_hooks` feature on the crate that owns the #[chain]
// declaration. Unit tests inside the declaring crate get it automatically
// via cfg(test).

fn hook_mod_ident(name: &Ident) -> Ident {
    format_ident!("__artisan_{}_hooks", name)
}

fn hook_cell_path(name: &Ident) -> TokenStream2 {
    let m = hook_mod_ident(name);
    quote! { #m::FORCED_MAX_TIER }
}

fn build_test_hooks(name: &Ident, variants: &[Ident]) -> TokenStream2 {
    let mod_ident = hook_mod_ident(name);
    let capitalized = {
        let s = name.to_string();
        let mut it = s.chars();
        match it.next() {
            Some(c) => c.to_ascii_uppercase().to_string() + it.as_str(),
            None => s,
        }
    };
    let enum_ident = format_ident!("{}Tier", capitalized);
    let scope_ident = format_ident!("{}Scope", capitalized);
    let force_fn_ident = format_ident!("{}_force_max_tier", name);

    // Variant indices: 0 = first declared (highest), ascending = lower tiers.
    // We also append `Default` as one past the last, so forcing Default disables
    // every tier.
    let variant_arms: Vec<TokenStream2> = variants
        .iter()
        .enumerate()
        .map(|(i, v)| {
            let idx = i as u8;
            quote! { #enum_ident::#v => #idx }
        })
        .collect();
    let default_idx = variants.len() as u8;

    let variant_defs = variants.iter();

    quote! {
        #[cfg(any(test, feature = "artisan_test_hooks"))]
        #[doc(hidden)]
        pub mod #mod_ident {
            ::std::thread_local! {
                pub(super) static FORCED_MAX_TIER: ::core::cell::Cell<::core::option::Option<u8>>
                    = const { ::core::cell::Cell::new(::core::option::Option::None) };
            }
        }

        /// Test-hook tier enum for this chain. Variants listed in declared
        /// order across all arches (highest tier first), plus `Default` as the
        /// no-SIMD fallback.
        #[cfg(any(test, feature = "artisan_test_hooks"))]
        #[derive(Debug, Clone, Copy, PartialEq, Eq)]
        pub enum #enum_ident {
            #(#variant_defs,)*
            Default,
        }

        /// RAII scope guard returned by [`#force_fn_ident`]. Dropping restores the
        /// previous forced-max-tier setting on the current thread.
        #[cfg(any(test, feature = "artisan_test_hooks"))]
        #[must_use = "dropping the TierScope immediately restores the previous override"]
        pub struct #scope_ident {
            prev: ::core::option::Option<u8>,
        }

        #[cfg(any(test, feature = "artisan_test_hooks"))]
        impl ::core::ops::Drop for #scope_ident {
            fn drop(&mut self) {
                #mod_ident::FORCED_MAX_TIER.with(|c| c.set(self.prev));
            }
        }

        /// Force this chain to dispatch no higher than `tier` on the current
        /// thread. Returns an RAII guard; drop to restore.
        ///
        /// Sound because this can only DISABLE tiers the CPU supports, never
        /// fabricate tiers the CPU lacks. See `SPEC-TEST-HOOKS.md` for the
        /// concurrency and soundness properties.
        #[cfg(any(test, feature = "artisan_test_hooks"))]
        pub fn #force_fn_ident(tier: #enum_ident) -> #scope_ident {
            let idx: u8 = match tier {
                #(#variant_arms,)*
                #enum_ident::Default => #default_idx,
            };
            let prev = #mod_ident::FORCED_MAX_TIER.with(|c| c.replace(::core::option::Option::Some(idx)));
            #scope_ident { prev }
        }
    }
}
