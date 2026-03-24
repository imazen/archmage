//! `incant!` — runtime dispatch to platform-specific variants.

use proc_macro::TokenStream;
use quote::{format_ident, quote};
use syn::{
    Ident, Token,
    parse::{Parse, ParseStream},
};

use crate::common::*;
use crate::tiers::*;

/// Input for the incant! macro
pub(crate) struct IncantInput {
    /// Function path to call (e.g. `func` or `module::func`)
    pub(crate) func_path: syn::Path,
    /// Arguments to pass
    pub(crate) args: Vec<syn::Expr>,
    /// Optional token variable for passthrough mode
    with_token: Option<syn::Expr>,
    /// Optional explicit tier list (None = default tiers)
    pub(crate) tiers: Option<(Vec<String>, proc_macro2::Span)>,
}

// suffix_path → moved to common.rs

impl Parse for IncantInput {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        // Parse: function_path(arg1, arg2, ...) [with token_expr] [, [tier1, tier2, ...]]
        let func_path: syn::Path = input.parse()?;

        // Parse parenthesized arguments
        let content;
        syn::parenthesized!(content in input);
        let args = content
            .parse_terminated(syn::Expr::parse, Token![,])?
            .into_iter()
            .collect();

        // Check for optional "with token"
        let with_token = if input.peek(Ident) {
            let kw: Ident = input.parse()?;
            if kw != "with" {
                return Err(syn::Error::new_spanned(kw, "expected `with` keyword"));
            }
            Some(input.parse()?)
        } else {
            None
        };

        // Check for optional tier list: , [tier1, tier2(feature), ...]
        // tier(feature) wraps dispatch in #[cfg(feature = "feature")].
        // Example: [v4(avx512), v3, neon(simd), scalar]
        let tiers = if input.peek(Token![,]) {
            let _: Token![,] = input.parse()?;
            let bracket_content;
            let bracket = syn::bracketed!(bracket_content in input);
            let mut tier_names = Vec::new();
            while !bracket_content.is_empty() {
                tier_names.push(parse_one_tier(&bracket_content)?);
                if bracket_content.peek(Token![,]) {
                    let _: Token![,] = bracket_content.parse()?;
                }
            }
            Some((tier_names, bracket.span.join()))
        } else {
            None
        };

        Ok(IncantInput {
            func_path,
            args,
            with_token,
            tiers,
        })
    }
}

/// Dispatch to platform-specific SIMD variants.
///
/// # Entry Point Mode (no token yet)
///
/// Summons tokens and dispatches to the best available variant:
///
/// ```rust,ignore
/// pub fn public_api(data: &[f32]) -> f32 {
///     incant!(dot(data))
/// }
/// ```
///
/// Expands to runtime feature detection + dispatch to `dot_v3`, `dot_v4`,
/// `dot_neon`, `dot_wasm128`, or `dot_scalar`.
///
/// # Explicit Tiers
///
/// Specify which tiers to dispatch to:
///
/// ```rust,ignore
/// // Only dispatch to v1, v3, neon, and scalar
/// pub fn api(data: &[f32]) -> f32 {
///     incant!(process(data), [v1, v3, neon, scalar])
/// }
/// ```
///
/// Always include `scalar` in explicit tier lists — `incant!` always
/// emits a `fn_scalar()` call as the final fallback, and listing it
/// documents this dependency. Currently auto-appended if omitted;
/// will become a compile error in v1.0. Unknown tier names cause a
/// compile error. Tiers are automatically sorted into correct
/// dispatch order (highest priority first).
///
/// Known tiers: `v1`, `v2`, `v3`, `v4`, `v4x`, `neon`, `neon_aes`,
/// `neon_sha3`, `neon_crc`, `wasm128`, `wasm128_relaxed`, `scalar`.
///
/// # Passthrough Mode (already have token)
///
/// Uses compile-time dispatch via `IntoConcreteToken`:
///
/// ```rust,ignore
/// #[arcane]
/// fn outer(token: X64V3Token, data: &[f32]) -> f32 {
///     incant!(inner(data) with token)
/// }
/// ```
///
/// Also supports explicit tiers:
///
/// ```rust,ignore
/// fn inner<T: IntoConcreteToken>(token: T, data: &[f32]) -> f32 {
///     incant!(process(data) with token, [v3, neon, scalar])
/// }
/// ```
///
/// The compiler monomorphizes the dispatch, eliminating non-matching branches.
///
/// # Variant Naming
///
/// Functions must have suffixed variants matching the selected tiers:
/// - `_v1` for `X64V1Token`
/// - `_v2` for `X64V2Token`
/// - `_v3` for `X64V3Token`
/// - `_v4` for `X64V4Token` (requires `avx512` feature)
/// - `_v4x` for `X64V4xToken` (requires `avx512` feature)
/// - `_neon` for `NeonToken`
/// - `_neon_aes` for `NeonAesToken`
/// - `_neon_sha3` for `NeonSha3Token`
/// - `_neon_crc` for `NeonCrcToken`
/// - `_wasm128` for `Wasm128Token`
/// - `_scalar` for `ScalarToken`

pub(crate) fn incant_impl(input: IncantInput) -> TokenStream {
    let func_path = &input.func_path;
    let args = &input.args;

    // Resolve tiers
    let tier_names: Vec<String> = match &input.tiers {
        Some((names, _)) => names.clone(),
        None => DEFAULT_TIER_NAMES.iter().map(|s| s.to_string()).collect(),
    };
    let last_segment_span = func_path
        .segments
        .last()
        .map(|s| s.ident.span())
        .unwrap_or_else(proc_macro2::Span::call_site);
    let error_span = input
        .tiers
        .as_ref()
        .map(|(_, span)| *span)
        .unwrap_or(last_segment_span);

    // When the user specifies explicit tiers without `scalar` or `default` in
    // incant!, emit a deprecation warning. In v1.0 this will become a compile error.
    // A fallback tier is always auto-appended, but not listing it explicitly
    // hides the fact that a _scalar/_default function is required.
    let scalar_warning = if let Some((names, _span)) = &input.tiers {
        // Additive mode (+tier) inherits scalar from defaults — no warning needed.
        let is_additive = names.iter().all(|n| n.starts_with('+'));
        if !is_additive
            && !names.iter().any(|n| {
                let base = n
                    .strip_prefix('+')
                    .unwrap_or(n)
                    .split('(')
                    .next()
                    .unwrap_or(n);
                base == "scalar" || base == "default"
            })
        {
            quote! {
                #[deprecated(since = "0.9.9", note = "\
                    explicit incant! tier lists should include `scalar`. \
                    incant! always calls fn_scalar() as the final fallback. \
                    This will become a compile error in v1.0. \
                    Example: incant!(foo(x), [v3, neon, scalar])")]
                #[allow(non_upper_case_globals)]
                const __incant_missing_scalar_tier: () = ();
                let _ = __incant_missing_scalar_tier;
            }
        } else {
            quote! {}
        }
    } else {
        quote! {}
    };

    // Apply default feature gates: tiers with cfg_feature (v4→avx512) auto-get
    // the gate unless the user explicitly wrote tier(feature). This is true for
    // BOTH default and explicit tier lists — backwards compatible with published
    // crates using [v4, v3, neon] where _v4 is behind #[cfg(feature = "avx512")].
    // Users with unconditional _v4 functions use v4(!) or just don't cfg-gate them.
    let tiers = match resolve_tiers(&tier_names, error_span, true) {
        Ok(t) => t,
        Err(e) => return e.to_compile_error().into(),
    };

    // Group tiers by architecture for cfg-guarded blocks
    // Within each arch, tiers are already sorted by priority (highest first)
    let dispatch: TokenStream = if let Some(token_expr) = &input.with_token {
        gen_incant_passthrough(func_path, args, token_expr, &tiers)
    } else {
        gen_incant_entry(func_path, args, &tiers)
    };

    if scalar_warning.is_empty() {
        dispatch
    } else {
        // Wrap dispatch in a block that includes the deprecation warning
        let dispatch2: proc_macro2::TokenStream = dispatch.into();
        quote! {
            {
                #scalar_warning
                #dispatch2
            }
        }
        .into()
    }
}

/// Generate incant! passthrough mode (already have a token).
pub(crate) fn gen_incant_passthrough(
    func_path: &syn::Path,
    args: &[syn::Expr],
    token_expr: &syn::Expr,
    tiers: &[ResolvedTier],
) -> TokenStream {
    let mut dispatch_arms = Vec::new();

    // Group non-fallback tiers by target_arch for cfg blocks
    let mut arch_groups: Vec<(Option<&str>, Vec<&ResolvedTier>)> = Vec::new();
    for rt in tiers {
        if rt.name == "scalar" || rt.name == "default" {
            continue; // Handle fallback separately at the end
        }
        if let Some(group) = arch_groups.iter_mut().find(|(a, _)| *a == rt.target_arch) {
            group.1.push(rt);
        } else {
            arch_groups.push((rt.target_arch, vec![rt]));
        }
    }

    for (target_arch, group_tiers) in &arch_groups {
        let mut tier_checks = Vec::new();
        for rt in group_tiers {
            let fn_suffixed = suffix_path(func_path, rt.suffix);
            let as_method = format_ident!("{}", rt.as_method);

            let check = quote! {
                if let Some(__t) = __incant_token.#as_method() {
                    break '__incant #fn_suffixed(__t, #(#args),*);
                }
            };

            if let Some(feat) = &rt.feature_gate {
                let allow_attr = if rt.allow_unexpected_cfg {
                    quote! { #[allow(unexpected_cfgs)] }
                } else {
                    quote! {}
                };
                tier_checks.push(quote! {
                    #allow_attr
                    #[cfg(feature = #feat)]
                    { #check }
                });
            } else {
                tier_checks.push(check);
            }
        }

        let inner = quote! { #(#tier_checks)* };

        if let Some(arch) = target_arch {
            dispatch_arms.push(quote! {
                #[cfg(target_arch = #arch)]
                { #inner }
            });
        } else {
            dispatch_arms.push(inner);
        }
    }

    // Fallback (always last): scalar (with token) or default (tokenless)
    let has_default = tiers.iter().any(|t| t.name == "default");
    let fallback_arm = if has_default {
        let fn_default = suffix_path(func_path, "default");
        quote! {
            break '__incant #fn_default(#(#args),*);
        }
    } else if tiers.iter().any(|t| t.name == "scalar") {
        let fn_scalar = suffix_path(func_path, "scalar");
        quote! {
            if let Some(__t) = __incant_token.as_scalar() {
                break '__incant #fn_scalar(__t, #(#args),*);
            }
            unreachable!("Token did not match any known variant")
        }
    } else {
        quote! { unreachable!("Token did not match any known variant") }
    };

    let expanded = quote! {
        '__incant: {
            use archmage::IntoConcreteToken;
            let __incant_token = #token_expr;
            #(#dispatch_arms)*
            #fallback_arm
        }
    };
    expanded.into()
}

/// Generate incant! entry point mode (summon tokens).
pub(crate) fn gen_incant_entry(
    func_path: &syn::Path,
    args: &[syn::Expr],
    tiers: &[ResolvedTier],
) -> TokenStream {
    let mut dispatch_arms = Vec::new();

    // Group non-fallback tiers by target_arch for cfg blocks.
    let mut arch_groups: Vec<(Option<&str>, Vec<&ResolvedTier>)> = Vec::new();
    for rt in tiers {
        if rt.name == "scalar" || rt.name == "default" {
            continue;
        }
        if let Some(group) = arch_groups.iter_mut().find(|(a, _)| *a == rt.target_arch) {
            group.1.push(rt);
        } else {
            arch_groups.push((rt.target_arch, vec![rt]));
        }
    }

    for (target_arch, group_tiers) in &arch_groups {
        let mut tier_checks = Vec::new();
        for rt in group_tiers {
            let fn_suffixed = suffix_path(func_path, rt.suffix);
            let token_path: syn::Path = syn::parse_str(rt.token_path).unwrap();

            let check = quote! {
                if let Some(__t) = #token_path::summon() {
                    break '__incant #fn_suffixed(__t, #(#args),*);
                }
            };

            if let Some(feat) = &rt.feature_gate {
                let allow_attr = if rt.allow_unexpected_cfg {
                    quote! { #[allow(unexpected_cfgs)] }
                } else {
                    quote! {}
                };
                tier_checks.push(quote! {
                    #allow_attr
                    #[cfg(feature = #feat)]
                    { #check }
                });
            } else {
                tier_checks.push(check);
            }
        }

        let inner = quote! { #(#tier_checks)* };

        if let Some(arch) = target_arch {
            dispatch_arms.push(quote! {
                #[cfg(target_arch = #arch)]
                { #inner }
            });
        } else {
            dispatch_arms.push(inner);
        }
    }

    // Fallback: scalar (with ScalarToken) or default (tokenless)
    let has_default = tiers.iter().any(|rt| rt.name == "default");
    let fallback_call = if has_default {
        let fn_default = suffix_path(func_path, "default");
        quote! { #fn_default(#(#args),*) }
    } else {
        let fn_scalar = suffix_path(func_path, "scalar");
        quote! { #fn_scalar(archmage::ScalarToken, #(#args),*) }
    };

    let expanded = quote! {
        '__incant: {
            use archmage::SimdToken;
            #(#dispatch_arms)*
            #fallback_call
        }
    };
    expanded.into()
}
