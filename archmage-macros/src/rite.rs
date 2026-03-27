//! `#[rite]` — adds `#[target_feature]` + `#[inline]` directly.
//!
//! Single-tier, multi-tier, and stub modes.

use proc_macro::TokenStream;
use quote::{format_ident, quote};
use syn::{
    Attribute, Ident, Token,
    parse::{Parse, ParseStream},
    parse_quote,
};

use crate::common::*;
use crate::generated::{
    canonical_token_to_tier_suffix, tier_to_canonical_token, token_to_arch, token_to_features,
    token_to_magetypes_namespace,
};
use crate::token_discovery::*;

#[derive(Default)]
pub(crate) struct RiteArgs {
    /// Generate an `unreachable!()` stub on the wrong architecture.
    /// Default is false (cfg-out: no function emitted on wrong arch).
    pub(crate) stub: bool,
    /// Inject `use archmage::intrinsics::{arch}::*;` (includes safe memory ops).
    pub(crate) import_intrinsics: bool,
    /// Inject `use magetypes::simd::{ns}::*;`, `use magetypes::simd::generic::*;`,
    /// and `use magetypes::simd::backends::*;`.
    import_magetypes: bool,
    /// Tiers specified directly (e.g., `#[rite(v3)]` or `#[rite(v3, v4, neon)]`).
    /// Stored as canonical token names (e.g., "X64V3Token").
    /// Single tier: generates one function (no suffix, no token parameter needed).
    /// Multiple tiers: generates suffixed variants (e.g., `fn_v3`, `fn_v4`, `fn_neon`).
    tier_tokens: Vec<String>,
    /// Additional cargo feature gate (same as arcane's cfg_feature).
    pub(crate) cfg_feature: Option<String>,
}

impl Parse for RiteArgs {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let mut args = RiteArgs::default();

        while !input.is_empty() {
            let ident: Ident = input.parse()?;
            match ident.to_string().as_str() {
                "stub" => args.stub = true,
                "import_intrinsics" => args.import_intrinsics = true,
                "import_magetypes" => args.import_magetypes = true,
                "cfg" => {
                    let content;
                    syn::parenthesized!(content in input);
                    let feat: Ident = content.parse()?;
                    args.cfg_feature = Some(feat.to_string());
                }
                other => {
                    if let Some(canonical) = tier_to_canonical_token(other) {
                        args.tier_tokens.push(String::from(canonical));
                    } else {
                        return Err(syn::Error::new(
                            ident.span(),
                            format!(
                                "unknown rite argument: `{}`. Supported: tier names \
                                 (v1, v2, v3, v4, neon, arm_v2, wasm128, ...), \
                                 `stub`, `import_intrinsics`, `import_magetypes`, `cfg(feature)`.",
                                other
                            ),
                        ));
                    }
                }
            }
            if input.peek(Token![,]) {
                let _: Token![,] = input.parse()?;
            }
        }

        Ok(args)
    }
}

/// Implementation for the `#[rite]` macro.
pub(crate) fn rite_impl(input_fn: LightFn, args: RiteArgs) -> TokenStream {
    // Multi-tier mode: generate suffixed variants for each tier
    if args.tier_tokens.len() > 1 {
        return rite_multi_tier_impl(input_fn, &args);
    }

    // Single-tier or token-param mode
    rite_single_impl(input_fn, args)
}

/// Generate a single `#[rite]` function (single tier or token-param mode).
pub(crate) fn rite_single_impl(mut input_fn: LightFn, args: RiteArgs) -> TokenStream {
    // Resolve features: either from tier name or from token parameter
    let TokenParamInfo {
        features,
        target_arch,
        token_type_name: _token_type_name,
        magetypes_namespace,
        ..
    } = if let Some(tier_token) = args.tier_tokens.first() {
        // Tier specified directly (e.g., #[rite(v3)]) — no token param needed
        let features = token_to_features(tier_token)
            .expect("tier_to_canonical_token returned invalid token name")
            .to_vec();
        let target_arch = token_to_arch(tier_token);
        let magetypes_namespace = token_to_magetypes_namespace(tier_token);
        TokenParamInfo {
            ident: Ident::new("_", proc_macro2::Span::call_site()),
            features,
            target_arch,
            token_type_name: Some(tier_token.clone()),
            magetypes_namespace,
        }
    } else {
        match find_token_param(&input_fn.sig) {
            Some(result) => result,
            None => {
                // Check for specific misuse: featureless traits like SimdToken
                if let Some(trait_name) = diagnose_featureless_token(&input_fn.sig) {
                    let msg = format!(
                        "`{trait_name}` cannot be used as a token bound in #[rite] \
                         because it doesn't specify any CPU features.\n\
                         \n\
                         #[rite] needs concrete features to generate #[target_feature]. \
                         Use a concrete token, a feature trait, or a tier name:\n\
                         \n\
                         Concrete tokens: X64V3Token, Desktop64, NeonToken, Arm64V2Token, ...\n\
                         Feature traits:  impl HasX64V2, impl HasNeon, impl HasArm64V3, ...\n\
                         Tier names:      #[rite(v3)], #[rite(neon)], #[rite(v4)], ..."
                    );
                    return syn::Error::new_spanned(&input_fn.sig, msg)
                        .to_compile_error()
                        .into();
                }
                let msg = "rite requires a token parameter or a tier name. Supported forms:\n\
                     - Tier name: `#[rite(v3)]`, `#[rite(neon)]`\n\
                     - Multi-tier: `#[rite(v3, v4, neon)]` (generates suffixed variants)\n\
                     - Concrete: `token: X64V3Token`\n\
                     - impl Trait: `token: impl HasX64V2`\n\
                     - Generic: `fn foo<T: HasX64V2>(token: T, ...)`";
                return syn::Error::new_spanned(&input_fn.sig, msg)
                    .to_compile_error()
                    .into();
            }
        }
    };

    // Check: import_intrinsics with AVX-512 features requires the avx512 cargo feature.
    // Check resolved features (not token name) for uniform handling of concrete/trait/generic.
    #[cfg(not(feature = "avx512"))]
    if args.import_intrinsics && features.iter().any(|f| f.starts_with("avx512")) {
        let token_desc = _token_type_name.as_deref().unwrap_or("an AVX-512 token");
        let msg = format!(
            "Using {token_desc} with `import_intrinsics` requires the `avx512` feature.\n\
             \n\
             Add to your Cargo.toml:\n\
             \x20 archmage = {{ version = \"...\", features = [\"avx512\"] }}\n\
             \n\
             Without it, 512-bit safe memory ops (_mm512_loadu_ps etc.) are not available.\n\
             If you only need value intrinsics (no memory ops), remove `import_intrinsics`."
        );
        return syn::Error::new_spanned(&input_fn.sig, msg)
            .to_compile_error()
            .into();
    }

    // Build a single target_feature attribute with all features comma-joined
    let features_csv = features.join(",");
    let target_feature_attrs: Vec<Attribute> =
        vec![parse_quote!(#[target_feature(enable = #features_csv)])];

    // Always use #[inline] - #[inline(always)] + #[target_feature] requires nightly
    let inline_attr: Attribute = parse_quote!(#[inline]);

    // Prepend attributes to the function, filtering user #[inline] to avoid duplicates
    let mut new_attrs = target_feature_attrs;
    new_attrs.push(inline_attr);
    for attr in filter_inline_attrs(&input_fn.attrs) {
        new_attrs.push(attr.clone());
    }
    input_fn.attrs = new_attrs;

    // Prepend import statements to body if requested
    let body_imports = generate_imports(
        target_arch,
        magetypes_namespace,
        args.import_intrinsics,
        args.import_magetypes,
    );
    if !body_imports.is_empty() {
        let original_body = &input_fn.body;
        input_fn.body = quote! {
            #body_imports
            #original_body
        };
    }

    // If we know the target arch, generate cfg-gated impl (+ optional stub)
    let cfg_guard = gen_cfg_guard(target_arch, args.cfg_feature.as_deref());
    if target_arch.is_some() {
        let vis = &input_fn.vis;
        let sig = &input_fn.sig;
        let attrs = &input_fn.attrs;
        let body = &input_fn.body;

        let stub = if args.stub {
            let not_cfg = match (target_arch, args.cfg_feature.as_deref()) {
                (Some(arch), Some(feat)) => {
                    quote! { #[cfg(not(all(target_arch = #arch, feature = #feat)))] }
                }
                (Some(arch), None) => quote! { #[cfg(not(target_arch = #arch))] },
                _ => quote! {},
            };
            quote! {
                #not_cfg
                #vis #sig {
                    unreachable!("This function requires a specific architecture and feature set")
                }
            }
        } else {
            quote! {}
        };

        quote! {
            #cfg_guard
            #(#attrs)*
            #vis #sig {
                #body
            }

            #stub
        }
        .into()
    } else {
        // No specific arch (trait bounds) - just emit the annotated function
        quote!(#input_fn).into()
    }
}

/// Generate multiple suffixed `#[rite]` variants for multi-tier mode.
///
/// `#[rite(v3, v4, neon)]` on `fn process(...)` generates:
/// - `fn process_v3(...)` with `#[target_feature(enable = "avx2,fma,...")]`
/// - `fn process_v4(...)` with `#[target_feature(enable = "avx512f,...")]`
/// - `fn process_neon(...)` with `#[target_feature(enable = "neon")]`
///
/// Each variant is cfg-gated to its architecture and gets `#[inline]`.
pub(crate) fn rite_multi_tier_impl(input_fn: LightFn, args: &RiteArgs) -> TokenStream {
    let fn_name = &input_fn.sig.ident;
    let mut variants = proc_macro2::TokenStream::new();

    for tier_token in &args.tier_tokens {
        let features = match token_to_features(tier_token) {
            Some(f) => f,
            None => {
                return syn::Error::new_spanned(
                    &input_fn.sig,
                    format!("unknown token `{tier_token}` in multi-tier #[rite]"),
                )
                .to_compile_error()
                .into();
            }
        };
        let target_arch = token_to_arch(tier_token);
        let magetypes_namespace = token_to_magetypes_namespace(tier_token);

        // Check: import_intrinsics with AVX-512 features requires the avx512 cargo feature.
        #[cfg(not(feature = "avx512"))]
        if args.import_intrinsics && features.iter().any(|f| f.starts_with("avx512")) {
            let msg = format!(
                "Using {tier_token} with `import_intrinsics` requires the `avx512` feature.\n\
                 \n\
                 Add to your Cargo.toml:\n\
                 \x20 archmage = {{ version = \"...\", features = [\"avx512\"] }}\n\
                 \n\
                 Without it, 512-bit safe memory ops (_mm512_loadu_ps etc.) are not available.\n\
                 If you only need value intrinsics (no memory ops), remove `import_intrinsics`."
            );
            return syn::Error::new_spanned(&input_fn.sig, msg)
                .to_compile_error()
                .into();
        }

        let suffix = canonical_token_to_tier_suffix(tier_token)
            .expect("canonical token must have a tier suffix");

        // Build suffixed function name
        let suffixed_ident = format_ident!("{}_{}", fn_name, suffix);

        // Clone and rename the function
        let mut variant_fn = input_fn.clone();
        variant_fn.sig.ident = suffixed_ident;

        // Build a single target_feature attribute with all features comma-joined
        let features_csv = features.join(",");
        let target_feature_attrs: Vec<Attribute> =
            vec![parse_quote!(#[target_feature(enable = #features_csv)])];
        let inline_attr: Attribute = parse_quote!(#[inline]);

        let mut new_attrs = target_feature_attrs;
        new_attrs.push(inline_attr);
        for attr in filter_inline_attrs(&variant_fn.attrs) {
            new_attrs.push(attr.clone());
        }
        variant_fn.attrs = new_attrs;

        // Prepend import statements if requested
        let body_imports = generate_imports(
            target_arch,
            magetypes_namespace,
            args.import_intrinsics,
            args.import_magetypes,
        );
        if !body_imports.is_empty() {
            let original_body = &variant_fn.body;
            variant_fn.body = quote! {
                #body_imports
                #original_body
            };
        }

        // Emit cfg-gated variant
        let variant_cfg = gen_cfg_guard(target_arch, args.cfg_feature.as_deref());
        if target_arch.is_some() {
            let vis = &variant_fn.vis;
            let sig = &variant_fn.sig;
            let attrs = &variant_fn.attrs;
            let body = &variant_fn.body;

            variants.extend(quote! {
                #variant_cfg
                #(#attrs)*
                #vis #sig {
                    #body
                }
            });

            if args.stub {
                let not_cfg = match (target_arch, args.cfg_feature.as_deref()) {
                    (Some(arch), Some(feat)) => {
                        quote! { #[cfg(not(all(target_arch = #arch, feature = #feat)))] }
                    }
                    (Some(arch), None) => quote! { #[cfg(not(target_arch = #arch))] },
                    _ => quote! {},
                };
                let arch_str = target_arch.unwrap_or("unknown");
                variants.extend(quote! {
                    #not_cfg
                    #vis #sig {
                        unreachable!(concat!(
                            "This function requires ",
                            #arch_str,
                            " architecture"
                        ))
                    }
                });
            }
        } else {
            // No specific arch — just emit the annotated function
            variants.extend(quote!(#variant_fn));
        }
    }

    variants.into()
}
