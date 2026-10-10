//! Legacy arcane syntax; expansion is shared with attune(wrap).
use crate::common::*;
pub(crate) use crate::engine::boundary::BoundaryOptions as ArcaneArgs;
use proc_macro2::TokenStream;
use syn::{
    Ident, Token,
    parse::{Parse, ParseStream},
};

impl Parse for crate::engine::boundary::BoundaryOptions {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let mut args = ArcaneArgs::default();

        while !input.is_empty() {
            let ident: Ident = input.parse()?;
            if !parse_shared_option(&ident, input, &mut args.shared)? {
                match ident.to_string().as_str() {
                    "suppress_const_test" => args.suppress_const_test = true,
                    "inline_always" => args.inline_always = true,
                    "nested" | "in_trait" => args.nested = true,
                    "in_impl" => args.in_impl = true,
                    "_self" => {
                        let _: Token![=] = input.parse()?;
                        args.self_type = Some(input.parse()?);
                    }
                    other => {
                        return Err(syn::Error::new(
                            ident.span(),
                            format!(
                                "unknown arcane argument: `{other}`. Supported: `in_impl`, \
                                 `in_trait` (or `nested`), `_self = Type`, `import_intrinsics`, \
                                 `import_magetypes`, `cfg(feature)`, `suppress_const_test`."
                            ),
                        ));
                    }
                }
            }
            // Consume optional comma
            if input.peek(Token![,]) {
                let _: Token![,] = input.parse()?;
            }
        }

        // _self = Type implies nested (inner fn needed for Self replacement)
        if args.self_type.is_some() {
            args.nested = true;
        }
        if args.in_impl && args.nested {
            return Err(syn::Error::new(
                input.span(),
                "`in_impl` is for a receiver-less function in an inherent impl, where the \
                 sibling expansion applies; `in_trait`/`nested`/`_self` already avoid the \
                 sibling. Use one of them.",
            ));
        }

        Ok(args)
    }
}

pub(crate) fn arcane_impl(input: LightFn, macro_name: &str, args: ArcaneArgs) -> TokenStream {
    crate::engine::boundary::expand(input, macro_name, args)
}
