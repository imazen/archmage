//! Legacy arcane syntax; expansion is shared with attune(wrap).
use crate::common::*;
pub(crate) use crate::engine::boundary::BoundaryOptions as ArcaneArgs;
use proc_macro2::TokenStream;
use syn::{
    Ident, Token,
    parse::{Parse, ParseStream},
};

impl Parse for crate::engine::boundary::BoundaryOptions {
    fn parse(input: ParseStream<'_>) -> syn::Result<Self> {
        let mut args = ArcaneArgs::default();

        while !input.is_empty() {
            let ident: Ident = input.parse()?;
            if !parse_shared_option(&ident, input, &mut args.shared)? {
                with_ident_str(&ident, |s| match s {
                    "suppress_const_test" => {
                        args.suppress_const_test = true;
                        Ok(())
                    }
                    "inline_always" => {
                        args.inline_always = true;
                        Ok(())
                    }
                    "nested" | "in_trait" => {
                        args.nested = true;
                        Ok(())
                    }
                    "in_impl" => {
                        args.in_impl = true;
                        Ok(())
                    }
                    "_self" => {
                        let _: Token![=] = input.parse()?;
                        args.self_type = Some(input.parse()?);
                        Ok(())
                    }
                    other => Err(syn::Error::new(
                        ident.span(),
                        format!(
                            "unknown arcane argument: `{other}`. Supported: `in_impl`, \
                             `in_trait` (or `nested`), `_self = Type`, `import_intrinsics`, \
                             `import_magetypes`, `cfg(feature)`, `suppress_const_test`."
                        ),
                    )),
                })?;
            }
            if input.peek(Token![,]) {
                let _: Token![,] = input.parse()?;
            }
        }

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
