//! Explicit inline policy, resolved before lowering operations to private helpers.
use proc_macro2::TokenStream;
use quote::quote;
use syn::Visibility;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum InlinePolicy {
    Default,
    None,
    Hint,
    Always,
    Never,
}

impl InlinePolicy {
    /// Freeze visibility-dependent policy before introducing a synthetic helper.
    pub(crate) fn resolve(self, visibility: &Visibility) -> Self {
        match self {
            Self::Default if matches!(visibility, Visibility::Public(_)) => Self::Hint,
            Self::Default => Self::None,
            policy => policy,
        }
    }

    /// Direct token emission on hot paths (zero `parse_quote!` / `syn::Attribute` parsing).
    pub(crate) fn tokens(self, visibility: &Visibility) -> TokenStream {
        match self.resolve(visibility) {
            Self::Default => unreachable!("default policy was resolved"),
            Self::None => TokenStream::new(),
            Self::Hint => quote!(#[inline]),
            Self::Always => quote!(#[inline(always)]),
            Self::Never => quote!(#[inline(never)]),
        }
    }
}
