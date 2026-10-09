//! Explicit inline policy, resolved only after the output visibility is known.
use syn::{Attribute, Visibility, parse_quote};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum InlinePolicy {
    Default,
    None,
    Hint,
    Always,
    Never,
}

impl InlinePolicy {
    pub(crate) fn attribute(self, visibility: &Visibility) -> Option<Attribute> {
        match self {
            Self::Default if matches!(visibility, Visibility::Public(_)) => {
                Some(parse_quote!(#[inline]))
            }
            Self::Default | Self::None => None,
            Self::Hint => Some(parse_quote!(#[inline])),
            Self::Always => Some(parse_quote!(#[inline(always)])),
            Self::Never => Some(parse_quote!(#[inline(never)])),
        }
    }
}
