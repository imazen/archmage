//! Explicit inline policy, resolved before lowering operations to private helpers.
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
    /// Freeze visibility-dependent policy before introducing a synthetic helper.
    pub(crate) fn resolve(self, visibility: &Visibility) -> Self {
        match self {
            Self::Default if matches!(visibility, Visibility::Public(_)) => Self::Hint,
            Self::Default => Self::None,
            policy => policy,
        }
    }

    pub(crate) fn attribute(self, visibility: &Visibility) -> Option<Attribute> {
        match self.resolve(visibility) {
            Self::Default => unreachable!("default policy was resolved"),
            Self::None => None,
            Self::Hint => Some(parse_quote!(#[inline])),
            Self::Always => Some(parse_quote!(#[inline(always)])),
            Self::Never => Some(parse_quote!(#[inline(never)])),
        }
    }
}
