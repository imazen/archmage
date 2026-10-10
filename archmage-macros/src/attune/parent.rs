//! Signature-local proof inference. Never inspect enclosing source or local bindings.
use std::cell::OnceCell;

use proc_macro2::{Span, TokenStream};
use quote::{format_ident, quote};
use syn::{FnArg, Ident, Pat, Signature, Type};

use crate::tiers::TierDescriptor;
use crate::token_discovery::{self, TokenTypeInfo};

struct Proof {
    value: TokenStream,
    concrete: Option<&'static str>,
}

enum Candidate {
    Missing,
    One(Proof),
    Ambiguous,
}

pub(super) struct Parent<'a> {
    signature: Option<&'a Signature>,
    candidate: OnceCell<Candidate>,
    binding: OnceCell<Ident>,
}

pub(super) enum Access {
    /// A concrete token proves this exact tier or a registry ancestor.
    Guaranteed(TokenStream),
    /// Generic IntoConcreteToken follows the existing exact-type selection contract.
    Conditional(TokenStream),
}

impl Candidate {
    fn discover(signature: Option<&Signature>) -> Self {
        let mut candidate = Candidate::Missing;
        if let Some(signature) = signature {
            for arg in signature.inputs.pairs() {
                let FnArg::Typed(arg) = arg.value() else {
                    continue;
                };
                let Pat::Ident(pattern) = arg.pat.as_ref() else {
                    continue;
                };
                let Some(info) = token_discovery::extract_token_type_info(&arg.ty) else {
                    continue;
                };
                let concrete = match info {
                    TokenTypeInfo::Concrete(name) => {
                        let Some(tier) = crate::generated::canonical_token_to_tier_suffix(&name)
                        else {
                            continue;
                        };
                        Some(tier)
                    }
                    TokenTypeInfo::ImplTrait(bounds) => {
                        if !bounds.iter().any(|b| b == "IntoConcreteToken") {
                            continue;
                        }
                        None
                    }
                    TokenTypeInfo::Generic(name) => {
                        if signature.generics.params.is_empty()
                            && signature.generics.where_clause.is_none()
                        {
                            continue;
                        }
                        if !token_discovery::find_generic_bounds(signature, &name)
                            .is_some_and(|bounds| bounds.iter().any(|b| b == "IntoConcreteToken"))
                        {
                            continue;
                        }
                        None
                    }
                };
                if !matches!(candidate, Candidate::Missing) {
                    candidate = Candidate::Ambiguous;
                    break;
                }
                let name = &pattern.ident;
                let mut value = quote!(#name);
                let mut ty = arg.ty.as_ref();
                while let Type::Reference(reference) = ty {
                    value = quote!(*(#value));
                    ty = &reference.elem;
                }
                if pattern.by_ref.is_some() {
                    value = quote!(*(#value));
                }
                candidate = Candidate::One(Proof { value, concrete });
            }
        }
        candidate
    }
}

impl<'a> Parent<'a> {
    pub(super) fn new(signature: Option<&'a Signature>) -> Self {
        Self {
            signature,
            candidate: OnceCell::new(),
            binding: OnceCell::new(),
        }
    }

    pub(super) fn access(&self, tier: &TierDescriptor) -> Result<Option<Access>, TokenStream> {
        match self
            .candidate
            .get_or_init(|| Candidate::discover(self.signature))
        {
            Candidate::Missing => Ok(None),
            Candidate::Ambiguous => Err(quote!(compile_error!(
                "attuned!: multiple parent proof parameters; select one with using(token)"
            ))),
            Candidate::One(proof) => {
                let downgrade = match proof.concrete {
                    Some(parent) if parent == tier.suffix => false,
                    Some(parent) if crate::generated::can_downgrade_tier(parent, tier.suffix) => {
                        true
                    }
                    Some(_) => return Ok(None),
                    None => false,
                };
                let binding = self
                    .binding
                    .get_or_init(|| Ident::new("__attune_parent_proof", Span::mixed_site()));
                Ok(Some(if proof.concrete.is_none() {
                    Access::Conditional(quote!(#binding))
                } else if downgrade {
                    let method = format_ident!("{}", tier.suffix);
                    Access::Guaranteed(quote!(#binding.#method()))
                } else {
                    Access::Guaranteed(quote!(#binding))
                }))
            }
        }
    }

    pub(super) fn capture(&self) -> Option<TokenStream> {
        let binding = self.binding.get()?;
        let Candidate::One(proof) = self.candidate.get()? else {
            return None;
        };
        let value = &proof.value;
        // Snapshot the parameter before user-local shadowing. Tokens are Copy;
        // borrowed proofs are copied without retaining the caller's borrow.
        Some(quote!(let #binding = #value;))
    }
}
