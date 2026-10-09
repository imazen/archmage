//! Signature-local proof inference. Never inspect enclosing source or local bindings.
use std::cell::{Cell, OnceCell};

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
    binding: Ident,
    used: Cell<bool>,
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
            for arg in &signature.inputs {
                let FnArg::Typed(arg) = arg else { continue };
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
            binding: Ident::new("__attune_parent_proof", Span::mixed_site()),
            used: Cell::new(false),
        }
    }

    pub(super) fn access(&self, tier: &TierDescriptor) -> Result<Option<Access>, TokenStream> {
        let binding = &self.binding;
        match self
            .candidate
            .get_or_init(|| Candidate::discover(self.signature))
        {
            Candidate::Missing => Ok(None),
            Candidate::Ambiguous => Err(quote!(compile_error!(
                "attuned!: multiple parent proof parameters; select one with using(token)"
            ))),
            Candidate::One(proof) => {
                let access = match proof.concrete {
                    Some(parent) if parent == tier.suffix => Access::Guaranteed(quote!(#binding)),
                    Some(parent) if crate::generated::can_downgrade_tier(parent, tier.suffix) => {
                        let method = format_ident!("{}", tier.suffix);
                        Access::Guaranteed(quote!(#binding.#method()))
                    }
                    Some(_) => return Ok(None),
                    None => Access::Conditional(quote!(#binding)),
                };
                self.used.set(true);
                Ok(Some(access))
            }
        }
    }

    pub(super) fn capture(&self) -> TokenStream {
        if self.used.get()
            && let Some(Candidate::One(proof)) = self.candidate.get()
        {
            let binding = &self.binding;
            let value = &proof.value;
            // Snapshot the parameter before user-local shadowing. Tokens are Copy;
            // borrowed proofs are copied without retaining the caller's borrow.
            quote!(let #binding = #value;)
        } else {
            TokenStream::new()
        }
    }
}
