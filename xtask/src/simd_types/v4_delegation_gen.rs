//! Concrete-impl delegation macros for `impls/x86_v4_f32_delegated.rs`,
//! `impls/x86_v3_delegated.rs`, and `impls/x86_v4_w512_delegated.rs`.
//!
//! Each generated file defines a crate-private `macro_rules!` whose arms are
//! derived directly from the generated backend trait ASTs (so a new trait
//! method can never silently fall back to its scalar default body, issue #60),
//! followed by one explicit macro invocation per `(Trait, Token)` pair.
//!
//! ## Intentional default-method exceptions
//!
//! Only `F32x4Backend::store_rgba_bytes` and `F32x8Backend::store_rgba_bytes`
//! are intentionally omitted from forwarding so they inherit their trait
//! default bodies, which dispatch through `Self::store_rgba_u8_into` (preserving
//! the selected AVX-512VL pixel-packing override on `F32x4Internal`). The
//! generator requires every listed exception to carry a default body on the
//! parsed trait and panics if an exception disappears or is unmatched.
//!
//! ## Auditability invariants
//!
//! - Every invocation expands to an ordinary concrete `impl Trait for
//!   archmage::<Token>` (no blanket impls, no coherence/method-lookup changes,
//!   concrete rustdoc rosters preserved).
//! - The macro accepts only an `archmage::$child:ident` token identifier and
//!   (for f32 overrides) a two-arm `v4_parent!` proof extractor (`identity` or
//!   `v4`).
//! - The generator validates at generation time that there are no duplicate
//!   tokens, duplicate trait specs, duplicate overrides, or unmatched overrides
//!   (catching renamed/removed trait methods immediately).
//! - The generator lexically scans the full emitted file (including helper
//!   macros and invocations, ignoring comments) and asserts zero forbidden
//!   identifiers (`unsafe`, `target_feature`, `arcane`, `attune`, `rite`,
//!   `from_context`, `summon`, `macro_export`) and zero raw pointer types
//!   (`*const` / `*mut`, regardless of whitespace).

use std::collections::{BTreeMap, BTreeSet};
use std::io::Write;
use std::process::{Command, Stdio};

use indoc::formatdoc;
use quote::{ToTokens, format_ident, quote};
use syn::{FnArg, Item, Pat, TraitItem};

/// How an AVX-512 child token obtains the `X64V4Token` proof expected by
/// `impls/x86_v4_f32_overrides.rs`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum V4ProofMode {
    /// The token *is* `X64V4Token` (`self`).
    Identity,
    /// The token is a strict superset of `X64V4Token` (`self.v4()`).
    ViaV4,
}

impl V4ProofMode {
    const fn as_macro_tag(self) -> &'static str {
        match self {
            Self::Identity => "identity",
            Self::ViaV4 => "v4",
        }
    }
}

/// A concrete `archmage::*` token that delegates narrow W128/W256 traits to
/// `archmage::X64V3Token`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct DelegatedToken {
    /// Unqualified identifier under `archmage::` (e.g. `"X64V4Token"`).
    pub(crate) ident: &'static str,
    /// How this token supplies an `X64V4Token` proof to AVX-512VL overrides.
    pub(crate) v4_mode: V4ProofMode,
}

/// The AVX-512 tokens that delegate narrow W128/W256 backends to V3
/// (gated on `feature = "avx512"`).
const AVX512_TOKENS: &[DelegatedToken] = &[
    DelegatedToken {
        ident: "X64V4Token",
        v4_mode: V4ProofMode::Identity,
    },
    DelegatedToken {
        ident: "X64V4xToken",
        v4_mode: V4ProofMode::ViaV4,
    },
    DelegatedToken {
        ident: "Avx512Fp16Token",
        v4_mode: V4ProofMode::ViaV4,
    },
];

/// Parent token and zero-cost proof-extraction method forwarded to by a
/// delegation macro (e.g. `X64V3Token` via `self.v3()`, or `X64V4Token` via
/// `self.v4()`).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct DelegationParent {
    token: &'static str,
    method: &'static str,
}

const V3_PARENT: DelegationParent = DelegationParent {
    token: "X64V3Token",
    method: "v3",
};

const V4_PARENT: DelegationParent = DelegationParent {
    token: "X64V4Token",
    method: "v4",
};

/// Pure-V3 extension tokens that delegate ALL 34 W128/W256 backends (including
/// f32 without AVX-512VL overrides) to `X64V3Token`, independent of
/// `feature = "avx512"`.
const V3_ONLY_TOKENS: &[&str] = &["X64V3VnniToken"];

/// AVX-512 superset tokens that delegate the 12 base W512 backends
/// (`F32x16Backend`, `F32x16Internal`, `F64x8Backend`, `I8x64..U64x8Backend`,
/// `F32x16Convert`) to `X64V4Token` via `self.v4()`.
const V4_W512_DELEGATED_TOKENS: &[DelegatedToken] = &[DelegatedToken {
    ident: "X64V4xToken",
    v4_mode: V4ProofMode::ViaV4,
}];

/// Methods that take the AVX-512VL path in `impls/x86_v4_f32_overrides.rs`:
/// `(trait_name, method_name, helper_fn)`.
const OVERRIDES: &[(&str, &str, &str)] = &[
    ("F32x4Backend", "to_u8_bytes", "to_u8_bytes_x4_v4"),
    (
        "F32x4Internal",
        "store_rgba_u8_into",
        "store_rgba_u8_into_x4_v4",
    ),
    (
        "F32x4Internal",
        "store_rgb_u8_into",
        "store_rgb_u8_into_x4_v4",
    ),
    ("F32x8Backend", "to_u8_bytes", "to_u8_bytes_x8_v4"),
    ("F32x4Backend", "min", "min_x4_v4"),
    ("F32x4Backend", "max", "max_x4_v4"),
    ("F32x4Internal", "to_u16_lanes", "to_u16_lanes_x4_v4"),
    ("F32x4Internal", "to_u8_lanes", "to_u8_lanes_x4_v4"),
    ("F32x4Internal", "to_u32_lanes", "to_u32_lanes_x4_v4"),
    ("F32x4Internal", "from_u32_lanes", "from_u32_lanes_x4_v4"),
    ("F32x8Backend", "min", "min_x8_v4"),
    ("F32x8Backend", "max", "max_x8_v4"),
    ("F32x8Internal", "to_u16_lanes", "to_u16_lanes_x8_v4"),
    ("F32x8Internal", "to_u8_lanes", "to_u8_lanes_x8_v4"),
    ("F32x8Internal", "to_u32_lanes", "to_u32_lanes_x8_v4"),
    ("F32x8Internal", "from_u32_lanes", "from_u32_lanes_x8_v4"),
];

/// Trait methods intentionally left to their trait default body rather than
/// forwarded to `X64V3Token`. Each calls `Self::store_rgba_u8_into`, which
/// routes through `F32x4Internal` / `F32x8Internal` and preserves overrides.
const DEFAULT_BODY_EXCEPTIONS: &[(&str, &str)] = &[
    ("F32x4Backend", "store_rgba_bytes"),
    ("F32x8Backend", "store_rgba_bytes"),
];

const CHILD_PLACEHOLDER: &str = "__DELEGATE_CHILD__";
const V4_PLACEHOLDER: &str = "__DELEGATE_V4__";
const ARM_SPLIT_MARKER: &str = "// __DELEGATE_ARM_SPLIT__";

/// Parsed index of generated backend traits, parsed once per trait source file
/// and shared across all delegation outputs.
pub(crate) struct TraitIndex {
    traits: BTreeMap<String, syn::ItemTrait>,
}

impl TraitIndex {
    pub(crate) fn from_files<'a>(sources: impl IntoIterator<Item = &'a str>) -> Self {
        let mut traits = BTreeMap::new();
        for src in sources {
            let file = syn::parse_file(src).expect("generated backend trait source must parse");
            for item in file.items {
                if let Item::Trait(t) = item {
                    let name = t.ident.to_string();
                    let prev = traits.insert(name.clone(), t);
                    assert!(
                        prev.is_none(),
                        "duplicate trait definition `{name}` across generated backend sources"
                    );
                }
            }
        }
        Self { traits }
    }

    fn get(&self, name: &str) -> &syn::ItemTrait {
        self.traits
            .get(name)
            .unwrap_or_else(|| panic!("trait {name} not found in generated source"))
    }
}

/// Specification for one delegation `macro_rules!` definition.
struct DelegationSpec<'a> {
    macro_name: &'a str,
    parent: DelegationParent,
    trait_names: &'a [&'a str],
    overrides: &'a [(&'a str, &'a str, &'a str)],
    default_exceptions: &'a [(&'a str, &'a str)],
    with_v4_overrides: bool,
}

/// Generate `impls/x86_v4_f32_delegated.rs` from the pre-parsed trait index.
pub(crate) fn generate(traits: &TraitIndex) -> String {
    let trait_names = [
        "F32x4Backend",
        "F32x4Internal",
        "F32x8Backend",
        "F32x8Internal",
    ];
    let macro_and_invocations = traits.build_delegation_body(
        &DelegationSpec {
            macro_name: "delegate_v3_f32_backend",
            parent: V3_PARENT,
            trait_names: &trait_names,
            overrides: OVERRIDES,
            default_exceptions: DEFAULT_BODY_EXCEPTIONS,
            with_v4_overrides: true,
        },
        AVX512_TOKENS,
    );

    let code_body = formatdoc! {r#"
        use super::x86_v4_f32_overrides as overrides;
        use crate::simd::backends::{{F32x4Backend, F32x4Internal, F32x8Backend, F32x8Internal}};

        macro_rules! v4_parent {{
            ($token:ident, identity) => {{
                $token
            }};
            ($token:ident, v4) => {{
                $token.v4()
            }};
        }}

        {macro_and_invocations}
    "#};
    assert_source_audit_invariants(&code_body);

    formatdoc! {r#"
        //! W128 / W256 f32 backend delegation for `X64V4Token` / `X64V4xToken` /
        //! `Avx512Fp16Token`. Generated by xtask from the `F32x4Backend`,
        //! `F32x4Internal`, `F32x8Backend`, and `F32x8Internal` definitions;
        //! do not edit.
        //!
        //! ## Why this file exists
        //!
        //! The AVX-512 tokens advertise W512 support but have no implementations
        //! of their own for the narrower [`F32x4Backend`] / [`F32x8Backend`]
        //! traits. Without these, generic code parameterized on `T: F32x8Backend`
        //! cannot accept a V4 token, and the cross-width
        //! [`F32x16FromHalves`](crate::simd::generic::F32x16FromHalves) primitive
        //! cannot either, though V4's native `_mm512_insertf32x8` is its best path.
        //!
        //! ## Deductive safety proof
        //!
        //! `X64V3Token`'s feature set is a strict subset of `X64V4Token`'s (the
        //! registry adds AVX-512 F/BW/CD/DQ/VL, PCLMULQDQ and AES on top of V3), so
        //! every instruction that is safe to execute under V3's proof is safe under
        //! V4's. [`F32x4Backend`] / [`F32x8Backend`] for `X64V3Token` use only V3
        //! intrinsics, so forwarding V4's methods to them is sound; `Self::Repr`
        //! stays `__m128` / `__m256`, so callers see no representation change.
        //! Each inherited method forwards with the `.v3()` extractor, so the proof
        //! chain is explicit at every call. `X64V4xToken` and `Avx512Fp16Token` are
        //! V4 plus further AVX-512 sub-features, supersets of V3 by the same argument.
        //!
        //! ## AVX-512VL overrides and default-method exceptions
        //!
        //! Sixteen narrow methods call the hand-written AVX-512VL helpers in
        //! `x86_v4_f32_overrides.rs`, which take a concrete `X64V4Token`
        //! (`v4_parent!(self, identity)` for V4, `v4_parent!(self, v4)` for V4x
        //! and FP16):
        //! - `F32x4Backend`: `to_u8_bytes`, `min`, `max`
        //! - `F32x4Internal`: `store_rgba_u8_into`, `store_rgb_u8_into`,
        //!   `to_u16_lanes`, `to_u8_lanes`, `to_u32_lanes`, `from_u32_lanes`
        //! - `F32x8Backend`: `to_u8_bytes`, `min`, `max`
        //! - `F32x8Internal`: `to_u16_lanes`, `to_u8_lanes`, `to_u32_lanes`,
        //!   `from_u32_lanes`
        //!
        //! `F32x4Backend::store_rgba_bytes` and `F32x8Backend::store_rgba_bytes`
        //! are intentionally left to their trait default bodies, which dispatch to
        //! `Self::store_rgba_u8_into` so the selected packing backend is preserved.
        //!
        //! ## Concrete-impl macro auditability
        //!
        //! `delegate_v3_f32_backend!` is crate-private and expands only to plain
        //! concrete `impl Trait for archmage::$child` blocks — zero `unsafe`,
        //! zero raw pointers, zero target-feature boundaries, and no blanket impls.

        #![cfg(all(target_arch = "x86_64", target_feature = "sse2", feature = "avx512"))]

        {code_body}
    "#}
}

/// Generate `impls/x86_v3_delegated.rs`: pure-V3 W128/W256 delegation without
/// AVX-512VL overrides.
///
/// - `V3_ONLY_TOKENS` (`X64V3VnniToken`) receive concrete impls for all 34
///   traits (4 f32 + 30 f64/integer/convert/bitcast) unconditionally under
///   `all(target_arch = "x86_64", target_feature = "sse2")`, with no dependency
///   on `feature = "avx512"`.
/// - `AVX512_TOKENS` (`X64V4Token`, `X64V4xToken`, `Avx512Fp16Token`) receive
///   concrete impls for the 30 non-f32 traits under `#[cfg(feature = "avx512")]`
///   (their 4 f32 traits with AVX-512VL overrides live in
///   `x86_v4_f32_delegated.rs`).
///
/// All macro arms are derived from the generated `backends/*.rs` sources, so a
/// new method cannot fall back to its scalar default body (issue #60).
pub(crate) fn generate_v3_delegated(traits: &TraitIndex, narrow_names: &[&str]) -> String {
    let mut all_names: Vec<&str> = Vec::with_capacity(4 + narrow_names.len());
    all_names.extend([
        "F32x4Backend",
        "F32x4Internal",
        "F32x8Backend",
        "F32x8Internal",
    ]);
    all_names.extend_from_slice(narrow_names);

    let v3_only_tokens: Vec<DelegatedToken> = V3_ONLY_TOKENS
        .iter()
        .map(|&ident| DelegatedToken {
            ident,
            v4_mode: V4ProofMode::Identity,
        })
        .collect();

    let macro_def = traits.build_delegation_macro_arms(&DelegationSpec {
        macro_name: "delegate_v3_backend",
        parent: V3_PARENT,
        trait_names: &all_names,
        overrides: &[],
        default_exceptions: DEFAULT_BODY_EXCEPTIONS,
        with_v4_overrides: false,
    });

    let mut invocations = String::new();
    let v3_only_total = v3_only_tokens.len() * all_names.len();
    invocations.push_str(&format!(
        "// Pure-V3 extension tokens ({} token(s) × {} traits = {v3_only_total} impls, no `avx512` feature required).\n",
        v3_only_tokens.len(),
        all_names.len(),
    ));
    validate_tokens(&v3_only_tokens);
    for token in &v3_only_tokens {
        let ident = token.ident;
        invocations.push_str(&format!("\n// archmage::{ident}\n"));
        for &trait_name in &all_names {
            invocations.push_str(&format!("delegate_v3_backend!({trait_name} => {ident});\n"));
        }
    }

    let avx512_total = AVX512_TOKENS.len() * narrow_names.len();
    invocations.push_str(&format!(
        "\n// AVX-512 tokens' non-f32 narrow traits ({} tokens × {} traits = {avx512_total} impls;\n\
         // their 4 f32 traits with AVX-512VL overrides live in `x86_v4_f32_delegated.rs`).\n",
        AVX512_TOKENS.len(),
        narrow_names.len(),
    ));
    validate_tokens(AVX512_TOKENS);
    for token in AVX512_TOKENS {
        let ident = token.ident;
        invocations.push_str(&format!("\n// archmage::{ident}\n"));
        for &trait_name in narrow_names {
            invocations.push_str(&format!(
                "#[cfg(feature = \"avx512\")]\ndelegate_v3_backend!({trait_name} => {ident});\n"
            ));
        }
    }

    let code_body = formatdoc! {r#"
        // Wildcard: the impls' signatures name supertraits of the
        // implemented traits (`<Self as F32x8Backend>::Repr` inside a
        // `F32x8Convert` impl, `<Self as U8x16Backend>::Repr` inside an
        // `I8x16Bitcast` impl, …) — every sibling backend trait must be
        // in scope, not just the implemented set.
        use crate::simd::backends::*;

        {macro_def}
        {invocations}
    "#};
    assert_source_audit_invariants(&code_body);

    formatdoc! {r#"
        //! W128 / W256 backend delegation to `X64V3Token` for V3-superset x86-64
        //! tokens (`X64V3VnniToken`, and under `feature = "avx512"` the non-f32
        //! narrow traits of `X64V4Token` / `X64V4xToken` / `Avx512Fp16Token`).
        //! Generated by xtask from the backend trait definitions; do not edit.
        //!
        //! ## Why this file exists
        //!
        //! `X64V3VnniToken` extends `X64V3Token` without requiring AVX-512, so it
        //! needs concrete W128/W256 f32, f64, integer, conversion, and bitcast
        //! backend impls outside `feature = "avx512"`. The AVX-512 tokens share the
        //! same 30 non-f32 narrow arms under `#[cfg(feature = "avx512")]` (while
        //! their 4 f32 traits with AVX-512VL overrides live in
        //! `x86_v4_f32_delegated.rs`).
        //!
        //! ## Deductive safety proof
        //!
        //! `X64V3Token`'s feature set is a strict subset of every delegated token's
        //! feature set (`X64V3VnniToken` extends `X64V3GfniCryptoToken` ⊃ `X64V3Token`,
        //! and `X64V4Token` / `X64V4xToken` / `Avx512Fp16Token` extend `X64V3Token`),
        //! so every instruction the V3 impls execute is safe under the child token's
        //! proof. `Self::Repr` stays the V3 representation, and each method forwards
        //! explicitly via `self.v3()`.
        //!
        //! `F32x4Backend::store_rgba_bytes` and `F32x8Backend::store_rgba_bytes`
        //! are intentionally left to their trait default bodies, which dispatch to
        //! `Self::store_rgba_u8_into`.
        //!
        //! Popcnt extension backends are NOT delegated: they exist only
        //! for `X64V4xToken` (native AVX-512 VPOPCNTDQ) and have no V3
        //! impls to forward to.
        //!
        //! ## Concrete-impl macro auditability
        //!
        //! `delegate_v3_backend!` is crate-private and expands only to plain
        //! concrete `impl Trait for archmage::$child` blocks — zero `unsafe`,
        //! zero raw pointers, zero target-feature boundaries, and no blanket impls.

        #![cfg(all(target_arch = "x86_64", target_feature = "sse2"))]

        {code_body}
    "#}
}

/// Generate `impls/x86_v4_w512_delegated.rs`: W512 backend delegation to
/// `X64V4Token` via `self.v4()` for AVX-512 superset tokens (`X64V4xToken`).
///
/// Popcnt extension backends are NOT delegated: they exist natively on
/// `X64V4xToken` (VPOPCNTDQ + BITALG) in `x86_v4.rs`.
pub(crate) fn generate_v4_w512_delegated(traits: &TraitIndex, w512_names: &[&str]) -> String {
    let macro_and_invocations = traits.build_delegation_body(
        &DelegationSpec {
            macro_name: "delegate_v4_w512_backend",
            parent: V4_PARENT,
            trait_names: w512_names,
            overrides: &[],
            default_exceptions: &[],
            with_v4_overrides: false,
        },
        V4_W512_DELEGATED_TOKENS,
    );

    let code_body = formatdoc! {r#"
        // Wildcard: W512 widen/narrow and conversion traits name sibling backend
        // traits in their signatures (`<Self as F32x16Backend>::Repr`,
        // `<Self as I16x32Backend>::Repr`, …).
        use crate::simd::backends::*;

        {macro_and_invocations}
    "#};
    assert_source_audit_invariants(&code_body);

    formatdoc! {r#"
        //! W512 backend delegation to `X64V4Token` for AVX-512 superset tokens
        //! (`X64V4xToken`). Generated by xtask from the W512 backend trait
        //! definitions; do not edit.
        //!
        //! ## Why this file exists
        //!
        //! `X64V4xToken` adds VPOPCNTDQ / BITALG / VBMI2 extensions on top of
        //! `X64V4Token`, while all 12 base W512 backends (`F32x16Backend`,
        //! `F32x16Internal`, `F64x8Backend`, `I8x64..U64x8Backend`,
        //! `F32x16Convert`) execute identical base AVX-512 (F/BW/CD/DQ/VL)
        //! instructions. Delegating them to `X64V4Token` via `self.v4()` keeps
        //! the native AVX-512 intrinsic surface in `x86_v4.rs` single-sourced.
        //!
        //! ## Deductive safety proof
        //!
        //! `X64V4Token`'s feature set is a strict subset of `X64V4xToken`'s, so
        //! every instruction `X64V4Token`'s W512 backends execute is safe under
        //! `X64V4xToken`'s proof. `Self::Repr` stays the native `__m512*` type
        //! and every method forwards explicitly via `self.v4()`.
        //!
        //! Popcnt extension backends are NOT delegated: they live natively on
        //! `X64V4xToken` in `x86_v4.rs`.
        //!
        //! ## Concrete-impl macro auditability
        //!
        //! `delegate_v4_w512_backend!` is crate-private and expands only to plain
        //! concrete `impl Trait for archmage::$child` blocks — zero `unsafe`,
        //! zero raw pointers, zero target-feature boundaries, and no blanket impls.

        #![cfg(all(
            target_arch = "x86_64",
            target_feature = "sse2",
            feature = "avx512",
            feature = "w512"
        ))]

        {code_body}
    "#}
}

impl TraitIndex {
    fn build_delegation_macro_arms(&self, spec: &DelegationSpec<'_>) -> String {
        validate_specs_and_overrides(
            spec.trait_names,
            spec.overrides,
            spec.default_exceptions,
            spec.with_v4_overrides,
        );

        let mut used_overrides: BTreeSet<(&str, &str)> = BTreeSet::new();
        let mut used_exceptions: BTreeSet<(&str, &str)> = BTreeSet::new();
        let mut raw_impls: Vec<(&str, String)> = Vec::with_capacity(spec.trait_names.len());

        for &trait_name in spec.trait_names {
            let trait_item = self.get(trait_name);
            let raw_impl = delegating_impl_template(
                trait_name,
                spec.parent,
                trait_item,
                spec.overrides,
                spec.default_exceptions,
                &mut used_overrides,
                &mut used_exceptions,
            );
            raw_impls.push((trait_name, raw_impl));
        }

        // Completeness check: every configured override and default-body exception
        // must match an actual trait method.
        for &(ov_trait, ov_method, ov_helper) in spec.overrides {
            assert!(
                used_overrides.contains(&(ov_trait, ov_method)),
                "unmatched override ({ov_trait}::{ov_method} -> {ov_helper}): trait or method not found"
            );
        }
        for &(ex_trait, ex_method) in spec.default_exceptions {
            assert!(
                used_exceptions.contains(&(ex_trait, ex_method)),
                "unmatched default-body exception ({ex_trait}::{ex_method}): trait or method not found"
            );
        }

        let formatted_impls = format_impl_templates(&raw_impls);

        let mut out = String::new();
        out.push_str(&format!("macro_rules! {} {{\n", spec.macro_name));
        for ((trait_name, _), formatted) in raw_impls.iter().zip(formatted_impls.iter()) {
            let arm_body = formatted
                .replace(CHILD_PLACEHOLDER, "$child")
                .replace(V4_PLACEHOLDER, "$v4");
            if spec.with_v4_overrides {
                out.push_str(&format!(
                    "    ({trait_name} => $child:ident, v4 = $v4:ident) => {{\n"
                ));
            } else {
                out.push_str(&format!("    ({trait_name} => $child:ident) => {{\n"));
            }
            for line in arm_body.lines() {
                if line.is_empty() {
                    out.push('\n');
                } else {
                    out.push_str("        ");
                    out.push_str(line);
                    out.push('\n');
                }
            }
            out.push_str("    };\n");
        }
        out.push_str("}\n");
        out
    }

    fn build_delegation_body(
        &self,
        spec: &DelegationSpec<'_>,
        tokens: &[DelegatedToken],
    ) -> String {
        validate_tokens(tokens);
        let mut out = self.build_delegation_macro_arms(spec);
        out.push('\n');

        let macro_name = spec.macro_name;
        let total_impls = tokens.len() * spec.trait_names.len();
        let token_word = if tokens.len() == 1 { "token" } else { "tokens" };
        out.push_str(&format!(
            "// Concrete trait implementations ({} {token_word} × {} traits = {total_impls} impls).\n",
            tokens.len(),
            spec.trait_names.len(),
        ));
        for token in tokens {
            let ident = token.ident;
            if spec.with_v4_overrides {
                let tag = token.v4_mode.as_macro_tag();
                out.push_str(&format!("\n// archmage::{ident} (v4 = {tag})\n"));
                for &trait_name in spec.trait_names {
                    out.push_str(&format!(
                        "{macro_name}!({trait_name} => {ident}, v4 = {tag});\n"
                    ));
                }
            } else {
                out.push_str(&format!("\n// archmage::{ident}\n"));
                for &trait_name in spec.trait_names {
                    out.push_str(&format!("{macro_name}!({trait_name} => {ident});\n"));
                }
            }
        }

        assert_source_audit_invariants(&out);
        out
    }
}

fn validate_tokens(tokens: &[DelegatedToken]) {
    assert!(
        !tokens.is_empty(),
        "delegation token list must not be empty"
    );
    let mut seen_tokens = BTreeSet::new();
    for token in tokens {
        assert!(
            !token.ident.is_empty()
                && token
                    .ident
                    .chars()
                    .all(|c| c.is_ascii_alphanumeric() || c == '_'),
            "invalid token identifier `{}` in delegation list",
            token.ident
        );
        assert!(
            seen_tokens.insert(token.ident),
            "duplicate token `{}` in delegation token list",
            token.ident
        );
    }
}

fn validate_specs_and_overrides(
    trait_names: &[&str],
    overrides: &[(&str, &str, &str)],
    default_exceptions: &[(&str, &str)],
    with_v4_overrides: bool,
) {
    assert!(
        !trait_names.is_empty(),
        "delegation trait spec list must not be empty"
    );
    if !with_v4_overrides {
        assert!(
            overrides.is_empty(),
            "overrides supplied for non-override delegation macro"
        );
    }

    let mut seen_traits = BTreeSet::new();
    for &trait_name in trait_names {
        assert!(
            !trait_name.is_empty(),
            "delegation trait name must not be empty"
        );
        assert!(
            seen_traits.insert(trait_name),
            "duplicate trait `{trait_name}` in delegation spec list"
        );
    }

    let mut seen_overrides = BTreeSet::new();
    for &(ov_trait, ov_method, ov_helper) in overrides {
        assert!(
            !ov_helper.is_empty(),
            "empty override helper for {ov_trait}::{ov_method}"
        );
        assert!(
            seen_overrides.insert((ov_trait, ov_method)),
            "duplicate override for {ov_trait}::{ov_method}"
        );
    }

    let mut seen_exceptions = BTreeSet::new();
    for &(ex_trait, ex_method) in default_exceptions {
        assert!(
            seen_exceptions.insert((ex_trait, ex_method)),
            "duplicate default-body exception for {ex_trait}::{ex_method}"
        );
        assert!(
            !seen_overrides.contains(&(ex_trait, ex_method)),
            "{ex_trait}::{ex_method} cannot be both an override and a default-body exception"
        );
    }
}

/// Mechanical safety/auditability guard: lexically scan emitted code (skipping
/// comments and string literals) so checks are independent of whitespace or
/// formatting. Rejects forbidden identifiers (`unsafe`, `target_feature`,
/// `arcane`, `attune`, `rite`, `from_context`, `summon`, `macro_export`) and
/// raw pointer syntax (`*const` / `*mut` with any intervening whitespace).
fn assert_source_audit_invariants(source: &str) {
    const FORBIDDEN_IDENTS: &[&str] = &[
        "unsafe",
        "target_feature",
        "arcane",
        "attune",
        "rite",
        "from_context",
        "summon",
        "macro_export",
    ];

    let bytes = source.as_bytes();
    let len = bytes.len();
    let mut i = 0;
    let mut prev_star = false;

    while i < len {
        let b = bytes[i];

        // Skip whitespace without resetting `prev_star`, so `* const` / `* mut`
        // are caught identically to `*const` / `*mut`.
        if b.is_ascii_whitespace() {
            i += 1;
            continue;
        }

        // Skip `//` line comments.
        if b == b'/' && i + 1 < len && bytes[i + 1] == b'/' {
            i += 2;
            while i < len && bytes[i] != b'\n' {
                i += 1;
            }
            continue;
        }

        // Skip `/* ... */` block comments (supporting nesting).
        if b == b'/' && i + 1 < len && bytes[i + 1] == b'*' {
            i += 2;
            let mut depth = 1usize;
            while i + 1 < len && depth > 0 {
                if bytes[i] == b'/' && bytes[i + 1] == b'*' {
                    depth += 1;
                    i += 2;
                } else if bytes[i] == b'*' && bytes[i + 1] == b'/' {
                    depth -= 1;
                    i += 2;
                } else {
                    i += 1;
                }
            }
            continue;
        }

        // Skip double-quoted string literals (used in `#![cfg(...)]`).
        if b == b'"' {
            prev_star = false;
            i += 1;
            while i < len {
                if bytes[i] == b'\\' {
                    i += 2;
                    continue;
                }
                if bytes[i] == b'"' {
                    i += 1;
                    break;
                }
                i += 1;
            }
            continue;
        }

        // Track `*` for raw-pointer detection (`*const` / `*mut`).
        if b == b'*' {
            prev_star = true;
            i += 1;
            continue;
        }

        // Scan whole Rust identifiers (including raw `r#ident`).
        if b.is_ascii_alphabetic() || b == b'_' {
            let start = i;
            i += 1;
            while i < len && (bytes[i].is_ascii_alphanumeric() || bytes[i] == b'_') {
                i += 1;
            }
            let ident = &source[start..i];
            if prev_star && matches!(ident, "const" | "mut") {
                panic!("delegation source contains forbidden raw pointer `*{ident}`");
            }
            prev_star = false;
            if FORBIDDEN_IDENTS.contains(&ident) {
                panic!("delegation source contains forbidden identifier `{ident}`");
            }
            continue;
        }

        prev_star = false;
        i += 1;
    }
}

fn delegating_impl_template<'a>(
    trait_name: &'a str,
    parent: DelegationParent,
    trait_item: &syn::ItemTrait,
    overrides: &[(&'a str, &'a str, &'a str)],
    default_exceptions: &[(&'a str, &'a str)],
    used_overrides: &mut BTreeSet<(&'a str, &'a str)>,
    used_exceptions: &mut BTreeSet<(&'a str, &'a str)>,
) -> String {
    let trait_ident = format_ident!("{trait_name}");
    let parent_token = format_ident!("{}", parent.token);
    let parent_method = format_ident!("{}", parent.method);
    let child_placeholder = format_ident!("{CHILD_PLACEHOLDER}");
    let v4_placeholder = format_ident!("{V4_PLACEHOLDER}");
    let mut methods = Vec::new();
    let mut seen_methods = BTreeSet::new();

    for item in &trait_item.items {
        let TraitItem::Fn(method) = item else {
            continue;
        };
        // `convert_int.rs` qualifies supertraits with `super::` (its own
        // module is `backends`); copied into `impls/` that path would
        // resolve to `crate::simd::impls` — retarget to the trait home.
        let mut sig: syn::Signature = syn::parse_str(
            &method
                .sig
                .to_token_stream()
                .to_string()
                .replace("super ::", "crate :: simd :: backends ::"),
        )
        .expect("method signature re-parse");
        let name = sig.ident.clone();
        assert!(
            seen_methods.insert(name.to_string()),
            "duplicate method `{name}` in trait {trait_name}"
        );
        if let Some(&(ex_trait, ex_method)) = default_exceptions
            .iter()
            .find(|&&(t, m)| t == trait_name && name == m)
        {
            assert!(
                method.default.is_some(),
                "default-body exception {ex_trait}::{ex_method} has no default body on the trait"
            );
            used_exceptions.insert((ex_trait, ex_method));
            continue;
        }
        assert!(
            matches!(sig.inputs.first(), Some(FnArg::Receiver(_))),
            "{trait_name}::{name} has no receiver; the delegation needs `self`"
        );
        let args: Vec<syn::Ident> = sig
            .inputs
            .iter_mut()
            .skip(1)
            .enumerate()
            .map(|(idx, arg)| match arg {
                FnArg::Typed(pat) => match pat.pat.as_mut() {
                    Pat::Ident(p) => {
                        p.mutability = None;
                        p.ident.clone()
                    }
                    other => {
                        // Destructured trait parameters (e.g. `[a, b, c]: [T; 3]` on
                        // `F32x16Internal::permute_rgb_wide`) forward verbatim under a
                        // synthetic binding.
                        let ident = format_ident!("__arg{idx}");
                        *other = syn::parse_quote!(#ident);
                        ident
                    }
                },
                FnArg::Receiver(_) => unreachable!("receiver after the first parameter"),
            })
            .collect();
        let sig = &sig;
        let turbofish = {
            let params: Vec<_> = sig
                .generics
                .params
                .iter()
                .map(|p| match p {
                    syn::GenericParam::Type(t) => t.ident.to_token_stream(),
                    syn::GenericParam::Const(c) => c.ident.to_token_stream(),
                    syn::GenericParam::Lifetime(l) => l.lifetime.to_token_stream(),
                })
                .collect();
            if params.is_empty() {
                quote! {}
            } else {
                quote! { ::<#(#params),*> }
            }
        };
        let override_match = overrides
            .iter()
            .find(|&&(t, m, _)| t == trait_name && name == m);
        let body = match override_match {
            Some(&(ov_trait, ov_method, helper)) => {
                used_overrides.insert((ov_trait, ov_method));
                let helper_ident = format_ident!("{helper}");
                quote! { overrides::#helper_ident #turbofish(v4_parent!(self, #v4_placeholder), #(#args),*) }
            }
            None => {
                quote! { <archmage::#parent_token as #trait_ident>::#name #turbofish(self.#parent_method(), #(#args),*) }
            }
        };
        methods.push(quote! {
            #[inline(always)]
            #sig { #body }
        });
    }

    // `type Repr` exists on the backend traits but not on the
    // float↔int conversion traits (whose methods name supertrait Reprs
    // in signature position) — emit it only when the trait declares it.
    let assoc_ty = trait_item.items.iter().find_map(|item| match item {
        TraitItem::Type(ty) => Some(&ty.ident),
        _ => None,
    });
    let repr_impl = assoc_ty.map(|assoc| {
        quote! { type #assoc = <archmage::#parent_token as #trait_ident>::#assoc; }
    });
    let tokens = quote! {
        impl #trait_ident for archmage::#child_placeholder {
            #repr_impl
            #(#methods)*
        }
    };
    tokens.to_string()
}

/// Format all generated `impl` templates in a single `rustfmt` invocation so
/// macro arm bodies are canonically indented and easy to audit. Fails loudly
/// with `rustfmt` stderr rather than silently falling back to unformatted
/// single-line token strings.
fn format_impl_templates(raw_impls: &[(&str, String)]) -> Vec<String> {
    let mut joined = String::new();
    for (idx, (_, src)) in raw_impls.iter().enumerate() {
        if idx > 0 {
            joined.push('\n');
            joined.push_str(ARM_SPLIT_MARKER);
            joined.push('\n');
        }
        joined.push_str(src);
        joined.push('\n');
    }

    let formatted = run_rustfmt(&joined);
    let parts: Vec<String> = formatted
        .split(ARM_SPLIT_MARKER)
        .map(|part| part.trim().to_string())
        .collect();
    assert_eq!(
        parts.len(),
        raw_impls.len(),
        "rustfmt split marker count mismatch (expected {}, got {})",
        raw_impls.len(),
        parts.len()
    );
    parts
}

fn run_rustfmt(input: &str) -> String {
    let mut child = Command::new("rustfmt")
        .arg("--edition")
        .arg("2024")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("failed to spawn `rustfmt --edition 2024` for delegation templates");
    if let Some(mut stdin) = child.stdin.take() {
        stdin
            .write_all(input.as_bytes())
            .expect("failed to write delegation templates to rustfmt stdin");
    }
    let output = child
        .wait_with_output()
        .expect("failed to wait on rustfmt for delegation templates");
    assert!(
        output.status.success(),
        "rustfmt failed on delegation templates (status {}):\n{}",
        output.status,
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8(output.stdout).expect("rustfmt stdout must be valid UTF-8")
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE_TRAIT: &str = r#"
        pub trait DemoBackend {
            type Repr;
            fn splat(self, v: f32) -> Self::Repr;
            fn min(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
            fn write(self, a: Self::Repr, out: &mut [f32; 4]);
            fn default_helper(self, a: Self::Repr) -> Self::Repr {
                a
            }
        }
    "#;

    const SAMPLE_TOKENS: &[DelegatedToken] = &[
        DelegatedToken {
            ident: "X64V4Token",
            v4_mode: V4ProofMode::Identity,
        },
        DelegatedToken {
            ident: "X64V4xToken",
            v4_mode: V4ProofMode::ViaV4,
        },
    ];

    fn sample_index() -> TraitIndex {
        TraitIndex::from_files([SAMPLE_TRAIT])
    }

    #[test]
    fn generated_macros_emit_exact_concrete_invocations_and_pass_audit() {
        let index = sample_index();
        let with_ov = index.build_delegation_body(
            &DelegationSpec {
                macro_name: "delegate_demo_f32",
                parent: V3_PARENT,
                trait_names: &["DemoBackend"],
                overrides: &[("DemoBackend", "min", "min_demo_v4")],
                default_exceptions: &[("DemoBackend", "default_helper")],
                with_v4_overrides: true,
            },
            SAMPLE_TOKENS,
        );
        assert!(with_ov.contains("macro_rules! delegate_demo_f32 {"));
        assert!(with_ov.contains("impl DemoBackend for archmage::$child {"));
        assert!(with_ov.contains("fn write("));
        assert!(!with_ov.contains("fn default_helper("));
        assert!(with_ov.contains("overrides::min_demo_v4(v4_parent!(self, $v4), a, b)"));
        assert!(with_ov.contains("delegate_demo_f32!(DemoBackend => X64V4Token, v4 = identity);"));
        assert!(with_ov.contains("delegate_demo_f32!(DemoBackend => X64V4xToken, v4 = v4);"));

        let without_ov = index.build_delegation_body(
            &DelegationSpec {
                macro_name: "delegate_demo_narrow",
                parent: V3_PARENT,
                trait_names: &["DemoBackend"],
                overrides: &[],
                default_exceptions: &[],
                with_v4_overrides: false,
            },
            SAMPLE_TOKENS,
        );
        assert!(without_ov.contains("macro_rules! delegate_demo_narrow {"));
        assert!(without_ov.contains("(DemoBackend => $child:ident) => {"));
        assert!(without_ov.contains("fn default_helper("));
        assert!(without_ov.contains("delegate_demo_narrow!(DemoBackend => X64V4Token);"));
        assert!(without_ov.contains("delegate_demo_narrow!(DemoBackend => X64V4xToken);"));

        let v4_parent = index.build_delegation_body(
            &DelegationSpec {
                macro_name: "delegate_demo_v4",
                parent: V4_PARENT,
                trait_names: &["DemoBackend"],
                overrides: &[],
                default_exceptions: &[],
                with_v4_overrides: false,
            },
            V4_W512_DELEGATED_TOKENS,
        );
        assert!(v4_parent.contains("macro_rules! delegate_demo_v4 {"));
        assert!(v4_parent.contains("<archmage::X64V4Token as DemoBackend>::Repr"));
        assert!(v4_parent.contains("<archmage::X64V4Token as DemoBackend>::splat(self.v4(), v)"));
        assert!(v4_parent.contains("delegate_demo_v4!(DemoBackend => X64V4xToken);"));
    }

    #[test]
    #[should_panic(expected = "duplicate override")]
    fn duplicate_override_panics() {
        let index = sample_index();
        let _ = index.build_delegation_body(
            &DelegationSpec {
                macro_name: "delegate_demo",
                parent: V3_PARENT,
                trait_names: &["DemoBackend"],
                overrides: &[
                    ("DemoBackend", "min", "min_a"),
                    ("DemoBackend", "min", "min_b"),
                ],
                default_exceptions: &[],
                with_v4_overrides: true,
            },
            SAMPLE_TOKENS,
        );
    }

    #[test]
    #[should_panic(expected = "unmatched override")]
    fn unknown_override_panics() {
        let index = sample_index();
        let _ = index.build_delegation_body(
            &DelegationSpec {
                macro_name: "delegate_demo",
                parent: V3_PARENT,
                trait_names: &["DemoBackend"],
                overrides: &[("DemoBackend", "nonexistent_method", "helper_v4")],
                default_exceptions: &[],
                with_v4_overrides: true,
            },
            SAMPLE_TOKENS,
        );
    }

    #[test]
    #[should_panic(expected = "has no default body")]
    fn default_exception_without_default_body_panics() {
        let index = sample_index();
        let _ = index.build_delegation_body(
            &DelegationSpec {
                macro_name: "delegate_demo",
                parent: V3_PARENT,
                trait_names: &["DemoBackend"],
                overrides: &[],
                default_exceptions: &[("DemoBackend", "splat")],
                with_v4_overrides: false,
            },
            SAMPLE_TOKENS,
        );
    }

    #[test]
    #[should_panic(expected = "unmatched default-body exception")]
    fn unknown_default_exception_panics() {
        let index = sample_index();
        let _ = index.build_delegation_body(
            &DelegationSpec {
                macro_name: "delegate_demo",
                parent: V3_PARENT,
                trait_names: &["DemoBackend"],
                overrides: &[],
                default_exceptions: &[("DemoBackend", "no_such_default")],
                with_v4_overrides: false,
            },
            SAMPLE_TOKENS,
        );
    }

    #[test]
    #[should_panic(expected = "duplicate trait")]
    fn duplicate_trait_spec_panics() {
        let index = sample_index();
        let _ = index.build_delegation_body(
            &DelegationSpec {
                macro_name: "delegate_demo",
                parent: V3_PARENT,
                trait_names: &["DemoBackend", "DemoBackend"],
                overrides: &[],
                default_exceptions: &[],
                with_v4_overrides: false,
            },
            SAMPLE_TOKENS,
        );
    }

    #[test]
    #[should_panic(expected = "duplicate token")]
    fn duplicate_token_panics() {
        let index = sample_index();
        let dup = &[
            DelegatedToken {
                ident: "X64V4Token",
                v4_mode: V4ProofMode::Identity,
            },
            DelegatedToken {
                ident: "X64V4Token",
                v4_mode: V4ProofMode::ViaV4,
            },
        ];
        let _ = index.build_delegation_body(
            &DelegationSpec {
                macro_name: "delegate_demo",
                parent: V3_PARENT,
                trait_names: &["DemoBackend"],
                overrides: &[],
                default_exceptions: &[],
                with_v4_overrides: false,
            },
            dup,
        );
    }

    #[test]
    fn audit_invariants_allow_write_and_comments() {
        assert_source_audit_invariants(
            "// zero `unsafe`, zero *const, zero target_feature\n\
             fn write(x: f32) -> f32 { x * x }\n",
        );
    }

    #[test]
    #[should_panic(expected = "forbidden identifier `unsafe`")]
    fn audit_invariants_reject_unsafe() {
        assert_source_audit_invariants("macro_rules! bad { () => { unsafe { } }; }");
    }

    #[test]
    #[should_panic(expected = "forbidden identifier `target_feature`")]
    fn audit_invariants_reject_target_feature() {
        assert_source_audit_invariants(
            "macro_rules! bad { () => { #[target_feature(enable = \"avx2\")] fn f() {} }; }",
        );
    }

    #[test]
    #[should_panic(expected = "forbidden raw pointer `*const`")]
    fn audit_invariants_reject_spaced_raw_const_ptr() {
        assert_source_audit_invariants("fn bad(p: * const u8) {}");
    }

    #[test]
    #[should_panic(expected = "forbidden raw pointer `*mut`")]
    fn audit_invariants_reject_spaced_raw_mut_ptr() {
        assert_source_audit_invariants("fn bad(p: *   mut u8) {}");
    }
}
