//! Structural and unsafe-surface soundness verification.
//!
//! # Division of labor with `rustc`
//!
//! Value-intrinsic target-feature gating is enforced directly by `rustc` 1.85+
//! (`target_feature_11`): `magetypes/src/lib.rs` sets `#![deny(unsafe_code)]`
//! everywhere except `mod simd_storage;`, and every backend method executes
//! inside `#[arcane]` or `#[attune]`, so calling any value intrinsic whose
//! required feature is absent from the enclosing `#[target_feature]` is a hard
//! compile error across all CI targets (`x86_64`, `aarch64`, `wasm32`).
//!
//! This verifier enforces the invariants `rustc` does **not** prove on its own:
//!
//! - **STRUCTURAL RULES** (`structural_rules`): token fabrication routes
//!   (`MaybeUninit`, `mem::zeroed`, `forge_token_dangerously`); outside
//!   `simd_storage.rs`, the `unsafe` keyword in any form, bare `transmute`,
//!   gather/scatter intrinsics, or `unsafe impl Pod`; any `allow(unsafe_code)`
//!   other than `mod simd_storage;` in `magetypes/src/lib.rs`, or a missing
//!   crate-wide `#![deny(unsafe_code)]`; hidden CPU detection outside the
//!   audited whole-slice/scalar selectors; `impl_token_storage!` structs
//!   without `#[repr(C)]`; tokenless wrapper constructors (`Default`/serde/
//!   bytemuck); and backend-trait methods without a `self` receiver.
//! - **STORAGE GATHER/SCATTER AST GUARD** (`verify_storage_gather_scatter`):
//!   the 6 AVX-512 gather/scatter intrinsics in `magetypes/src/simd_storage.rs`
//!   are `unsafe fn` pointer ops called inside `unsafe { ... }` (where `rustc`
//!   does not reject a missing target feature). This AST check verifies every
//!   gather/scatter call in `simd_storage.rs` is in the audited `avx512f` set
//!   and sits inside an `#[arcane]`/`#[attune(wrap)]` fn taking `X64V4Token`.
//! - **`from_context()` AST GUARD** (`raw_context::verify`): verifies every
//!   `Token::from_context()` call in `magetypes/src/` sits inside a safe fn
//!   with matching or superset `#[target_feature]`/`#[attune]`/`#[rite]`.
//! - **SAFETY-COMMENT DISCIPLINE** (`safety_comment_rules`): every `unsafe {`
//!   block outside generated backend impls must carry an adjacent `// SAFETY:`
//!   comment within the 4 preceding lines.
//! - **VACUOUS-PASS GUARDS**: asserts at least [`MIN_SCANNED_FILES`] `.rs`
//!   files were scanned and all [`MIN_STORAGE_GATHER_SCATTER`] audited
//!   gather/scatter calls in `simd_storage.rs` were seen.
//! - **SNAPSHOT PORTABILITY** (`check_stderr_snapshot_portability`): ensures
//!   trybuild `.stderr` snapshots contain no host- or toolchain-specific text.
//!
//! Token `summon()` ↔ `token-registry.toml` feature consistency is verified
//! separately by `cmd::validate::validate_summon`.

use anyhow::{Context, Result, bail};
use regex::Regex;
use std::collections::HashSet;
use std::fs;
use std::path::{Path, PathBuf};
use syn::visit::{self, Visit};

use crate::registry::Registry;

mod raw_context;

/// Directories scanned for structural and unsafe-surface soundness rules.
const SCAN_ROOTS: &[&str] = &["src", "magetypes/src", "tests/expand"];

/// Minimum number of `.rs` files expected under [`SCAN_ROOTS`] (vacuous-pass guard).
const MIN_SCANNED_FILES: usize = 100;

/// Minimum audited AVX-512 gather/scatter calls in [`STORAGE_HOME`] (vacuous-pass guard).
const MIN_STORAGE_GATHER_SCATTER: usize = 6;

/// The audited `avx512f`-only gather/scatter intrinsics permitted in [`STORAGE_HOME`].
const AUDITED_STORAGE_GATHER_SCATTER: &[&str] = &[
    "_mm512_i32gather_epi32",
    "_mm512_i32gather_ps",
    "_mm512_mask_i32gather_epi32",
    "_mm512_mask_i32gather_ps",
    "_mm512_mask_i32scatter_epi32",
    "_mm512_mask_i32scatter_ps",
];

/// The one magetypes file allowed `unsafe` of any kind, `unsafe impl Pod`, and
/// gather/scatter intrinsics. Each `unsafe` there states its invariant next to
/// the others, so auditing magetypes' unsafe means reading one file. The crate
/// root denies `unsafe_code` everywhere else.
const STORAGE_HOME: &str = "magetypes/src/simd_storage.rs";

/// The magetypes crate root, which must deny `unsafe_code`.
const LIB_HOME: &str = "magetypes/src/lib.rs";

/// x86-64 ABI baseline — implicitly enabled on every x86-64 target, used when
/// auditing compile-time `#[cfg]` gates in `scalar_selector_summons`.
const X86_64_BASELINE: &[&str] = &["sse", "sse2"];

/// Result of checking one source file.
#[derive(Default)]
pub struct FileScan {
    pub errors: Vec<String>,
    pub storage_gather_scatter: usize,
}

/// Run all structural, AST, and safety-comment checks on one source file.
pub fn scan_source(reg: &Registry, rel: &str, raw: &str) -> FileScan {
    let text = strip_comments(raw);
    let mut out = FileScan::default();
    structural_rules(reg, rel, &text, &mut out.errors);
    if rel == STORAGE_HOME {
        out.storage_gather_scatter = verify_storage_gather_scatter(rel, raw, &mut out.errors);
    }
    raw_context::verify(reg, rel, &text, &mut out.errors);
    safety_comment_rules(rel, raw, &mut out.errors);
    out
}

/// AST-verify the `unsafe` AVX-512 gather/scatter intrinsic calls in `simd_storage.rs`.
///
/// Inside `unsafe { ... }`, `rustc` allows calling an `unsafe fn` intrinsic even
/// if the enclosing `#[target_feature]` is missing its required feature. Because
/// `structural_rules` forbids gather/scatter everywhere outside `simd_storage.rs`,
/// checking every call here closes the only intrinsic-gating gap left by `rustc`.
fn verify_storage_gather_scatter(rel: &str, raw: &str, errors: &mut Vec<String>) -> usize {
    let file = match syn::parse_file(raw) {
        Ok(file) => file,
        Err(err) => {
            errors.push(format!(
                "{rel}: cannot parse for gather/scatter audit: {err}"
            ));
            return 0;
        }
    };
    let mut visitor = GatherScatterVisitor {
        rel,
        errors,
        fn_name: String::new(),
        has_v4_boundary: false,
        verified: 0,
    };
    visitor.visit_file(&file);
    visitor.verified
}

struct GatherScatterVisitor<'a> {
    rel: &'a str,
    errors: &'a mut Vec<String>,
    fn_name: String,
    has_v4_boundary: bool,
    verified: usize,
}

impl GatherScatterVisitor<'_> {
    fn attr_is_boundary(attr: &syn::Attribute) -> bool {
        let Some(last) = attr.path().segments.last() else {
            return false;
        };
        if last.ident == "arcane" {
            return true;
        }
        if last.ident == "attune" {
            let mut is_wrap = false;
            let _ = attr.parse_nested_meta(|meta| {
                if meta.path.is_ident("wrap") {
                    is_wrap = true;
                }
                Ok(())
            });
            return is_wrap;
        }
        false
    }

    fn sig_takes_v4_token(sig: &syn::Signature) -> bool {
        sig.inputs.iter().any(|arg| {
            let syn::FnArg::Typed(pat_ty) = arg else {
                return false;
            };
            let syn::Type::Path(tp) = pat_ty.ty.as_ref() else {
                return false;
            };
            tp.path
                .segments
                .last()
                .is_some_and(|s| s.ident == "X64V4Token")
        })
    }

    fn check_fn(&mut self, attrs: &[syn::Attribute], sig: &syn::Signature, block: &syn::Block) {
        let prev_name = std::mem::replace(&mut self.fn_name, sig.ident.to_string());
        let prev_v4 = self.has_v4_boundary;
        self.has_v4_boundary =
            attrs.iter().any(Self::attr_is_boundary) && Self::sig_takes_v4_token(sig);
        self.visit_block(block);
        self.fn_name = prev_name;
        self.has_v4_boundary = prev_v4;
    }
}

impl<'ast> Visit<'ast> for GatherScatterVisitor<'_> {
    fn visit_item_fn(&mut self, item: &'ast syn::ItemFn) {
        self.check_fn(&item.attrs, &item.sig, &item.block);
    }

    fn visit_impl_item_fn(&mut self, item: &'ast syn::ImplItemFn) {
        self.check_fn(&item.attrs, &item.sig, &item.block);
    }

    fn visit_expr_call(&mut self, call: &'ast syn::ExprCall) {
        if let syn::Expr::Path(path) = call.func.as_ref()
            && let Some(seg) = path.path.segments.last()
        {
            let name = seg.ident.to_string();
            let is_gather_scatter =
                name.starts_with("_mm") && (name.contains("gather_") || name.contains("scatter_"));
            if is_gather_scatter {
                if !AUDITED_STORAGE_GATHER_SCATTER.contains(&name.as_str()) {
                    self.errors.push(format!(
                        "{}: STORAGE GATHER/SCATTER: `{}` in `{}` is not in the audited \
                         avx512f set ({:?})",
                        self.rel, name, self.fn_name, AUDITED_STORAGE_GATHER_SCATTER
                    ));
                } else if !self.has_v4_boundary {
                    self.errors.push(format!(
                        "{}: STORAGE GATHER/SCATTER: `{}` in `{}` must sit inside an \
                         `#[arcane]` or `#[attune(wrap)]` fn taking `X64V4Token`",
                        self.rel, name, self.fn_name
                    ));
                } else {
                    self.verified += 1;
                }
            }
        }
        visit::visit_expr_call(self, call);
    }
}

/// SAFETY-comment discipline, checked on the RAW source (comments intact).
///
/// - In handwritten/generated crate sources: every `unsafe {` block must
///   have a `SAFETY`-bearing comment within the four preceding lines (or
///   on the same line). In archmage `src/` these are the token
///   constructors' detection sites; in magetypes they are the storage
///   helpers in `simd_storage.rs`.
/// - The generated backend impls (`magetypes/src/simd/impls/*.rs`) are
///   checked for a file-header audit contract (emitted by the generator)
///   instead of per-block comments, wherever they contain `unsafe`. They
///   contain none today: the structural rules reject `unsafe` there.
/// - Macro-expansion snapshots (`.expanded.rs`) are exempt: comments
///   cannot survive tokenization, so proc-macro output structurally cannot
///   carry them. Snapshots still get full intrinsic-gating verification;
///   the macros' emitted `unsafe` is documented at its source in
///   `archmage-macros/src/`.
fn safety_comment_rules(rel: &str, raw: &str, errors: &mut Vec<String>) {
    let in_scope = rel.starts_with("src") || rel.starts_with("magetypes/src");
    if !in_scope {
        return;
    }

    let is_backend_impl = rel.starts_with("magetypes/src/simd/impls/");
    let lines: Vec<&str> = raw.lines().collect();

    if is_backend_impl {
        let has_unsafe = lines
            .iter()
            .any(|l| !l.trim_start().starts_with("//") && l.contains("unsafe"));
        if has_unsafe && !raw.contains("# Safety (audit contract") {
            errors.push(format!(
                "{}: missing the `# Safety (audit contract ...)` file header — \
                 generated backend impls must state the uniform justification \
                 for their unsafe blocks (emitted by impls_safety_contract() in \
                 xtask/src/simd_types/backend_gen.rs).",
                rel
            ));
        }
        return;
    }

    for (i, line) in lines.iter().enumerate() {
        let trimmed = line.trim_start();
        if trimmed.starts_with("//") {
            continue;
        }
        if !(line.contains("unsafe {") || line.contains("unsafe{")) {
            continue;
        }
        let window_start = i.saturating_sub(4);
        let window = &lines[window_start..=i];
        if !window.iter().any(|l| l.contains("SAFETY")) {
            errors.push(format!(
                "{}:{}: `unsafe` block without an adjacent `// SAFETY:` comment — \
                 every unsafe block outside the generated backend impls must state \
                 its invariant (within the 4 preceding lines).",
                rel,
                i + 1
            ));
        }
    }
}

/// Audit the specific summon call in each scalar selector against its module's cfg.
fn scalar_selector_summons(reg: &Registry, rel: &str, text: &str) -> HashSet<usize> {
    use syn::{Expr, Item, Meta, Stmt};

    let Ok(file) = syn::parse_file(text) else {
        return HashSet::new();
    };
    let mut permitted = HashSet::new();
    for item in file.items {
        let Item::Mod(module) = item else { continue };
        if module.ident != "selected" {
            continue;
        }
        let Some((_, items)) = module.content else {
            continue;
        };
        for item in items {
            let Item::Fn(function) = item else { continue };
            if function.sig.ident != "token" {
                continue;
            }
            let [Stmt::Expr(Expr::Call(call), None)] = function.block.stmts.as_slice() else {
                continue;
            };
            let Expr::Path(callee) = call.func.as_ref() else {
                continue;
            };
            let segments: Vec<_> = callee.path.segments.iter().collect();
            let [krate, token, summon] = segments.as_slice() else {
                continue;
            };
            if callee.qself.is_some()
                || callee.path.leading_colon.is_none()
                || krate.ident != "archmage"
                || summon.ident != "summon"
                || !call.args.is_empty()
                || segments.iter().any(|s| !s.arguments.is_empty())
            {
                continue;
            }
            let token_name = token.ident.to_string();
            let arch = match token_name.as_str() {
                "NeonToken" => "aarch64",
                "Wasm128Token" => "wasm32",
                "X64V2Token" if rel == "magetypes/src/scalar_round.rs" => "x86_64",
                _ => continue,
            };
            let Some(def) = reg.find_token(&token_name) else {
                continue;
            };
            // Only positive conjuncts provide proof. In particular, features
            // mentioned in any(...) or not(...) cannot authorize a summon.
            let cfg = module
                .attrs
                .iter()
                .filter(|attr| attr.path().is_ident("cfg"))
                .filter_map(|attr| match attr.parse_args::<Meta>().ok()? {
                    Meta::List(list) if list.path.is_ident("all") => Some(list),
                    _ => None,
                })
                .filter_map(|list| {
                    list.parse_args_with(
                        syn::punctuated::Punctuated::<Meta, syn::Token![,]>::parse_terminated,
                    )
                    .ok()
                })
                .flatten()
                .filter_map(|meta| {
                    let Meta::NameValue(pair) = meta else {
                        return None;
                    };
                    let Expr::Lit(value) = pair.value else {
                        return None;
                    };
                    let syn::Lit::Str(value) = value.lit else {
                        return None;
                    };
                    Some((pair.path.get_ident()?.to_string(), value.value()))
                })
                .collect::<Vec<_>>();
            let has_cfg = |key, value| cfg.iter().any(|(k, v)| k == key && v == value);
            if has_cfg("target_arch", arch)
                && def.features.iter().all(|feature| {
                    (arch == "x86_64" && X86_64_BASELINE.contains(&feature.as_str()))
                        || has_cfg("target_feature", feature.as_str())
                })
            {
                permitted.insert(summon.ident.span().byte_range().start);
            }
        }
    }
    permitted
}

/// Structural soundness rules for magetypes sources, beyond intrinsic
/// gating. Each rule fences off a way the token-as-proof model could be
/// bypassed without any intrinsic-vs-feature mismatch being visible.
fn structural_rules(reg: &Registry, rel: &str, text: &str, errors: &mut Vec<String>) {
    let in_magetypes = rel.starts_with("magetypes/src");
    if !in_magetypes {
        return;
    }

    // Token fabrication routes. Tokens must only come from summon() or an
    // explicit archmage forge. Safe from_context calls are checked separately
    // against the enclosing function, never against an impl or token parameter.
    for (pat, why) in [
        (
            r"\bforge_token_dangerously\b",
            "token forging inside magetypes",
        ),
        (
            r"\bMaybeUninit\b",
            "uninitialized construction (potential token/Repr fabrication)",
        ),
        (
            r"\bmem::zeroed\b",
            "zeroed construction (potential token fabrication)",
        ),
        (
            r"\bmem::uninitialized\b",
            "deprecated uninitialized construction",
        ),
    ] {
        let re = Regex::new(pat).expect("structural rule regex");
        for m in re.find_iter(text) {
            errors.push(format!(
                "{}:{}: STRUCTURAL RULE: `{}` — {}. If genuinely needed, it must \
                 live in archmage with an explicit safety argument, not magetypes.",
                rel,
                line_of(text, m.start()),
                m.as_str(),
                why
            ));
        }
    }

    // Whole-slice f16 conversion deliberately selects once per buffer.
    // Per-vector arithmetic and register conversions must never pay that cost.
    // The one measured exception is scalar f64 portable FMA's cached token
    // check (soft_fma.rs `token`, approved 2026-10-24: identical output,
    // 0.09-0.43x the software time with FMA, about 1.0x without). These named
    // functions are the only accepted detection sites.
    let detection = Regex::new(
        r"\b(?:summon|is_x86_feature_detected|is_aarch64_feature_detected|__cpuid|__cpuid_count)\b",
    )
    .expect("CPU detection rule regex");
    let audited: &[&str] = match rel {
        "magetypes/src/simd/generic/convert_f16.rs" => &[
            "neon_f16_decode_select",
            "neon_f16_encode_select",
            "f16c_decode_select",
            "f16c_encode_select",
        ],
        "magetypes/src/soft_fma.rs" => &["token"],
        // Approved 2026-11-10: compile-time-proven baseline sqrt selectors,
        // no runtime upgrades; token disabling exercises software fallbacks.
        "magetypes/src/scalar_sqrt.rs" => &["sqrtf", "sqrt"],
        // Compile-time relaxed-SIMD availability with test-only suppression.
        "magetypes/src/wasm_fma.rs" => &["relaxed_token"],
        _ => &[],
    };
    let mut detections = detection.find_iter(text).peekable();
    let scalar_selectors = match rel {
        "magetypes/src/scalar_round.rs" | "magetypes/src/scalar_extrema.rs"
            if detections.peek().is_some() =>
        {
            scalar_selector_summons(reg, rel, text)
        }
        _ => HashSet::new(),
    };
    for m in detections {
        let in_slice_selector = audited.iter().any(|name| {
            let signature = format!("fn {name}(");
            text.match_indices(&signature).any(|(start, _)| {
                let Some(open) = text[start..].find('{').map(|i| start + i) else {
                    return false;
                };
                find_matching_brace(text, open)
                    .is_some_and(|close| open < m.start() && m.start() < close)
            })
        });
        if !in_slice_selector && !scalar_selectors.contains(&m.start()) {
            errors.push(format!(
                "{}:{}: STRUCTURAL RULE: `{}` — hidden CPU detection in magetypes; \
                 accept a caller-supplied proof or use an audited whole-slice selector.",
                rel,
                line_of(text, m.start()),
                m.as_str()
            ));
        }
    }

    // Every `unsafe` in magetypes lives in the storage file, where each one
    // states its invariant next to the others: blocks, impls, traits and
    // functions alike. Code that needs `unsafe` gets a helper there instead.
    // The crate root's `#![deny(unsafe_code)]` enforces the same thing for what
    // the host compiles; this textual check also covers code cfg'd out for the
    // host's architecture.
    if rel != STORAGE_HOME {
        let re = Regex::new(r"\bunsafe\b").expect("unsafe keyword regex");
        for m in re.find_iter(text) {
            errors.push(format!(
                "{}:{}: STRUCTURAL RULE: `unsafe` outside {} — magetypes keeps all \
                 of its unsafe code in that file; add a helper there instead.",
                rel,
                line_of(text, m.start()),
                STORAGE_HOME
            ));
        }
    }

    // Lint attributes that would let unsafe code back in. The one sanctioned
    // `allow(unsafe_code)` outside the storage file is the attribute on
    // `mod simd_storage;` in lib.rs; the storage file's own macros carry the
    // rest. lib.rs must keep the crate-wide deny.
    if rel != STORAGE_HOME {
        let re = Regex::new(r"\b(?:allow|expect|warn)\s*\([^)]*\bunsafe_code\b[^)]*\)\s*\]")
            .expect("unsafe_code lint attribute regex");
        for m in re.find_iter(text) {
            let sanctioned = rel == LIB_HOME
                && text[m.end()..]
                    .trim_start()
                    .starts_with("mod simd_storage;");
            if !sanctioned {
                errors.push(format!(
                    "{}:{}: STRUCTURAL RULE: `{}` outside {} — only `mod simd_storage;` \
                     in {} may allow unsafe code.",
                    rel,
                    line_of(text, m.start()),
                    m.as_str(),
                    STORAGE_HOME,
                    LIB_HOME
                ));
            }
        }
    }
    if rel == LIB_HOME
        && !text.contains("#![deny(unsafe_code)]")
        && !text.contains("#![forbid(unsafe_code)]")
    {
        errors.push(format!(
            "{}: STRUCTURAL RULE: missing `#![deny(unsafe_code)]` — magetypes denies \
             unsafe code everywhere except {}.",
            LIB_HOME, STORAGE_HOME
        ));
    }

    // A bare `transmute` needs `unsafe`, so outside the storage file it is
    // already rejected above; this names the specific problem. Inside the
    // storage file the audited idiom is `transmute_copy` behind compile-time
    // size checks.
    if rel != STORAGE_HOME {
        let re = Regex::new(r"\btransmute\b").expect("transmute regex");
        for m in re.find_iter(text) {
            // `transmute_copy` is a different, length-checked-by-Dst idiom.
            if text[m.end()..].starts_with("_copy") {
                continue;
            }
            errors.push(format!(
                "{}:{}: STRUCTURAL RULE: bare `transmute` outside simd_storage.rs — \
                 use the size-asserted byte-cast helpers there.",
                rel,
                line_of(text, m.start())
            ));
        }
    }

    // Token-less wrapper construction: no Default / serde / bytemuck on
    // SIMD wrapper types. These would let safe code materialize a
    // token-carrying wrapper without a genuine token.
    for (pat, why) in [
        (
            r"impl(?:\s*<[^>{]*>)?\s+Default\s+for\s+\w+x\d+",
            "Default on a SIMD wrapper bypasses token-gated construction",
        ),
        (
            r"#\[derive\([^)]*(?:Deserialize|Pod|Zeroable)",
            "serde/bytemuck derives allow token-less construction",
        ),
    ] {
        let re = Regex::new(pat).expect("wrapper rule regex");
        for m in re.find_iter(text) {
            errors.push(format!(
                "{}:{}: STRUCTURAL RULE: {} ({}).",
                rel,
                line_of(text, m.start()),
                m.as_str().trim(),
                why
            ));
        }
    }

    // Gather/scatter intrinsics address memory at `base + offset * scale`
    // for per-lane offsets, so a borrowed slice proves nothing about the
    // lanes. Their bounds proof is the offset masking or lane enabling in the
    // storage helpers; anywhere else, including the generated impls, the
    // offsets would be unchecked.
    if rel != STORAGE_HOME {
        let re =
            Regex::new(r"\b_mm\w*_i(?:32|64)(?:lo)?(?:gather|scatter)_\w+").expect("gather regex");
        for m in re.find_iter(text) {
            errors.push(format!(
                "{}:{}: STRUCTURAL RULE: `{}` outside {} — gather/scatter offsets \
                 must be bounded against the borrowed slice, which only that file does.",
                rel,
                line_of(text, m.start()),
                m.as_str(),
                STORAGE_HOME
            ));
        }
    }

    // `Pod` says "every bit pattern is valid" — implementing it for a
    // token-bearing wrapper would let `copy`/`cast`/`view` manufacture a
    // proof out of arbitrary bytes. The only legitimate impls are the
    // `impl_pod!` ones on raw scalars and stdarch vector types.
    if rel != STORAGE_HOME {
        let re =
            Regex::new(r"unsafe\s+impl(?:\s*<[^>{]*>)?\s+(?:\w+::)*Pod\b").expect("pod impl regex");
        for m in re.find_iter(text) {
            errors.push(format!(
                "{}:{}: STRUCTURAL RULE: `unsafe impl Pod` outside simd_storage.rs — \
                 Pod asserts every bit pattern is valid, which would let the byte-cast \
                 helpers fabricate a token-bearing wrapper from arbitrary bytes.",
                rel,
                line_of(text, m.start())
            ));
        }
    }

    // `TokenStorage`'s contract is "repr(C) pair of a Pod representation and a
    // 1-ZST token". `impl_token_storage!` checks the layout at compile time
    // (field arity and types, Repr at offset 0, sizes); as a second line,
    // require the attribute that guarantees that layout on each struct it is
    // invoked for.
    {
        let impl_re = Regex::new(r"impl_token_storage!\s*\(\s*(\w+)")
            .expect("token storage invocation regex");
        for m in impl_re.captures_iter(text) {
            let ty = m.get(1).unwrap().as_str();
            let struct_re = Regex::new(&format!(
                r"((?:#\[[^\]]*\]\s*)*)(?:pub(?:\([^)]*\))?\s+)?struct\s+{}\b",
                regex::escape(ty)
            ))
            .expect("struct decl regex");
            let attrs = struct_re
                .captures(text)
                .map(|c| c.get(1).unwrap().as_str().to_string())
                .unwrap_or_default();
            if !(attrs.contains("repr(C)") || attrs.contains("repr(transparent)")) {
                errors.push(format!(
                    "{}:{}: STRUCTURAL RULE: `impl_token_storage!({}, ..)` but `{}` \
                     is not declared `#[repr(C)]` in this file — TokenStorage's contract \
                     requires the Repr-then-token layout the attribute guarantees.",
                    rel,
                    line_of(text, m.get(0).unwrap().start()),
                    ty,
                    ty
                ));
            }
        }
    }

    // Backend trait methods must take `self` — the token IS the receiver.
    // A method without a receiver could be called as
    // `<Token as Backend>::method(...)` without holding a token value.
    // (The compile-fail tests in bypass_adversarial.rs prove a sample of
    // methods reject this; this rule covers all of them mechanically.)
    if rel.starts_with("magetypes/src/simd/backends/") {
        let trait_re = Regex::new(r"\btrait\s+\w+(?:Backend|Internal|Convert|Bitcast)\b[^{]*")
            .expect("trait regex");
        let fn_re = Regex::new(r"\bfn\s+(\w+)\s*\(([^)]*)").expect("trait fn regex");
        for m in trait_re.find_iter(text) {
            let Some(open) = find_char_from(text, m.end(), b'{') else {
                continue;
            };
            let Some(close) = find_matching_brace(text, open) else {
                continue;
            };
            let body = &text[open..close];
            for f in fn_re.captures_iter(body) {
                let params = f.get(2).unwrap().as_str();
                let first = params.split(',').next().unwrap_or("").trim();
                if !(first == "self"
                    || first.starts_with("self")
                    || first.starts_with("&self")
                    || first.starts_with("mut self"))
                {
                    errors.push(format!(
                        "{}:{}: STRUCTURAL RULE: backend trait method `{}` does not \
                         take `self` — tokenless associated fns can be invoked via \
                         UFCS without holding a token.",
                        rel,
                        line_of(text, open + f.get(0).unwrap().start()),
                        f.get(1).unwrap().as_str()
                    ));
                }
            }
        }
    }
}

/// Full-repo verification: walk the scan roots, run all structural/AST/safety checks.
pub fn verify(reg: &Registry) -> Result<()> {
    println!("=== Structural & Unsafe-Surface Soundness Verification ===\n");

    let mut files: Vec<PathBuf> = Vec::new();
    for root in SCAN_ROOTS {
        collect_rs_files(Path::new(root), &mut files)?;
    }
    // Under tests/expand, only the committed macro *output* snapshots are
    // meaningful — the inputs are pre-expansion (their gating attribute has
    // not been applied yet).
    files.retain(|f| {
        !f.starts_with("tests/expand") || f.to_string_lossy().ends_with(".expanded.rs")
    });
    files.sort();

    let mut errors: Vec<String> = Vec::new();
    let mut storage_gather_scatter = 0usize;

    for file in &files {
        let raw = fs::read_to_string(file)
            .with_context(|| format!("Failed to read {}", file.display()))?;
        let rel = file.display().to_string();
        let scan = scan_source(reg, &rel, &raw);
        errors.extend(scan.errors);
        storage_gather_scatter += scan.storage_gather_scatter;
    }

    println!(
        "Scanned {} files ({} audited gather/scatter calls in {STORAGE_HOME})",
        files.len(),
        storage_gather_scatter
    );

    if files.len() < MIN_SCANNED_FILES {
        errors.push(format!(
            "VACUOUS PASS GUARD: scanned only {} files (floor {MIN_SCANNED_FILES}). \
             Check SCAN_ROOTS in xtask/src/soundness.rs.",
            files.len()
        ));
    }
    if storage_gather_scatter < MIN_STORAGE_GATHER_SCATTER {
        errors.push(format!(
            "VACUOUS PASS GUARD: verified only {} gather/scatter calls in {STORAGE_HOME} \
             (floor {MIN_STORAGE_GATHER_SCATTER}). If gather/scatter helpers were intentionally \
             removed, update MIN_STORAGE_GATHER_SCATTER in xtask/src/soundness.rs.",
            storage_gather_scatter
        ));
    }

    if errors.is_empty() {
        println!("\n✓ Soundness verification PASSED");
        println!("  All structural, from_context, gather/scatter, and SAFETY rules satisfied.");
        Ok(())
    } else {
        println!(
            "\n✗ Soundness verification FAILED — {} error(s):\n",
            errors.len()
        );
        for e in &errors {
            eprintln!("  {}", e);
        }
        bail!(
            "Soundness verification failed with {} error(s)",
            errors.len()
        )
    }
}

/// Recursively collect `.rs` files under `dir`, skipping `target/`.
fn collect_rs_files(dir: &Path, out: &mut Vec<PathBuf>) -> Result<()> {
    if !dir.exists() {
        bail!(
            "Scan root {} does not exist — update SCAN_ROOTS in xtask/src/soundness.rs",
            dir.display()
        );
    }
    for entry in fs::read_dir(dir).with_context(|| format!("read_dir {}", dir.display()))? {
        let entry = entry?;
        let path = entry.path();
        let name = entry.file_name();
        let name = name.to_string_lossy();
        if path.is_dir() {
            if name == "target" || name.starts_with('.') {
                continue;
            }
            collect_rs_files(&path, out)?;
        } else if name.ends_with(".rs") {
            out.push(path);
        }
    }
    Ok(())
}

/// Replace comment bytes with spaces (newlines preserved) so offsets and
/// line numbers remain valid. Handles `//`-to-EOL and nested `/* */`.
fn strip_comments(src: &str) -> String {
    let bytes = src.as_bytes();
    let mut out = bytes.to_vec();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'/' && i + 1 < bytes.len() && bytes[i + 1] == b'/' {
            while i < bytes.len() && bytes[i] != b'\n' {
                out[i] = b' ';
                i += 1;
            }
        } else if bytes[i] == b'/' && i + 1 < bytes.len() && bytes[i + 1] == b'*' {
            let mut depth = 1;
            out[i] = b' ';
            out[i + 1] = b' ';
            i += 2;
            while i < bytes.len() && depth > 0 {
                if bytes[i] == b'/' && i + 1 < bytes.len() && bytes[i + 1] == b'*' {
                    depth += 1;
                    out[i] = b' ';
                    out[i + 1] = b' ';
                    i += 2;
                } else if bytes[i] == b'*' && i + 1 < bytes.len() && bytes[i + 1] == b'/' {
                    depth -= 1;
                    out[i] = b' ';
                    out[i + 1] = b' ';
                    i += 2;
                } else {
                    if bytes[i] != b'\n' {
                        out[i] = b' ';
                    }
                    i += 1;
                }
            }
        } else {
            i += 1;
        }
    }
    String::from_utf8(out).expect("comment stripping preserved UTF-8")
}

/// Index of the first `needle` byte at or after `from`.
fn find_char_from(text: &str, from: usize, needle: u8) -> Option<usize> {
    text.as_bytes()[from..]
        .iter()
        .position(|&b| b == needle)
        .map(|p| from + p)
}

/// Given the index of an opening brace, return the index of its match.
fn find_matching_brace(text: &str, open: usize) -> Option<usize> {
    let bytes = text.as_bytes();
    debug_assert_eq!(bytes[open], b'{');
    let mut depth = 0i32;
    for (i, &b) in bytes.iter().enumerate().skip(open) {
        match b {
            b'{' => depth += 1,
            b'}' => {
                depth -= 1;
                if depth == 0 {
                    return Some(i);
                }
            }
            _ => {}
        }
    }
    None
}

/// 1-based line number of a byte offset.
fn line_of(text: &str, offset: usize) -> usize {
    text.as_bytes()[..offset]
        .iter()
        .filter(|&&b| b == b'\n')
        .count()
        + 1
}

/// Guard the committed `.stderr` snapshots against portability rot.
///
/// `trybuild` compares rustc's rendered diagnostic byte-for-byte, so a case
/// whose message varies by platform or toolchain passes on the machine that
/// blessed it and fails everywhere else. That is not hypothetical: the
/// `from_context()` rejection cases were blessed on Linux x86-64, whose
/// diagnostic ends
///
/// ```text
/// = note: the sse and sse2 target features being enabled in the build
///         configuration does not remove the requirement to list them ...
/// ```
///
/// while macOS-Intel renders `the cmpxchg16b, sse, sse2, sse3, sse4.1, and
/// ssse3` for the same code, because the two targets enable different baseline
/// features. They now live in `tests/soundness_exploits.rs`, which asserts an
/// error code plus message fragments and is stable across platforms and rustc
/// releases.
///
/// This check keeps a case like that from being added back. It only inspects
/// snapshots consumed by trybuild (`tests/compile_fail/`, `tests/ui/`); the
/// fragment harness reads its sources from `tests/soundness/` and ignores any
/// `.stderr` sitting beside them.
pub fn check_stderr_snapshot_portability() -> Result<()> {
    println!("=== Compile-fail snapshot portability ===\n");

    // Each rule is a thing rustc renders differently depending on where or with
    // which toolchain it runs.
    let rules: &[(&str, &str, &str)] = &[
        (
            r"target features? ",
            "names target features",
            "the enabled-feature baseline differs per target (Linux x86-64 vs \
             macOS-Intel vs i686)",
        ),
        (
            r"enabled in the build configuration",
            "quotes the build configuration's feature set",
            "that set is target-specific",
        ),
        (
            r"(?:/home/|/Users/|[A-Za-z]:\\)",
            "contains an absolute path",
            "paths differ per machine and per CI runner",
        ),
        (
            r"(?:\.cargo[/\\]registry|toolchains[/\\])",
            "contains a toolchain or registry path",
            "those paths embed the toolchain version and the runner's layout",
        ),
        (
            r"rust-\d+\.\d+",
            "embeds a rustc version",
            "clippy help URLs carry the version and change on every bump",
        ),
        (
            r"size of `?usize`?|\b(?:32|64)-bit\b",
            "depends on pointer width",
            "i686 and armv7 are 32-bit CI targets",
        ),
    ];

    let mut errors = Vec::new();
    let mut checked = 0usize;
    for dir in ["tests/compile_fail", "tests/ui"] {
        let path = Path::new(dir);
        if !path.is_dir() {
            continue;
        }
        let mut entries: Vec<PathBuf> = fs::read_dir(path)?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.extension().is_some_and(|e| e == "stderr"))
            .collect();
        entries.sort();
        for file in entries {
            checked += 1;
            let text = fs::read_to_string(&file)?;
            for (pat, what, why) in rules {
                let re = Regex::new(pat).expect("snapshot portability regex");
                if let Some(m) = re.find(&text) {
                    errors.push(format!(
                        "{}:{}: snapshot {} (`{}`) — {}. trybuild matches stderr \
                         byte-for-byte, so this passes only where it was blessed. \
                         Move the case to tests/soundness_exploits.rs, which asserts \
                         an error code plus message fragments.",
                        file.display(),
                        line_of(&text, m.start()),
                        what,
                        m.as_str().trim(),
                        why
                    ));
                }
            }
        }
    }

    if errors.is_empty() {
        println!("Checked {checked} committed .stderr snapshots\n");
        println!("✓ Snapshot portability PASSED");
        println!("  No snapshot depends on the host platform or toolchain version.");
        return Ok(());
    }
    println!("✗ Snapshot portability FAILED — {} issue(s):", errors.len());
    for e in &errors {
        println!("  {e}");
    }
    bail!("{} non-portable compile-fail snapshot(s)", errors.len());
}

// ============================================================================
// Tests — the tester is itself tested. Every failure class must fire on a
// planted violation, and every gating context must admit a valid call.
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn test_registry() -> Registry {
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .join("token-registry.toml");
        Registry::load(&path).expect("load token-registry.toml")
    }

    fn scan_at(rel: &str, src: &str) -> FileScan {
        let reg = test_registry();
        scan_source(&reg, rel, src)
    }

    #[test]
    fn hidden_cpu_detection_is_rejected_in_magetypes() {
        let reg = test_registry();
        for call in [
            "X64V3Token::summon()",
            "<X64V3Token as SimdToken>::summon()",
            "std::is_x86_feature_detected!(\"fma\")",
            "std::arch::is_aarch64_feature_detected!(\"fp16\")",
            "core::arch::x86_64::__cpuid(1)",
        ] {
            let mut errors = Vec::new();
            structural_rules(
                &reg,
                "magetypes/src/nostd_math.rs",
                &format!("fn arithmetic() {{ let _ = {call}; }}"),
                &mut errors,
            );
            assert!(
                errors.iter().any(|e| e.contains("hidden CPU detection")),
                "{call}: {errors:?}"
            );
        }
        let mut errors = Vec::new();
        structural_rules(
            &reg,
            "magetypes/src/nostd_math.rs",
            "fn arithmetic(t: Arm64V2Token) { let _ = t.neon(); }",
            &mut errors,
        );
        assert!(errors.is_empty(), "{errors:?}");
        structural_rules(
            &reg,
            "src/tokens/generated/x86.rs",
            "fn detect() { X64V3Token::summon(); }",
            &mut errors,
        );
        assert!(
            errors.is_empty(),
            "explicit archmage detection stays permitted: {errors:?}"
        );
    }

    #[test]
    fn slice_detection_exception_does_not_cover_vector_methods() {
        let reg = test_registry();
        let mut errors = Vec::new();
        structural_rules(
            &reg,
            "magetypes/src/simd/generic/convert_f16.rs",
            "fn neon_f16_decode_select() { if true { Arm64V2Token::summon(); } }",
            &mut errors,
        );
        assert!(errors.is_empty(), "{errors:?}");
        structural_rules(
            &reg,
            "magetypes/src/simd/generic/convert_f16.rs",
            "fn neon_f16_decode_select() {} fn register_convert() { Arm64V2Token::summon(); }",
            &mut errors,
        );
        assert_eq!(errors.len(), 1, "{errors:?}");
        assert!(errors[0].contains("hidden CPU detection"));
    }

    #[test]
    fn soft_fma_detection_exception_is_one_function() {
        let reg = test_registry();
        let mut errors = Vec::new();
        structural_rules(
            &reg,
            "magetypes/src/soft_fma.rs",
            "mod hw { pub(super) fn token() -> Option<Token> { Token::summon() } }",
            &mut errors,
        );
        assert!(errors.is_empty(), "{errors:?}");
        structural_rules(
            &reg,
            "magetypes/src/soft_fma.rs",
            "fn token() {} fn fmaf_lanes() { X64V3Token::summon(); }",
            &mut errors,
        );
        assert_eq!(errors.len(), 1, "{errors:?}");
        assert!(errors[0].contains("hidden CPU detection"));
        let mut errors = Vec::new();
        structural_rules(
            &reg,
            "magetypes/src/nostd_math.rs",
            "fn token() { X64V3Token::summon(); }",
            &mut errors,
        );
        assert_eq!(errors.len(), 1, "the exception is per file: {errors:?}");
    }

    #[test]
    fn scalar_selectors_require_the_matching_compile_time_gate() {
        let reg = test_registry();
        const ROUND: &str = "magetypes/src/scalar_round.rs";
        const EXTREMA: &str = "magetypes/src/scalar_extrema.rs";
        const V2: &str = r#"target_arch = "x86_64", target_feature = "sse3",
            target_feature = "ssse3", target_feature = "sse4.1", target_feature = "sse4.2",
            target_feature = "popcnt", target_feature = "cmpxchg16b""#;
        let checked = |file, gate: &str, token, extra| {
            let src = format!(
                "{gate}\nmod selected {{ fn token() -> Option<{token}> {{
                    ::archmage::{token}::summon()
                }} }}\n{extra}"
            );
            let mut errors = Vec::new();
            structural_rules(&reg, file, &src, &mut errors);
            errors
        };

        let v2_gate = format!("#[cfg(all({V2}))]");
        for (file, gate, token, extra) in [
            (ROUND, "", "X64V2Token", ""),
            (
                ROUND,
                r#"#[cfg(all(target_arch = "x86_64", target_feature = "sse4.1"))]"#,
                "X64V2Token",
                "",
            ),
            (ROUND, &format!("#[cfg(any({V2}))]"), "X64V2Token", ""),
            (ROUND, &v2_gate, "X64V3Token", ""),
            (EXTREMA, &v2_gate, "X64V2Token", ""),
            (
                ROUND,
                &v2_gate,
                "X64V2Token",
                "mod other { fn token() { ::archmage::X64V2Token::summon(); } }",
            ),
        ] {
            let errors = checked(file, gate, token, extra);
            assert_eq!(errors.len(), 1, "{file} {gate} {token} {extra}: {errors:?}");
            assert!(errors[0].contains("hidden CPU detection"), "{errors:?}");
        }

        for (file, gate, token) in [
            (ROUND, v2_gate.as_str(), "X64V2Token"),
            (
                EXTREMA,
                r#"#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]"#,
                "NeonToken",
            ),
            (
                EXTREMA,
                r#"#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]"#,
                "Wasm128Token",
            ),
        ] {
            let errors = checked(file, gate, token, "");
            assert!(errors.is_empty(), "{file} {gate} {token}: {errors:?}");
        }
    }

    #[test]
    fn storage_gather_scatter_requires_v4_boundary_and_audited_op() {
        let ok_arcane = "#[archmage::arcane(import_intrinsics)]\n\
            fn gather(_t: X64V4Token, p: &[i32], idx: __m512i) -> __m512i {\n\
                // SAFETY: bounded.\n\
                unsafe { _mm512_i32gather_epi32::<4>(idx, p.as_ptr()) }\n\
            }\n";
        let res = scan_at(STORAGE_HOME, ok_arcane);
        assert!(res.errors.is_empty(), "{:?}", res.errors);
        assert_eq!(res.storage_gather_scatter, 1);

        let ok_attune = "#[archmage::attune(wrap, import_intrinsics)]\n\
            fn scatter(_t: X64V4Token, dst: &mut [f32], k: u16, idx: __m512i, v: __m512) {\n\
                // SAFETY: bounded.\n\
                unsafe { _mm512_mask_i32scatter_ps::<4>(dst.as_mut_ptr(), k, idx, v) }\n\
            }\n";
        let res = scan_at(STORAGE_HOME, ok_attune);
        assert!(res.errors.is_empty(), "{:?}", res.errors);
        assert_eq!(res.storage_gather_scatter, 1);

        let wrong_token = "#[archmage::arcane(import_intrinsics)]\n\
            fn gather(_t: X64V3Token, p: &[i32], idx: __m512i) -> __m512i {\n\
                // SAFETY: bounded.\n\
                unsafe { _mm512_i32gather_epi32::<4>(idx, p.as_ptr()) }\n\
            }\n";
        let res = scan_at(STORAGE_HOME, wrong_token);
        assert!(
            res.errors
                .iter()
                .any(|e| e.contains("STORAGE GATHER/SCATTER") && e.contains("X64V4Token")),
            "{:?}",
            res.errors
        );

        let missing_attr = "fn gather(_t: X64V4Token, p: &[i32], idx: __m512i) -> __m512i {\n\
                // SAFETY: bounded.\n\
                unsafe { _mm512_i32gather_epi32::<4>(idx, p.as_ptr()) }\n\
            }\n";
        let res = scan_at(STORAGE_HOME, missing_attr);
        assert!(
            res.errors
                .iter()
                .any(|e| e.contains("STORAGE GATHER/SCATTER") && e.contains("X64V4Token")),
            "{:?}",
            res.errors
        );

        let unknown_op = "#[archmage::arcane(import_intrinsics)]\n\
            fn gather(_t: X64V4Token, p: &[f64], idx: __m512i) -> __m512d {\n\
                // SAFETY: bounded.\n\
                unsafe { _mm512_i32gather_pd::<8>(idx, p.as_ptr()) }\n\
            }\n";
        let res = scan_at(STORAGE_HOME, unknown_op);
        assert!(
            res.errors
                .iter()
                .any(|e| e.contains("STORAGE GATHER/SCATTER") && e.contains("audited avx512f set")),
            "{:?}",
            res.errors
        );
    }

    #[test]
    fn structural_rule_no_maybe_uninit_in_magetypes() {
        let scan = scan_at(
            "magetypes/src/simd/generic/foo.rs",
            "fn f() { let x: MaybeUninit<u8> = MaybeUninit::uninit(); }\n",
        );
        assert!(
            scan.errors
                .iter()
                .any(|e| e.contains("STRUCTURAL RULE") && e.contains("MaybeUninit")),
            "{:?}",
            scan.errors
        );
    }

    #[test]
    fn structural_rule_no_bare_transmute_outside_storage_home() {
        let scan = scan_at(
            "magetypes/src/simd/generic/foo.rs",
            "fn f(x: [f32; 4]) -> Y { unsafe { core::mem::transmute(x) } }\n",
        );
        assert!(
            scan.errors.iter().any(|e| e.contains("bare `transmute`")),
            "{:?}",
            scan.errors
        );
        // transmute_copy is the audited idiom and stays allowed (in the one
        // file where hand-written unsafe blocks may live).
        let ok = scan_at(
            STORAGE_HOME,
            "fn f(x: &[u8; 16]) -> Y { unsafe { core::mem::transmute_copy(x) } }\n",
        );
        assert!(
            !ok.errors.iter().any(|e| e.contains("STRUCTURAL RULE")),
            "{:?}",
            ok.errors
        );
    }

    #[test]
    fn structural_rule_transmute_rejected_in_backend_impls() {
        // The generated backends hold no unsafe at all, so a bitcast there
        // goes through the storage helpers like everywhere else.
        let scan = scan_at(
            "magetypes/src/simd/impls/x86_v3.rs",
            "//! # Safety (audit contract — test fixture)\n\
             impl F32x4Backend for archmage::X64V3Token {\n\
             fn from_array(self, arr: [f32; 4]) -> R { unsafe { core::mem::transmute(arr) } }\n\
             }\n",
        );
        assert!(
            scan.errors.iter().any(|e| e.contains("bare `transmute`")),
            "{:?}",
            scan.errors
        );
        assert!(
            scan.errors.iter().any(|e| e.contains("`unsafe` outside")),
            "{:?}",
            scan.errors
        );
    }

    #[test]
    fn structural_rule_gather_scatter_only_in_storage_home() {
        let gather = "fn f(t: X64V4Token, i: __m512i, p: &[i32]) -> __m512i {\n\
                      unsafe { _mm512_i32gather_epi32::<4>(i, p.as_ptr()) } }\n";
        let bad = scan_at("magetypes/src/simd/generic/foo.rs", gather);
        assert!(
            bad.errors
                .iter()
                .any(|e| e.contains("STRUCTURAL RULE") && e.contains("_mm512_i32gather_epi32")),
            "{:?}",
            bad.errors
        );
        let scatter = scan_at(
            "magetypes/src/simd/impls/x86_v4.rs",
            "fn f(t: X64V4Token) { unsafe { _mm512_mask_i32scatter_ps::<4>(p, k, i, v) } }\n",
        );
        assert!(
            scatter
                .errors
                .iter()
                .any(|e| e.contains("STRUCTURAL RULE") && e.contains("_mm512_mask_i32scatter_ps")),
            "{:?}",
            scatter.errors
        );
        let ok = scan_at(STORAGE_HOME, gather);
        assert!(
            !ok.errors.iter().any(|e| e.contains("STRUCTURAL RULE")),
            "{:?}",
            ok.errors
        );
    }

    #[test]
    fn structural_rule_unsafe_only_in_storage_home() {
        let src = "fn f(x: &[u8; 4]) -> u32 {\n\
                   // SAFETY: test fixture.\n\
                   unsafe { core::mem::transmute_copy(x) }\n\
                   }\n";
        let unsafe_outside = |scan: &FileScan| {
            scan.errors
                .iter()
                .any(|e| e.contains("STRUCTURAL RULE") && e.contains("`unsafe` outside"))
        };
        let bad = scan_at("magetypes/src/simd/generic/foo.rs", src);
        assert!(unsafe_outside(&bad), "{:?}", bad.errors);
        let ok = scan_at(STORAGE_HOME, src);
        assert!(ok.errors.is_empty(), "{:?}", ok.errors);
        // The generated backend impls get no exemption.
        let impls = scan_at("magetypes/src/simd/impls/x86_v3.rs", src);
        assert!(unsafe_outside(&impls), "{:?}", impls.errors);
        // Nor do the other forms of unsafe code.
        for form in [
            "unsafe impl Send for Foo {}\n",
            "unsafe fn f() {}\n",
            "pub unsafe trait Bar {}\n",
            "pub trait Baz { unsafe fn up(self); }\n",
        ] {
            let scan = scan_at("magetypes/src/cast.rs", form);
            assert!(unsafe_outside(&scan), "{form}: {:?}", scan.errors);
        }
        // Lint names and comments are not the keyword.
        let names = scan_at(
            "magetypes/src/simd/generic/foo.rs",
            "#![deny(unsafe_op_in_unsafe_fn)]\n// unsafe in a comment\nfn f() {}\n",
        );
        assert!(!unsafe_outside(&names), "{:?}", names.errors);
    }

    #[test]
    fn structural_rule_unsafe_code_lint_attributes() {
        let allows = |scan: &FileScan| {
            scan.errors
                .iter()
                .any(|e| e.contains("may allow unsafe code"))
        };
        let stray = scan_at(
            "magetypes/src/simd/generic/foo.rs",
            "#[allow(unsafe_code)]\nfn f() {}\n",
        );
        assert!(allows(&stray), "{:?}", stray.errors);
        let expect = scan_at(
            "magetypes/src/simd/generic/foo.rs",
            "#[expect(dead_code, unsafe_code)]\nfn f() {}\n",
        );
        assert!(allows(&expect), "{:?}", expect.errors);
        // The one sanctioned allow: on `mod simd_storage;`, under a crate-wide deny.
        let lib = scan_at(
            LIB_HOME,
            "#![deny(unsafe_code)]\n#[allow(unsafe_code)]\nmod simd_storage;\nmod types;\n",
        );
        assert!(
            !lib.errors.iter().any(|e| e.contains("STRUCTURAL RULE")),
            "{:?}",
            lib.errors
        );
        let other_module = scan_at(
            LIB_HOME,
            "#![deny(unsafe_code)]\n#[allow(unsafe_code)]\nmod types;\n",
        );
        assert!(allows(&other_module), "{:?}", other_module.errors);
        let no_deny = scan_at(LIB_HOME, "#[allow(unsafe_code)]\nmod simd_storage;\n");
        assert!(
            no_deny
                .errors
                .iter()
                .any(|e| e.contains("missing `#![deny(unsafe_code)]`")),
            "{:?}",
            no_deny.errors
        );
        // The storage file's own macros may carry the allow.
        let storage = scan_at(
            STORAGE_HOME,
            "macro_rules! m { () => { #[allow(unsafe_code)] unsafe impl Foo for Bar {} }; }\n",
        );
        assert!(!allows(&storage), "{:?}", storage.errors);
    }

    #[test]
    fn structural_rule_token_storage_needs_repr_c() {
        let rule = |scan: &FileScan| {
            scan.errors
                .iter()
                .any(|e| e.contains("is not declared `#[repr(C)]`"))
        };
        let missing = scan_at(
            "magetypes/src/simd/generic/generated/foo_impl.rs",
            "#[derive(Clone, Copy)]\npub struct foo<T: FooBackend>(pub(crate) T::Repr, pub(crate) T);\n\
             crate::simd_storage::impl_token_storage!(foo, FooBackend);\n",
        );
        assert!(rule(&missing), "{:?}", missing.errors);
        let present = scan_at(
            "magetypes/src/simd/generic/generated/foo_impl.rs",
            "#[derive(Clone, Copy)]\n#[repr(C)]\npub struct foo<T: FooBackend>(pub(crate) T::Repr, pub(crate) T);\n\
             crate::simd_storage::impl_token_storage!(foo, FooBackend);\n",
        );
        assert!(!rule(&present), "{:?}", present.errors);
    }

    #[test]
    fn structural_rule_no_default_on_wrappers() {
        let scan = scan_at(
            "magetypes/src/simd/generic/generated/f32x8_impl.rs",
            "impl<T: F32x8Backend> Default for f32x8<T> { fn default() -> Self { todo!() } }\n",
        );
        assert!(
            scan.errors
                .iter()
                .any(|e| e.contains("Default on a SIMD wrapper")),
            "{:?}",
            scan.errors
        );
    }

    #[test]
    fn safety_rule_flags_uncommented_unsafe_outside_impls() {
        let scan = scan_at(
            "magetypes/src/simd/generic/foo.rs",
            "fn f(x: u32) -> f32 { unsafe { core::mem::transmute_copy(&x) } }\n",
        );
        assert!(
            scan.errors
                .iter()
                .any(|e| e.contains("without an adjacent `// SAFETY:`")),
            "{:?}",
            scan.errors
        );
        let ok = scan_at(
            STORAGE_HOME,
            "fn f(x: u32) -> f32 {\n\
             // SAFETY: u32 and f32 are the same size; all bit patterns valid.\n\
             unsafe { core::mem::transmute_copy(&x) }\n\
             }\n",
        );
        assert!(ok.errors.is_empty(), "{:?}", ok.errors);
    }

    #[test]
    fn safety_rule_requires_contract_header_in_backend_impls() {
        let scan = scan_at(
            "magetypes/src/simd/impls/x86_v3.rs",
            "impl F32x4Backend for archmage::X64V3Token {\n\
             fn add(self, a: R, b: R) -> R { unsafe { _mm_add_ps(a, b) } }\n\
             }\n",
        );
        assert!(
            scan.errors
                .iter()
                .any(|e| e.contains("# Safety (audit contract")),
            "{:?}",
            scan.errors
        );
    }

    #[test]
    fn structural_rule_backend_trait_methods_take_self() {
        let scan = scan_at(
            "magetypes/src/simd/backends/f32x8.rs",
            "pub trait F32x8Backend: SimdToken + Sealed {\n\
             fn splat(self, v: f32) -> Self::Repr;\n\
             fn rogue(v: f32) -> Self::Repr;\n\
             }\n",
        );
        assert_eq!(
            scan.errors
                .iter()
                .filter(|e| e.contains("does not take `self`"))
                .count(),
            1,
            "{:?}",
            scan.errors
        );
        assert!(scan.errors[0].contains("rogue"));

        let internal = scan_at(
            "magetypes/src/simd/backends/f32x8.rs",
            "pub(crate) trait F32x8Internal: SimdToken + Sealed {\n\
             fn rogue_internal(v: f32) -> Self::InternalRepr;\n\
             }\n",
        );
        assert!(
            internal
                .errors
                .iter()
                .any(|e| e.contains("does not take `self`") && e.contains("rogue_internal")),
            "{:?}",
            internal.errors
        );
    }

    #[test]
    fn full_repo_scan_passes_and_meets_floors() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap();
        std::env::set_current_dir(root).expect("chdir to repo root");
        let reg = test_registry();
        verify(&reg).expect("repo soundness verification");
    }
}
