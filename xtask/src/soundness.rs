//! Structure-aware intrinsic soundness verification.
//!
//! # What this verifies
//!
//! The archmage/magetypes safety model rests on one invariant:
//!
//! > **Every call to a feature-gated CPU intrinsic is enclosed by a proof
//! > that the feature is available.**
//!
//! A proof is one of:
//!
//! 1. An `impl <Trait> for <Token>` block — the token type (e.g.
//!    `archmage::X64V3Token`) is a zero-sized runtime proof: it can only be
//!    obtained from `summon()` (which performs CPU detection) or an
//!    explicitly-dangerous forge. Backend trait methods take `self` (the
//!    token), so executing the body implies the caller held the proof.
//! 2. A `#[target_feature(enable = "...")]` attribute on the enclosing
//!    function — reaching the body implies the caller discharged the
//!    feature obligation (`unsafe` call or matching-feature safe call).
//! 3. A token-typed parameter in the enclosing `fn` signature (e.g. the
//!    `_fast` overloads that take `X64V4Token`) — same proof-by-existence
//!    argument as (1).
//!
//! Contexts union: a `fn` taking `X64V4Token` inside an
//! `impl ... for X64V3Token` block proves V3 ∪ V4 features (the caller held
//! both proofs).
//!
//! For every intrinsic call found, the intrinsic's required features (from
//! the stdarch-extracted database `docs/intrinsics/complete_intrinsics.csv`)
//! must be a subset of the features proven by the union of enclosing
//! contexts — plus, for x86 intrinsics only, the x86-64 ABI baseline
//! (`sse`, `sse2`).
//!
//! # Failure conditions (all hard errors)
//!
//! - **FEATURE MISMATCH** — an intrinsic requires features its gating
//!   context does not prove. This is the unsoundness this tool exists to
//!   catch.
//! - **UNGATED** — an intrinsic with non-baseline feature requirements
//!   appears outside any gating context (e.g. in a backend trait *default
//!   body*, where `Self` could be `ScalarToken`).
//! - **UNKNOWN INTRINSIC** — an identifier shaped like an intrinsic
//!   (`_mm*`, NEON `v*_f32`-style, WASM `f32x4_*`-style) is called but is
//!   not in the database and not in the audited allowlist. Either the
//!   database is stale (`just intrinsics-refresh`) or the name is a typo.
//! - **STRUCTURAL RULE** — a magetypes source pattern that the soundness
//!   model forbids (see [`structural_rules`]): token fabrication routes
//!   (`MaybeUninit`, `mem::zeroed`, forging); outside `simd_storage.rs`, the
//!   `unsafe` keyword in any form, a bare `transmute`, gather/scatter
//!   intrinsics or `unsafe impl Pod`; an `allow(unsafe_code)` other than the
//!   one on `mod simd_storage;`, or a crate root without
//!   `#![deny(unsafe_code)]`; an `impl_token_storage!` struct without
//!   `#[repr(C)]`; token-less wrapper constructors
//!   (`Default`/serde/bytemuck); or backend-trait methods without a `self`
//!   receiver (which would allow calling intrinsics via UFCS without
//!   holding a token).
//! - **SAFETY-COMMENT DISCIPLINE** — every `unsafe {` block outside the
//!   generated backend impls must carry an adjacent `// SAFETY:` comment
//!   (see [`safety_comment_rules`]); a generated impls file that contains
//!   `unsafe` must carry the file-header audit contract instead.
//! - **VACUOUS PASS** — the total number of verified calls fell below
//!   [`MIN_VERIFIED_CALLS`], or a file listed in [`REQUIRED_FILE_FLOORS`]
//!   produced fewer verified calls than its floor. This guards against the
//!   scanner silently rotting after a refactor (which happened once: the
//!   pre-2026-07 checker was driven by registry file mappings that no
//!   longer existed and "passed" while verifying zero calls).
//!
//! # Testing the tester
//!
//! The unit tests at the bottom of this file feed synthetic sources with
//! planted violations through the same [`Scanner`] used in production and
//! assert each failure class actually fires. `cargo test -p xtask` runs in
//! CI; if the scanner rots, the tests fail even when the repo scan passes.
//!
//! # Known limitations (documented for auditors)
//!
//! - Text-based, not AST-based: comments are stripped, but intrinsic names
//!   inside string literals would be (harmlessly) scanned, and pathological
//!   formatting could confuse the brace matcher. Generated code is
//!   formatting-stable, making this reliable in practice; the floors ensure
//!   the scanner can never silently stop seeing the bulk of the code.
//! - It verifies *feature availability*, not memory safety: `unsafe` blocks
//!   whose obligation is pointer validity/layout (loads, stores,
//!   transmutes) are confined to `simd_storage.rs` by the structural rules
//!   and exercised under Miri, not proven here.

use anyhow::{Context, Result, bail};
use regex::Regex;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};

use crate::IntrinsicEntry;
use crate::registry::Registry;

mod raw_context;

/// Directories scanned for intrinsic calls, relative to the repo root.
///
/// `src` is the archmage crate itself (detection code), `magetypes/src` is
/// where all backend intrinsic calls live, and `tests/expand` holds the
/// committed macro-expansion snapshots — scanning them means macro *output*
/// is subject to the same verification as handwritten code.
const SCAN_ROOTS: &[&str] = &["src", "magetypes/src", "tests/expand"];

/// Global floor on verified intrinsic calls (measured 4478 at
/// introduction, 2026-07-14). If a legitimate refactor reduces coverage
/// below this, update the constant in the same commit and explain why in
/// the commit message. A scanning failure drops coverage to ~0 and always
/// trips this.
const MIN_VERIFIED_CALLS: usize = 4000;

/// Per-file verified-call floors (measured counts in comments). Every file
/// carrying a substantial portion of the unsafe surface must individually
/// stay visible to the scanner. Floors sit ~25% below measured so normal
/// churn doesn't trip them.
const REQUIRED_FILE_FLOORS: &[(&str, usize)] = &[
    ("magetypes/src/simd/impls/x86_v3.rs", 750), // measured 1260
    // Measured 589 after delegating X64V4xToken's duplicate base W512 backends
    // to X64V4Token in x86_v4_w512_delegated.rs (was 1056 with duplicated impls).
    ("magetypes/src/simd/impls/x86_v4.rs", 440),
    ("magetypes/src/simd/impls/arm_neon.rs", 1080), // measured 1705
    ("magetypes/src/simd/impls/wasm128.rs", 700),   // measured 1227
    ("magetypes/src/simd/impls/wasm128_fma_overrides.rs", 5), // measured 6
    ("magetypes/src/simd/generic/cross_width.rs", 6), // measured 8
    ("magetypes/src/simd/generic/convert_f16.rs", 18), // measured 26
    ("magetypes/src/simd_storage.rs", 13),          // measured 18 (gather/scatter)
    ("magetypes/src/wasm_fma.rs", 45),              // measured 62 (WasmFmaOps + relaxed_probe)
    ("src/tokens/generated/wasm.rs", 8),            // measured 12 (runtime FMA probe)
];

/// The one magetypes file allowed `unsafe` of any kind, `unsafe impl Pod`, and
/// gather/scatter intrinsics. Each `unsafe` there states its invariant next to
/// the others, so auditing magetypes' unsafe means reading one file. The crate
/// root denies `unsafe_code` everywhere else.
const STORAGE_HOME: &str = "magetypes/src/simd_storage.rs";

/// The magetypes crate root, which must deny `unsafe_code`.
const LIB_HOME: &str = "magetypes/src/lib.rs";

/// Identifiers that look like intrinsics but are deliberately not verified.
/// Every entry must carry a justification. Keep this list short — it is
/// printed in the report so auditors see exactly what was waived.
const ALLOWLIST: &[(&str, &str)] = &[
    (
        "_MM_SHUFFLE",
        "const fn shuffle-immediate helper; computes an i32, executes no instruction",
    ),
    (
        "__cpuid",
        "CPUID leaf query; available on all x86-64, used by feature detection itself",
    ),
    (
        "__cpuid_count",
        "CPUID subleaf query; available on all x86-64, used by feature detection itself",
    ),
];

/// x86-64 ABI baseline — every x86-64 CPU has these, so any gated context
/// implicitly proves them for x86 intrinsics.
const X86_64_BASELINE: &[&str] = &["sse", "sse2"];

/// A byte range of a scanned file proving a set of features.
struct GatingSpan {
    start: usize,
    end: usize,
    features: HashSet<String>,
    origin: String,
}

/// Result of scanning one source file.
#[derive(Default)]
pub struct FileScan {
    pub errors: Vec<String>,
    pub verified: usize,
    pub allowlisted: BTreeMap<String, usize>,
}

/// Compiled scanning machinery, reusable across files (and unit-testable).
pub struct Scanner<'r> {
    reg: &'r Registry,
    db: HashMap<String, IntrinsicEntry>,
    candidate_re: Regex,
    impl_re: Regex,
    tf_re: Regex,
    quoted_re: Regex,
    fn_re: Regex,
    token_name_re: Regex,
    shape_x86_re: Regex,
    shape_wasm_re: Regex,
    shape_arm_re: Regex,
    allow: BTreeMap<&'static str, &'static str>,
}

impl<'r> Scanner<'r> {
    pub fn new(reg: &'r Registry) -> Result<Self> {
        let db = crate::load_intrinsic_database()?;
        let token_names = reg.all_token_names();
        Ok(Scanner {
            reg,
            db,
            // Any identifier used as a call or turbofish — DB membership
            // then decides whether it is an intrinsic. This cannot miss an
            // intrinsic the way a hand-maintained prefix whitelist can.
            candidate_re: Regex::new(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*(?:\(|::<)")
                .expect("candidate regex"),
            impl_re: Regex::new(
                r"\bimpl(?:\s*<[^>{]*>)?\s+[A-Za-z_][\w:]*(?:\s*<[^>{]*>)?\s+for\s+((?:[A-Za-z_]\w*\s*::\s*)*)([A-Za-z_]\w*)",
            )
            .expect("impl regex"),
            tf_re: Regex::new(r"#\[target_feature\(([^)]*)\)\]").expect("target_feature regex"),
            quoted_re: Regex::new(r#""([^"]*)""#).expect("quoted regex"),
            fn_re: Regex::new(r"\bfn\s+[A-Za-z_]\w*").expect("fn regex"),
            token_name_re: Regex::new(&format!(r"\b({})\b", token_names.join("|")))
                .expect("token name regex"),
            shape_x86_re: Regex::new(r"^_(?:_)?mm\w*$").expect("x86 shape regex"),
            shape_wasm_re: Regex::new(r"^(?:[iuf](?:8x16|16x8|32x4|64x2)|v128)_[a-z0-9_]+$")
                .expect("wasm shape"),
            shape_arm_re: Regex::new(
                r"^v[a-z][a-z0-9]*_(?:[a-z0-9]+_)*(?:[sup](?:8|16|32|64)|f(?:16|32|64)|p(?:8|16|64|128)|bf16)$",
            )
            .expect("arm shape regex"),
            allow: ALLOWLIST.iter().copied().collect(),
        })
    }

    /// Scan one file's source text. `rel` is used in error messages only.
    pub fn scan_source(&self, rel: &str, raw: &str) -> FileScan {
        let text = strip_comments(raw);
        let mut out = FileScan::default();
        let spans = self.gating_spans(rel, &text, &mut out.errors);
        let arch_aliases = arch_path_aliases(&text);

        for cap in self.candidate_re.captures_iter(&text) {
            let name_match = cap.get(1).unwrap();
            let name = name_match.as_str();
            let pos = name_match.start();

            // Definitions (`fn f32x4_splat(...)`) and method calls
            // (`self.f32x4_splat(...)`) are not intrinsic invocations, even
            // when the name collides with an intrinsic (magetypes' width
            // traits deliberately reuse WASM intrinsic naming). For
            // path-qualified calls, only `<arch module>::name(...)` counts
            // as an intrinsic; `Self::name(...)` / `f32x4::name(...)` are
            // trait/assoc calls.
            match preceding_context(&text, pos, &arch_aliases) {
                Preceding::FnKeyword | Preceding::Dot | Preceding::NonArchPath => continue,
                Preceding::ArchPath | Preceding::Bare => {}
            }

            let Some(entry) = self.db.get(name) else {
                let looks_like_intrinsic = self.shape_x86_re.is_match(name)
                    || self.shape_wasm_re.is_match(name)
                    || self.shape_arm_re.is_match(name);
                if looks_like_intrinsic {
                    if self.allow.contains_key(name) {
                        *out.allowlisted.entry(name.to_string()).or_default() += 1;
                    } else {
                        out.errors.push(format!(
                            "{}:{}: UNKNOWN INTRINSIC `{}` — intrinsic-shaped but not in \
                             complete_intrinsics.csv and not allowlisted. Stale database? \
                             Run `just intrinsics-refresh`.",
                            rel,
                            line_of(&text, pos),
                            name
                        ));
                    }
                }
                continue;
            };

            let required: HashSet<&str> = entry
                .features
                .split(',')
                .map(|s| s.trim())
                .filter(|s| !s.is_empty())
                .collect();

            let enclosing: Vec<&GatingSpan> = spans
                .iter()
                .filter(|s| pos > s.start && pos < s.end)
                .collect();

            let mut provided: HashSet<&str> = HashSet::new();
            for s in &enclosing {
                provided.extend(s.features.iter().map(|f| f.as_str()));
            }
            // The sse/sse2 ABI baseline applies only to x86 intrinsics —
            // granting it inside e.g. a NeonToken impl would let a stray
            // SSE2 intrinsic pass unnoticed.
            let is_x86 = entry.arch.starts_with("x86");
            if !enclosing.is_empty() && is_x86 {
                provided.extend(X86_64_BASELINE.iter().copied());
            }

            let baseline_only = is_x86
                && required
                    .iter()
                    .all(|f| X86_64_BASELINE.contains(f) || f.is_empty());

            if enclosing.is_empty() {
                if baseline_only {
                    // e.g. sse2 helpers in detection code — sound on the
                    // x86-64 ABI without any token.
                    out.verified += 1;
                } else {
                    out.errors.push(format!(
                        "{}:{}: UNGATED intrinsic `{}` (requires [{}]) — no enclosing \
                         impl-for-token block, #[target_feature] fn, or token-typed \
                         parameter proves these features. If this is a trait default \
                         body, move the intrinsic into per-token impls.",
                        rel,
                        line_of(&text, pos),
                        name,
                        entry.features
                    ));
                }
                continue;
            }

            let missing: Vec<&&str> = required
                .iter()
                .filter(|f| !provided.contains(**f))
                .collect();

            if missing.is_empty() {
                out.verified += 1;
            } else {
                let origins: Vec<&str> = enclosing.iter().map(|s| s.origin.as_str()).collect();
                out.errors.push(format!(
                    "{}:{}: FEATURE MISMATCH `{}` requires [{}] but context ({}) proves \
                     only [{}] — missing {:?}",
                    rel,
                    line_of(&text, pos),
                    name,
                    entry.features,
                    origins.join(" + "),
                    {
                        let mut p: Vec<&str> = provided.iter().copied().collect();
                        p.sort_unstable();
                        p.join(", ")
                    },
                    missing
                ));
            }
        }

        structural_rules(rel, &text, &mut out.errors);
        raw_context::verify(self.reg, rel, &text, &mut out.errors);
        safety_comment_rules(rel, raw, &mut out.errors);
        out
    }

    /// Collect every byte range of `text` that proves a feature set.
    fn gating_spans(&self, rel: &str, text: &str, errors: &mut Vec<String>) -> Vec<GatingSpan> {
        let mut spans: Vec<GatingSpan> = Vec::new();

        // (1) impl-for-token blocks.
        for cap in self.impl_re.captures_iter(text) {
            let type_name = cap.get(2).unwrap().as_str();
            let Some(token) = self.reg.find_token(type_name) else {
                continue;
            };
            let header_end = cap.get(0).unwrap().end();
            let Some(open) = find_char_from(text, header_end, b'{') else {
                continue;
            };
            let Some(close) = find_matching_brace(text, open) else {
                errors.push(format!(
                    "{}:{}: unbalanced braces after `impl ... for {}`",
                    rel,
                    line_of(text, header_end),
                    type_name
                ));
                continue;
            };
            spans.push(GatingSpan {
                start: open,
                end: close,
                features: token.features.iter().cloned().collect(),
                origin: format!("impl for {}", token.name),
            });
        }

        // (2) #[target_feature(enable = "...")] function bodies.
        for cap in self.tf_re.captures_iter(text) {
            let attr_end = cap.get(0).unwrap().end();
            let mut features: HashSet<String> = HashSet::new();
            for q in self.quoted_re.captures_iter(cap.get(1).unwrap().as_str()) {
                for f in q.get(1).unwrap().as_str().split(',') {
                    features.insert(f.trim().to_string());
                }
            }
            // The attribute is followed (possibly after more attributes and
            // the signature) by the fn body's opening brace.
            let Some(open) = find_char_from(text, attr_end, b'{') else {
                continue;
            };
            let Some(close) = find_matching_brace(text, open) else {
                continue;
            };
            spans.push(GatingSpan {
                start: open,
                end: close,
                features,
                origin: "#[target_feature]".to_string(),
            });
        }

        // (3) fn input signatures mentioning token types (parameters or generics;
        // return types after `->` never prove a feature context, and WASM token
        // parameters require a WASM feature-enabling attribute on free fns).
        for m in self.fn_re.find_iter(text) {
            let sig_start = m.end();
            let Some((sig_end, has_body)) = find_signature_end(text, sig_start) else {
                continue;
            };
            if !has_body {
                continue;
            }
            let input_sig = trim_return_type(&text[sig_start..sig_end]);
            let has_wasm_attr = has_wasm_fn_attr(text, m.start());
            let mut features: HashSet<String> = HashSet::new();
            let mut names: Vec<&str> = Vec::new();
            for cap in self.token_name_re.captures_iter(input_sig) {
                let name = cap.get(1).unwrap().as_str();
                if let Some(token) = self.reg.find_token(name) {
                    if token.arch == "wasm" && !has_wasm_attr {
                        continue;
                    }
                    features.extend(token.features.iter().cloned());
                    names.push(&token.name);
                }
            }
            if features.is_empty() {
                continue;
            }
            let open = sig_end; // index of `{`
            let Some(close) = find_matching_brace(text, open) else {
                continue;
            };
            spans.push(GatingSpan {
                start: open,
                end: close,
                features,
                origin: format!("fn param tokens [{}]", names.join(", ")),
            });
        }

        // (4) #[attune(...)] / #[rite(...)] / #[token_target_features(...)] bodies.
        self.attune_rite_spans(text, &mut spans);

        // (5) Positive compile-time #[cfg(... target_feature = "..." ...)] guards.
        cfg_target_feature_spans(text, &mut spans);

        spans
    }

    fn attune_rite_spans(&self, text: &str, spans: &mut Vec<GatingSpan>) {
        let mut search = 0;
        while let Some(rel_hash) = text[search..].find('#') {
            let hash = search + rel_hash;
            search = hash + 1;
            let bytes = text.as_bytes();
            let mut i = hash + 1;
            while i < bytes.len() && bytes[i].is_ascii_whitespace() {
                i += 1;
            }
            if i >= bytes.len() || bytes[i] != b'[' {
                continue;
            }
            let Some(attr_close) = find_matching_bracket(text, i) else {
                continue;
            };
            let attr_inner = text[i + 1..attr_close].trim();
            let Some((path, args_inner)) = split_attr_call(attr_inner) else {
                continue;
            };
            let attr_ident = path.rsplit("::").next().unwrap_or(path).trim();
            if !matches!(attr_ident, "attune" | "rite" | "token_target_features") {
                continue;
            }
            let Some(features) = resolve_macro_attr_features(self.reg, args_inner) else {
                continue;
            };
            if features.is_empty() {
                continue;
            }
            let after_attrs = skip_attrs_and_ws(text, attr_close + 1);
            if let Some((start, end, true)) = find_guarded_span(text, after_attrs) {
                spans.push(GatingSpan {
                    start,
                    end,
                    features,
                    origin: format!("#[{attr_ident}]"),
                });
            }
        }
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

/// Structural soundness rules for magetypes sources, beyond intrinsic
/// gating. Each rule fences off a way the token-as-proof model could be
/// bypassed without any intrinsic-vs-feature mismatch being visible.
fn structural_rules(rel: &str, text: &str, errors: &mut Vec<String>) {
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
        _ => &[],
    };
    for m in detection.find_iter(text) {
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
        if !in_slice_selector {
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
        let trait_re = Regex::new(r"\btrait\s+\w+Backend\b[^{]*").expect("trait regex");
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

/// Full-repo verification: walk the scan roots, aggregate, apply floors.
pub fn verify(reg: &Registry) -> Result<()> {
    println!("=== Intrinsic Soundness Verification (structure-aware) ===\n");

    let scanner = Scanner::new(reg)?;
    println!(
        "Loaded {} intrinsics from stdarch database",
        scanner.db.len()
    );

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
    let mut stats: BTreeMap<String, usize> = BTreeMap::new();
    let mut allowlisted_used: BTreeMap<String, usize> = BTreeMap::new();

    for file in &files {
        let raw = fs::read_to_string(file)
            .with_context(|| format!("Failed to read {}", file.display()))?;
        let rel = file.display().to_string();
        let scan = scanner.scan_source(&rel, &raw);
        errors.extend(scan.errors);
        if scan.verified > 0 {
            stats.insert(rel.clone(), scan.verified);
        }
        for (name, count) in scan.allowlisted {
            *allowlisted_used.entry(name).or_default() += count;
        }
    }

    // ---- Report ----
    let total_verified: usize = stats.values().sum();

    println!("\nPer-file verified intrinsic calls:");
    for (file, verified) in &stats {
        println!("  {:>5}  {}", verified, file);
    }
    if !allowlisted_used.is_empty() {
        println!("\nAllowlisted (audited, not feature-verified):");
        let allow: BTreeMap<&str, &str> = ALLOWLIST.iter().copied().collect();
        for (name, count) in &allowlisted_used {
            let why = allow.get(name.as_str()).copied().unwrap_or("");
            println!("  {} ×{} — {}", name, count, why);
        }
    }
    println!(
        "\nTotals: {} verified across {} files scanned",
        total_verified,
        files.len()
    );

    // ---- Vacuous-pass guards ----
    for (file, floor) in REQUIRED_FILE_FLOORS {
        let seen = stats.get(*file).copied().unwrap_or(0);
        if seen < *floor {
            errors.push(format!(
                "VACUOUS PASS GUARD: {} verified only {} intrinsic calls (floor {}). \
                 If a refactor legitimately moved/removed this code, update \
                 REQUIRED_FILE_FLOORS in xtask/src/soundness.rs in the same commit.",
                file, seen, floor
            ));
        }
    }
    if total_verified < MIN_VERIFIED_CALLS {
        errors.push(format!(
            "VACUOUS PASS GUARD: only {} total verified intrinsic calls (floor {}). \
             The scanner may no longer be seeing the generated backends.",
            total_verified, MIN_VERIFIED_CALLS
        ));
    }

    if errors.is_empty() {
        println!("\n✓ Soundness verification PASSED");
        println!("  Every intrinsic call is gated by a context proving its required features.");
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

/// Scan forward from a fn name to the end of its signature. Returns
/// `(index, has_body)` where index points at the body `{` or the trailing
/// `;` of a bodyless declaration. Bracket depth is tracked so the `;` in
/// array types like `&[f32; 8]` doesn't terminate the signature early.
fn find_signature_end(text: &str, from: usize) -> Option<(usize, bool)> {
    let bytes = text.as_bytes();
    let mut i = from;
    let mut bracket_depth = 0i32;
    // Hard cap: no real signature in this codebase approaches this length;
    // prevents quadratic blowup on malformed input.
    let cap = (from + 4096).min(bytes.len());
    while i < cap {
        match bytes[i] {
            b'[' | b'(' => bracket_depth += 1,
            b']' | b')' => bracket_depth -= 1,
            b'{' => return Some((i, true)),
            b';' if bracket_depth == 0 => return Some((i, false)),
            _ => {}
        }
        i += 1;
    }
    None
}

/// What immediately precedes an identifier, for call-site classification.
enum Preceding {
    /// `fn name(` — a definition, not a call.
    FnKeyword,
    /// `.name(` — a method call on a value, never a bare intrinsic.
    Dot,
    /// `seg::name(` where `seg` is a `core::arch` module — an intrinsic call.
    ArchPath,
    /// `seg::name(` where `seg` is anything else (`Self`, a type, a module
    /// of ours) — an associated/module call, not an intrinsic.
    NonArchPath,
    /// Plain `name(` — resolved by import; treated as a potential intrinsic.
    Bare,
}

fn preceding_context(text: &str, ident_start: usize, arch_aliases: &HashSet<String>) -> Preceding {
    let bytes = text.as_bytes();
    let mut i = ident_start;
    while i > 0 && bytes[i - 1].is_ascii_whitespace() {
        i -= 1;
    }
    if i >= 2 && &bytes[i - 2..i] == b"fn" && (i == 2 || !bytes[i - 3].is_ascii_alphanumeric()) {
        return Preceding::FnKeyword;
    }
    if i >= 1 && bytes[i - 1] == b'.' {
        return Preceding::Dot;
    }
    if i >= 2 && &bytes[i - 2..i] == b"::" {
        // Walk back over the preceding path segment.
        let mut j = i - 2;
        while j > 0 && (bytes[j - 1].is_ascii_alphanumeric() || bytes[j - 1] == b'_') {
            j -= 1;
        }
        let seg = &text[j..i - 2];
        if is_arch_mod_name(seg) || arch_aliases.contains(seg) {
            return Preceding::ArchPath;
        }
        return Preceding::NonArchPath;
    }
    Preceding::Bare
}

fn is_arch_mod_name(seg: &str) -> bool {
    matches!(
        seg,
        "x86_64" | "x86" | "aarch64" | "arm" | "wasm32" | "wasm"
    )
}

/// Collect aliases bound to `core::arch::<arch>` / `std::arch::<arch>` via `use` items.
fn arch_path_aliases(text: &str) -> HashSet<String> {
    use syn::visit::{self, Visit};
    struct Collector(HashSet<String>);

    fn walk_tree(tree: &syn::UseTree, arch_mod: bool, in_arch: bool, out: &mut HashSet<String>) {
        match tree {
            syn::UseTree::Path(p) => {
                let ident = p.ident.to_string();
                if arch_mod {
                    if is_arch_mod_name(&ident) {
                        walk_tree(&p.tree, false, true, out);
                    }
                } else if ident == "arch" {
                    walk_tree(&p.tree, true, false, out);
                } else {
                    walk_tree(&p.tree, false, false, out);
                }
            }
            syn::UseTree::Group(g) => {
                for item in &g.items {
                    walk_tree(item, arch_mod, in_arch, out);
                }
            }
            syn::UseTree::Rename(r)
                if (arch_mod && is_arch_mod_name(&r.ident.to_string()))
                    || (in_arch && r.ident == "self") =>
            {
                out.insert(r.rename.to_string());
            }
            _ => {}
        }
    }

    impl<'ast> Visit<'ast> for Collector {
        fn visit_item_use(&mut self, i: &'ast syn::ItemUse) {
            // Only treat top-level `core::arch::...` / `std::arch::...` paths cleanly.
            if let syn::UseTree::Path(root) = &i.tree
                && matches!(root.ident.to_string().as_str(), "core" | "std")
            {
                walk_tree(&root.tree, false, false, &mut self.0);
            }
            visit::visit_item_use(self, i);
        }
    }

    let mut collector = Collector(HashSet::new());
    if let Ok(file) = syn::parse_file(text) {
        collector.visit_file(&file);
    }
    collector.0
}

/// Strip the top-level `-> ReturnType [where ...]` suffix from a `fn` signature slice
/// so return types mentioning tokens never grant feature access.
fn trim_return_type(sig: &str) -> &str {
    let bytes = sig.as_bytes();
    let mut paren_depth = 0i32;
    let mut bracket_depth = 0i32;
    let mut saw_params = false;
    let mut i = 0;
    while i < bytes.len() {
        match bytes[i] {
            b'(' => {
                paren_depth += 1;
                saw_params = true;
            }
            b')' => paren_depth -= 1,
            b'[' => bracket_depth += 1,
            b']' => bracket_depth -= 1,
            b'-' if saw_params
                && paren_depth == 0
                && bracket_depth == 0
                && i + 1 < bytes.len()
                && bytes[i + 1] == b'>' =>
            {
                return &sig[..i];
            }
            _ => {}
        }
        i += 1;
    }
    sig
}

/// True when `fn` at `fn_pos` carries an attribute enabling target features
/// (`#[arcane]`, `#[attune]`, `#[rite]`, `#[target_feature]`, `#[token_target_features]`).
fn has_wasm_fn_attr(text: &str, fn_pos: usize) -> bool {
    let bytes = text.as_bytes();
    let mut i = fn_pos;
    loop {
        while i > 0 && bytes[i - 1].is_ascii_whitespace() {
            i -= 1;
        }
        if i == 0 {
            return false;
        }
        // Closing `]` of an attribute `#[...]`
        if bytes[i - 1] == b']' {
            let Some(open) = find_matching_bracket_backwards(text, i - 1) else {
                return false;
            };
            let mut j = open;
            while j > 0 && bytes[j - 1].is_ascii_whitespace() {
                j -= 1;
            }
            if j == 0 || bytes[j - 1] != b'#' {
                return false;
            }
            let inner = text[open + 1..i - 1].trim();
            let path = inner.split(['(', ' ']).next().unwrap_or("").trim();
            let ident = path.rsplit("::").next().unwrap_or(path);
            if matches!(
                ident,
                "arcane" | "attune" | "rite" | "target_feature" | "token_target_features"
            ) {
                return true;
            }
            i = j - 1;
            continue;
        }
        // `pub(crate)` / visibility parens
        if bytes[i - 1] == b')' {
            let mut depth = 1i32;
            let mut k = i - 1;
            while k > 0 && depth > 0 {
                k -= 1;
                if bytes[k] == b')' {
                    depth += 1;
                } else if bytes[k] == b'(' {
                    depth -= 1;
                }
            }
            if depth == 0 {
                i = k;
                continue;
            }
            return false;
        }
        // ABI string literal in `extern "C"`
        if bytes[i - 1] == b'"' {
            let mut k = i - 1;
            while k > 0 {
                k -= 1;
                if bytes[k] == b'"' && (k == 0 || bytes[k - 1] != b'\\') {
                    i = k;
                    break;
                }
            }
            continue;
        }
        // Qualifiers (`pub`, `unsafe`, `const`, `async`, `extern`, `default`)
        if bytes[i - 1].is_ascii_alphanumeric() || bytes[i - 1] == b'_' {
            let end = i;
            while i > 0 && (bytes[i - 1].is_ascii_alphanumeric() || bytes[i - 1] == b'_') {
                i -= 1;
            }
            if matches!(
                &text[i..end],
                "pub" | "unsafe" | "const" | "async" | "extern" | "default"
            ) {
                continue;
            }
        }
        return false;
    }
}

fn find_matching_bracket(text: &str, open: usize) -> Option<usize> {
    let bytes = text.as_bytes();
    debug_assert_eq!(bytes[open], b'[');
    let mut depth = 0i32;
    let mut in_str = false;
    let mut i = open;
    while i < bytes.len() {
        let b = bytes[i];
        if in_str {
            if b == b'\\' {
                i += 2;
                continue;
            }
            if b == b'"' {
                in_str = false;
            }
        } else {
            match b {
                b'"' => in_str = true,
                b'[' => depth += 1,
                b']' => {
                    depth -= 1;
                    if depth == 0 {
                        return Some(i);
                    }
                }
                _ => {}
            }
        }
        i += 1;
    }
    None
}

fn find_matching_bracket_backwards(text: &str, close: usize) -> Option<usize> {
    let bytes = text.as_bytes();
    debug_assert_eq!(bytes[close], b']');
    let mut depth = 0i32;
    let mut i = close + 1;
    while i > 0 {
        i -= 1;
        match bytes[i] {
            b']' => depth += 1,
            b'[' => {
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

fn split_attr_call(inner: &str) -> Option<(&str, &str)> {
    let lparen = inner.find('(')?;
    let rparen = inner.rfind(')')?;
    if rparen < lparen {
        return None;
    }
    Some((inner[..lparen].trim(), inner[lparen + 1..rparen].trim()))
}

fn resolve_macro_attr_features(reg: &Registry, args: &str) -> Option<HashSet<String>> {
    let mut items = Vec::new();
    let mut start = 0usize;
    let mut depth = 0i32;
    for (idx, b) in args.bytes().enumerate() {
        match b {
            b'(' | b'[' => depth += 1,
            b')' | b']' => depth -= 1,
            b',' if depth == 0 => {
                items.push(args[start..idx].trim());
                start = idx + 1;
            }
            _ => {}
        }
    }
    let last = args[start..].trim();
    if !last.is_empty() {
        items.push(last);
    }

    let mut intersection: Option<HashSet<String>> = None;
    for item in items {
        if item.is_empty() {
            continue;
        }
        let head = item.split(['(', '=', ' ']).next().unwrap_or(item).trim();
        if matches!(
            head,
            "wrap"
                | "dispatch"
                | "nested"
                | "in_impl"
                | "in_trait"
                | "import_intrinsics"
                | "import_magetypes"
                | "suppress_const_test"
                | "crate"
                | "inline"
                | "cfg"
                | "pub"
                | "define"
                | "_self"
                | "self"
        ) || item.contains('(')
            || item.contains('=')
        {
            continue;
        }
        if head.starts_with("_*") {
            return None;
        }
        let clean = head.trim_start_matches('+').trim_start_matches('_');
        if matches!(clean, "scalar" | "default") {
            return None;
        }
        let token = reg.token.iter().find(|t| {
            t.short_name.as_deref() == Some(clean)
                || t.extraction_aliases.iter().any(|a| a == clean)
        })?;
        if token.runtime_probe {
            return None;
        }
        let set: HashSet<String> = token.features.iter().cloned().collect();
        intersection = Some(match intersection {
            None => set,
            Some(prev) => prev.intersection(&set).cloned().collect(),
        });
    }
    intersection
}

fn skip_attrs_and_ws(text: &str, mut idx: usize) -> usize {
    let bytes = text.as_bytes();
    loop {
        while idx < bytes.len() && bytes[idx].is_ascii_whitespace() {
            idx += 1;
        }
        if idx < bytes.len() && bytes[idx] == b'#' {
            let mut j = idx + 1;
            if j < bytes.len() && bytes[j] == b'!' {
                j += 1;
            }
            while j < bytes.len() && bytes[j].is_ascii_whitespace() {
                j += 1;
            }
            if j < bytes.len()
                && bytes[j] == b'['
                && let Some(close) = find_matching_bracket(text, j)
            {
                idx = close + 1;
                continue;
            }
        }
        return idx;
    }
}

/// Find `(start, end, is_block)` of the item or statement following an attribute,
/// stopping at `;` when a semicolon-terminated item precedes any block `{ ... }`.
fn find_guarded_span(text: &str, from: usize) -> Option<(usize, usize, bool)> {
    let bytes = text.as_bytes();
    let mut paren_depth = 0i32;
    let mut bracket_depth = 0i32;
    let mut i = from;
    while i < bytes.len() {
        match bytes[i] {
            b'(' => paren_depth += 1,
            b')' => paren_depth -= 1,
            b'[' => bracket_depth += 1,
            b']' => bracket_depth -= 1,
            b'{' if paren_depth == 0 && bracket_depth == 0 => {
                let close = find_matching_brace(text, i)?;
                return Some((from, close, true));
            }
            b';' if paren_depth == 0 && bracket_depth == 0 => {
                return Some((from, i, false));
            }
            _ => {}
        }
        i += 1;
    }
    None
}

fn cfg_target_feature_spans(text: &str, spans: &mut Vec<GatingSpan>) {
    fn extract_features(meta: &syn::Meta, out: &mut HashSet<String>) {
        match meta {
            syn::Meta::NameValue(nv) if nv.path.is_ident("target_feature") => {
                if let syn::Expr::Lit(syn::ExprLit {
                    lit: syn::Lit::Str(s),
                    ..
                }) = &nv.value
                {
                    out.insert(s.value());
                }
            }
            syn::Meta::List(list) if list.path.is_ident("all") => {
                if let Ok(nested) = list.parse_args_with(
                    syn::punctuated::Punctuated::<syn::Meta, syn::Token![,]>::parse_terminated,
                ) {
                    for child in nested {
                        extract_features(&child, out);
                    }
                }
            }
            _ => {}
        }
    }

    let mut search = 0;
    while let Some(rel_hash) = text[search..].find('#') {
        let hash = search + rel_hash;
        search = hash + 1;
        let bytes = text.as_bytes();
        let mut i = hash + 1;
        while i < bytes.len() && bytes[i].is_ascii_whitespace() {
            i += 1;
        }
        if i >= bytes.len() || bytes[i] != b'[' {
            continue;
        }
        let Some(attr_close) = find_matching_bracket(text, i) else {
            continue;
        };
        let attr_inner = text[i + 1..attr_close].trim();
        let Some((path, args_inner)) = split_attr_call(attr_inner) else {
            continue;
        };
        if path != "cfg" {
            continue;
        }
        let Ok(cfg_arg) = syn::parse_str::<syn::Meta>(args_inner) else {
            continue;
        };
        let mut features = HashSet::new();
        extract_features(&cfg_arg, &mut features);
        if features.is_empty() {
            continue;
        }
        let after_attrs = skip_attrs_and_ws(text, attr_close + 1);
        if let Some((start, end, _is_block)) = find_guarded_span(text, after_attrs) {
            spans.push(GatingSpan {
                start,
                end,
                features,
                origin: "#[cfg(target_feature)]".to_string(),
            });
        }
    }
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

    fn scan(src: &str) -> FileScan {
        // Fixtures default to a backend-impls path (per-block SAFETY
        // comments not required there); satisfy the file-header rule.
        let with_header = format!("//! # Safety (audit contract — test fixture)\n{src}");
        scan_at("magetypes/src/simd/impls/test_input.rs", &with_header)
    }

    fn scan_at(rel: &str, src: &str) -> FileScan {
        // The CSV path in load_intrinsic_database is repo-root-relative;
        // tests run with CWD=xtask, so chdir up if needed.
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap();
        std::env::set_current_dir(root).expect("chdir to repo root");
        let reg = test_registry();
        let scanner = Scanner::new(&reg).expect("scanner");
        scanner.scan_source(rel, src)
    }

    #[test]
    fn hidden_cpu_detection_is_rejected_in_magetypes() {
        for call in [
            "X64V3Token::summon()",
            "<X64V3Token as SimdToken>::summon()",
            "std::is_x86_feature_detected!(\"fma\")",
            "std::arch::is_aarch64_feature_detected!(\"fp16\")",
            "core::arch::x86_64::__cpuid(1)",
        ] {
            let mut errors = Vec::new();
            structural_rules(
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
            "magetypes/src/nostd_math.rs",
            "fn arithmetic(t: Arm64V2Token) { let _ = t.neon(); }",
            &mut errors,
        );
        assert!(errors.is_empty(), "{errors:?}");
        structural_rules(
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
        let mut errors = Vec::new();
        structural_rules(
            "magetypes/src/simd/generic/convert_f16.rs",
            "fn neon_f16_decode_select() { if true { Arm64V2Token::summon(); } }",
            &mut errors,
        );
        assert!(errors.is_empty(), "{errors:?}");
        structural_rules(
            "magetypes/src/simd/generic/convert_f16.rs",
            "fn neon_f16_decode_select() {} fn register_convert() { Arm64V2Token::summon(); }",
            &mut errors,
        );
        assert_eq!(errors.len(), 1, "{errors:?}");
        assert!(errors[0].contains("hidden CPU detection"));
    }

    #[test]
    fn soft_fma_detection_exception_is_one_function() {
        let mut errors = Vec::new();
        structural_rules(
            "magetypes/src/soft_fma.rs",
            "mod hw { pub(super) fn token() -> Option<Token> { Token::summon() } }",
            &mut errors,
        );
        assert!(errors.is_empty(), "{errors:?}");
        structural_rules(
            "magetypes/src/soft_fma.rs",
            "fn token() {} fn fmaf_lanes() { X64V3Token::summon(); }",
            &mut errors,
        );
        assert_eq!(errors.len(), 1, "{errors:?}");
        assert!(errors[0].contains("hidden CPU detection"));
        let mut errors = Vec::new();
        structural_rules(
            "magetypes/src/nostd_math.rs",
            "fn token() { X64V3Token::summon(); }",
            &mut errors,
        );
        assert_eq!(errors.len(), 1, "the exception is per file: {errors:?}");
    }

    #[test]
    fn catches_feature_mismatch_in_impl() {
        let scan = scan(
            "impl F32x4Backend for archmage::X64V3Token {\n\
             fn f(self) { let _ = _mm512_setzero_ps(); }\n\
             }\n",
        );
        assert_eq!(scan.errors.len(), 1, "{:?}", scan.errors);
        assert!(
            scan.errors[0].contains("FEATURE MISMATCH"),
            "{:?}",
            scan.errors
        );
        assert!(scan.errors[0].contains("_mm512_setzero_ps"));
    }

    #[test]
    fn catches_ungated_intrinsic() {
        let scan = scan("fn helper() { let _ = vaddq_f32(a, b); }\n");
        assert_eq!(scan.errors.len(), 1, "{:?}", scan.errors);
        assert!(scan.errors[0].contains("UNGATED"), "{:?}", scan.errors);
    }

    #[test]
    fn catches_ungated_in_trait_default_body() {
        let scan = scan(
            "pub trait F32x8Backend: SimdToken {\n\
             fn rcp(self, a: Self::Repr) -> Self::Repr { _mm256_rcp_ps(a) }\n\
             }\n",
        );
        // Trait blocks are not gating contexts — Self could be ScalarToken.
        assert_eq!(scan.errors.len(), 1, "{:?}", scan.errors);
        assert!(scan.errors[0].contains("UNGATED"), "{:?}", scan.errors);
    }

    #[test]
    fn ungated_x86_baseline_intrinsics_are_sound() {
        // sse/sse2 are the x86-64 ABI baseline — sound without any proof.
        let scan = scan("fn helper(a: __m128) -> __m128 { _mm_rcp_ps(a) }\n");
        assert!(scan.errors.is_empty(), "{:?}", scan.errors);
        assert_eq!(scan.verified, 1);
    }

    #[test]
    fn accepts_gated_by_impl_token() {
        let scan = scan(
            "impl F32x8Backend for archmage::X64V3Token {\n\
             fn add(self, a: R, b: R) -> R { _mm256_add_ps(a, b) }\n\
             }\n",
        );
        assert!(scan.errors.is_empty(), "{:?}", scan.errors);
        assert_eq!(scan.verified, 1);
    }

    #[test]
    fn accepts_gated_by_target_feature_attr() {
        let scan = scan(
            "#[target_feature(enable = \"avx2,fma\")]\n\
             fn kernel(a: &[f32; 8]) -> f32 { let v = _mm256_fmadd_ps(x, y, z); 0.0 }\n",
        );
        assert!(scan.errors.is_empty(), "{:?}", scan.errors);
        assert_eq!(scan.verified, 1);
    }

    #[test]
    fn accepts_gated_by_token_param_with_array_types_in_signature() {
        // Regression: the `;` inside `&[f32; 8]` must not truncate the
        // signature scan (it did, in the first version of this scanner).
        let scan = scan(
            "fn process(token: X64V3Token, a: &[f32; 8], b: &[f32; 8]) -> [f32; 8] {\n\
             let va = _mm256_loadu_ps(a);\n\
             [0.0; 8]\n\
             }\n",
        );
        assert!(scan.errors.is_empty(), "{:?}", scan.errors);
        assert_eq!(scan.verified, 1);
    }

    #[test]
    fn union_of_impl_and_param_token_contexts() {
        // V3 impl + V4-token-taking fast-path method: V4 features available.
        let scan = scan(
            "impl I64x2Backend for archmage::X64V3Token {\n\
             fn min_fast(self, a: R, b: R, _t: X64V4Token) -> R {\n\
             _mm_min_epi64(a, b)\n\
             }\n\
             }\n",
        );
        assert!(scan.errors.is_empty(), "{:?}", scan.errors);
        assert_eq!(scan.verified, 1);
    }

    #[test]
    fn method_name_collisions_are_not_intrinsics() {
        // magetypes width traits name methods after WASM intrinsics.
        let scan = scan(
            "impl WidthDispatch for archmage::NeonToken {\n\
             fn f32x4_splat(self, v: f32) -> Self::F32x4 { f32x4::splat_t(self, v) }\n\
             fn call_it(self) { let _ = self.f32x4_splat(1.0); }\n\
             }\n",
        );
        assert!(scan.errors.is_empty(), "{:?}", scan.errors);
        assert_eq!(scan.verified, 0);
    }

    #[test]
    fn unknown_intrinsic_shape_is_reported() {
        let scan = scan(
            "impl F32x4Backend for archmage::NeonToken {\n\
             fn f(self) { let _ = vfrobnicateq_f32(a); }\n\
             }\n",
        );
        assert_eq!(scan.errors.len(), 1, "{:?}", scan.errors);
        assert!(
            scan.errors[0].contains("UNKNOWN INTRINSIC"),
            "{:?}",
            scan.errors
        );
    }

    #[test]
    fn comments_are_ignored() {
        let scan = scan(
            "// let _ = _mm512_setzero_ps();\n\
             /* vaddq_f32(a, b) */\n\
             /// doc: _mm256_add_ps(x, y)\n\
             fn nothing() {}\n",
        );
        assert!(scan.errors.is_empty(), "{:?}", scan.errors);
        assert_eq!(scan.verified, 0);
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
    }

    #[test]
    fn wasm_alias_cfg_and_return_type_guards() {
        // Aliased core::arch::wasm32 path is recognized and caught when ungated.
        let bad_alias = scan(
            "use core::arch::wasm32::{self as w, v128};\n\
             fn bad(a: v128, b: v128) -> v128 { w::f32x4_add(a, b) }\n",
        );
        assert_eq!(bad_alias.errors.len(), 1, "{:?}", bad_alias.errors);
        assert!(bad_alias.errors[0].contains("UNGATED"));

        // Return type mentioning a token never grants features.
        let ret_tok =
            scan("fn forge() -> X64V3Token { let _ = _mm256_add_ps(x, y); unreachable!() }\n");
        assert_eq!(ret_tok.errors.len(), 1, "{:?}", ret_tok.errors);
        assert!(ret_tok.errors[0].contains("UNGATED"));

        // Free fn taking Wasm128Token without #[arcane]/#[attune]/#[rite]/#[target_feature] is rejected.
        let bare_wasm = scan(
            "use core::arch::wasm32::{self as w, v128};\n\
             fn bad(_t: Wasm128Token, a: v128, b: v128) -> v128 { w::f32x4_add(a, b) }\n",
        );
        assert_eq!(bare_wasm.errors.len(), 1, "{:?}", bare_wasm.errors);
        assert!(bare_wasm.errors[0].contains("UNGATED"));

        // #[attune(wasm128)] enables simd128; relaxed-simd requires #[cfg(all(...))] or wasm128_relaxed.
        let attune_relaxed_bad = scan(
            "use core::arch::wasm32::{self as w, v128};\n\
             #[archmage::attune(wasm128)]\n\
             fn bad(a: v128, b: v128, c: v128) -> v128 { w::f32x4_relaxed_madd(a, b, c) }\n",
        );
        assert!(
            attune_relaxed_bad
                .errors
                .iter()
                .any(|e| e.contains("FEATURE MISMATCH")),
            "{:?}",
            attune_relaxed_bad.errors
        );

        // #[cfg(not(...))] and #[cfg(any(...))] never grant features, nor does #[cfg(...)] use ...; leak.
        for bad_cfg in [
            "#[cfg(not(target_feature = \"simd128\"))]\nfn bad(a: v128, b: v128) -> v128 { core::arch::wasm32::f32x4_add(a, b) }\n",
            "#[cfg(any(target_feature = \"simd128\", target_arch = \"x86_64\"))]\nfn bad(a: v128, b: v128) -> v128 { core::arch::wasm32::f32x4_add(a, b) }\n",
            "#[cfg(all(target_arch = \"wasm32\", target_feature = \"simd128\"))]\nuse core::arch::wasm32 as w;\nfn bad(a: w::v128, b: w::v128) -> w::v128 { w::f32x4_add(a, b) }\n",
        ] {
            let s = scan(bad_cfg);
            assert!(!s.errors.is_empty(), "accepted {bad_cfg}");
        }

        // Positive #[cfg(all(target_feature = "simd128", target_feature = "relaxed-simd"))] grants both.
        let ok_cfg = scan(
            "#[cfg(all(target_feature = \"simd128\", target_feature = \"relaxed-simd\"))]\n\
             mod relaxed {\n\
                 use core::arch::wasm32::{self as w, v128};\n\
                 pub fn f(a: v128, b: v128, c: v128) -> v128 { w::f32x4_relaxed_madd(a, b, c) }\n\
             }\n",
        );
        assert!(ok_cfg.errors.is_empty(), "{:?}", ok_cfg.errors);
        assert_eq!(ok_cfg.verified, 1);
    }

    #[test]
    fn full_repo_scan_passes_and_meets_floors() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap();
        std::env::set_current_dir(root).expect("chdir to repo root");
        let reg = test_registry();
        verify(&reg).expect("repo soundness verification");
    }
}
