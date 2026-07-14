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
//!   (`MaybeUninit`, `mem::zeroed`, forging), `transmute` outside the
//!   audited backend impls, token-less wrapper constructors
//!   (`Default`/serde/bytemuck), or backend-trait methods without a `self`
//!   receiver (which would allow calling intrinsics via UFCS without
//!   holding a token).
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
//!   transmutes) are counted and reported by `cargo xtask audit`
//!   (`unsafe`-inventory) and exercised under Miri, not proven here.

use anyhow::{Context, Result, bail};
use regex::Regex;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};

use crate::IntrinsicEntry;
use crate::registry::Registry;

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
    ("magetypes/src/simd/impls/x86_v3.rs", 750), // measured 993
    ("magetypes/src/simd/impls/x86_v4.rs", 790), // measured 1056
    ("magetypes/src/simd/impls/arm_neon.rs", 1080), // measured 1445
    ("magetypes/src/simd/impls/wasm128.rs", 700), // measured 942
    ("magetypes/src/simd/generic/cross_width.rs", 6), // measured 8
    ("magetypes/src/simd/generic/convert_f16.rs", 18), // measured 26
];

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
            match preceding_context(&text, pos) {
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

        // (3) fn signatures mentioning token types (parameters or generics).
        for m in self.fn_re.find_iter(text) {
            let sig_start = m.end();
            let Some((sig_end, has_body)) = find_signature_end(text, sig_start) else {
                continue;
            };
            if !has_body {
                continue;
            }
            let signature = &text[sig_start..sig_end];
            let mut features: HashSet<String> = HashSet::new();
            let mut names: Vec<&str> = Vec::new();
            for cap in self.token_name_re.captures_iter(signature) {
                let name = cap.get(1).unwrap().as_str();
                if let Some(token) = self.reg.find_token(name) {
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

        spans
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
    let is_backend_impl = rel.starts_with("magetypes/src/simd/impls/");

    // Token fabrication routes. Tokens must only come from summon() or an
    // explicit archmage forge — magetypes has no business creating them.
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

    // `transmute` is allowed only inside the backend impls (array <-> Repr
    // bitcasts inside impl-for-token blocks, verified by the intrinsic
    // scanner's context machinery + Miri). Elsewhere in magetypes the
    // audited idiom is `transmute_copy` on size-asserted types; a bare
    // `transmute` outside impls/ is a red flag.
    if !is_backend_impl {
        let re = Regex::new(r"\btransmute\b").expect("transmute regex");
        for m in re.find_iter(text) {
            // `transmute_copy` is a different, length-checked-by-Dst idiom.
            if text[m.end()..].starts_with("_copy") {
                continue;
            }
            errors.push(format!(
                "{}:{}: STRUCTURAL RULE: bare `transmute` outside simd/impls/ — use \
                 the audited size-asserted byte-cast helpers or move the bitcast \
                 into a backend impl.",
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

fn preceding_context(text: &str, ident_start: usize) -> Preceding {
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
        return match seg {
            "x86_64" | "x86" | "aarch64" | "arm" | "wasm32" | "wasm" => Preceding::ArchPath,
            _ => Preceding::NonArchPath,
        };
    }
    Preceding::Bare
}

/// 1-based line number of a byte offset.
fn line_of(text: &str, offset: usize) -> usize {
    text.as_bytes()[..offset]
        .iter()
        .filter(|&&b| b == b'\n')
        .count()
        + 1
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
        scan_at("magetypes/src/simd/impls/test_input.rs", src)
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
    fn catches_feature_mismatch_in_impl() {
        let scan = scan(
            "impl F32x4Backend for archmage::X64V3Token {\n\
             fn f(self) { let _ = unsafe { _mm512_setzero_ps() }; }\n\
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
        let scan = scan("fn helper() { let _ = unsafe { vaddq_f32(a, b) }; }\n");
        assert_eq!(scan.errors.len(), 1, "{:?}", scan.errors);
        assert!(scan.errors[0].contains("UNGATED"), "{:?}", scan.errors);
    }

    #[test]
    fn catches_ungated_in_trait_default_body() {
        let scan = scan(
            "pub trait F32x8Backend: SimdToken {\n\
             fn rcp(self, a: Self::Repr) -> Self::Repr { unsafe { _mm256_rcp_ps(a) } }\n\
             }\n",
        );
        // Trait blocks are not gating contexts — Self could be ScalarToken.
        assert_eq!(scan.errors.len(), 1, "{:?}", scan.errors);
        assert!(scan.errors[0].contains("UNGATED"), "{:?}", scan.errors);
    }

    #[test]
    fn ungated_x86_baseline_intrinsics_are_sound() {
        // sse/sse2 are the x86-64 ABI baseline — sound without any proof.
        let scan = scan("fn helper(a: __m128) -> __m128 { unsafe { _mm_rcp_ps(a) } }\n");
        assert!(scan.errors.is_empty(), "{:?}", scan.errors);
        assert_eq!(scan.verified, 1);
    }

    #[test]
    fn accepts_gated_by_impl_token() {
        let scan = scan(
            "impl F32x8Backend for archmage::X64V3Token {\n\
             fn add(self, a: R, b: R) -> R { unsafe { _mm256_add_ps(a, b) } }\n\
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
             unsafe { _mm_min_epi64(a, b) }\n\
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
             fn f32x4_splat(self, v: f32) -> Self::F32x4 { f32x4::splat(self, v) }\n\
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
             fn f(self) { let _ = unsafe { vfrobnicateq_f32(a) }; }\n\
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
    fn structural_rule_no_bare_transmute_outside_impls() {
        let scan = scan_at(
            "magetypes/src/simd/generic/foo.rs",
            "fn f(x: [f32; 4]) -> Y { unsafe { core::mem::transmute(x) } }\n",
        );
        assert!(
            scan.errors.iter().any(|e| e.contains("bare `transmute`")),
            "{:?}",
            scan.errors
        );
        // transmute_copy is the audited idiom and stays allowed.
        let ok = scan_at(
            "magetypes/src/simd/generic/foo.rs",
            "fn f(x: &[u8; 16]) -> Y { unsafe { core::mem::transmute_copy(x) } }\n",
        );
        assert!(
            !ok.errors.iter().any(|e| e.contains("STRUCTURAL RULE")),
            "{:?}",
            ok.errors
        );
    }

    #[test]
    fn structural_rule_transmute_allowed_in_backend_impls() {
        let scan = scan_at(
            "magetypes/src/simd/impls/x86_v3.rs",
            "impl F32x4Backend for archmage::X64V3Token {\n\
             fn from_array(self, arr: [f32; 4]) -> R { unsafe { core::mem::transmute(arr) } }\n\
             }\n",
        );
        assert!(scan.errors.is_empty(), "{:?}", scan.errors);
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
    fn full_repo_scan_passes_and_meets_floors() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap();
        std::env::set_current_dir(root).expect("chdir to repo root");
        let reg = test_registry();
        verify(&reg).expect("repo soundness verification");
    }
}
