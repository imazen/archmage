//! Post-pass over generated backend impl files: rewrite each trait-impl
//! method into the nested-`#[target_feature]` form so value intrinsics are
//! **compiler-verified** against the token's feature set (Rust 1.87+ makes
//! them safe inside a matching `#[target_feature]` fn, and E0133 rejects a
//! mismatched one), leaving exactly one mechanical boundary
//! `unsafe { __tf(...) }` per method whose justification is uniform: the
//! receiver token proves the features at runtime.
//!
//! Before:
//! ```text
//! #[inline(always)]
//! fn splat(self, v: f32) -> __m128 {
//!     unsafe { _mm_set1_ps(v) }
//! }
//! ```
//! After:
//! ```text
//! #[inline(always)]
//! fn splat(self, v: f32) -> __m128 {
//!     #[target_feature(enable = "sse,sse2,...,movbe")]
//!     #[inline]  // inline(always)+target_feature rejected: rust#145574
//!     fn __tf(_this: archmage::X64V3Token, v: f32) -> __m128 {
//!         _mm_set1_ps(v)
//!     }
//!     // SAFETY: `self` proves the enabled target features at runtime.
//!     unsafe { __tf(self, v) }
//! }
//! ```
//!
//! `unsafe` blocks whose content is genuinely unsafe regardless of target
//! features (raw-pointer loads/stores, `transmute`) are kept; all other
//! (value-intrinsic) blocks are unwrapped so the compiler checks them.
//! `#[target_feature]` still cannot go on safe trait impl methods (E0053,
//! re-verified on rustc 1.98.1), which is why the nested shape is required.

use std::path::Path;
use std::sync::OnceLock;

use crate::registry::Registry;

/// Substrings marking an `unsafe` block as genuinely unsafe (pointer or
/// layout operations) — these keep their `unsafe`. Everything else in these
/// generated files is value-intrinsic calls, safe inside `__tf`.
const KEEP_UNSAFE: &[&str] = &[
    "as_ptr",
    "as_mut_ptr",
    "transmute",
    "read_unaligned",
    "write_unaligned",
    "copy_nonoverlapping",
    "from_raw",
    "ptr::",
    "*const",
    "*mut",
    ".add(",
    ".offset(",
];

fn registry() -> &'static Registry {
    static REG: OnceLock<Registry> = OnceLock::new();
    REG.get_or_init(|| {
        Registry::load(Path::new("token-registry.toml"))
            .expect("tf_inner: token-registry.toml must load")
    })
}

/// Comma-joined `#[target_feature(enable = ...)]` list for a token name.
fn features_for(token: &str) -> String {
    let reg = registry();
    let feats = reg
        .features_for(token)
        .unwrap_or_else(|| panic!("tf_inner: unknown token {token}"));
    feats.join(",")
}

/// Apply the transform to one generated impl file.
pub fn apply(text: &str) -> String {
    let lines: Vec<&str> = text.lines().collect();
    let mut out = String::with_capacity(text.len() * 2);
    let mut current_token: Option<String> = None;
    let mut i = 0;

    while i < lines.len() {
        let line = lines[i];

        // Track which token the enclosing impl block targets.
        if line.starts_with("impl ") {
            current_token = line
                .split(" for archmage::")
                .nth(1)
                .map(|rest| rest.trim_end_matches([' ', '{']).to_string());
        }

        let is_method_start = line == "    #[inline(always)]"
            && lines.get(i + 1).is_some_and(|l| l.starts_with("    fn "));

        if let (true, Some(token)) = (is_method_start, current_token.clone()) {
            // Collect signature lines until the one that opens the body.
            let mut sig = String::new();
            let mut j = i + 1;
            loop {
                let l = lines[j];
                sig.push_str(l);
                sig.push('\n');
                j += 1;
                if l.trim_end().ends_with('{') {
                    break;
                }
            }
            // Collect body lines until the method's closing brace.
            let body_start = j;
            while lines[j] != "    }" {
                j += 1;
            }
            let body = lines[body_start..j].join("\n");

            if body.contains("unsafe") {
                out.push_str(&transform_method(&sig, &body, &token));
            } else {
                // Pure composition (no unsafe, no direct intrinsics needing
                // feature context) — leave untouched.
                out.push_str("    #[inline(always)]\n");
                out.push_str(&sig);
                out.push_str(&body);
                out.push_str("\n    }\n");
            }
            i = j + 1;
            continue;
        }

        out.push_str(line);
        out.push('\n');
        i += 1;
    }
    out
}

/// Split a parameter list on top-level commas.
fn split_params(params: &str) -> Vec<String> {
    let mut parts = Vec::new();
    let mut depth = 0i32;
    let mut cur = String::new();
    for c in params.chars() {
        match c {
            '(' | '[' | '<' => depth += 1,
            ')' | ']' | '>' => depth -= 1,
            ',' if depth == 0 => {
                parts.push(cur.trim().to_string());
                cur.clear();
                continue;
            }
            _ => {}
        }
        cur.push(c);
    }
    if !cur.trim().is_empty() {
        parts.push(cur.trim().to_string());
    }
    parts
}

/// Replace whole-word occurrences of `from` with `to` (identifier semantics).
fn replace_word(text: &str, from: &str, to: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let bytes = text.as_bytes();
    let mut i = 0;
    let is_ident = |b: u8| b == b'_' || b.is_ascii_alphanumeric();
    while let Some(pos) = text[i..].find(from) {
        let start = i + pos;
        let end = start + from.len();
        let ok_before = start == 0 || !is_ident(bytes[start - 1]);
        let ok_after = end == text.len() || !is_ident(bytes[end]);
        out.push_str(&text[i..start]);
        if ok_before && ok_after {
            out.push_str(to);
        } else {
            out.push_str(from);
        }
        i = end;
    }
    out.push_str(&text[i..]);
    out
}

/// Unwrap `unsafe { ... }` blocks that contain only value-intrinsic work;
/// keep blocks doing pointer/layout operations.
fn strip_value_unsafe(body: &str) -> String {
    let mut out = String::with_capacity(body.len());
    let bytes = body.as_bytes();
    let mut i = 0;
    while let Some(pos) = body[i..].find("unsafe") {
        let start = i + pos;
        let end_kw = start + "unsafe".len();
        // Must be the keyword (not part of an identifier) followed by `{`.
        let ok_before =
            start == 0 || !(bytes[start - 1] == b'_' || bytes[start - 1].is_ascii_alphanumeric());
        let after = body[end_kw..].trim_start();
        if !ok_before || !after.starts_with('{') {
            out.push_str(&body[i..end_kw]);
            i = end_kw;
            continue;
        }
        let brace_open = end_kw + (body[end_kw..].len() - after.len());
        // Brace-match to find the block's end.
        let mut depth = 0usize;
        let mut k = brace_open;
        let close = loop {
            assert!(
                k < bytes.len(),
                "tf_inner: unbalanced braces after `unsafe` at byte {start} in body:\n{body}"
            );
            match bytes[k] {
                b'{' => depth += 1,
                b'}' => {
                    depth -= 1;
                    if depth == 0 {
                        break k;
                    }
                }
                _ => {}
            }
            k += 1;
        };
        let content = &body[brace_open + 1..close];
        out.push_str(&body[i..start]);
        if KEEP_UNSAFE.iter().any(|m| content.contains(m)) {
            // Genuinely unsafe — keep verbatim.
            out.push_str(&body[start..=close]);
        } else if content.contains(';') || content.contains("let ") {
            // Multi-statement value block: keep the braces, drop `unsafe`.
            out.push_str(&body[brace_open..=close]);
        } else {
            // Single expression: unwrap fully.
            out.push_str(content.trim());
        }
        i = close + 1;
    }
    out.push_str(&body[i..]);
    out
}

fn transform_method(sig: &str, body: &str, token: &str) -> String {
    let token_path = format!("archmage::{token}");

    // --- Parse the signature ---------------------------------------------
    let sig_flat = sig.trim_end().trim_end_matches('{').trim_end();
    let after_fn = sig_flat
        .trim_start()
        .strip_prefix("fn ")
        .expect("method sig starts with fn");
    let name_end = after_fn.find(['(', '<']).expect("method sig has ( or <");
    let name = &after_fn[..name_end];

    // Optional generics `<...>` (only const generics occur in these files).
    let (generics, rest) = if after_fn[name_end..].starts_with('<') {
        let mut depth = 0i32;
        let mut end = name_end;
        for (off, c) in after_fn[name_end..].char_indices() {
            match c {
                '<' => depth += 1,
                '>' => {
                    depth -= 1;
                    if depth == 0 {
                        end = name_end + off + 1;
                        break;
                    }
                }
                _ => {}
            }
        }
        (&after_fn[name_end..end], &after_fn[end..])
    } else {
        ("", &after_fn[name_end..])
    };

    // Parameter list between the first balanced parens.
    let popen = rest.find('(').expect("param list");
    let mut depth = 0i32;
    let mut pclose = popen;
    for (off, c) in rest[popen..].char_indices() {
        match c {
            '(' => depth += 1,
            ')' => {
                depth -= 1;
                if depth == 0 {
                    pclose = popen + off;
                    break;
                }
            }
            _ => {}
        }
    }
    let params_text = &rest[popen + 1..pclose];
    let ret = rest[pclose + 1..].trim(); // e.g. `-> __m128` or empty

    let params = split_params(params_text);
    assert_eq!(
        params.first().map(String::as_str),
        Some("self"),
        "backend method must take self: {sig_flat}"
    );

    // Inner signature: `self` becomes an explicit token parameter.
    let mut inner_params = vec![format!("_this: {token_path}")];
    let mut fwd_args = vec!["self".to_string()];
    for p in &params[1..] {
        inner_params.push(p.clone());
        let pname = p
            .split(':')
            .next()
            .unwrap()
            .trim()
            .trim_start_matches("mut ")
            .to_string();
        fwd_args.push(pname);
    }

    // Const-generic forwarding: `<const N: i32>` -> `::<N>`.
    let turbofish = if generics.is_empty() {
        String::new()
    } else {
        let names: Vec<&str> = generics
            .trim_matches(['<', '>'])
            .split(',')
            .map(|g| {
                g.trim()
                    .trim_start_matches("const ")
                    .split(':')
                    .next()
                    .unwrap()
                    .trim()
            })
            .collect();
        format!("::<{}>", names.join(", "))
    };

    // --- Transform the body ----------------------------------------------
    let body = strip_value_unsafe(body);
    let body = replace_word(&body, "Self", &token_path);
    let body = replace_word(&body, "self", "_this");
    // Re-indent the body one level deeper (8 -> 12 spaces).
    let body_indented: String = body
        .lines()
        .map(|l| {
            if l.is_empty() {
                String::from("\n")
            } else {
                format!("    {l}\n")
            }
        })
        .collect();

    let features = features_for(token);
    let ret_sp = if ret.is_empty() {
        String::new()
    } else {
        let ret_norm = ret.split_whitespace().collect::<Vec<_>>().join(" ");
        format!(" {ret_norm}")
    };
    let inner_params_j = inner_params.join(", ");
    let fwd = fwd_args.join(", ");

    // The outer method keeps its original attribute + signature verbatim
    // (fmt-stable); only the inner fn is reconstructed.
    let _ = (name, params_text);
    format!(
        "    #[inline(always)]\n{sig}        \
         #[target_feature(enable = \"{features}\")]\n        #[inline]\n        \
         fn __tf{generics}({inner_params_j}){ret_sp} {{\n{body_indented}        }}\n        \
         // SAFETY: `self` is a runtime-detection token proving the enabled\n        \
         // target features (compiler-verified intrinsic/feature matching).\n        \
         unsafe {{ __tf{turbofish}({fwd}) }}\n    }}\n"
    )
}
