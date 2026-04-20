//! Macro-expansion snapshot tests.
//!
//! Each `tests/expand/*.rs` file is an input; its `.expanded.rs` sibling is
//! the committed snapshot of what artisan-macros expands it into. This test
//! drives `macrotest` over the glob and fails if any expansion drifts.
//!
//! Each snapshot has the original input source pinned to the top as a
//! commented-out block between `// === INPUT ===` and `// === END INPUT ===`
//! markers, so a reviewer can see input + expansion in a single file without
//! tab-switching.
//!
//! ## Regenerating
//!
//! ```text
//! MACROTEST=overwrite cargo test -p artisan-macros --test expand
//! ```
//!
//! The overwrite pass runs macrotest, then prepends each input file's source
//! to the generated `.expanded.rs`. Commit all `*.expanded.rs` changes in
//! the same diff as the macro change.
//!
//! ## Requires cargo-expand
//!
//! `macrotest` shells out to `cargo expand`. Install with
//! `cargo install cargo-expand --locked`; CI does this.
//!
//! ## x86_64-only
//!
//! `cargo expand` resolves `#[cfg(target_arch = "...")]` against the host
//! target BEFORE printing. Committed snapshots are therefore x86_64-resolved;
//! the test is gated on `target_arch = "x86_64"`. Aarch64 and wasm32 code
//! paths are covered by the cross-compile CI jobs (which test the macros
//! directly; snapshots are for shape, not coverage).

use std::fs;
use std::path::{Path, PathBuf};

const INPUT_START: &str = "// === INPUT ===";
const INPUT_END: &str = "// === END INPUT ===";

#[cfg(target_arch = "x86_64")]
#[test]
fn snapshots() {
    let overwrite = std::env::var("MACROTEST").as_deref() == Ok("overwrite");

    // Run macrotest first. In overwrite mode it regenerates snapshots without
    // the input prefix; we add the prefix in our post-process step below.
    // In compare mode, macrotest compares against the committed snapshots
    // (which already have the prefix). For that comparison to succeed, the
    // committed snapshots must be post-processed — and the
    // verify_snapshots_have_input_prefix pass below enforces that invariant.
    //
    // Order matters: under overwrite, macrotest regenerates FIRST, then we
    // prepend. Under compare, macrotest compares FIRST (against prefixed
    // snapshots), which would always fail because macrotest's own rendering
    // has no prefix. Hence in compare mode we temporarily strip prefixes
    // before calling macrotest::expand — see run_macrotest_compare below.
    if overwrite {
        macrotest::expand("tests/expand/*.rs");
        postprocess_all_snapshots();
    } else {
        run_macrotest_compare();
        verify_snapshots_have_input_prefix();
    }
}

#[cfg(not(target_arch = "x86_64"))]
#[test]
fn snapshots_are_x86_64_only() {
    println!("expansion snapshots are x86_64-only; skipping on this host");
}

fn input_files() -> Vec<PathBuf> {
    let mut out = Vec::new();
    for entry in fs::read_dir("tests/expand").expect("tests/expand must exist") {
        let path = entry.expect("readable dir entry").path();
        let name = path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("")
            .to_string();
        if name.ends_with(".rs") && !name.ends_with(".expanded.rs") {
            out.push(path);
        }
    }
    out.sort();
    out
}

fn expanded_path_for(input: &Path) -> PathBuf {
    let stem = input.file_stem().and_then(|s| s.to_str()).unwrap();
    input.with_file_name(format!("{stem}.expanded.rs"))
}

fn build_prefix(input_source: &str) -> String {
    let mut out = String::new();
    out.push_str(INPUT_START);
    out.push('\n');
    for line in input_source.lines() {
        if line.is_empty() {
            out.push_str("//");
            out.push('\n');
        } else {
            out.push_str("// ");
            out.push_str(line);
            out.push('\n');
        }
    }
    out.push_str(INPUT_END);
    out.push('\n');
    out.push('\n');
    out
}

fn strip_existing_prefix(content: &str) -> &str {
    if let Some(rest) = content.strip_prefix(INPUT_START) {
        // Trim leading newline after INPUT_START, then look for END marker.
        let rest = rest.strip_prefix('\n').unwrap_or(rest);
        if let Some(idx) = rest.find(INPUT_END) {
            let after_marker = &rest[idx + INPUT_END.len()..];
            // Skip the newline that follows INPUT_END and any blank line.
            let after_marker = after_marker.strip_prefix('\n').unwrap_or(after_marker);
            let after_marker = after_marker.strip_prefix('\n').unwrap_or(after_marker);
            return after_marker;
        }
    }
    content
}

fn postprocess_all_snapshots() {
    for input in input_files() {
        let expanded = expanded_path_for(&input);
        if !expanded.exists() {
            continue;
        }
        let input_source = fs::read_to_string(&input).expect("read input");
        let current = fs::read_to_string(&expanded).expect("read expanded");
        let body = strip_existing_prefix(&current);
        let new = format!("{}{}", build_prefix(&input_source), body);
        if new != current {
            fs::write(&expanded, new).expect("write expanded");
        }
    }
}

fn run_macrotest_compare() {
    // macrotest compares against file contents verbatim. Our committed
    // snapshots have an input-source prefix that macrotest's fresh rendering
    // won't match. To let macrotest do its comparison, we stage stripped
    // copies in a sibling temp directory and point macrotest at them. If
    // macrotest finds a drift there, propagate the failure.
    //
    // For simplicity in the current draft we just strip prefixes in-place,
    // run macrotest, then restore. If macrotest panics we restore in a
    // PanicGuard.
    let inputs = input_files();
    let pairs: Vec<(PathBuf, String)> = inputs
        .iter()
        .filter_map(|input| {
            let expanded = expanded_path_for(input);
            if expanded.exists() {
                let original = fs::read_to_string(&expanded).ok()?;
                Some((expanded, original))
            } else {
                None
            }
        })
        .collect();

    // Write stripped versions.
    for (path, original) in &pairs {
        let stripped = strip_existing_prefix(original);
        if stripped != original.as_str() {
            fs::write(path, stripped).expect("write stripped snapshot");
        }
    }

    // Guard: restore originals on panic or normal exit.
    struct Restore<'a>(&'a [(PathBuf, String)]);
    impl Drop for Restore<'_> {
        fn drop(&mut self) {
            for (path, original) in self.0 {
                let _ = fs::write(path, original);
            }
        }
    }
    let _guard = Restore(&pairs);

    macrotest::expand("tests/expand/*.rs");
}

fn verify_snapshots_have_input_prefix() {
    let mut failures = Vec::new();
    for input in input_files() {
        let expanded = expanded_path_for(&input);
        if !expanded.exists() {
            failures.push(format!(
                "missing snapshot: {}\n  run MACROTEST=overwrite cargo test -p artisan-macros --test expand to generate",
                expanded.display()
            ));
            continue;
        }
        let input_source = fs::read_to_string(&input).expect("read input");
        let snapshot = fs::read_to_string(&expanded).expect("read expanded");
        let expected_prefix = build_prefix(&input_source);
        if !snapshot.starts_with(&expected_prefix) {
            failures.push(format!(
                "snapshot {} is missing the input-source prefix, or the prefix doesn't match {}\n  regenerate with: MACROTEST=overwrite cargo test -p artisan-macros --test expand",
                expanded.display(),
                input.display()
            ));
        }
    }
    if !failures.is_empty() {
        panic!(
            "expansion snapshot integrity check failed:\n{}",
            failures.join("\n")
        );
    }
}
