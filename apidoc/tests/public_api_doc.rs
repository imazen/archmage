//! Per-target public-API surface snapshots for the PARENT workspace, written
//! to `docs/public-api/<arch>/`. Shared implementation + snapshot format
//! docs: the `zenutils-apidoc` crate.
//!
//! # Why per-target (issue #75)
//!
//! This workspace's public surface is a function of `target_arch`: the
//! `__m128`/`__m256`/`__m512` constructors exist only on x86_64, the
//! `int8x16_t` family only on aarch64, `v128` only on wasm32.
//! `zenutils-apidoc` builds rustdoc JSON for the HOST, so a regeneration on
//! an aarch64 box silently dropped every x86 entry (and vice versa) — the
//! snapshot's content was a function of whoever last ran the generator.
//!
//! This runner pins the compilation target instead: one snapshot directory
//! per target triple, each produced with `CARGO_BUILD_TARGET=<triple>`, so
//! the output is a function of the target and regeneration is byte-identical
//! on any host (given the same nightly toolchain).
//!
//! # Mechanics
//!
//! `zenutils-apidoc` 0.1.x has no `--target` plumbing, but every cargo
//! process it spawns inherits this process's environment, and cargo honors
//! `CARGO_BUILD_TARGET`/`CARGO_TARGET_DIR` from the environment. Two details
//! make that work here:
//!
//! - **Env is set per child process, not via `env::set_var`** (unsafe in
//!   edition 2024, and racy). The test re-executes its own binary once per
//!   target with the env prepared on the `Command`.
//! - **JSON path bridge:** with a build target set, cargo writes rustdoc
//!   JSON to `<target-dir>/<triple>/doc/`, while zenutils-apidoc reads
//!   `<target-dir>/doc/`. A `doc -> <triple>/doc` symlink inside a dedicated
//!   `CARGO_TARGET_DIR` (`target/public-api/`) bridges the two; the parent
//!   retargets it before each child run. If zenutils-apidoc ever grows
//!   native `--target` support, this shim collapses to builder calls.
//!
//! Modes are unchanged (`ZEN_API_DOC` = regen default / `check` / `off`;
//! unset under `GITHUB_ACTIONS` skips). The nightly toolchain and the three
//! rustup target stdlibs are auto-installed via rustup.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::{env, fs};

/// (snapshot subdirectory, target triple). The triples are canonical
/// representatives per architecture — the API surface is `target_arch`-gated,
/// not `target_os`-gated, so one triple per arch is enough. They match the
/// targets CI already tests (linux x64/arm64 runners, wasm32-wasip1 under
/// wasmtime).
const TARGETS: &[(&str, &str)] = &[
    ("x86_64", "x86_64-unknown-linux-gnu"),
    ("aarch64", "aarch64-unknown-linux-gnu"),
    ("wasm32", "wasm32-wasip1"),
];

/// Explicit crate list: archmage-macros surfaces through archmage's
/// re-exports; xtask + tests/no-features-crate are internal.
const CRATES: [&str; 2] = ["archmage", "magetypes"];

/// Snapshot file suffixes zenutils-apidoc writes per crate.
const SUFFIXES: [&str; 3] = [".txt", ".features.txt", ".internal.txt"];

/// Hand-maintained files allowed at the `docs/public-api/` root next to the
/// generated per-target directories.
const EXTRA_COMMITTED: [&str; 3] = ["README.md", "ABLATION-archmage.md", "ABLATION-magetypes.md"];

/// Same default as zenutils-apidoc: `ZEN_API_DOC_TOOLCHAIN` env override, or
/// the tracking `nightly`.
fn toolchain() -> String {
    env::var("ZEN_API_DOC_TOOLCHAIN").unwrap_or_else(|_| "nightly".to_owned())
}

/// Mirror of zenutils-apidoc's mode gate, so the parent skips the rustup /
/// child-spawn work in exactly the cases the inner runner would skip.
fn skip() -> bool {
    match env::var("ZEN_API_DOC").as_deref() {
        Ok("off") => true,
        Ok("check" | "regen") => false,
        Ok(other) => panic!("unknown ZEN_API_DOC value {other:?} (off|check|regen)"),
        Err(_) if env::var_os("GITHUB_ACTIONS").is_some() => {
            eprintln!(
                "ZEN_API_DOC unset under GITHUB_ACTIONS — snapshot regen skipped \
                 (a dedicated api-doc job should set ZEN_API_DOC=check)"
            );
            true
        }
        Err(_) => false,
    }
}

fn workspace_root() -> PathBuf {
    // cargo runs test binaries with cwd = the package root (apidoc/).
    Path::new("..")
        .canonicalize()
        .expect("canonicalize workspace root (..)")
}

#[test]
fn public_api_surface_docs_are_current() {
    if skip() {
        return;
    }
    if let Ok(sub) = env::var("APIDOC_TARGET") {
        run_one_target(&sub);
        return;
    }

    let toolchain = toolchain();
    for (_, triple) in TARGETS {
        let st = Command::new("rustup")
            .args(["target", "add", "--toolchain"])
            .arg(&toolchain)
            .arg(triple)
            .status()
            .expect("failed to run rustup target add");
        assert!(
            st.success(),
            "rustup target add --toolchain {toolchain} {triple} failed; \
             set ZEN_API_DOC=off to skip the public-API snapshot test"
        );
    }

    let ws = workspace_root();
    let target_dir = ws.join("target").join("public-api");
    let exe = env::current_exe().expect("current_exe");
    for (sub, triple) in TARGETS {
        retarget_doc_symlink(&target_dir, triple);
        let st = Command::new(&exe)
            .args([
                "public_api_surface_docs_are_current",
                "--exact",
                "--nocapture",
            ])
            .env("APIDOC_TARGET", sub)
            .env("CARGO_BUILD_TARGET", triple)
            .env("CARGO_TARGET_DIR", &target_dir)
            .status()
            .expect("failed to re-exec the test binary for a per-target run");
        assert!(
            st.success(),
            "per-target public-API snapshot run failed for {sub} ({triple}) — see output above"
        );
    }

    check_no_stray_files(&ws);
}

/// Child mode: one zenutils-apidoc run for one target. `CARGO_BUILD_TARGET`
/// and `CARGO_TARGET_DIR` were set on this process by the parent, so every
/// cargo the inner runner spawns builds rustdoc JSON for that triple.
fn run_one_target(sub: &str) {
    let (_, triple) = TARGETS
        .iter()
        .find(|(s, _)| s == &sub)
        .unwrap_or_else(|| panic!("unknown APIDOC_TARGET {sub:?}"));
    eprintln!("=== public-API snapshots for {triple} -> docs/public-api/{sub}/ ===");
    let mut doc = zenutils_apidoc::ApiDoc::new()
        .workspace_dir("..")
        .crates(CRATES)
        .out_dir(&format!("docs/public-api/{sub}"));
    // The packaging invariant is target-independent — checking it once (on
    // the x86_64 run) is enough.
    if sub != "x86_64" {
        for c in CRATES {
            doc = doc.skip_packaging_check(c);
        }
    }
    doc.run();
}

/// Point `<target_dir>/doc` at `<triple>/doc` (relative symlink), replacing
/// whatever the previous run left. `target/public-api/` is exclusively ours,
/// so anything else at that path is a leftover safe to remove.
fn retarget_doc_symlink(target_dir: &Path, triple: &str) {
    // Pre-create the real destination so the link is never dangling.
    fs::create_dir_all(target_dir.join(triple).join("doc"))
        .expect("create target/public-api/<triple>/doc");
    let link = target_dir.join("doc");
    if let Ok(meta) = fs::symlink_metadata(&link) {
        if meta.file_type().is_symlink() || !meta.is_dir() {
            fs::remove_file(&link).expect("remove stale doc symlink");
        } else {
            fs::remove_dir_all(&link).expect("remove stale doc dir");
        }
    }
    let dest = Path::new(triple).join("doc");
    #[cfg(unix)]
    std::os::unix::fs::symlink(&dest, &link).expect("symlink target/public-api/doc");
    #[cfg(windows)]
    std::os::windows::fs::symlink_dir(&dest, &link).unwrap_or_else(|e| {
        panic!(
            "symlink target/public-api/doc failed ({e}); on Windows enable \
             Developer Mode, or regenerate from WSL, or set ZEN_API_DOC=off"
        )
    });
}

/// Every file under `docs/public-api/` must be either a generated per-target
/// snapshot or one of the known hand-maintained docs. Orphans (e.g. the old
/// host-dependent top-level `<crate>.txt` layout, or files for a crate that
/// was removed) would otherwise sit stale forever without failing `check`.
fn check_no_stray_files(ws: &Path) {
    let root = ws.join("docs").join("public-api");
    let mut expected: BTreeSet<PathBuf> = EXTRA_COMMITTED.iter().map(|f| root.join(f)).collect();
    for (sub, _) in TARGETS {
        for c in CRATES {
            for suffix in SUFFIXES {
                expected.insert(root.join(sub).join(format!("{c}{suffix}")));
            }
        }
    }
    let mut actual = BTreeSet::new();
    collect_files(&root, &mut actual);
    let stray: Vec<&PathBuf> = actual.difference(&expected).collect();
    assert!(
        stray.is_empty(),
        "unexpected files under docs/public-api/ (delete them, or add them to \
         EXTRA_COMMITTED in apidoc/tests/public_api_doc.rs): {stray:#?}"
    );
}

fn collect_files(dir: &Path, out: &mut BTreeSet<PathBuf>) {
    let entries = fs::read_dir(dir).unwrap_or_else(|e| panic!("read_dir {}: {e}", dir.display()));
    for entry in entries {
        let path = entry.expect("dir entry").path();
        if path.is_dir() {
            collect_files(&path, out);
        } else {
            out.insert(path);
        }
    }
}
