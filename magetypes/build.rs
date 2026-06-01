//! Nightly-opportunistic capability-probe scaffold for magetypes.
//!
//! # Why this build script is *nightly-only* now
//!
//! The previous incarnation of this script probed **stable** newer-stable
//! intrinsics (e.g. the aarch64 NEON-f16 converters `vcvt_f32_f16` /
//! `vcvt_f16_f32`, stable since 1.94). That case has moved to a
//! `rustversion`-plus-`target_arch` gate in `src/simd/generic/convert_f16.rs`,
//! verified by a CI matrix that builds + tests **both sides** of the 1.94
//! boundary — no build script, a trusted dep-light proc-macro, and the matrix
//! makes the named version bound load-bearing. See `MSRV.md`.
//!
//! A try-compile probe still wins in exactly **one** situation: a **nightly-only**
//! intrinsic that has *no stable version yet*. There, you can't name a version,
//! and a blind `#![feature(<gate>)]` breaks whenever the nightly feature gate is
//! renamed or removed between nightlies. The robust answer is "does *this*
//! nightly actually compile the feature + intrinsic?" — a try-compile probe,
//! run **only under nightly**, that enables the path iff it compiles and falls
//! back otherwise.
//!
//! # Status: scaffold, not yet load-bearing
//!
//! archmage has **no nightly-only intrinsic to gate today**, so this script is a
//! documented scaffold:
//!
//! - On a **nightly** toolchain it runs [`probe_nightly`] (which currently
//!   try-compiles a trivially-stable placeholder snippet, so it always succeeds)
//!   and emits `archmage_nightly_probe_example`. This keeps the try-compile
//!   machinery exercised by the CI nightly cell without inventing a fake
//!   intrinsic.
//! - On a **non-nightly** toolchain it does nothing but declare the check-cfg,
//!   so stable/beta builds pay **zero** probe cost.
//!
//! The cfg is consumed by a `#[rustversion::nightly] #[cfg(archmage_nightly_probe_example)]`
//! item in `src/simd/generic/convert_f16.rs` (see `NIGHTLY_PROBE_OK`) so the
//! wiring pattern is real and compiles end-to-end.
//!
//! # Adding a real nightly-only intrinsic
//!
//! 1. Replace the placeholder snippet in [`probe_nightly`] with a
//!    `#![feature(<gate>)] #![no_std]` snippet that *uses* the intrinsic behind
//!    its `#[target_feature]`, and `probe_for_arch(...)` it for the right arch.
//! 2. Give the cfg a real name (e.g. `archmage_nightly_<name>`); add it to
//!    [`declare_check_cfgs`].
//! 3. Gate the hardware impl `#[rustversion::nightly] #[cfg(archmage_nightly_<name>)]`,
//!    wire it into the runtime token dispatch, and keep the software kernel as
//!    the fallback (`#[rustversion::not(nightly)]` or `cfg(not(...))`).
//! 4. The CI nightly cell already exercises that the scaffold compiles; it will
//!    cover the real probe too.
//!
//! No external build-dependency is used: the script shells out to the same
//! `rustc` Cargo invoked it with (`$RUSTC`), keeping a foundational crate's
//! dependency graph empty. Nightly detection is `$RUSTC --version` channel
//! sniffing (a build script can't use the `rustversion` proc-macro).

use std::env;
use std::fs;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    // The probe result depends only on the toolchain + target, not crate source.
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=RUSTC");

    declare_check_cfgs();

    // Nightly-only: stable/beta builds skip the probe entirely (zero cost).
    if !is_nightly() {
        return;
    }
    probe_nightly();
}

/// Declare every custom `cfg` this script may emit so Rust 1.80+'s
/// unexpected-`cfg` lint stays quiet for downstream crates and our own source.
/// Each name must appear here whether or not the probe sets it.
fn declare_check_cfgs() {
    println!("cargo:rustc-check-cfg=cfg(archmage_nightly_probe_example)");
}

/// The single nightly probe. Currently a **scaffold**: it try-compiles a
/// trivially-stable placeholder snippet (so it always succeeds, exercising the
/// machinery), and emits `archmage_nightly_probe_example` on success. Replace
/// the snippet with a real `#![feature(<gate>)]` + nightly-intrinsic snippet
/// when one is needed (see the module docs).
fn probe_nightly() {
    // PLACEHOLDER: a `#![no_std]` snippet that compiles on every toolchain. A
    // real nightly-only probe would put `#![feature(<gate>)]` here and *use* the
    // nightly intrinsic behind its `#[target_feature]`.
    let placeholder_snippet = r#"
#![no_std]
pub const fn _probe_marker() -> u32 { 0 }
"#;
    probe("archmage_nightly_probe_example", placeholder_snippet);
}

/// Try-compile `snippet` to metadata-only for the build target; emit
/// `cargo:rustc-cfg=<cfg_name>` on success, stay silent on failure (the software
/// fallback is selected by `cfg(not(..))` / `#[rustversion::not(nightly)]`).
fn probe(cfg_name: &str, snippet: &str) {
    let target = env::var("TARGET").unwrap_or_default();
    if target.is_empty() {
        return;
    }
    if try_compile(&target, snippet) {
        println!("cargo:rustc-cfg={cfg_name}");
    }
}

/// Compile `snippet` to metadata-only for `target` using the same `rustc` Cargo
/// invoked us with; return whether it type-checked. Never runs the artifact and
/// writes only into `OUT_DIR`.
fn try_compile(target: &str, snippet: &str) -> bool {
    let rustc = env::var_os("RUSTC").unwrap_or_else(|| "rustc".into());
    let out_dir = PathBuf::from(env::var_os("OUT_DIR").expect("OUT_DIR set by cargo"));

    let src = out_dir.join(format!("nightly_probe_{}.rs", sanitize(target)));
    if fs::write(&src, snippet).is_err() {
        return false;
    }

    let mut cmd = Command::new(&rustc);
    cmd.arg("--edition=2021")
        .arg("--crate-type=lib")
        .arg("--emit=metadata")
        .arg("--target")
        .arg(target)
        .arg("--out-dir")
        .arg(&out_dir)
        // We only care about the exit status, not diagnostics.
        .arg("--cap-lints")
        .arg("allow")
        .arg(&src);

    match cmd.status() {
        Ok(status) => status.success(),
        Err(_) => false,
    }
}

/// Channel-sniff the active toolchain via `$RUSTC --version`. A build script
/// cannot use the `rustversion` proc-macro (it runs at build time, not compile
/// time of the crate), so we read the version string's `-nightly`/`-dev` marker.
fn is_nightly() -> bool {
    let rustc = env::var_os("RUSTC").unwrap_or_else(|| "rustc".into());
    let Ok(out) = Command::new(&rustc).arg("--version").output() else {
        return false;
    };
    let v = String::from_utf8_lossy(&out.stdout);
    // e.g. "rustc 1.96.0-nightly (abc 2026-..)" — stable/beta lack "-nightly".
    v.contains("-nightly") || v.contains("-dev")
}

fn sanitize(s: &str) -> String {
    s.chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
        .collect()
}
