//! Capability-probe build script for magetypes.
//!
//! # Why this exists
//!
//! magetypes pins a low MSRV (workspace `rust-version`, currently 1.89) so it
//! compiles on a broad range of toolchains. But some hardware SIMD paths use
//! `core::arch` intrinsics that only became `#[stable]` *after* that floor —
//! e.g. the aarch64 NEON half-precision converters `vcvt_f32_f16` /
//! `vcvt_f16_f32` are `#[stable(feature = "stdarch_neon_fp16", since =
//! "1.94.0")]`. A *static* `cfg` gate on those would force the whole crate's
//! MSRV up to 1.94.
//!
//! Instead we **probe capability by try-compiling**: for each newer-stable
//! intrinsic of interest, this script attempts to compile a tiny snippet for
//! the crate's build target. If it compiles, the corresponding `cfg` is emitted
//! and the hardware path lights up; if it does not (older toolchain), the crate
//! silently falls back to its already-shipped, branchless software kernel. The
//! MSRV floor never moves, and the same source builds clean on every toolchain
//! ≥ MSRV.
//!
//! This is the autocfg / "feature-detection-not-version-matching" approach: it
//! is robust against target variance, backports, and custom toolchains in a way
//! that a bare `rustc --version` comparison is not.
//!
//! # Why we probe the *build target*, not a fixed cross target
//!
//! A probe pinned to e.g. `--target aarch64-unknown-linux-gnu` would fail on a
//! capable toolchain whenever that target's precompiled `core` isn't installed
//! (a common case on an x86 host) — yielding the *wrong* answer (cfg off on a
//! 1.94+ toolchain). Each intrinsic's `cfg` is only ever consumed inside a
//! `#[cfg(target_arch = "…")]` block, so it only matters when the crate is
//! actually being *built for* that arch. We therefore run a given arch's probe
//! only when `CARGO_CFG_TARGET_ARCH` matches, and probe the real build `TARGET`
//! — whose `core` is guaranteed present because Cargo is compiling the crate
//! for it. (A native aarch64 build and an x86-host `cargo check --target
//! aarch64-…` both report `CARGO_CFG_TARGET_ARCH = aarch64`, so both are
//! covered.)
//!
//! # Adding the next newer-stable intrinsic
//!
//! 1. Write a `#![no_std]` snippet that *uses* the intrinsic behind the
//!    `#[target_feature]` it requires.
//! 2. Add a `probe_for_arch("archmage_has_<name>", "<arch>", snippet)` call in
//!    `main` (it self-gates on `CARGO_CFG_TARGET_ARCH`).
//! 3. Add the cfg name to `declare_check_cfgs` (Rust 1.80+ checks unexpected
//!    cfgs; the crate MSRV ≥ 1.89 supports `cargo:rustc-check-cfg`).
//! 4. Gate the hardware impl with `#[cfg(archmage_has_<name>)]`, wire it into
//!    the runtime token dispatch, and keep the software kernel as the
//!    `#[cfg(not(archmage_has_<name>))]` fallback.
//!
//! No external build-dependency is used: the probe shells out to the same
//! `rustc` Cargo invoked us with (`$RUSTC`), keeping a foundational crate's
//! dependency graph empty.

use std::env;
use std::fs;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    // Re-run only when this script changes; the probe result depends solely on
    // the toolchain + target (captured via the rustc invocation), not on any
    // crate source.
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=RUSTC");

    declare_check_cfgs();

    // --- aarch64 NEON f16 (stdarch_neon_fp16, stable since 1.94.0) ----------
    //
    // `vcvt_f32_f16` (decode) and `vcvt_f16_f32` (encode) require the `fp16`
    // target feature. The snippet enables `neon,fp16` and uses both. It only
    // needs to type-check, so we emit metadata (no codegen) and never run it.
    // Test *every* intrinsic the hardware kernel uses, not just one. They are
    // all gated behind the single `stdarch_neon_f16` library feature (the
    // `float16x4_t` type gates the whole set), so they stabilize atomically —
    // but probing all of them keeps the gate honest if that ever changes.
    let neon_f16_snippet = r#"
#![allow(internal_features)]
#![no_std]
use core::arch::aarch64::{
    float16x4_t, float32x4_t, uint16x4_t,
    vcvt_f32_f16, vcvt_f16_f32, vreinterpret_f16_u16, vreinterpret_u16_f16,
};
#[target_feature(enable = "neon,fp16")]
unsafe fn _decode(h: float16x4_t) -> float32x4_t { vcvt_f32_f16(h) }
#[target_feature(enable = "neon,fp16")]
unsafe fn _encode(f: float32x4_t) -> float16x4_t { vcvt_f16_f32(f) }
#[target_feature(enable = "neon,fp16")]
unsafe fn _cast_in(x: uint16x4_t) -> float16x4_t { vreinterpret_f16_u16(x) }
#[target_feature(enable = "neon,fp16")]
unsafe fn _cast_out(x: float16x4_t) -> uint16x4_t { vreinterpret_u16_f16(x) }
"#;
    probe_for_arch("archmage_has_neon_f16", "aarch64", neon_f16_snippet);
}

/// Declare every custom `cfg` this script may emit so that Rust 1.80+'s
/// unexpected-`cfg` lint stays quiet for downstream crates and our own source.
/// Each name must appear here whether or not the probe sets it.
fn declare_check_cfgs() {
    println!("cargo:rustc-check-cfg=cfg(archmage_has_neon_f16)");
}

/// If the crate is being built for `arch`, try-compile `snippet` for the real
/// build `TARGET` and emit `cargo:rustc-cfg=<cfg_name>` on success. Skipped
/// entirely (cfg left unset) when building for a different arch — which is
/// correct, because the cfg is only consumed inside a matching
/// `#[cfg(target_arch = "…")]` block.
fn probe_for_arch(cfg_name: &str, arch: &str, snippet: &str) {
    let build_arch = env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();
    if build_arch != arch {
        return;
    }
    let target = env::var("TARGET").unwrap_or_default();
    if target.is_empty() {
        return;
    }
    if try_compile(&target, snippet) {
        println!("cargo:rustc-cfg={cfg_name}");
    }
    // else: stay silent; the software fallback is selected by `cfg(not(..))`.
    // We deliberately do NOT emit a `cargo:warning` on success — it would fire
    // on every build of this target and spam downstream consumers. The cfg
    // itself is the observable signal (visible via `cargo build -vv`).
}

/// Compile `snippet` to metadata-only for `target`, returning whether it
/// type-checked. Uses the same `rustc` Cargo invoked us with. Never runs the
/// produced artifact and writes only into `OUT_DIR`.
fn try_compile(target: &str, snippet: &str) -> bool {
    let rustc = env::var_os("RUSTC").unwrap_or_else(|| "rustc".into());
    let out_dir = PathBuf::from(env::var_os("OUT_DIR").expect("OUT_DIR set by cargo"));

    let src = out_dir.join(format!("probe_{}.rs", sanitize(target)));
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

fn sanitize(s: &str) -> String {
    s.chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
        .collect()
}
