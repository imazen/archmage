//! Compile-time intrinsic soundness entrypoint and Miri UB-detection runner.

use anyhow::{Context, Result, bail};
use std::path::PathBuf;

use crate::{registry, soundness};

// ============================================================================
// Compile-Time Intrinsic Soundness Verification
// ============================================================================

/// Verify intrinsic soundness — see `soundness.rs` for the model and rules.
pub(crate) fn verify_intrinsic_soundness() -> Result<()> {
    let reg = registry::Registry::load(&PathBuf::from("token-registry.toml"))?;
    soundness::verify(&reg)
}

/// The nightly Miri runs on: `ARCHMAGE_MIRI_TOOLCHAIN`, else the pin in
/// `xtask/miri-nightly.txt` that CI installs, else the host's `nightly`.
pub(crate) fn miri_toolchain() -> String {
    if let Ok(toolchain) = std::env::var("ARCHMAGE_MIRI_TOOLCHAIN") {
        return toolchain;
    }
    std::fs::read_to_string("xtask/miri-nightly.txt")
        .map(|s| s.trim().to_string())
        .ok()
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "nightly".into())
}

/// Expensive magetypes tests that Miri does not run. Each one either reaches no
/// `unsafe` code or reaches only paths that a cheaper test, still run under
/// Miri, already covers, so skipping it loses no undefined-behavior coverage.
/// Each entry names a test binary and, with `Some`, one test in it; `None`
/// skips the whole binary. The first two were about 95% of Miri's interpreted
/// work (measured 2026-10-03). Native, SDE and cross-architecture CI still run
/// them in full.
const MIRI_SKIPPED_TESTS: &[(&str, Option<&str>, &str)] = &[
    (
        "int_widen_narrow",
        Some("scalar_backend"),
        "exhaustive i16/u16 loops; the unsafe bitcast path they reach is covered \
         by scalar_w512_bitcast_values, which still runs under Miri",
    ),
    (
        "fused_arithmetic",
        None,
        "software-FMA comparisons against std; safe code only",
    ),
    (
        "cbrt_range",
        Some("cbrt_midp_precise_covers_the_whole_range"),
        "range sweeps, about 33,000 cbrt calls per function (432 s under Miri); \
         on the scalar backend Miri runs they are safe code only",
    ),
    (
        "int_uniform_shift_saturating",
        Some("scalar_backend"),
        "every boundary shift count over every 8/16-bit type and width on the \
         scalar backend (54 s under Miri, 2026-10-07): safe lane code, and its \
         loads and stores are the paths x64v3_backend in the same binary runs \
         under Miri in under a second",
    ),
    (
        "precise_reciprocals",
        None,
        "ULP sweeps of recip/rsqrt over thousands of inputs (22 s under Miri): \
         safe code apart from the loads and stores generic_all_types runs under \
         Miri for every type",
    ),
    (
        "exp2_lowp_range",
        None,
        "a range sweep over [126, 128) (12 s under Miri): safe code only",
    ),
    (
        "sqrt_token_testing",
        None,
        "requires archmage/testable_dispatch (run explicitly by just sqrt-token-testing); \
         safe code only (#![forbid(unsafe_code)])",
    ),
];

/// magetypes integration-test targets, minus the binaries [`MIRI_SKIPPED_TESTS`]
/// skips whole. Fails if an entry no longer names a real target or a test that
/// target defines, so a rename cannot leave it stale.
fn miri_test_targets() -> Result<Vec<String>> {
    let out = std::process::Command::new("cargo")
        .args(["metadata", "--no-deps", "--format-version", "1"])
        .output()
        .context("Failed to run cargo metadata")?;
    if !out.status.success() {
        bail!("cargo metadata failed");
    }
    let meta: serde_json::Value = serde_json::from_slice(&out.stdout)?;
    let package = meta["packages"]
        .as_array()
        .and_then(|packages| packages.iter().find(|p| p["name"] == "magetypes"))
        .context("magetypes is missing from cargo metadata")?;
    let tests: Vec<(String, String)> = package["targets"]
        .as_array()
        .context("magetypes has no targets in cargo metadata")?
        .iter()
        .filter(|t| {
            t["kind"]
                .as_array()
                .is_some_and(|k| k.iter().any(|k| k == "test"))
        })
        .filter_map(|t| {
            Some((
                t["name"].as_str()?.to_owned(),
                t["src_path"].as_str()?.to_owned(),
            ))
        })
        .collect();
    for (name, test, _) in MIRI_SKIPPED_TESTS {
        let Some((_, src_path)) = tests.iter().find(|(t, _)| t == name) else {
            bail!("MIRI_SKIPPED_TESTS names `{name}`, which is not a magetypes test target");
        };
        if let Some(test) = test {
            let src = std::fs::read_to_string(src_path)
                .with_context(|| format!("Failed to read {src_path}"))?;
            if !src.contains(&format!("fn {test}(")) {
                bail!("MIRI_SKIPPED_TESTS names test `{test}`, which {src_path} does not define");
            }
        }
    }
    Ok(tests
        .into_iter()
        .map(|(name, _)| name)
        .filter(|t| {
            !MIRI_SKIPPED_TESTS
                .iter()
                .any(|(skip, test, _)| skip == t && test.is_none())
        })
        .collect())
}

/// Run native raw tests with their required virtual CPU, then the baseline
/// suite (unit tests and the integration tests [`MIRI_SKIPPED_TESTS`] does not
/// skip), then the doctests.
pub(crate) fn run_miri() -> Result<()> {
    use std::process::Command;
    let toolchain = format!("+{}", miri_toolchain());
    println!("=== Running Miri on magetypes ({toolchain}) ===");
    let check = Command::new("cargo")
        .args([&toolchain, "miri", "--version"])
        .status()?;
    if !check.success() {
        bail!("Miri is unavailable for {toolchain}; install its miri component");
    }

    let targets = miri_test_targets()?;
    println!("Not run under Miri (unsafe paths covered by retained tests; native CI runs them):");
    for (bin, test, why) in MIRI_SKIPPED_TESTS {
        match test {
            Some(test) => println!("  {bin}::{test}: {why}"),
            None => println!("  {bin} (whole binary): {why}"),
        }
    }
    // A binary with per-test skips runs on its own: libtest's --skip applies to
    // every binary in one cargo invocation.
    let mut partial: Vec<(&str, Vec<&str>)> = Vec::new();
    for (bin, test, _) in MIRI_SKIPPED_TESTS {
        if let Some(test) = test {
            match partial.iter_mut().find(|(b, _)| b == bin) {
                Some((_, skips)) => skips.push(test),
                None => partial.push((bin, vec![test])),
            }
        }
    }
    let mut baseline = Command::new("cargo");
    baseline.args([
        &toolchain,
        "miri",
        "test",
        "-p",
        "magetypes",
        "--features",
        "avx512",
        "--lib",
    ]);
    for target in targets
        .iter()
        .filter(|t| !partial.iter().any(|(b, _)| b == t))
    {
        baseline.args(["--test", target]);
    }
    if cfg!(target_arch = "x86_64") {
        // This test deliberately requires a native V3 token. Give the interpreter
        // that virtual CPU; retain every roundtrip assertion. Run it first so a
        // failure cannot hide behind the much longer baseline suite.
        let mut raw = Command::new("cargo");
        raw.args([
            &toolchain,
            "miri",
            "test",
            "-p",
            "magetypes",
            "--features",
            "avx512",
            "--test",
            "raw_interop",
        ]);
        if let Ok(flags) = std::env::var("CARGO_ENCODED_RUSTFLAGS") {
            raw.env(
                "CARGO_ENCODED_RUSTFLAGS",
                format!("{flags}\x1f-Ctarget-cpu=x86-64-v3"),
            );
        } else {
            let flags = std::env::var("RUSTFLAGS").unwrap_or_default();
            raw.env("RUSTFLAGS", format!("{flags} -Ctarget-cpu=x86-64-v3"));
        }
        if !raw
            .status()
            .context("Failed to run native raw Miri tests")?
            .success()
        {
            bail!("Native raw Miri tests failed; see diagnostics above");
        }
        // The named raw test has already executed above. All other tests retain
        // the baseline virtual CPU. Non-x86 hosts run their entire native suite.
        baseline.args(["--", "--skip", "native_raw_roundtrips"]);
    }
    let started = std::time::Instant::now();
    let baseline = baseline
        .status()
        .context("Failed to run baseline Miri tests")?;
    if !baseline.success() {
        bail!("Baseline Miri tests failed; see diagnostics above");
    }
    println!(
        "  miri: baseline suite {:.0} s",
        started.elapsed().as_secs_f64()
    );
    for (bin, skips) in &partial {
        let started = std::time::Instant::now();
        let mut run = Command::new("cargo");
        run.args([
            &toolchain,
            "miri",
            "test",
            "-p",
            "magetypes",
            "--features",
            "avx512",
            "--test",
            bin,
            "--",
            "--exact",
        ]);
        for test in skips {
            run.args(["--skip", test]);
        }
        if !run
            .status()
            .with_context(|| format!("Failed to run Miri on {bin}"))?
            .success()
        {
            bail!("Miri tests in {bin} failed; see diagnostics above");
        }
        println!("  miri: {bin} {:.0} s", started.elapsed().as_secs_f64());
    }
    // Cover the configurable math contracts and high-precision powf path under Miri.
    let migrated = Command::new("cargo")
        .args([
            &toolchain,
            "miri",
            "test",
            "-p",
            "magemath",
            "--features",
            "std avx512",
            "--lib",
            "--test",
            "powf_highp",
            "--test",
            "expf_logf_contracts",
        ])
        .status()
        .context("Failed to run Magemath Miri contracts")?;
    if !migrated.success() {
        bail!("Magemath Miri contracts failed");
    }
    let started = std::time::Instant::now();
    // Cargo cannot mix --doc with explicit targets, so the doctests that the
    // former all-targets invocation covered get their own run.
    for (package, features) in [("magetypes", "avx512"), ("magemath", "std avx512")] {
        let doc = Command::new("cargo")
            .args([
                &toolchain,
                "miri",
                "test",
                "-p",
                package,
                "--features",
                features,
                "--doc",
            ])
            .status()
            .with_context(|| format!("Failed to run {package} Miri doctests"))?;
        if !doc.success() {
            bail!("{package} Miri doctests failed; see diagnostics above");
        }
    }
    println!("  miri: doctests {:.0} s", started.elapsed().as_secs_f64());
    println!("Miri tests passed for native raw interop, the baseline suite, and the doctests");
    Ok(())
}
